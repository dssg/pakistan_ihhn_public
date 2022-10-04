from config.model_settings import FeatureGeneratorConfig
from src.preprocessing.text_preprocess import PreprocessText
from typing import Any, Optional, List
from abc import ABC, abstractmethod
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from setup_environment import db_dict, get_dbengine
from src.utils.utils import get_data
from joblib import dump
import pandas as pd
import os
from numpy import float64
import logging

logging.basicConfig(level=logging.INFO)


class FeatureGeneratorBase(ABC):
    def __init__(self, schema_name, table_name, text_features_path):
        self.schema_name = schema_name
        self.table_name = table_name
        self.text_features_path = text_features_path

    @abstractmethod
    def execute_features(self, *args: Any) -> pd.DataFrame:
        """Execute generating features"""

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name


class FeatureGeneratorSklearn(FeatureGeneratorBase):
    def __init__(
        self,
        id_column_list,
        text_column_list,
        cat_column_list,
        cont_column_list,
        special_column_list,
        ignore_punctuation,
        process_functions,
        text_features_path,
        all_model_features: Optional[List[str]] = None,
    ):
        self.id_column_list = id_column_list
        self.text_column_list = text_column_list
        self.cat_column_list = cat_column_list
        self.cont_column_list = cont_column_list
        self.special_column_list = special_column_list
        self.all_model_features = all_model_features
        self.ignore_punctuation = ignore_punctuation
        self.process_functions = process_functions
        self.text_features_path = text_features_path
        super().__init__(
            FeatureGeneratorConfig.SCHEMA_NAME,
            FeatureGeneratorConfig.TABLE_NAME,
            text_features_path,
        )
        engine = get_dbengine(**db_dict)
        self.engine = engine

    @classmethod
    def from_dataclass_config(
        cls,
        config: FeatureGeneratorConfig,
        text_features_path,
    ) -> "FeatureGeneratorSklearn":
        return cls(
            id_column_list=config.ID_COLUMN_LIST,
            text_column_list=config.TEXT_COLUMN_LIST,
            cat_column_list=config.CAT_COLUMN_LIST,
            cont_column_list=config.CONT_COLUMN_LIST,
            special_column_list=config.SPECIAL_COLUMN_LIST,
            all_model_features=config.ALL_MODEL_FEATURES,
            ignore_punctuation=config.IGNORE_PUNCTUATION,
            process_functions=config.PROCESS_FUNCTIONS,
            text_features_path=text_features_path,
        )

    def execute_features(
        self, train_validation_set, run_date, schema_type, table_name=None
    ):
        self.get_schema_name(schema_type)

        if table_name is not None:
            self.table_name = table_name

        df = self._query_features(train_validation_set)
        df_train, df_valid = (
            df.pipe(self._change_to_categorical_type)
            .pipe(self._generate_cont_columns, train_validation_set, run_date)
            .pipe(self._generate_special_columns, train_validation_set, run_date)
            .drop(
                ["new_mr", "new_er", "triage_datetime"] + self.text_column_list, axis=1
            )
            .pipe(self._generate_ohe_columns, train_validation_set, run_date)
        )

        self._generate_text_columns(df, train_validation_set, run_date)
        feature_train_id, feature_valid_id = self._get_uniqueids(df)

        return df_train, df_valid, feature_train_id, feature_valid_id

    def _query_features(self, train_validation_set):
        query = """select new_er, new_mr, c.triage_datetime, {id_cols},
            {columns}
            from {schema}.cohorts c
            left join {schema}.{table} as features_and_labels
                on c.unique_id = features_and_labels.unique_id
                where train_validation_set={i};""".format(
            schema=self.schema_name,
            table=self.table_name,
            id_cols=",".join(["c." + x for x in self.id_column_list]),
            columns=",".join(
                ["features_and_labels." + x for x in self.all_model_features]
            ),
            i=train_validation_set,
        )

        return get_data(query)

    def _query_investigations(self):
        query = """select ri.new_mr, ri.new_er, ri.lab_test,
                lab_order_date, lab_result_date,t.category
                from raw.investigations ri
                inner join {schema}.{table} t
                on ri.new_er = t.new_er and ri.new_mr = t.new_mr
                order by lab_test;""".format(
            schema=self.schema_name, table=self.table_name
        )

        return get_data(query)

    def _change_to_categorical_type(self, df: pd.DataFrame) -> pd.DataFrame:
        for cat_col in self.cat_column_list:
            df.loc[:, cat_col] = df[cat_col].astype("category")

        return df

    def _get_uniqueids(self, df):
        train_ids = df.loc[
            df["cohort_type"] == "training", ["unique_id", "cohort", "cohort_type"]
        ]
        valid_ids = df.loc[
            df["cohort_type"] == "validation", ["unique_id", "cohort", "cohort_type"]
        ]
        return train_ids, valid_ids

    def _split_train_valid(self, df):
        df_train = df.loc[df["cohort_type"] == "training"]
        df_valid = df.loc[df["cohort_type"] == "validation"]
        return df_train, df_valid

    def _generate_ohe_columns(self, df, train_validation_set, run_date):
        """
        This function generates one hot encoded variables
        from the categorical variables
        and renames the columns to the unique categories
        """
        enc = OneHotEncoder(handle_unknown="ignore")
        encode_df = pd.DataFrame(enc.fit_transform(df[self.cat_column_list]).toarray())
        # rename columns to original columns.

        cat_column_names = [item for items in enc.categories_ for item in items]
        encode_df.columns = cat_column_names

        assert len(df) == len(encode_df)
        encod_orig_df = pd.concat([df.reset_index(drop=True), encode_df], axis=1).drop(
            self.cat_column_list, axis=1
        )

        self._feature_arrays_to_db(
            cat_column_names, None, train_validation_set, run_date
        )

        df_train, df_valid = self._split_train_valid(encod_orig_df)
        return df_train, df_valid

    def _generate_cont_columns(self, cont_df, train_validation_set, run_date):
        """
        This function checks whether the columns are numeric,
        cleans braces from the columns
        and casts them as float
        """
        for i in self.cont_column_list:
            if cont_df[i].dtypes != "float64":
                logging.error("Not a numerical variable")
                cont_df[i] = cont_df[i].apply(lambda x: x[1:-1].split(","))
                cont_df = (
                    cont_df.applymap(lambda x: x[0] if isinstance(x, list) else x)
                    .replace("NULL", 0)
                    .astype(float64)
                )
            else:
                cont_df[i] = cont_df[i]
                cont_df = cont_df.applymap(
                    lambda x: x[0] if isinstance(x, list) else x
                ).replace("NULL", 0)

        logging.info(
            f"""Writing cont to the database for
            train_validation_set: {train_validation_set}"""
        )
        self._feature_arrays_to_db(
            self.cont_column_list, None, train_validation_set, run_date
        )

        return cont_df

    def _generate_text_columns(self, df, train_validation_set, run_date):
        # cleaning
        df_with_placeholder_text = self._features_with_placeholder_text(df)

        # tfidf
        self._vectorise_feature_cols(
            df, df_with_placeholder_text, train_validation_set, run_date
        )

    def _generate_special_columns(self, df, train_validation_set, run_date):
        # extra arguments: as_of_date, window
        """This generates the three special features"""

        # special_feature_df = df['new_mr','new_er']
        for i in range(len(self.special_column_list)):
            if self.special_column_list[i] == "past_visit_count":
                output_df = get_data(
                    """select unique_id, count(*)
                    over (partition by new_mr order by triage_datetime asc rows
                    between unbounded preceding and
                    current row exclude current row)
                    as num_visits from {schema}.{table};""".format(
                        schema=self.schema_name, table=self.table_name
                    )
                )
                df = pd.merge(df, output_df, on=["unique_id"], how="left")
            elif self.special_column_list[i] == "season":
                df["season_"] = df.triage_datetime.map(self._get_season)
                df["season"] = df["season_"]
                df["season"].mask(df.season_ == "spring", 1, inplace=True)
                df["season"].mask(df.season_ == "summer", 2, inplace=True)
                df["season"].mask(df.season_ == "autumn", 3, inplace=True)
                df["season"].mask(df.season_ == "winter", 4, inplace=True)
                df["season"] = df["season"].astype(int)
                df.drop("season_", axis=1, inplace=True)

        self._feature_arrays_to_db(
            self.special_column_list, None, train_validation_set, run_date
        )
        return df

    def _features_with_placeholder_text(self, features_data):
        """Fill missing valuesin string data with placeholder text"""
        features_data[self.text_column_list] = (
            features_data[self.text_column_list]
            .replace(",", " ")
            .fillna("PLACEHOLDER_TEXT")
            .apply(lambda x: x.str.lower())
        )
        return features_data

    def _get_vectoriser(self):
        """Get count vectorizer"""
        return TfidfVectorizer(min_df=2)

    def _vectorise_feature_cols(
        self, features_data, df_with_placeholder_text, train_validation_set, run_date
    ):
        vec = self._get_vectoriser()

        for col in self.text_column_list:
            logging.info(f"Generating vectorized features for {col}")
            self._vectorise_one_col(
                col, vec, df_with_placeholder_text, train_validation_set, run_date
            )

        return features_data.drop(self.text_column_list, axis=1)

    def _vectorise_one_col(self, col, vec, df, train_validation_set, run_date):
        X = vec.fit_transform(df[col])

        # split into train, valid
        for type in ["training", "validation"]:
            X_filt = X[df[df["cohort_type"] == type].index.values]

            self._write_features_as_pickle(
                X_filt, col, train_validation_set, type, run_date
            )
        self._feature_arrays_to_db(
            vec.get_feature_names_out().tolist(), col, train_validation_set, run_date
        )

    def _feature_arrays_to_db(self, features, col, train_validation_set, run_date):
        with self.engine.connect() as conn:
            conn.execute("""SET ROLE "pakistan-ihhn-role" """)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS {schema}.feature_arrays
                            (col text,
                            features text[],
                            train_validation_set int,
                            run_date timestamp)""".format(
                    schema=self.schema_name
                )
            )
            q = text(
                """insert into {schema}.feature_arrays
                (col, features, train_validation_set, run_date)
                    values (:c, :f, :t, :r);""".format(
                    schema=self.schema_name
                )
            )
            conn.execute(
                q,
                c=col,
                f=features,
                t=int(train_validation_set),
                r=run_date,
            )

    def _preprocess_text_df(self, df):
        """Process all text columns using text processor"""
        text_preprocessor = PreprocessText()

        for i in self.text_column_list:
            df[i + "_qm"] = df[i].apply(text_preprocessor.count_question_marks)
            df[i] = self._process_text(df, i, text_preprocessor)
        return df

    def _process_text(self, df, column_name, text_preprocessor, ignore_punct=[]):
        """Process text for an individual column"""
        if column_name in self.ignore_punctuation.keys():
            ignore_punct = self.ignore_punctuation[column_name]

        process_func = self.process_functions[column_name]

        return text_preprocessor.run_preprocessing_steps(
            df[column_name],
            process_func,
            ignore_punct,
        )

    def _write_features_as_pickle(
        self, X, col, train_validation_set, cohort_type, run_date
    ):
        # save as pickle
        filename = "_".join(
            [
                col,
                str(train_validation_set),
                cohort_type,
                run_date.strftime("%Y%m%d_%H%M%S%f"),
            ]
        )
        dump(
            X,
            os.path.join(self.text_features_path, filename + ".joblib"),
        )

    def _get_season(self, date_column):
        year = str(date_column.year)
        seasons = {
            "spring": pd.date_range(start="21/03/" + year, end="20/06/" + year),
            "summer": pd.date_range(start="21/06/" + year, end="22/09/" + year),
            "autumn": pd.date_range(start="23/09/" + year, end="20/12/" + year),
        }
        if date_column in seasons["spring"]:
            return "spring"
        if date_column in seasons["summer"]:
            return "summer"
        if date_column in seasons["autumn"]:
            return "autumn"
        else:
            return "winter"
