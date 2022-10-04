import logging
import string
import os
import psutil
import itertools
from joblib import dump, Parallel, delayed
from typing import Optional, List
from sqlalchemy import text

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from config.model_settings import (
    FeatureGeneratorConfig,
    ModelTrainerConfig,
    RetrainingConfig,
)


import xgboost as xgb

from setup_environment import db_dict, get_dbengine
from config.model_settings import HyperparamConfig

logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(
        self,
        schema_name,
        model_names_list: List,
        random_state: int,
        id_cols_to_remove: List,
        special_column_list: List,
        best_model: str,
        best_model_hyperparams: List,
        all_model_features: Optional[List[str]] = None,
    ) -> None:
        self.schema_name = schema_name
        self.model_names_list = model_names_list
        self.random_state = random_state
        self.id_cols_to_remove = id_cols_to_remove
        self.special_column_list = special_column_list
        self.best_model = best_model
        self.best_model_hyperparams = best_model_hyperparams
        self.all_model_features = all_model_features
        engine = get_dbengine(**db_dict)
        self.engine = engine

    @classmethod
    def from_dataclass_config(cls, config: ModelTrainerConfig) -> "ModelTrainer":
        return cls(
            schema_name=config.SCHEMA_NAME,
            model_names_list=config.MODEL_NAMES_LIST,
            random_state=config.RANDOM_STATE,
            id_cols_to_remove=config.ID_COLS_TO_REMOVE,
            all_model_features=FeatureGeneratorConfig().ALL_MODEL_FEATURES,
            special_column_list=FeatureGeneratorConfig().SPECIAL_COLUMN_LIST,
            best_model=RetrainingConfig().BEST_MODEL,
            best_model_hyperparams=RetrainingConfig().BEST_MODEL_HYPERPARAMS,
        )

    def train_best_model(
        self, cohort_id, X_train, Y_train, model_path, run_date, schema_type
    ):
        self.get_schema_name(schema_type)
        return self.execute_one_model(
            cohort_id,
            self.best_model,
            X_train,
            Y_train,
            model_path,
            run_date,
            self.best_model_hyperparams,
            schema_type,
        )

    def train_all_models(
        self, cohort_id, X_train, Y_train, model_path, run_date, schema_type
    ):
        """Loop through all models and save each trained model to server"""
        self.get_schema_name(schema_type)

        logging.info("Training all models")
        logging.info(Y_train.head())
        model_output = []
        if schema_type == "dev":
            model_output += self._parallelize_dtc(
                cohort_id,
                "DTC",
                X_train,
                Y_train,
                model_path,
                run_date,
                [(5,)],
                schema_type,
            )
        else:
            for model in self.model_names_list:

                model_params = HyperparamConfig.MODEL_HYPERPARAMS[model]
                hyperparams = itertools.product(*list(model_params.values()))

                if model == "DTC":
                    model_output += self._parallelize_dtc(
                        cohort_id,
                        model,
                        X_train,
                        Y_train,
                        model_path,
                        run_date,
                        hyperparams,
                        schema_type,
                    )
                else:
                    for hp in hyperparams:
                        model_output += [
                            self.execute_one_model(
                                cohort_id,
                                model,
                                X_train,
                                Y_train,
                                model_path,
                                run_date,
                                hp,
                                schema_type,
                            )
                        ]

        return model_output

    def execute_one_model(
        self,
        cohort_id,
        model_name,
        X_train,
        Y_train,
        model_path,
        run_date,
        hp,
        schema_type,
    ):
        """This is a docstring that describes the overall function:
        Arguments
        ---------
            model_id : str
                      A model_id that identifies the model
                      `model_name` and `hyperparameters`
            train_model: model
                     model instance to use in training
            model_name: str
                      A name identifying the algorithm

        Returns
        -------
            train_model: model
                     model instance to use in training
            mlb: MultiLabelBinarizer()
                      A label binariser to generate category matrix.
            model_id: str
                      A model_id that identifies the model based on the `cohort_id` and
                      `hyperparameters`"""
        logging.info(f"Training model {model_name} with hyperparameters {hp}")
        # split by labels and features
        text_clf = self.get_train_pipeline(model_name, hp)
        logging.info("Fitting model")
        logging.info(f"Current memory usage: {psutil.virtual_memory()}")
        logging.info(f"Shape of X data: {X_train.shape}")
        logging.info(f"Shape of Y data: {Y_train.shape}")
        train_model = self.fit_model(text_clf, X_train, Y_train)

        hp_id = self._build_hyperparameters_id(model_name, hp)
        # get model_id
        model_id, model_set = self._generate_model_id(
            train_model, model_name, cohort_id, hp_id, schema_type
        )

        # write model to server
        self._save_trained_model(train_model, model_path, model_id, run_date)

        # write model metadata
        self._generate_model_metadata(
            model_id,
            model_set,
            run_date,
            list(Y_train.columns),
            hp_id,
        )
        return model_id, model_name, cohort_id

    def get_train_pipeline(self, model_name, hp):
        """
        Create pipeline based on model name and instantiation

        Arguments
        ---------
            model_name : dict

        Returns
        -------
            pipeline
            model_name : str
        """
        text_clf = Pipeline(
            [
                # ("vect", CountVectorizer()),
                # ("tfidf", TfidfTransformer()),
                (f"{model_name}", self._get_model(model_name, hp)),
            ]
        )
        return text_clf

    def fit_model(self, text_clf, X_train, y_train):
        return text_clf.fit(X_train, y_train)

    def _drop_id_label_cols(self, df, mlb_categories):
        return df.drop(
            list(mlb_categories) + self.id_cols_to_remove,
            axis=1,
        )

    def _remove_punctuation(self, text_column):
        free_text = "".join([i for i in text_column if i not in string.punctuation])
        return free_text

    def _get_model(self, model_name, hp):
        if model_name == "DTC":
            return DecisionTreeClassifier(
                max_depth=hp[0], random_state=self.random_state
            )
        elif model_name == "RFC":
            return RandomForestClassifier(
                n_jobs=-3,
                n_estimators=hp[0],
                max_depth=hp[1],
                random_state=self.random_state,
            )
        elif model_name == "XGB":
            return xgb.XGBClassifier(
                n_jobs=-3, n_estimators=hp[0], max_depth=hp[1], learning_rate=hp[2]
            )
        elif model_name == "MNB":
            model = MultinomialNB(alpha=hp[0])
            return MultiOutputClassifier(model)
        elif model_name == "MLR":
            model = LogisticRegression(
                penalty=hp[0], C=hp[1], solver=hp[2], max_iter=hp[3]
            )
            return MultiOutputClassifier(model)
        else:
            logging.info(f"Model name {model_name} not exist")

    def _generate_model_id(
        self, train_model, model_name, cohort_id, hp_id, schema_type
    ):
        """Generate model id based on model name, cohort ID"""
        model_name = train_model.named_steps[f"{model_name}"].__class__.__name__
        model_id = "_".join(
            list(
                map(
                    self._clean_for_model_id,
                    [model_name, hp_id, cohort_id],
                    [None, None, schema_type],
                )
            )
        )
        model_set = "_".join(
            list(
                map(
                    self._clean_for_model_id,
                    [model_name, hp_id],
                    [None, None, schema_type],
                )
            )
        )
        return model_id, model_set

    def _clean_for_model_id(self, word, schema_type):
        """Clean word arguments for model id and concatenate together."""
        word = str(word).replace("-", "")
        if schema_type == "dev":
            return f"{word.lower()}_dev"
        else:
            return word.lower()

    def _save_trained_model(self, train_model, model_path, model_id, run_date):
        filename = (
            "_".join([model_id, run_date.strftime("%Y%m%d_%H%M%S%f")]) + ".joblib"
        )
        dump(train_model, os.path.join(model_path, filename))

    def _build_hyperparameters_id(self, model_name, hp):
        if model_name == "DTC":
            return f"max_depth{hp[0]}"
        elif model_name == "RFC":
            return f"n_estimators{hp[0]}_max_depth{hp[1]}"
        elif model_name == "MNB":
            return f"alpha{hp[0]}"
        elif model_name == "MLR":
            return f"penalty{hp[0]}_C{hp[1]}"

    def _generate_model_metadata(
        self,
        model_id,
        model_set,
        run_date,
        labels,
        hp_id,
    ):
        with self.engine.connect() as conn:
            conn.execute("""SET ROLE "pakistan-ihhn-role" """)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS {schema}.model_metadata
                            (model_id text,
                            model_set text,
                            features varchar[],
                            labels varchar[],
                            hyperparameters varchar,
                            run_date timestamp)""".format(
                    schema=self.schema_name
                )
            )

        with self.engine.connect() as conn:
            logging.info("Inserting model information to database")
            q = text(
                """insert into {schema}.model_metadata
                (model_id, model_set, features, labels, hyperparameters, run_date)
                    values (:m1, :m2, :f, :l, :h, :r);""".format(
                    schema=self.schema_name
                )
            )
            conn.execute(
                q,
                m1=model_id,
                m2=model_set,
                f=self.all_model_features + self.special_column_list,
                l=labels,
                h=hp_id,
                r=run_date,
            )

    def _parallelize_dtc(
        self,
        cohort_id,
        model,
        X_train,
        Y_train,
        model_path,
        run_date,
        hyperparams,
        schema_type,
    ):
        return Parallel(n_jobs=-2, backend="threading")(
            delayed(self.execute_one_model)(
                cohort_id,
                model,
                X_train,
                Y_train,
                model_path,
                run_date,
                hp,
                schema_type,
            )
            for hp in hyperparams
        )

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name
