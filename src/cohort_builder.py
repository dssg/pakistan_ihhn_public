import logging
from itertools import chain
from typing import List, Dict, Any

import pandas as pd
from joblib import Parallel, delayed


from config.model_settings import CohortBuilderConfig
from src.preprocess import Preprocess
from src.utils.utils import get_data, write_to_db


class CohortBuilder:
    def __init__(
        self,
        date_col: str,
        table_name: str,
        entity_id_cols: List[str],
        filter_dict: Dict[str, Any],
        schema_name: str,
    ) -> None:
        self.date_col = date_col
        self.table_name = table_name
        self.entity_id_cols = entity_id_cols
        self.filter_dict = filter_dict
        self.schema_name = schema_name

    @classmethod
    def from_dataclass_config(cls, config: CohortBuilderConfig) -> "CohortBuilder":
        return cls(
            date_col=config.DATE_COL,
            table_name=config.TABLE_NAME,
            entity_id_cols=config.ENTITY_ID_COLS,
            filter_dict=config.FILTER_DICT,
            schema_name=config.SCHEMA_NAME,
        )

    def execute(self, train_validation_dict, schema_type, engine, table_name=None):
        self.get_schema_name(schema_type)

        if table_name is not None:
            # redefine if non-missing
            self.table_name = table_name

        filter_cols = ", ".join(
            set(list(chain.from_iterable(self.filter_dict.values())))
        )

        cohorts_df = pd.concat(
            Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
                delayed(self.cohort_builder)(
                    cohort_type, train_validation_dict, filter_cols
                )
                for cohort_type in train_validation_dict.keys()
            ),
            axis=0,
        ).reset_index(drop=True)

        filtered_cohorts_df = Preprocess.from_options(
            list(self.filter_dict.keys())
        ).execute(cohorts_df)

        self._results_to_db(engine, filtered_cohorts_df, filter_cols)

    def cohort_builder(
        self, cohort_type, train_validation_dict, filter_cols
    ) -> pd.DataFrame:
        """
        Retrieve coded er data data from train data.

        Ensures that dataframe always has columns mentioned in ENTITY_ID_COLUMNS
        even if dataframe is empty.

        Returns
        -------
        pd.DataFrame
            Cohort dataframe for er-visits
        """
        date_tup_list = list(train_validation_dict[f"{cohort_type}"])
        df_list = []
        for index, date_tuple in enumerate(date_tup_list):
            query = """SELECT DISTINCT {entity_id_cols},{date_col},{filter_cols}
                FROM {schema}.{table}
                WHERE {date_col}
                BETWEEN '{start_date}'
                AND '{end_date}';""".format(
                schema=self.schema_name,
                table=self.table_name,
                date_col=self.date_col,
                start_date=date_tuple[0],
                end_date=date_tuple[1],
                entity_id_cols=", ".join(self.entity_id_cols),
                filter_cols=filter_cols,
            )
            df = get_data(query)
            df["train_validation_set"] = index
            df["cohort"] = f"{index}_{date_tuple[0].date()}_{date_tuple[1].date()}"
            df["cohort_type"] = f"{cohort_type}"
            if df.empty:
                logging.info(
                    f"""No er-visit data found for
                     {date_tuple[0].date()}_{date_tuple[1].date()} time window"""
                )

            df_list.append(df)
        cohort_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return cohort_df

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _results_to_db(self, engine, filtered_cohorts_df, filter_cols):
        """Write model results to the database for all cohorts"""
        filtered_cohorts_df_no_features = filtered_cohorts_df.drop(
            labels=list(filter_cols.split(", ")), axis=1
        )
        write_to_db(
            filtered_cohorts_df_no_features,
            engine,
            "cohorts",
            self.schema_name,
            "replace",
        )

    def filter_priority_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows which contain non-priority categories
        """
        priority_cats = self._list_priority_categories()

        return df.assign(
            priority_cats=(
                df.category.apply(
                    lambda cat: any(
                        str_.lower() in cat[1:-1].split(",") for str_ in priority_cats
                    )
                )
            )
        ).query("priority_cats == True")

    def _list_priority_categories(self):
        """Getting priority codes as a list"""
        priority_cat_query = """select icd_10_cm from {schema_name}.{table}
        where count >= {no_of_occurrences};""".format(
            schema_name=self.priority_schema_name,
            table=self.priority_table_name,
            no_of_occurrences=self.no_of_occurrences,
        )
        priority_cats = get_data(priority_cat_query)
        return list(priority_cats["icd_10_cm"])
