import pandas as pd
import string
from tqdm import tqdm
from joblib import Parallel, delayed
from ast import literal_eval
import logging

from src.baselines.run_baseline_base import BaseFlow
from src.utils.utils import get_data, get_official_codes, write_to_db
from config.model_settings import BaselineConfig


class BaselineDescOverlap(BaseFlow):
    def __init__(self, metric: str):
        self.distance_metric = metric
        self.as_of_date = None
        self.official_code_data = get_official_codes()
        super().__init__(
            BaselineConfig.CONSTRAINT,
            BaselineConfig.CATEGORY_COL,
            BaselineConfig.SCHEMA_NAME,
            BaselineConfig.TEXT_COLS,
        )

    def execute(self, engine):
        """For each given baseline name, generate the precision and recall
        for each contraint number, and write these to a table.
        """
        tqdm.pandas()

        df_list = []

        for text_col in self.text_cols:
            category_frequency = self.get_freq()
            text_col_df = pd.concat(
                Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
                    delayed(self.execute_for_constraint)(
                        category_frequency,
                        text_col,
                        constraint,
                    )
                    for constraint in self.constraint
                )
            )
            df_list.append(text_col_df)

        full_df = pd.concat(df_list, axis=0)
        write_to_db(
            full_df,
            engine,
            f"baseline_{self.distance_metric}_distance_similarity",
            self.schema_name,
            "replace",
        )

    def execute_for_constraint(
        self,
        df,
        text_col,
        constraint,
    ):
        """For a given constraint, compute filter only the top codes
        within a given constraint that are selected based on each filter
        distance.

        Parameters
        ----------
        df: the data we are using to assign 'probable' codes
        text_col: the column containing the text used for the baseline
        constraint: the number of 'probable' codes to retrieve

        """
        logging.info(
            f"Getting distances for column: {text_col}, and constraint: {constraint}"
        )
        df_list = []
        for index, row in tqdm(df.iterrows()):
            df_list.append(self.filter_distance(row, text_col, constraint))
            constrain_df = pd.concat(df_list, axis=0)

        return constrain_df

    def filter_distance(self, row, text_col, constraint):
        """filters each of the data by the number of 'probable' categories
        based on a constraint using each of the different 'distance_metrics'

        Parameters
        ----------
        row: the row data
        text_col: column containing the text we will use to cancluate 'distance'
        constraint: the number of rows to filter.
        """
        dist_dict = dict(
            map(
                lambda long_desc: (
                    long_desc,
                    self._calculate_desc_overlap(
                        row[f"{text_col}"].translate(
                            str.maketrans("", "", string.punctuation)
                        ),
                        long_desc,
                    ),
                ),
                self.official_code_data["description_long"],
            )
        )
        select_df = self.official_code_data.loc[
            self.official_code_data["description_long"].isin(
                self._get_n_highest_similarity(dist_dict, constraint)
            )
        ]

        top_k_codes = list(select_df["icd_10_cm"].drop_duplicates().str.lower())
        precision, recall = self._precision_recall_from_lists(
            top_k_codes, literal_eval(row["category"].lower())
        )

        return pd.DataFrame.from_dict(
            {
                "text_column": text_col,
                "constraint_val": constraint,
                "top_dist_dict": [str(select_df.to_dict("records"))],
                "top_icd_matches": [top_k_codes],
                "actual_codes": str(row["category"]),
                "precision": precision,
                "recall": recall,
            }
        )

    def _calculate_desc_overlap(self, s1, s2):
        """Calculating the percentage of overlap between two strings"""
        overlap = set(s1.split()) & set(s2.split())
        return (len(overlap) / len(s1)) * 100

    def get_freq(
        self,
    ):
        if self.as_of_date is not None:
            cat_frequency_query = """select {frequency_cols}, {category}, count(*)
            from {schema}.train where triage_datetime < {date} and {frequency_cols}
            IS NOT NULL group by ({frequency_cols}, {category});""".format(
                frequency_cols=", ".join(self.text_cols),
                date=self.as_of_date,
                category=self.category_col,
                schema=self.schema_name,
            )
        else:
            cat_frequency_query = """select {frequency_cols}, {category}, count(*)
            from {schema}.train WHERE {frequency_cols} IS NOT NULL
            group by ({frequency_cols}, {category});""".format(
                frequency_cols=", ".join(self.text_cols),
                category=self.category_col,
                schema=self.schema_name,
            )
        return get_data(cat_frequency_query).sort_values(by="count", ascending=False)
