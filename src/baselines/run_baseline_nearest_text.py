from joblib import Parallel, delayed
import logging
import pandas as pd
from tqdm import tqdm
import string

from src.utils.utils import get_data, get_official_codes, write_to_db
from src.baselines.run_baseline_base import BaseFlow
from config.model_settings import BaselineConfig


class BaselineNearestText(BaseFlow):
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
            category_frequency_explode = self._explode_cat_col(category_frequency)

            text_col_df = pd.concat(
                Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
                    delayed(self.execute_for_constraint)(
                        category_frequency_explode,
                        text_col,
                        constraint,
                    )
                    for constraint in self.constraint
                )
            )
            df_list.append(text_col_df)

        full_df = pd.concat([text_col_df], axis=0)
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
        category_frequency = self.get_freq()
        category_frequency_explode = self._explode_cat_col(category_frequency)

        categories_merged, text_appearances = self._shortlist_overlapping_text(
            row, category_frequency_explode, text_col
        )

        dist_dict = dict(
            map(
                lambda text: (
                    text,
                    self._calculate_nearest_diagnosis(
                        row[f"{text_col}"].translate(
                            str.maketrans("", "", string.punctuation)
                        ),
                        text,
                    ),
                ),
                text_appearances[f"{text_col}"],
            )
        )
        select_df = self.get_constraint_based_matches(
            categories_merged, text_col, dist_dict, constraint
        )
        top_k_codes = list(select_df["icd_10_cm"].drop_duplicates())
        select_df[f"{self.category_col}"] = select_df[f"{self.category_col}"].apply(
            lambda x: x[1:-1].upper().split(",")
        )
        if select_df.empty:
            precision, recall = (0, 0)
        else:
            precision, recall = self.precision_recall_from_lists(
                top_k_codes, row["category"]
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

    def get_freq(
        self,
    ):
        """Get the frequency of category appearances"""
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

    def get_constraint_based_matches(
        self, categories_merged, text_col, dist_dict, constraint
    ):
        """filtering the df with the dictionary of similarity scored given
        a constraint"""
        return categories_merged.loc[
            categories_merged[f"{text_col}"].isin(
                self._get_n_highest_similarity(dist_dict, constraint)
            )
        ].head(constraint)

    def _shortlist_overlapping_text(self, row, exploded_df, text_col):
        """Merge the categories with the text and calculate the count
        of each text appearance."""
        categories_merged = exploded_df.merge(
            self.official_code_data,
            how="left",
            left_on="category",
            right_on="icd_10_cm",
        )
        text_overlap = pd.DataFrame(
            categories_merged[f"{text_col}"][
                (categories_merged["count"] > 1)
                & (categories_merged[f"{text_col}"].notnull())
            ]
        )
        text_appearances = text_overlap[
            text_overlap[f"{text_col}"].str.contains(row[f"{text_col}"])
        ]
        return categories_merged, text_appearances

    def _calculate_nearest_diagnosis(self, s1, s2):
        """Calculating the percentage instersection between two strings"""
        overlap = set(s1).intersection(set(s2))
        return (len(overlap) / len(s2)) * 100

    def _explode_cat_col(self, df):
        df[f"{self.category_col}"] = df[f"{self.category_col}"].apply(
            lambda x: x[1:-1].upper().split(",")
        )
        return df.explode(f"{self.category_col}")

    def precision_recall_from_lists(self, top_k_codes, actual_codes):
        code_intersect = len(set(actual_codes.split(",")) & set(top_k_codes))
        recall = code_intersect / len(actual_codes)
        precision = code_intersect / len(top_k_codes)
        return precision, recall
