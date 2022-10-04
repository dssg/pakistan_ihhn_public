from tqdm import tqdm
import string
from math import floor
import logging
import pandas as pd
from joblib import Parallel, delayed

from src.utils.utils import get_data, get_official_codes, write_to_db
from src.baselines.run_baseline_base import BaseFlow
from config.model_settings import BaselineConfig


class BaselineJaro(BaseFlow):
    def __init__(self, metric: str):
        self.distance_metric = metric
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
            raw_text = self.get_text_data()
            raw_text_no_dupe = raw_text.drop_duplicates()
            raw_text_no_dupe = self.list_cat_col(raw_text_no_dupe)
            text_col_df = pd.concat(
                Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
                    delayed(self.execute_for_constraint)(
                        raw_text_no_dupe, text_col, constraint
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
                    self._calculate_jaro(
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
            top_k_codes, [x.lower().replace("'", "") for x in row["category"]]
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

    def _calculate_jaro(self, s1, s2):
        """For each row in the data, compute the Jaccard distance
        between the text column and the official description field.
        Then take the categories given a constraint that returns the maximum
        value of the Jaro distance as the "'predicted' category"""
        # If the s are equal
        if s1 == s2:
            return 1.0

        # Length of two s
        len1 = len(s1)
        len2 = len(s2)

        # Maximum distance upto which matching
        # is allowed
        max_dist = floor(max(len1, len2) / 2) - 1

        # Count of matches
        match = 0

        # Hash for matches
        hash_s1 = [0] * len(s1)
        hash_s2 = [0] * len(s2)

        # Traverse through the first
        for i in range(len1):

            # Check if there is any matches
            for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):

                # If there is a match
                if s1[i] == s2[j] and hash_s2[j] == 0:
                    hash_s1[i] = 1
                    hash_s2[j] = 1
                    match += 1
                    break

        # If there is no match
        if match == 0:
            return 0.0

        # Number of transpositions
        t = 0
        point = 0

        # Count number of occurrences
        # where two characters match but
        # there is a third matched character
        # in between the indices
        for i in range(len1):
            if hash_s1[i]:

                # Find the next matched character
                # in second
                while hash_s2[point] == 0:
                    point += 1

                if s1[i] != s2[point]:
                    t += 1
                point += 1
        t = t // 2

        # Return the Jaro Similarity
        return (match / len1 + match / len2 + (match - t) / match) / 3.0

    def get_text_data(self):
        text_query = """select ed_dx,
        category from {schema}.train;""".format(
            schema=self.schema_name
        )
        raw_text = get_data(text_query)
        return raw_text

    def list_cat_col(self, df):
        df[f"{self.category_col}"] = df[f"{self.category_col}"].apply(
            lambda x: x[1:-1].upper().split(",")
        )
        return df

    def _get_frequency(self):
        frequency_query = """select {frequency_cols}, count (*)
        from {schema}.train  group by {frequency_cols};""".format(
            frequency_cols=", ".join(self.text_cols),
            schema=self.schema_name,
        )
        return get_data(frequency_query).sort_values(by="count", ascending=False)
