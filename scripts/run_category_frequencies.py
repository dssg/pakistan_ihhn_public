import numpy as np
import sys
from tqdm import tqdm
from collections import Counter
import pandas as pd

from src.utils.utils import parallel_apply, get_data


class CategoryFrequencies:
    def __init__(self, cat_cols=["icd_10_cm", "count"]):
        self.cat_cols = cat_cols

    def execute(self, folder_path):
        """
        1. length of list where category in `icd_10_cm` is in `categories`
        2. the key if a dictionary as the `ed_dx` value of the row in which
        the `category` came from

        """
        tqdm.pandas()

        raw_labels = self.get_category_data()
        raw_labels_no_dupe = raw_labels.drop_duplicates()
        official_code_data = self.get_official_codes()
        category_list = np.concatenate(
            parallel_apply(raw_labels, self.get_category_frequencies, axis=1)
        )
        df_categories = self.create_df_from_array(category_list)

        df_categories.columns = self.cat_cols
        official_code_data["ed_dx_json"] = official_code_data.progress_apply(
            lambda row: self.get_ed_dx_match(row, raw_labels_no_dupe), axis=1
        )
        df_official_code_frequencies_in_data = df_categories.merge(
            official_code_data, how="left", on="icd_10_cm"
        )
        df_official_code_frequencies_in_data.to_csv(
            f"{folder_path}/official_codes_with_ed_dx_json.csv"
        )

    def get_category_data(self):
        category_query = """select category, ed_dx from model_output.train;"""
        raw_labels = get_data(category_query)
        return raw_labels

    def get_official_codes(self):
        official_code_query = """
        select icd_10_cm, description_long
        from raw.icd10cm_order_2023 where LENGTH(icd_10_cm) = 3;"""

        official_code_data = get_data(official_code_query)
        return official_code_data

    def get_category_frequencies(self, row):
        categories = filter(
            lambda cat: len(cat) == 3, (row["category"].strip("{}").upper().split(","))
        )
        return [*categories]

    def get_ed_dx_match(self, row, raw_labels):
        categories_ed_dx = filter(
            lambda raw_cat: self._match_category(
                row["icd_10_cm"], raw_cat[0].strip("{}").upper().split(",")
            )
            == True,  # noqa: E712
            zip(
                raw_labels["category"],
                raw_labels["ed_dx"],
            ),
        )

        return dict([*categories_ed_dx])

    def create_df_from_array(self, category_list):
        dict = Counter(category_list)
        return pd.DataFrame([dict]).transpose().reset_index()

    def _match_category(self, row_cat, raw_cat):
        return row_cat in raw_cat


if __name__ == "__main__":
    folder_path = sys.argv[1]
    CategoryFrequencies().execute(folder_path)
