import logging
import re
import string
from typing import List

from config.model_settings import ProcessCodesConfig
from src.utils.utils import execute_hash, get_data, parallel_apply, write_to_db

logging.basicConfig(level=logging.INFO)


class ProcessCodes:
    def __init__(
        self,
        table_name: str,
        schema_name: str,
        columns_to_keep: List,
    ):
        self.table_name = table_name
        self.schema_name = schema_name
        self.columns_to_keep = columns_to_keep

    @classmethod
    def from_dataclass_config(cls, config: ProcessCodesConfig) -> "ProcessCodes":

        return cls(
            table_name=config.TABLE_NAME,
            schema_name=config.SCHEMA_NAME,
            columns_to_keep=config.COLUMNS_TO_KEEP,
        )

    def execute(self, engine):
        """Execute function called in main script
        Args:
            engine: psql engine to connect to database
        """
        query = """SELECT ed_dx, hopi_ as hopi, age_years,
            triagecomplaint, STRING_AGG("code", ',')
            as code FROM raw.{table} GROUP BY ed_dx,
            hopi_, age_years, triagecomplaint;""".format(
            table=self.table_name
        )

        df = get_data(query)
        self.get_official_codes()

        df_clean = self.clean_df(df)
        df_no_whitespace = self.remove_whitespace_and_null_codes(df_clean)
        df_clean_codes = parallel_apply(
            df_no_whitespace, self.execute_for_row, n_jobs=-1, axis=1
        )
        df_clean_codes = execute_hash(df_clean_codes, "ed_dx")
        df_clean_codes_no_empty_string = df_clean_codes.drop(
            labels=list(self.remove_empty_string_codes(df_clean_codes))
        ).reset_index(drop=True)

        write_to_db(
            df_clean_codes_no_empty_string[self.columns_to_keep],
            engine,
            f"{self.table_name}",
            self.schema_name,
            "replace",
        )
        logging.info("Finished codes data")

    def execute_for_row(self, row):
        """Executes cleaning private functions row-wise on code cells
        Args:
            row: a row of the dataframe
        Returns:
            row with processing complete
        """
        row["code"] = self._separate_codes(row["code"])
        row["category"] = list(self._get_category(row["code"]))
        row["category"] = self._separate_six_digit_categories(row["category"], 3)
        return row

    def remove_whitespace_and_null_codes(self, df):
        """Cleans whitespace codes and then drops any non-codes"""
        df["code"] = df["code"].str.strip()
        df["code"].dropna(inplace=True)
        return df

    def clean_df(self, df):
        df_no_duplicates = df.drop_duplicates()
        return df_no_duplicates.dropna()

    def _get_category(self, code):
        """Takes in a code and creates a category"""
        return [
            re.sub(r"\..*", "", individual_code)
            for individual_code in set(list(code.split(",")))
        ]

    def _separate_codes(self, code):
        """Takes in a code and converts it to a clean string of codes"""
        code = code.replace(" ", "")
        code = code.replace(" , ", "")
        for punctuation in string.punctuation.replace(".", ""):
            if punctuation in code:
                code = code.replace(str(punctuation), ",")
        code = re.sub(",+", ",", code)
        code = code.lstrip(",")
        code = code.rstrip(",").lower()
        return code

    def remove_empty_string_codes(self, df):
        return df[df["code"] == ""].index

    def get_official_codes(self):
        official_code_query = """
        select icd_10_cm
        from raw.icd10cm_order_2023;"""
        return get_data(official_code_query)

    def _separate_six_digit_categories(self, category, splitat):
        """finding strings of len()==6 and separating"""
        return [
            ", ".join([f"{i[:splitat]}", f"{i[splitat:]}"])
            if len(i) == 6 and i[3].isalpha()
            else ", ".join([i])
            for i in set(category)
        ]
