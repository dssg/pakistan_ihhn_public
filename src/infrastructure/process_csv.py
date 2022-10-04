import logging
import os
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed

from config.model_settings import ProcessCsvConfig
from src.utils.utils import read_csv, write_csv

logging.basicConfig(level=logging.INFO)


class ProcessCsv:
    def __init__(
        self,
        filestems: Dict,
        admissions_col_to_join: List,
        doctors_col_to_join: List,
    ) -> None:
        self.filestems = filestems
        self.admissions_col_to_join = admissions_col_to_join
        self.doctors_col_to_join = doctors_col_to_join

    @classmethod
    def from_dataclass_config(cls, config: ProcessCsvConfig) -> "ProcessCsv":

        return cls(
            filestems=config.FILESTEMS,
            admissions_col_to_join=config.ADMISSIONS_COLUMNS_TO_JOIN,
            doctors_col_to_join=config.DOCTORS_COLUMNS_TO_JOIN,
        )

    def execute(self, raw_csvs_directory, processed_csvs_directory):
        """Iterates through files within the `raw_csvs_directory`, to create
        processed csvs which will be used to populate the raw schema.

        Args:
            raw_csvs_directory: directory containing raw csvs to be processed
            processed_csvs_directory: directory that processed csvs will be saved to.
        Returns:
            Processed csvs saved in `processed_csvs_directory`
        """
        for filestem, processed_csv_dict in self.filestems.items():
            for csv, col_list in processed_csv_dict.items():
                logging.info(f"Processing {csv} data")
                df = (
                    pd.concat(
                        Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
                            delayed(self.execute_for_file)(
                                raw_csvs_directory,
                                col_list,
                                file,
                            )
                            for file in os.listdir(raw_csvs_directory)
                            if os.path.isfile(os.path.join(raw_csvs_directory, file))
                            and str(filestem) in file
                        )
                    )
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                if str(csv) == str("admissions"):
                    if filestem == str("diag"):
                        admissions_df = df
                    elif filestem == str("procedure"):
                        df = pd.merge(
                            admissions_df,
                            df,
                            how="outer",
                            # merge on new_mr and admission_no
                            on=self.admissions_col_to_join,
                        )
                elif str(csv) == str("doctors"):
                    if filestem == str("diag"):
                        doctors_df = df
                    elif filestem == str("HMIS_visit"):
                        df = pd.merge(
                            doctors_df,
                            df,
                            how="outer",
                            # merge on new_mr and admission_no
                            on=self.doctors_col_to_join,
                        )
                write_csv(df, f"{processed_csvs_directory}/{str(csv)}.csv")

    def execute_for_file(self, raw_csvs_directory, col_list, file):
        """Executes cleaning private functions on files
        Args:
            raw_csvs_directory: directory containing raw csvs
                generated from `xlsx` and `xls`
            col_list: list of columns to keep when generating processed csvs
            file: the name of the raw csv in the raw csv directory
        Returns:
            df from the csv filepath
        """
        csv_filepath = self._select_csvs(raw_csvs_directory, file)
        try:
            return read_csv(csv_filepath, usecols=col_list)
        except UnicodeDecodeError:
            pass

    def _select_csvs(self, raw_csvs_directory, file):
        """Joins the file and the directory path into a python-path readable"""
        return os.path.join(raw_csvs_directory, file)
