import logging
import os
from typing import Dict, List

import pandas as pd
from styleframe import StyleFrame, utils
import numpy as np
from joblib import Parallel, delayed

from config.model_settings import ProcessXlsxConfig
from src.utils.utils import read_excel, write_csv

logging.basicConfig(level=logging.INFO)


class ProcessXlsx:
    def __init__(
        self,
        non_xlsx_filetypes: List,
        sheet_name: str,
        icd_column_sheet_mapping: List[Dict],
    ) -> None:
        self.non_xlsx_filetypes = non_xlsx_filetypes
        self.sheet_name = sheet_name
        self.icd_column_sheet_mapping = icd_column_sheet_mapping

    @classmethod
    def from_dataclass_config(cls, config: ProcessXlsxConfig) -> "ProcessXlsx":

        return cls(
            non_xlsx_filetypes=config.NON_XLSX_FILETYPES,
            sheet_name=config.SHEET_NAME,
            icd_column_sheet_mapping=config.ICD_COLUMNS_SHEET_MAPPING,
        )

    def execute(self, xlsx_directory, csvs_directory):
        """Iterates through files within the `xlsx_directory`, to
        create processed csvs which will be used to populate the raw schema.

        Args:
            xlsx_directory: directory containing raw `xlsx` files to generate csvs
            csvs_directory: directory that raw csvs will be saved to.
        Returns:
            Raw csvs saved in `csvs_directory`
        """
        Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
            delayed(self.execute_for_file)(csvs_directory, xlsx_directory, file)
            for file in os.listdir(xlsx_directory)
            if os.path.isfile(os.path.join(xlsx_directory, file))
        )

    def execute_for_file(self, csvs_directory, xlsx_directory, file):
        """iterates through `xlsx` files and writes them to csv
        Args:
            csvs_directory: directory to write raw csvs generated from `xlsx` files
            xlsx_directory: directory containing the raw `xlsx` files
            file: the name of the raw `xlsx` file
        Returns:
            raw csv saved in csv_directory
        """
        for non_xlsx_filetype in self.non_xlsx_filetypes:
            if not file.endswith(non_xlsx_filetype):
                xlsx_filepath = os.path.join(xlsx_directory, file)
        if xlsx_filepath:
            logging.info(f"Processing {xlsx_filepath}")
            if any(x in xlsx_filepath for x in ["diag_", "procedure_", "medicalTerms"]):
                df = read_excel(
                    xlsx_filepath,
                    sheet_name=None,
                )
                if len(df.keys()) > 1:
                    print("Multiple sheets for this excel workbook")
                else:
                    # should be only one sheet value
                    write_csv(list(df.values())[0], f"{csvs_directory}/{file[:-5]}.csv")
            elif "Coded_ICD" in xlsx_filepath:
                df = self.read_multiple_sheets_from_icd(xlsx_filepath)
                df = df.rename(
                    columns=self.icd_column_sheet_mapping[1]
                )  # accessing "March 28" to "raw.codes" column name mapping
                write_csv(df, f"{csvs_directory}/{file[:-5]}.csv")
            elif "official_codes_with_ed_dx" in xlsx_filepath:
                df = self.read_priority_icd_codes(xlsx_filepath)
                df = df.drop(columns=["Unnamed: 0"])
                write_csv(df, f"{csvs_directory}/{file[:-5]}.csv")
        else:
            pass

    def read_multiple_sheets_from_icd(self, xlsx_filepath):
        sheets_dict = read_excel(xlsx_filepath, sheet_name=None)
        all_sheets = []
        for name, sheet in sheets_dict.items():
            sheet["sheet"] = name
            sheet = sheet.rename(
                columns=self.icd_column_sheet_mapping[0]
            )  # accessing "Sheet 1" to "March 28" column name mapping
            all_sheets.append(sheet)
        return pd.concat(all_sheets)

    def read_priority_icd_codes(self, xlsx_filepath):
        sf = StyleFrame.read_excel(
            xlsx_filepath, read_style=True, use_openpyxl_styles=False
        )
        sf3 = StyleFrame(sf.applymap(self._only_cells_with_green_background))
        m = sf3.data_df.description_long.isin(sf.description_long)
        df = sf.data_df[m]
        return df

    def _only_cells_with_green_background(self, cell):
        return (
            cell if cell.style.bg_color in {utils.colors.yellow, "FFFFFF00"} else np.nan
        )
