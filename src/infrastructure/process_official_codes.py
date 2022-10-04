import logging
import csv
import os
from typing import List

import pandas as pd
from joblib import Parallel, delayed

from src.utils.utils import write_csv
from config.model_settings import ProcessOfficialCodesConfig

logging.basicConfig(level=logging.INFO)


class ProcessOfficialCodes:
    def __init__(
        self,
        txt_filetypes: List,
    ) -> None:
        self.txt_filetypes = txt_filetypes

    @classmethod
    def from_dataclass_config(
        cls, config: ProcessOfficialCodesConfig
    ) -> "ProcessOfficialCodes":
        return cls(
            txt_filetypes=config.TEXT_FILETYPES,
        )

    def execute(self, raw_txt_directory, processed_csvs_directory):
        """Iterates through text files within the `raw_csvs_directory`, to create
        processed csvs which will be used to populate the raw schema.

        Args:
            raw_txt_directory: directory containing raw textfiles to be processed
            processed_csvs_directory: directory that processed csvs will be saved to.
        Returns:
            Processed csvs saved in `processed_csvs_directory`
        """

        Parallel(n_jobs=-1, backend="multiprocessing", verbose=5)(
            delayed(self.execute_for_text_file)(
                raw_txt_directory, processed_csvs_directory, text_file
            )
            for text_file in os.listdir(raw_txt_directory)
            if os.path.isfile(os.path.join(raw_txt_directory, text_file))
        )

    def execute_for_text_file(
        self, raw_txt_directory, processed_csvs_directory, text_file
    ):
        for txt_filetype in self.txt_filetypes:
            if text_file.endswith(txt_filetype):
                text_filepath = os.path.join(raw_txt_directory, text_file)
                logging.info(f"Processing {text_filepath}")
                row_list = []

                csv.register_dialect("skip_space", skipinitialspace=True)
                with open(text_filepath, "r") as f:
                    reader = csv.reader(f, delimiter="\t", dialect="skip_space")
                    for item in reader:

                        if text_file == "icd10cm_codes_2023.txt":
                            code_description = self.read_icd10cm_codes(item)
                            row_list.append(code_description)
                        else:
                            code_description = self.split_icd10cm_order(item[0])
                            row_list.append(code_description)
                    if text_file == "icd10cm_codes_2023.txt":
                        df = pd.DataFrame(
                            row_list, columns=["icd_10_cm", "description_long"]
                        )
                        write_csv(
                            df, f"{processed_csvs_directory}/{text_file[:-4]}.csv"
                        )
                    else:
                        df = pd.DataFrame(
                            row_list,
                            columns=[
                                "order_number",
                                "icd_10_cm",
                                "valid_for_submission",
                                "description_short",
                                "description_long",
                            ],
                        )
                        write_csv(
                            df, f"{processed_csvs_directory}/{text_file[:-4]}.csv"
                        )

    def split_icd10cm_order(self, item):
        order_split = 5
        icd_code_split = 13
        header_split = 15
        short_desc_split = 77
        order, icd_code, header, short_desc, long_desc = (
            item[:order_split],
            item[order_split:icd_code_split],
            item[icd_code_split:header_split],
            item[header_split:short_desc_split],
            item[short_desc_split:],
        )
        return [
            order.strip(),
            icd_code.strip(),
            header.strip(),
            short_desc.lstrip().rstrip(),
            long_desc,
        ]

    def read_icd10cm_codes(self, item):
        row = item[0].split(" ", 1)
        stripped_row = list(map(str.strip, row))
        filtered_row = list(filter(len, stripped_row))
        return filtered_row
