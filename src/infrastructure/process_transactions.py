import logging
from typing import List

from config.model_settings import ProcessTransactionsConfig
from src.utils.utils import execute_hash, get_data, write_to_db

logging.basicConfig(level=logging.INFO)


class ProcessTransactions:
    def __init__(
        self,
        table_name: str,
        schema_name: str,
        columns_to_keep: List,
        columns_to_group: List,
    ):
        self.table_name = table_name
        self.schema_name = schema_name
        self.columns_to_keep = columns_to_keep
        self.columns_to_group = columns_to_group

    @classmethod
    def from_dataclass_config(
        cls, config: ProcessTransactionsConfig
    ) -> "ProcessTransactions":

        return cls(
            table_name=config.TABLE_NAME,
            schema_name=config.SCHEMA_NAME,
            columns_to_keep=config.COLUMNS_TO_KEEP,
            columns_to_group=config.COLUMNS_TO_GROUP,
        )

    def execute(self, engine):
        """Execute function called in main script
        Args:
            engine: psql engine to connect to database
        """

        query = """SELECT DISTINCT new_er,
        new_mr,
        gender,
        city,
        age_years,
        triage_datetime,
        triagecomplaint,
        bp,
        tr_pulse,
        tr_temp,
        tr_resp,
        acuity,
        visit_datetime,
        disposition,
        disposition_time,
        doctor_id,
        specialty,
        admission_date,
        admission_ward,
        discharge_ward,
        discharge_datetime,
        ed_dx,
        hopi,
        array_agg(DISTINCT systolic) as systolic_agg,
        array_agg(DISTINCT diastolic) as diastolic_agg,
        array_agg(DISTINCT temperature) as temperature_agg,
        array_agg(DISTINCT weight) as weight_agg,
        array_agg(DISTINCT o2sat) as o2sat_agg,
        array_agg(DISTINCT nurse_id) as nurse_id_agg,
        array_agg(DISTINCT hopi) as hopi_agg
    FROM raw.{table}
    GROUP BY {cols_group}""".format(
            table=self.table_name, cols_group=", ".join(self.columns_to_group)
        )
        df = get_data(query)
        df = execute_hash(df, "ed_dx")

        write_to_db(
            df[self.columns_to_keep],
            engine,
            f"{self.table_name}",
            self.schema_name,
            "replace",
        )
        logging.info("Finished transactions data")

    def execute_for_row(self, row):
        """Executes cleaning private functions row-wise on transactions cells
        Args:
            row: a row of the dataframe
        Returns:
            row with hash complete
        """
        try:
            row["ed_dx_hash"] = self._get_hash_of_string(row)
        except TypeError:
            row["ed_dx_hash"] = 0
        return row
