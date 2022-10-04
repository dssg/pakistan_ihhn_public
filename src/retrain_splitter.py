import logging
import pandas as pd
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import text

from config.model_settings import RetrainSplitterConfig
from src.utils.utils import get_data

logging.basicConfig(level=logging.INFO)


class RetrainSplitterBase(ABC):
    def __init__(
        self,
        date_col: str,
        table_name: str,
        schema_name: str,
    ):
        self.date_col = date_col
        self.table_name = table_name
        self.schema_name = schema_name

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def create_training_end_date(self, *args: Any) -> datetime:
        sql_query = """SELECT MAX("{date_col}") FROM {schema}.{table}
                        WHERE train_type = 'training';""".format(
            schema=self.schema_name, table=self.table_name, date_col=self.date_col
        )
        end_date = (get_data(sql_query).values[0])[0]  # Accessing timestamp string
        return datetime.strptime(f"{end_date}", "%Y-%m-%dT%H:%M:%S.000000000")

    def create_training_start_date(self, *args: Any) -> datetime:
        sql_query = """SELECT MIN("{date_col}") FROM {schema}.{table}
                        WHERE train_type = 'training';""".format(
            schema=self.schema_name, table=self.table_name, date_col=self.date_col
        )
        start_date = (get_data(sql_query).values[0])[0]  # Accessing timestamp string
        return datetime.strptime(f"{start_date}", "%Y-%m-%dT%H:%M:%S.000000000")


class RetrainSplitter(RetrainSplitterBase):
    def __init__(
        self,
        date_col: str,
        validation_start: str,
        validation_end: str,
        train_validation_dict: Dict[str, List[Any]],
    ) -> None:
        self.date_col = date_col
        self.validation_start = validation_start
        self.validation_end = validation_end
        self.train_validation_dict = train_validation_dict
        super().__init__(
            RetrainSplitterConfig.DATE_COL,
            RetrainSplitterConfig.TABLE_NAME,
            RetrainSplitterConfig.SCHEMA_NAME,
        )

    @classmethod
    def from_dataclass_config(cls, config: RetrainSplitterConfig) -> "RetrainSplitter":
        return cls(
            date_col=config.DATE_COL,
            validation_start=config.VALIDATION_START,
            validation_end=config.VALIDATION_END,
            train_validation_dict=config.TRAIN_VALIDATION_DICT,
        )

    def execute(self, schema_type, engine):
        """
        For retraining, identifies the time span of the data to be retrained on
        and selects 3 months for validation/prediction.
        """
        self.get_schema_name(schema_type)

        if schema_type == "dev":
            commands = self.create_dev_tables()
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))
                    for command in commands:
                        conn.execute(command)

        start_date = self.create_training_start_date(self.date_col, self.table_name)
        end_date = self.create_training_end_date(self.date_col, self.table_name)

        self.train_validation_dict["training"] += [
            (
                start_date,
                end_date,
            )
        ]

        self.train_validation_dict["validation"] += [
            (
                pd.to_datetime(self.validation_start),
                pd.to_datetime(self.validation_end),
            )
        ]

        return self.train_validation_dict, self.table_name

    def create_dev_tables(self):
        """create tables in the PostgreSQL database"""
        commands = (
            f"""
            CREATE SCHEMA IF NOT EXISTS {self.schema_name};
            """,
            f"""DROP TABLE IF EXISTS {self.schema_name}.retraining;"""
            f""" CREATE TABLE {self.schema_name}.retraining AS
            SELECT * FROM model_output.retraining limit 1000;
            """,
        )
        return commands
