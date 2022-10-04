import logging
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List
import sqlalchemy
from dateutil.relativedelta import relativedelta
from sqlalchemy import text

from config.model_settings import TimeSplitterConfig
from src.utils.utils import get_data

logging.basicConfig(level=logging.INFO)


class TimeSplitterBase(ABC):
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

    def create_end_date(self, *args: Any) -> datetime:
        sql_query = """SELECT MAX("{date_col}") FROM {schema}.{table};""".format(
            schema=self.schema_name, table=self.table_name, date_col=self.date_col
        )
        end_date = (get_data(sql_query).values[0])[0]  # Accessing timestamp string
        return datetime.strptime(f"{end_date}", "%Y-%m-%dT%H:%M:%S.000000000")

    def create_start_date(self, *args: Any) -> datetime:
        sql_query = """SELECT MIN("{date_col}") FROM {schema}.{table};""".format(
            schema=self.schema_name, table=self.table_name, date_col=self.date_col
        )
        start_date = (get_data(sql_query).values[0])[0]  # Accessing timestamp string
        return datetime.strptime(f"{start_date}", "%Y-%m-%dT%H:%M:%S.000000000")


class TimeSplitter(TimeSplitterBase):
    def __init__(
        self,
        date_col: str,
        time_window_length: int,
        within_window_sampler: int,
        window_count: int,
        train_validation_dict: Dict[str, List[Any]],
    ) -> None:
        self.date_col = date_col
        self.time_window_length = time_window_length
        self.within_window_sampler = within_window_sampler
        self.window_count = window_count
        self.train_validation_dict = train_validation_dict
        super().__init__(
            TimeSplitterConfig.DATE_COL,
            TimeSplitterConfig.TABLE_NAME,
            TimeSplitterConfig.SCHEMA_NAME,
        )

    @classmethod
    def from_dataclass_config(cls, config: TimeSplitterConfig) -> "TimeSplitter":
        return cls(
            date_col=config.DATE_COL,
            time_window_length=config.TIME_WINDOW_LENGTH,
            within_window_sampler=config.WITHIN_WINDOW_SAMPLER,
            window_count=config.WINDOW_COUNT,
            train_validation_dict=config.TRAIN_VALIDATION_DICT,
        )

    def execute(self, schema_type, engine):
        """
        Input
        ----
        creates list of dates between start and end date of each time
        window within a given window time length and a given number of months to sample
        ----
        The start and end dates for each time window
        """
        self.get_schema_name(schema_type)
        if schema_type == "dev":
            self.window_count = 2
            self.time_window_length = 1
            self.within_window_sampler = 1

        window_no = 0
        try:
            end_date = self.create_end_date(self.date_col, self.table_name)
            start_date = self.create_start_date(self.date_col, self.table_name)
        except sqlalchemy.exc.ProgrammingError:
            commands = self.create_dev_tables()
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))
                    for command in commands:
                        conn.execute(command)

            end_date = self.create_end_date(self.date_col, self.table_name)
            start_date = self.create_start_date(self.date_col, self.table_name)

        while window_no < self.window_count:
            window_start_date, window_end_date = self.get_validation_window(
                end_date, window_no
            )
            logging.info(
                f"Getting cohort between {window_start_date} and {window_end_date}"
            )
            if window_start_date < start_date:
                logging.warning(
                    f"""Date: {window_start_date.date()} is ealier than the first date
                     within data: {start_date.date()}"""
                )
                window_no += 1
            else:
                self.train_validation_dict["validation"] += [
                    (
                        window_start_date,
                        window_end_date,
                    )
                ]
                self.train_validation_dict["training"] += [
                    (start_date, window_start_date)
                ]
                window_no += 1
        return self.train_validation_dict

    def create_dev_tables(self):
        """create tables in the PostgreSQL database"""
        commands = (
            f"""
            CREATE SCHEMA {self.schema_name};
            """,
            f""" CREATE TABLE {self.schema_name}.train AS
            SELECT * FROM model_output.train LIMIT 1000;
            """,
        )
        return commands

    def get_validation_window(self, end_date, window_no):
        """Gets start and end date of each training window"""
        window_start_date = self._get_start_time_windows(end_date, window_no)
        window_end_date = self._get_end_time_windows(window_start_date)
        return window_start_date, window_end_date

    def _get_start_time_windows(self, window_date, window_no):
        """Gets start date of window based on the window length and the number of sample
        months used in the window"""
        return window_date - relativedelta(
            months=+window_no * self.time_window_length + self.within_window_sampler
        )

    def _get_end_time_windows(self, window_start_date):
        """Gets end date of window based on the window length and the number of sample
        months used in the window"""
        return window_start_date + relativedelta(months=+self.within_window_sampler)
