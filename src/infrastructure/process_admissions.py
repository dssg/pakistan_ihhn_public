import logging
from typing import List

from config.model_settings import ProcessAdmissionsConfig
from src.utils.utils import (
    execute_hash,
    get_data,
    iterate_categories,
    parallel_apply,
    write_to_db,
)

logging.basicConfig(level=logging.INFO)


class ProcessAdmissions:
    def __init__(
        self,
        table_name: str,
        schema_name: str,
        columns_to_keep: List,
        diagnosis_stems: List,
    ) -> None:
        self.table_name = table_name
        self.schema_name = schema_name
        self.columns_to_keep = columns_to_keep
        self.diagnosis_stems = diagnosis_stems

    @classmethod
    def from_dataclass_config(
        cls, config: ProcessAdmissionsConfig
    ) -> "ProcessAdmissions":

        return cls(
            table_name=config.TABLE_NAME,
            schema_name=config.SCHEMA_NAME,
            columns_to_keep=config.COLUMNS_TO_KEEP,
            diagnosis_stems=config.DIAGNOSIS_STEMS,
        )

    def execute(self, engine):
        """Execute function called in main script
        Args:
            engine: psql engine to connect to database
        """
        # create base query, concatenate select statements
        # using union all
        logging.info(f"Loading {self.table_name} from db")
        query = self._build_query()
        df = get_data(query)
        df = execute_hash(df, "diagnosis")
        df["category"] = parallel_apply(df["code"], iterate_categories, n_jobs=-1)

        # assert uniqueness on diagnosis, admission, and new_mr
        assert (
            df.loc[:, ["new_mr", "admission_no", "diagnosis_hash"]].duplicated().sum()
            == 0
        )

        logging.info(
            f"Writing {self.table_name} from db under schema {self.schema_name}"
        )
        write_to_db(
            df,
            engine,
            f"{self.table_name}",
            self.schema_name,
            "replace",
        )

    def _build_query(self):
        """returns query to fetch all required admissions data"""

        query_base = [
            """SELECT {cols_keep},
            {stem}_code as code,
            {stem} as diagnosis
            FROM raw.{table}""".format(
                table=self.table_name, cols_keep=", ".join(self.columns_to_keep), stem=x
            )
            for x in self.diagnosis_stems
        ]

        query_join = """ UNION ALL """.join(query_base)
        query = """WITH long_table as ({query_join})
                    SELECT {cols_keep},
                            array_agg(DISTINCT code) as code,
                            diagnosis
                    FROM long_table
                    GROUP BY {cols_keep},
                            diagnosis""".format(
            cols_keep=", ".join(self.columns_to_keep), query_join=query_join
        )
        return query
