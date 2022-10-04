import logging
from sqlalchemy import text
from src.evaluation.model_evaluator_base import ModelEvaluatorBase

from config.model_settings import (
    ModelEvaluatorConfig,
    ModelRankerConfig,
    ModelScorerConfig,
)


class ModelRanker(ModelEvaluatorBase):
    def __init__(self, table_name):
        self.table_name = table_name
        self.scores_table_name = ModelScorerConfig.TABLE_NAME
        super().__init__(ModelEvaluatorConfig.SCHEMA_NAME, ModelEvaluatorConfig.ID_VAR)

    @classmethod
    def from_dataclass_config(cls, config: ModelRankerConfig) -> "ModelRanker":
        """Imports data from the config class"""
        return cls(
            table_name=config.TABLE_NAME,
        )

    def execute(self, model_id, run_date, schema_type):
        """
        Returns the predicted values for each record and class combination,
        given a trained model and validation data
        """
        logging.info("Ranking results")
        self.get_schema_name(schema_type)

        with self.engine.begin() as conn:
            conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))

            conn.execute(
                text(
                    """CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        unique_id int,
                        model_id text,
                        variable text,
                        run_date timestamp,
                       rank_number numeric,
                       PRIMARY KEY(unique_id, model_id, variable, run_date))
                    """.format(
                        schema=self.schema_name, table=self.table_name
                    )
                )
            )

            conn.execute(
                f"""create index if not exists idx_{self.table_name}_uid
                        on {self.schema_name}.{self.table_name}(unique_id)"""
            )
            conn.execute(
                f"""create index if not exists idx_{self.table_name}_mid
                        on {self.schema_name}.{self.table_name}(model_id)"""
            )
            conn.execute(
                f"""create index if not exists idx_{self.table_name}_rdate
                        on {self.schema_name}.{self.table_name}(run_date)"""
            )

            conn.execute(
                text(
                    """INSERT INTO {schema}.{table}
                    SELECT
                        unique_id,
                        model_id,
                        variable,
                        run_date,
                        RANK () OVER (
                            partition by unique_id, model_id, run_date
                            ORDER BY value desc, random desc
                        ) rank_number
                    FROM {schema}.{scores_tbl}
                    WHERE model_id = '{model_id}' AND
                    run_date = '{run_date}'
                    """.format(
                        schema=self.schema_name,
                        table=self.table_name,
                        scores_tbl=self.scores_table_name,
                        model_id=model_id,
                        run_date=run_date,
                    )
                )
            )

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name
