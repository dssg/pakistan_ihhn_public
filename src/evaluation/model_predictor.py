import logging
import os
from sqlalchemy import text


from config.model_settings import (
    ModelEvaluatorConfig,
    RetrainingConfig,
    ModelRankerConfig,
    FeatureGeneratorConfig,
)
from src.evaluation.model_evaluator_base import ModelEvaluatorBase
from src.utils.utils import get_data


class ModelPredictor(ModelEvaluatorBase):
    def __init__(
        self,
        constraint,
        predictions_table_name,
        rank_table_name,
        schema_name,
        all_model_features,
    ) -> None:
        self.constraint = constraint
        self.predictions_table_name = predictions_table_name
        self.rank_table_name = rank_table_name
        self.schema_name = schema_name
        self.all_model_features = all_model_features
        super().__init__(ModelEvaluatorConfig.SCHEMA_NAME, ModelEvaluatorConfig.ID_VAR)

    @classmethod
    def from_dataclass_config(cls, config: RetrainingConfig) -> "ModelPredictor":
        """Imports data from the config class"""
        return cls(
            constraint=config.CONSTRAINT,
            predictions_table_name=config.PREDICTIONS_TABLE_NAME,
            rank_table_name=ModelRankerConfig().TABLE_NAME,
            schema_name=config.SCHEMA_NAME,
            all_model_features=FeatureGeneratorConfig().ALL_MODEL_FEATURES,
        )

    def execute(self, model_id, run_date, schema_type, output_path):
        """
        Generate model predictions and sample output in the database
        """
        logging.info("Generating predictions")
        self.get_schema_name(schema_type)
        self._generate_predictions(model_id, run_date)
        self.generate_sample_output(model_id, run_date, output_path)

    def generate_sample_output(self, model_id, run_date, output_path=None):
        """
        Generate sample output in the database with features, unique IDs,
        and ranked ICD-10 categories
        """
        with self.engine.begin() as conn:
            conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))
            conn.execute(
                text(
                    "DROP TABLE IF EXISTS {schema}.sample_output;".format(
                        schema=self.schema_name
                    )
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE {schema}.sample_output as
                    with codes as (
                        SELECT array_agg(description_long) as code_descriptions,
                        left(lower(btrim(icd_10_cm)), 3) as category
                        FROM raw.icd10cm_codes_2023
                        GROUP BY left(lower(btrim(icd_10_cm)), 3))
                        SELECT p.unique_id,
                        t.new_er,
                        t.new_mr,
                        p.model_id,
                        p.run_date,
                        p.variable as category,
                        c.code_descriptions,
                        p.rank_number as rank,
                        t.triage_datetime,
                        {feature_cols}
                        FROM {schema}.{rank_tbl} as p
                        LEFT JOIN {schema}.retraining as t using(unique_id)
                        LEFT JOIN codes as c on lower(btrim(p.variable)) = c.category
                    WHERE model_id = '{model_id}' AND
                    run_date = '{run_date}' AND
                    rank_number <= {constraint}
                    ORDER BY p.unique_id, p.rank_number;
                    """.format(
                        schema=self.schema_name,
                        rank_tbl=self.rank_table_name,
                        model_id=model_id,
                        run_date=run_date,
                        constraint=self.constraint,
                        feature_cols=", ".join(
                            ["t." + x for x in self.all_model_features]
                        ),
                    )
                )
            )

        # if output path is specified, write to the database
        if output_path is not None:
            df = get_data(f"""SELECT * FROM {self.schema_name}.sample_output;""")
            df.to_csv(os.path.join(output_path, "sample_output.csv"), index=False)

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _generate_predictions(self, model_id, run_date):
        """
        Create predictions table in database based on
        rankings for each model id and run date
        """
        with self.engine.begin() as conn:
            conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))

            conn.execute(
                text(
                    """CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        unique_id int,
                        model_id text,
                        category text,
                        rank_number int,
                        run_date timestamp,
                       PRIMARY KEY(unique_id, model_id, category, run_date))
                    """.format(
                        schema=self.schema_name, table=self.predictions_table_name
                    )
                )
            )

            conn.execute(
                text(
                    """
                    INSERT INTO {schema}.{table}
                    SELECT
                        unique_id,
                        model_id,
                        variable,
                        rank_number,
                        run_date
                    FROM {schema}.{rank_tbl}
                    WHERE model_id = '{model_id}' AND
                    run_date = '{run_date}' AND
                    rank_number <= {constraint}
                    ORDER BY unique_id, rank_number;
                    """.format(
                        schema=self.schema_name,
                        table=self.predictions_table_name,
                        rank_tbl=self.rank_table_name,
                        model_id=model_id,
                        run_date=run_date,
                        constraint=self.constraint,
                    )
                )
            )
