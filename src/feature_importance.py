import pandas as pd
import os
import logging
from joblib import load
from abc import ABC, abstractmethod

from config.model_settings import FeatureImportanceConfig, FeatureGeneratorConfig
from src.utils.utils import write_to_db, get_data
from setup_environment import db_dict, get_dbengine


class FeatureImportanceBase(ABC):
    def __init__(self, num_records, table_name, schema_name):
        self.num_records = num_records
        self.table_name = table_name
        self.schema_name = schema_name

        # initialize engine
        engine = get_dbengine(**db_dict)
        self.engine = engine

    @abstractmethod
    def execute(self):
        """Execute flow starting a run"""


class FeatureImportanceSklearn(FeatureImportanceBase):
    def __init__(self, num_records, table_name, schema_name):
        self.num_records = num_records
        self.table_name = table_name
        self.schema_name = schema_name
        self.text_column_list = FeatureGeneratorConfig().TEXT_COLUMN_LIST
        super().__init__(
            FeatureImportanceConfig.NUM_RECORDS,
            FeatureImportanceConfig.TABLE_NAME,
            FeatureImportanceConfig.SCHEMA_NAME,
        )

    @classmethod
    def from_dataclass_config(
        cls, config: FeatureImportanceConfig
    ) -> "FeatureImportanceSklearn":
        """Imports data from the config class"""
        return cls(
            num_records=config.NUM_RECORDS,
            table_name=config.TABLE_NAME,
            schema_name=config.SCHEMA_NAME,
        )

    def execute(
        self,
        model_name,
        model_path,
        model_id,
        run_date,
        train_validation_set,
        schema_type,
    ):
        """Return feature importance if attribute exists"""
        self.get_schema_name(schema_type)
        train_model = self._get_train_model(model_path, model_id, run_date)

        features = self._get_array(train_validation_set, run_date).tolist()

        logging.info(f"Number of features is {len(features)}")

        try:
            logging.info(
                f"""Number of importances is
                {len(train_model.named_steps[model_name].feature_importances_)}"""
            )

            importances = pd.DataFrame(
                {
                    "features": features,
                    "importance": train_model.named_steps[
                        model_name
                    ].feature_importances_,
                }
            )

            logging.info(f"Feature importance for {model_name}")
            logging.info(
                importances.sort_values("importance", ascending=False).head(
                    self.num_records
                )
            )
            logging.info(
                importances.sort_values("importance", ascending=True).head(
                    self.num_records
                )
            )

            importances["run_date"] = run_date
            importances["model_id"] = model_id
            write_to_db(
                importances, self.engine, self.table_name, self.schema_name, "append"
            )
        except Exception as error:
            logging.error(error)

    def _get_array(self, train_validation_set, run_date):
        return get_data(
            f"""with features as (select col, unnest(features)
            as features from {self.schema_name}.feature_arrays
            WHERE train_validation_set={train_validation_set} AND
            run_date = '{run_date}'::timestamp)
            SELECT CASE WHEN col is NULL THEN features
                            ELSE col ||'_' || features
                            end features
                            from features;"""
        )["features"]

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _get_train_model(self, model_path, model_id, run_date):
        filename = (
            "_".join([model_id, run_date.strftime("%Y%m%d_%H%M%S%f")]) + ".joblib"
        )
        return load(os.path.join(model_path, filename))
