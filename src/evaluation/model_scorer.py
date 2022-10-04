import pandas as pd
import numpy as np
import logging
import os
from joblib import load

from src.evaluation.model_evaluator_base import ModelEvaluatorBase
from src.utils.utils import write_to_db, get_data

from config.model_settings import (
    ModelEvaluatorConfig,
    ModelScorerConfig,
)


class ModelScorer(ModelEvaluatorBase):
    def __init__(self, table_name, seed, labels_path, features_path):
        self.table_name = table_name
        self.seed = seed
        self.labels_path = labels_path
        self.features_path = features_path
        super().__init__(ModelEvaluatorConfig.SCHEMA_NAME, ModelEvaluatorConfig.ID_VAR)

    @classmethod
    def from_dataclass_config(
        cls, config: ModelScorerConfig, labels_path, features_path
    ) -> "ModelScorer":
        """Imports data from the config class"""
        return cls(
            table_name=config.TABLE_NAME,
            seed=config.SEED,
            labels_path=labels_path,
            features_path=features_path,
        )

    def execute(
        self,
        model_id,
        model_name,
        run_date,
        model_path,
        schema_type,
        train_validation_set,
        return_df=True,
        class_value=1,
    ):
        """
        Return scores for each record and class combination,
        given a trained model and validation data
        """
        logging.info("Scoring model")
        self.get_schema_name(schema_type)
        train_model = self._get_train_model(model_path, model_id, run_date)
        features_valid, labels_valid, unique_id_column = self._get_valid(
            train_validation_set, run_date
        )

        # get mlb_categories
        mlb_categories = get_data(
            f"""select distinct categories from {self.schema_name}.labels
                                    where run_date='{run_date}'::timestamp
                                    and train_validation_set={train_validation_set}
                                    and cohort_type='validation'"""
        ).values[0]
        mlb_categories = str(mlb_categories[0]).split(",")

        # only use predict_proba if it exists for the classifier
        if hasattr(train_model, "predict_proba"):
            scores = train_model.predict_proba(features_valid)

            # return dataframe with scores for positive cases
            if return_df:

                # exclude cases where predict_proba only returns one class
                # these are cases where there is likely only one class value
                # in the training and test data for this code
                if model_name == "XGB":
                    scores_df = self._get_xgb_scores(scores, mlb_categories)
                else:
                    scores_list = self._scores_to_list(
                        features_valid, scores, class_value
                    )
                    scores_df = self._scores_to_df(
                        scores_list, mlb_categories, unique_id_column
                    )
                scores_df_melt = self._melt_df(scores_df)
                scores_df_rand = self._add_random_values(scores_df_melt)

                # add time stamp
                scores_df_rand["run_date"] = run_date
                scores_df_rand["model_id"] = model_id

                with self.engine.connect() as conn:
                    conn.execute("""SET ROLE "pakistan-ihhn-role" """)
                    conn.execute(
                        f"""CREATE TABLE IF NOT EXISTS
                        {self.schema_name}.{self.table_name}
                                    (unique_id int,
                                    variable text,
                                    value float8,
                                    random int,
                                    seed int,
                                    run_date timestamp,
                                    model_id text,
                                    PRIMARY KEY
                                    (unique_id, variable, run_date, model_id)
                                    )"""
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

                write_to_db(
                    scores_df_rand,
                    self.engine,
                    self.table_name,
                    self.schema_name,
                    "append",
                    index=False,
                )
                return labels_valid
            else:
                return labels_valid, scores
        else:
            logging.warning("Scoring function is not defined")

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _scores_to_list(self, features_valid, scores, class_value):
        """Generate a list of scores from the predict_proba output"""
        return [
            x[:, class_value] if x.shape[1] > 1 else np.zeros(features_valid.shape[0])
            for x in scores
        ]

    def _scores_to_df(self, scores_list, mlb_categories, unique_id_column):
        """Convert a list of scores into a dataframe"""
        transposed_scores = np.array(scores_list).transpose()
        scores_df = pd.DataFrame(transposed_scores, columns=mlb_categories)
        scores_df["unique_id"] = unique_id_column.reset_index(drop=True)
        return scores_df

    def _melt_df(self, scores_df):
        """Pivot the scores data long"""
        scores_df_melt = scores_df.melt(id_vars="unique_id")
        scores_df_melt["value"] = scores_df_melt["value"].astype(np.float16)
        return scores_df_melt

    def _add_random_values(self, df):
        """Generate random values for sorting based on a seed"""
        rng = np.random.default_rng(self.seed)
        df["random"] = rng.choice(df.shape[0], df.shape[0], replace=False)

        df["seed"] = self.seed
        return df

    def _get_xgb_scores(self, scores, mlb_categories):
        return pd.DataFrame(scores, columns=mlb_categories)

    def _get_train_model(self, model_path, model_id, run_date):
        filename = (
            "_".join([model_id, run_date.strftime("%Y%m%d_%H%M%S%f")]) + ".joblib"
        )
        return load(os.path.join(model_path, filename))

    def _get_valid(self, train_validation_set, run_date):
        features_filename = (
            "_".join(
                [
                    str(train_validation_set),
                    "validation",
                    run_date.strftime("%Y%m%d_%H%M%S%f"),
                ]
            )
            + ".joblib"
        )

        labels_filename = (
            "_".join(
                [
                    "labels",
                    run_date.strftime("%Y%m%d_%H%M%S%f"),
                    str(train_validation_set),
                    "validation",
                ]
            )
            + ".joblib"
        )

        features = load(os.path.join(self.features_path, features_filename))
        labels = load(os.path.join(self.labels_path, labels_filename))
        return features, labels, labels["unique_id"]
