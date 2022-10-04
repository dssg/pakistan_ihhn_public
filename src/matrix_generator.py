import logging
import os
import pandas as pd
import scipy.sparse as sp

from typing import Type, List
from joblib import load, dump

from src.utils.utils import get_data
from config.model_settings import FeatureGeneratorConfig
from src.feature_generator import FeatureGeneratorSklearn
from config.model_settings import (
    MatrixGeneratorConfig,
)

# Matrix generator
# Input: label, features, time_splits
# Output: train_df = numpy array, valid_df = numpy arr

# for each text column of interest
# run feature generator
# add resulting object to a list
# matrix generator: merge everything in the list together to get train_df and valid_df

logging.basicConfig(level=logging.INFO)


class MatrixGenerator:
    def __init__(
        self,
        algorithm: str,
        cohort_splits_list: List,
        schema_name: str,
        text_column_list: List,
        id_column_list: List,
        text_features_path: str,
        features_path: str,
        labels_path: str,
    ) -> None:
        self.algorithm = algorithm
        self.cohort_splits_list = cohort_splits_list
        self.schema_name = schema_name
        self.text_column_list = text_column_list
        self.id_column_list = id_column_list
        self.text_features_path = text_features_path
        self.features_path = features_path
        self.labels_path = labels_path

    @classmethod
    def from_dataclass_config(
        cls,
        config: MatrixGeneratorConfig,
        text_features_path,
        features_path,
        labels_path,
    ) -> "MatrixGenerator":
        return cls(
            algorithm=config.ALGORITHM,
            cohort_splits_list=config.COHORT_SPLITS_LIST,
            schema_name=config.SCHEMA_NAME,
            text_column_list=FeatureGeneratorConfig().TEXT_COLUMN_LIST,
            id_column_list=FeatureGeneratorConfig().ID_COLUMN_LIST,
            text_features_path=text_features_path,
            features_path=features_path,
            labels_path=labels_path,
        )

    def execute_train_valid_set(self, schema_type):
        self.get_schema_name(schema_type)
        cohorts_query = """select distinct "unique_id", "cohort", "cohort_type",
        "train_validation_set" from {schema}."cohorts";""".format(
            schema=self.schema_name
        )
        cohorts_df = get_data(cohorts_query)

        if self.cohort_splits_list == []:
            return cohorts_df.train_validation_set.unique()
        else:
            return self.cohort_splits_list

    def execute(self, train_valid_id, run_date, schema_type, table_name=None):
        cohorts_query = """select distinct "unique_id", "cohort", "cohort_type",
        "train_validation_set" from {schema}."cohorts";""".format(
            schema=self.schema_name
        )
        cohorts_df = get_data(cohorts_query)

        return self.execute_for_cohort(
            train_valid_id,
            cohorts_df,
            run_date,
            schema_type,
            table_name,
        )

    def execute_for_cohort(
        self, training_validation_id, cohorts_df, run_date, schema_type, table_name
    ):
        cohort_df = cohorts_df.loc[
            cohorts_df["train_validation_set"] == training_validation_id
        ]

        # load labels
        labels_df = self._load_all_labels(run_date)

        if cohort_df is not None:
            logging.info(f"Generating features for Cohort {training_validation_id}")
            # get feature ids
            (
                train_csr,
                validation_csr,
                feature_train_id,
                feature_valid_id,
            ) = self.matrix_generator(
                training_validation_id,
                run_date,
                schema_type,
                table_name,
            )
            logging.info(f"Rows in training features: {train_csr.shape[0]}")
            logging.info(f"Rows in validation features: {validation_csr.shape[0]}")

            # convert back to merge
            labels_valid_df = pd.merge(
                feature_valid_id,
                labels_df,
                on=["unique_id", "cohort_type", "cohort"],
                how="left",
            )
            labels_train_df = pd.merge(
                feature_train_id,
                labels_df,
                on=["unique_id", "cohort_type", "cohort"],
                how="left",
            )

            # write as pickle
            self._write_labels_as_pickle(
                labels_train_df,
                run_date,
                training_validation_id,
                "training",
            )
            self._write_labels_as_pickle(
                labels_valid_df,
                run_date,
                training_validation_id,
                "validation",
            )

            logging.info(f"Rows in training labels: {labels_train_df.shape[0]}")
            logging.info(f"Rows in validation labels: {labels_valid_df.shape[0]}")

            logging.info(feature_valid_id.head())
            logging.info(labels_valid_df.head())
            return validation_csr, train_csr, labels_valid_df, labels_train_df
        else:
            logging.info("training or validation cohort must be assigned")

    def matrix_generator(self, train_validation_set, run_date, schema_type, table_name):
        if self.algorithm == "sklearn":
            config = FeatureGeneratorConfig()
            df_train, df_valid, feature_train_id, feature_valid_id = (
                self._get_feature_generator()
                .from_dataclass_config(config, self.text_features_path)
                .execute_features(
                    train_validation_set, run_date, schema_type, table_name
                )
            )

            train_csr = self._add_csr(
                df_train, train_validation_set, "training", run_date
            )
            valid_csr = self._add_csr(
                df_valid, train_validation_set, "validation", run_date
            )

            return train_csr, valid_csr, feature_train_id, feature_valid_id

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _add_csr(self, df, train_validation_set, cohort_type, run_date):
        csr_list = self._get_csr(train_validation_set, cohort_type, run_date)
        csr = self._concat_csr(df, csr_list)
        filename = "_".join(
            [
                str(train_validation_set),
                cohort_type,
                run_date.strftime("%Y%m%d_%H%M%S%f"),
            ]
        )

        dump(
            csr,
            os.path.join(
                self.features_path,
                filename + ".joblib",
            ),
        )

        return csr

    def _get_feature_generator(self) -> Type[FeatureGeneratorSklearn]:
        if self.algorithm == "sklearn":
            return FeatureGeneratorSklearn
        else:
            raise ValueError(
                "The algorithm provided has no registered feature builder!"
            )

    def _get_csr(self, train_validation_set, cohort_type, run_date):
        filename = "_".join(
            [
                str(train_validation_set),
                cohort_type,
                run_date.strftime("%Y%m%d_%H%M%S%f"),
            ]
        )
        return [
            load(
                os.path.join(
                    self.text_features_path,
                    x + "_" + filename + ".joblib",
                )
            )
            for x in self.text_column_list
        ]

    def _concat_csr(self, X, csr_list):
        structured_csr = sp.csr_matrix(X.drop(self.id_column_list, axis=1))
        csr_list += [structured_csr]
        return sp.hstack(csr_list)

    def _load_all_labels(self, run_date):
        filename = "_".join(["labels", run_date.strftime("%Y%m%d_%H%M%S%f")])
        return load(os.path.join(self.labels_path, filename + ".joblib"))

    def _write_labels_as_pickle(self, y, run_date, training_validation_id, cohort_type):
        # save as pickle
        filename = "_".join(
            [
                "labels",
                run_date.strftime("%Y%m%d_%H%M%S%f"),
                str(training_validation_id),
                cohort_type,
            ]
        )
        dump(
            y,
            os.path.join(self.labels_path, filename + ".joblib"),
        )
