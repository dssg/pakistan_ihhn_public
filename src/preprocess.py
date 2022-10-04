import logging
import pandas as pd
from src.preprocessing.cohort_filter import Filter
from config.model_settings import CohortBuilderConfig


class Preprocess:
    def __init__(
        self,
        filter_pregnancies,
        filter_children,
        filter_non_standard_codes,
        filter_priority_categories,
    ):
        self.filter_pregnancies = filter_pregnancies
        self.filter_children = filter_children
        self.filter_non_standard_codes = filter_non_standard_codes
        self.filter_priority_categories = filter_priority_categories

    @classmethod
    def from_options(cls, filters) -> "Preprocess":
        filter_default = dict.fromkeys(
            [
                "filter_pregnancies",
                "filter_children",
                "filter_non_standard_codes",
                "filter_priority_categories",
            ],
            False,
        )
        for filter_ in filters:
            filter_default[filter_] = True
        return cls(**filter_default)

    def execute(self, training_validation_df: pd.DataFrame, **kwargs):
        preprocessed_training = self.preprocess_data(training_validation_df)
        return preprocessed_training

    def preprocess_data(self, training_validation_df: pd.DataFrame):
        if self.filter_pregnancies is True:
            training_validation_df = training_validation_df.pipe(
                Filter.filter_pregnant_patients
            )
            logging.info(
                f"""Total number of patients left after
                filtering pregnancies : {len(training_validation_df)}"""
            )
        if self.filter_children is True:
            training_validation_df = training_validation_df.pipe(
                Filter.filter_child_patients
            )
            logging.info(
                f"""Total number of patients left after
                filtering children {len(training_validation_df)}"""
            )
        if self.filter_non_standard_codes is True:
            training_validation_df = training_validation_df.pipe(
                Filter.filter_non_standard_codes
            )
            logging.info(
                f"""Total number of patients left after
                filtering non-standard codes {len(training_validation_df)}"""
            )
            # training_validation_df = training_validation_df.pipe(
            #     Filter.filter_priority_categories
            # )
        if self.filter_priority_categories is True:
            no_of_occurrences = CohortBuilderConfig.NO_OF_OCCURENCES
            training_validation_df = Filter.filter_priority_categories(
                training_validation_df, no_of_occurrences
            )
            logging.info(
                f"""Total number of patients left after
                filtering priority categories {len(training_validation_df)}"""
            )
        return training_validation_df
