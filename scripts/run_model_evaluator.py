import itertools
import os
import logging
import pandas as pd
from main import (
    ModelScorerFlow,
    ModelRankerFlow,
    ModelEvaluatorFlow,
    ModelVisualizerFlow,
)


if __name__ == "__main__":
    schema_type = os.environ.get("SCHEMA_TYPE")
    date_list = ["2022-08-16 13:31:11.227697"]
    for start_datetime in date_list:
        start_datetime = pd.to_datetime(start_datetime)
        model_output = itertools.product(
            *[  # model sets
                [
                    "randomforestclassifier_n_estimators500_max_depth150",
                    "randomforestclassifier_n_estimators300_max_depth150",
                ],
                ["RFC"],
                # list of cohorts
                list(range(2, 10)),
            ]
        )

        for model_id, model_name, i in model_output:
            try:
                model_id = model_id + "_" + str(i)
                model_scorer = ModelScorerFlow().execute()
                valid_labels = model_scorer.execute(
                    model_id,
                    model_name,
                    start_datetime,
                    "/mnt/data/projects/pakistan-ihhn/models/",
                    schema_type,
                    i,
                )

                model_ranker = ModelRankerFlow().execute()
                model_ranker.execute(model_id, start_datetime, schema_type)

                model_evaluator = ModelEvaluatorFlow().execute()
                model_evaluator.execute(
                    schema_type,
                    model_name=model_name,
                    model_id=model_id,
                    valid_y=valid_labels.drop(
                        ["cohort", "cohort_type", "train_validation_set"], axis=1
                    ),
                    run_date=start_datetime,
                )
            except Exception as Error:
                logging.error(Error)

        ModelVisualizerFlow(os.environ.get("PLOTPATH")).execute(
            start_datetime, schema_type
        )
