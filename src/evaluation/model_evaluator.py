import pandas as pd
import logging


from config.model_settings import ModelEvaluatorConfig, ModelRankerConfig
from src.utils.utils import get_data
from src.evaluation.model_evaluator_base import ModelEvaluatorBase


class ModelEvaluator(ModelEvaluatorBase):
    def __init__(
        self,
        metrics,
        constraint,
        summary,
        valid_models,
        table_name,
        micro_pr_table_name,
        validation_table_name,
        table_name_priority_cat,
        priority_categories,
        schema_name,
        id_var,
    ) -> None:
        self.metrics = metrics
        self.constraint = constraint
        self.summary = summary
        self.valid_models = valid_models
        self.table_name = table_name
        self.micro_pr_table_name = micro_pr_table_name
        self.rank_table_name = ModelRankerConfig.TABLE_NAME
        self.validation_table_name = validation_table_name
        self.table_name_priority_cat = table_name_priority_cat
        self.priority_categories = priority_categories
        self.schema_name = schema_name
        self.id_var = id_var
        super().__init__(ModelEvaluatorConfig.SCHEMA_NAME, ModelEvaluatorConfig.ID_VAR)

    @classmethod
    def from_dataclass_config(cls, config: ModelEvaluatorConfig) -> "ModelEvaluator":
        """Imports data from the config class"""
        return cls(
            metrics=config.METRICS,
            constraint=config.CONSTRAINT,
            summary=config.SUMMARY_METHOD,
            valid_models=config.VALID_MODELS,
            validation_table_name=config.VALIDATION_TABLE_NAME,
            micro_pr_table_name=config.MICRO_PR_TABLE_NAME,
            table_name=config.TABLE_NAME,
            table_name_priority_cat=config.TABLE_NAME_PRIORITY_CAT,
            priority_categories=config.PRIORITY_CATEGORIES,
            schema_name=config.SCHEMA_NAME,
            id_var=config.ID_VAR,
        )

    def execute(
        self,
        schema_type,
        model_name,
        model_id,
        valid_y,
        run_date,
    ):
        """
        Evaluate performance of trained model based on precision, recall, or accuracy.
        Writes the results table to the database

        Parameters
        ----------
        model_name : str
        model_id : str
        mlb_categories : list
        valid_y : dataframe
        """

        if model_name not in self.valid_models:
            logging.warning(
                f"Classifier {model_name} is not valid. Check valid models list."
            )
            return None

        logging.info("Evaluating all models")
        # iterate through all numeric constraints and metrics
        self.get_schema_name(schema_type)
        eval_list = []
        eval_list_priority = []

        for c in self.constraint:
            eval, eval_priority = self.evaluate_one_metric(
                model_id, valid_y, c, run_date
            )
            eval_list += [eval]
            eval_list_priority += [eval_priority]

        results_metrics_df = self._concat_eval_results(eval_list, "summary")
        results_metrics_priority_df = pd.concat(eval_list_priority, axis=0)

        # write the priority codes to the db
        if len(self.priority_categories) > 0:
            self._results_to_db(
                results_metrics_priority_df,
                self.table_name_priority_cat,
                run_date,
                index_label="variable",
            )

        # write the results to the db
        self._results_to_db(results_metrics_df, self.table_name, run_date)

    def evaluate_one_metric(
        self, model_id, valid_y, constraint, run_date, micro_pr=True
    ):
        """Calculate evaluation metrics for one metric and constraint"""
        ranked_df = get_data(
            """SELECT * FROM {schema}.{table}
            WHERE model_id = '{model_id}' AND
            run_date = '{run_date}'::timestamp""".format(
                schema=self.schema_name,
                table=self.rank_table_name,
                model_id=model_id,
                run_date=run_date,
            )
        )

        # select only the ranked values within the constraint
        ranked_df["pred"] = (ranked_df["rank_number"] <= constraint).astype(int)

        pred_comp = self._compare_pred(ranked_df, valid_y)

        metric_value_priority = self._iterate_priority_categories(pred_comp)
        metric_value_priority = self._add_info_cols(
            metric_value_priority, constraint, model_id
        )

        # calculate the metric (e.g., precision, recall, or accuracy)
        # for each row in the data
        metric_value = self._calc_metric(pred_comp)

        logging.info(f"writing pr to db for constraint {constraint}")

        metric_value["constraint_val"] = constraint
        metric_value["run_date"] = run_date
        self._output_to_db(
            metric_value, model_id, run_date, self.micro_pr_table_name, index=False
        )
        metric_value = metric_value.drop(["constraint_val", "run_date"], axis=1)

        if self.summary:
            metric_value = self._summ_metric(metric_value)
            metric_value = self._add_info_cols(metric_value, constraint, model_id)

        return metric_value, metric_value_priority

    def evaluate_priority_codes(
        self, model_id, run_date, valid_y, constraint, priority_codes=[]
    ):
        """Returns a dataframe with recall and precision for priority codes"""
        ranked_df = get_data(
            """SELECT * FROM {schema}.{table}
            WHERE model_id = '{model_id}' AND
            run_date = '{run_date}'::timestamp""".format(
                schema=self.schema_name,
                table=self.rank_table_name,
                model_id=model_id,
                run_date=run_date,
            )
        )
        ranked_df["pred"] = (ranked_df["rank_number"] <= constraint).astype(int)

        pred_comp = self._compare_pred(ranked_df, valid_y)
        return self._iterate_priority_categories(pred_comp, priority_codes)

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _compare_pred(self, pred, valid_y):
        """Compare predicted and real values, varies by metric for model performance"""

        valid_y_melt = valid_y.melt(id_vars="unique_id")

        comp_df = pd.merge(
            valid_y_melt,
            pred[["unique_id", "variable", "pred"]],
            on=["unique_id", "variable"],
            how="left",
        )

        comp_df["pred"] = comp_df["pred"].fillna(0)

        comp_df["tp"] = (comp_df["value"] == comp_df["pred"]) & (comp_df["value"] == 1)
        comp_df["comp"] = comp_df["value"] == comp_df["pred"]

        return comp_df

    def _calc_metric(self, df, id="unique_id"):
        """Calculate model performance based on comparing predicted and real values"""
        # convert to int for groupby
        comp_df = df.copy()
        comp_df["count"] = 1
        comp_df.loc[:, "value"] = comp_df.loc[:, "value"].astype(int)
        if comp_df[["tp", "comp", "value", "pred", "count"]].isna().any().any():
            logging.warning("Missing values in prediction and validation data")

        comp_df_sum = comp_df.groupby(id, as_index=False)[
            ["tp", "comp", "value", "pred", "count"]
        ].sum()

        comp_df_sum["precision"] = comp_df_sum["tp"] / comp_df_sum["pred"]
        comp_df_sum["recall"] = comp_df_sum["tp"] / comp_df_sum["value"]
        comp_df_sum["accuracy"] = comp_df_sum["comp"] / comp_df_sum["count"]

        return comp_df_sum.loc[:, [id, "precision", "recall", "accuracy"]]

    def _summ_metric(self, df):
        """Summarize model performance across all observations"""
        return (
            df[["precision", "recall", "accuracy"]]
            .agg(["mean", "min", "max", "median"])
            .reset_index(drop=False)
            .rename({"index": "summary"}, axis=1)
            .melt(id_vars="summary")
        )

    def _iterate_priority_categories(self, pred_comp, priority_categories=[]):
        """Generate precision, recall, and accuracy for each priority category"""

        if priority_categories == []:
            priority_categories = self.priority_categories

        if len(priority_categories) > 0:
            comp_df_filt = pred_comp.loc[
                pred_comp["variable"].isin(priority_categories)
            ]

            return self._calc_metric(comp_df_filt, id="variable")

    def _add_info_cols(self, df, constraint, model_id):
        """Add columns with constraint, metric, and model_id to dataframe"""
        df["constraint_val"] = constraint
        df["model_id"] = model_id
        return df

    def _concat_eval_results(self, list_df, index_col):
        """Concatenate dataframes of evaluation metrics and constraints"""
        df = pd.concat(list_df, axis=0)
        return df.pivot(
            index=[index_col, "model_id", "constraint_val"],
            columns="variable",
            values="value",
        ).reset_index(drop=False)
