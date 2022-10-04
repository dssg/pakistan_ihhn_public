import logging
import os
import matplotlib.pyplot as plt
import datetime as dt

from config.model_settings import ModelVisualizerConfig, ModelEvaluatorConfig
from src.utils.utils import get_data, write_to_db
from setup_environment import db_dict, get_dbengine


class ModelVisualizer:
    def __init__(
        self,
        plot,
        plot_metrics,
        plots_schema_name,
        plots_table_name,
        results_schema_name,
        results_table_name,
    ) -> None:
        self.plot = plot
        self.plot_metrics = plot_metrics
        self.plots_table_name = plots_table_name
        self.plots_schema_name = plots_schema_name
        self.results_schema_name = results_schema_name
        self.results_table_name = results_table_name

        # initialize engine
        engine = get_dbengine(**db_dict)
        self.engine = engine

    @classmethod
    def from_dataclass_config(
        cls, config: ModelVisualizerConfig, eval_config: ModelEvaluatorConfig
    ) -> "ModelVisualizer":
        """Imports data from the config class"""
        return cls(
            plot=config.PLOT,
            plot_metrics=config.PLOT_METRICS,
            plots_table_name=config.PLOTS_TABLE_NAME,
            plots_schema_name=config.PLOTS_SCHEMA_NAME,
            results_schema_name=eval_config.SCHEMA_NAME,
            results_table_name=eval_config.TABLE_NAME,
        )

    def execute(
        self,
        schema_type,
        path=None,
        all_models=True,
        last_run_model=None,
        run_date=None,
        model_ids=[],
        figsize=(10, 8),
    ):
        """
        Creates plot of average precision and recall for a model run and model ID.

        Arguments
        ---------
            schema_type: str
                the schema type to be used
            path : str
                file path on server to save plots
            all_models : boolean
                generate plots for all models
            run_date : date
                specific run date
            last_run_model : str
                model id for returning the most recent run
            model_ids : list
                list of model ids, empty if all_models = True
            figsize : tuple
                tuple for figure size of plot
        """
        df = self.get_results(
            schema_type, run_date=run_date, last_run_model=last_run_model
        )

        if all_models:
            model_ids = df["model_id"].unique()

        for id in model_ids:
            for metric in self.plot_metrics:
                logging.info("Generating plot for {} and {}".format(id, metric))

                if self.plot:
                    df_filt = self._filter_df(df, metric, id)
                    self._generate_plot(df_filt, id, metric, path=path, figsize=figsize)

    def get_results(self, schema_type, run_date=None, last_run_model=None):
        self.get_schema_name(schema_type)
        """Query the results data from the database for a specific run date and time"""
        if run_date is None and last_run_model is None:
            filter_query = """WHERE run_date = (SELECT MAX(run_date)
                        FROM {schema}.{table})""".format(
                schema=self.results_schema_name, table=self.results_table_name
            )
        elif last_run_model is not None:
            filter_query = """WHERE model_id = '{last_run_model}' AND
                                run_date = (SELECT MAX(run_date)
                                FROM {schema}.{table}
                                WHERE model_id = '{last_run_model}')""".format(
                schema=self.results_schema_name,
                table=self.results_table_name,
                last_run_model=last_run_model,
            )
        else:
            filter_query = f"""WHERE date_trunc('seconds', run_date) =
            TO_TIMESTAMP('{run_date}', 'YYYY-MM-DD HH24:MI:SS')"""

        df = get_data(
            """SELECT * FROM {schema}.{table}
                        {filter_query}""".format(
                schema=self.results_schema_name,
                table=self.results_table_name,
                filter_query=filter_query,
            )
        )

        if df.shape[0] == 0:
            logging.warning("Empty dataframe. Check run date or model_id if specified.")

        return df

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.plots_schema_name = f"{self.plots_schema_name}_{schema_type}"
            self.results_schema_name = f"{self.results_schema_name}_{schema_type}"
        else:
            self.plots_schema_name

    def _generate_plot(self, df, model_id, metric, path, figsize):
        """Generate plot of average precision and recall"""
        plt.figure(figsize=figsize)
        plt.plot(df["constraint_val"], df["precision"], label="precision")
        plt.plot(df["constraint_val"], df["recall"], label="recall")
        plt.ylabel(f"{metric} precision and recall", fontsize=16)
        plt.xlabel("Constraint", fontsize=16)
        plt.title(f"Precision and recall for {model_id}", fontsize=18)
        plt.ylim((0, 1))
        plt.legend(fontsize=16)
        plt.show()

        if path is not None:
            plot_path = self._plot_path(path, model_id, metric, df)
            plt.savefig(plot_path)

            # write to db
            self._generate_db_tbl(df, plot_path)

    def _filter_df(self, df, metric, model_id):
        "Filter dataframe of results for summary method and model ID"
        if metric not in df["summary"].unique():
            logging.warning(f"Summary value {metric} not in dataframe")
            return df

        return df.loc[(df["summary"] == metric) & (df["model_id"] == model_id), :]

    def _plot_path(self, path, model_id, metric, df):
        """Generate plot path fro model ID, metric, and run date"""
        run_date = df["run_date"].dt.strftime("%Y%m%d_%H%M%S%f").unique()[0]

        fname = "_".join([model_id, metric, run_date]) + ".png"
        return os.path.join(path, fname)

    def _generate_db_tbl(self, df, plot_path):
        """Write plot information and file path to the database"""
        plots_df = df.loc[:, ["model_id", "run_date", "summary"]].drop_duplicates()
        plots_df["date_generated"] = dt.datetime.now()
        plots_df["plot_location"] = plot_path

        write_to_db(
            plots_df,
            self.engine,
            self.plots_table_name,
            self.plots_schema_name,
            "append",
        )
