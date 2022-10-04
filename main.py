from setup_environment import db_dict, get_dbengine
from src.baselines.run_baseline_jaro import BaselineJaro
from src.baselines.run_baseline_nearest_text import BaselineNearestText
from src.baselines.run_baseline_description_overlap import BaselineDescOverlap
from src.model_visualizer import ModelVisualizer
from src.label_creator import LabelGenerator
from src.matrix_generator import MatrixGenerator
from src.time_splitter import TimeSplitter
from src.retrain_splitter import RetrainSplitter
from src.cohort_builder import CohortBuilder
from src.train_model import ModelTrainer
from src.evaluation.model_ranker import ModelRanker
from src.evaluation.model_scorer import ModelScorer
from src.evaluation.model_predictor import ModelPredictor
from src.evaluation.model_evaluator import ModelEvaluator
from src.feature_importance import FeatureImportanceSklearn
from src.infrastructure.process_official_codes import ProcessOfficialCodes
from src.infrastructure.process_admissions import ProcessAdmissions
from src.infrastructure.process_transactions import ProcessTransactions
from src.infrastructure.process_xlsx import ProcessXlsx
from src.infrastructure.process_csv import ProcessCsv
from src.infrastructure.process_codes import ProcessCodes
from config.model_settings import (
    BaselineConfig,
    CohortBuilderConfig,
    ProcessAdmissionsConfig,
    ProcessCodesConfig,
    ProcessCsvConfig,
    ProcessTransactionsConfig,
    ProcessXlsxConfig,
    TimeSplitterConfig,
    RetrainSplitterConfig,
    MatrixGeneratorConfig,
    ModelTrainerConfig,
    ModelEvaluatorConfig,
    FeatureImportanceConfig,
    ModelScorerConfig,
    ModelRankerConfig,
    RetrainingConfig,
    LabelGeneratorConfig,
    ModelVisualizerConfig,
    ProcessOfficialCodesConfig,
)
import click
import gc
import datetime as dt
import logging
from dotenv import load_dotenv

load_dotenv()


class ProcessXlsxFlow:
    def __init__(self, xlsx_directory, csvs_directory):
        self.xlsx_directory = xlsx_directory
        self.csvs_directory = csvs_directory
        self.config = ProcessXlsxConfig()

    def execute(self):
        xlsx_processor = ProcessXlsx.from_dataclass_config(
            self.config,
        )

        xlsx_processor.execute(self.xlsx_directory, self.csvs_directory)


class ProcessCsvFlow:
    def __init__(self, raw_csvs_directory, processed_csvs_directory):
        self.raw_csvs_directory = raw_csvs_directory
        self.processed_csvs_directory = processed_csvs_directory
        self.config = ProcessCsvConfig()

    def execute(self):
        csv_processor = ProcessCsv.from_dataclass_config(
            self.config,
        )
        csv_processor.execute(self.raw_csvs_directory, self.processed_csvs_directory)


class ProcessOfficialCodesFlow:
    def __init__(self, raw_csvs_directory, processed_csvs_directory):
        self.raw_csvs_directory = raw_csvs_directory
        self.processed_csvs_directory = processed_csvs_directory
        self.config = ProcessOfficialCodesConfig()

    def execute(self):
        csv_processor = ProcessOfficialCodes.from_dataclass_config(
            self.config,
        )
        csv_processor.execute(self.raw_csvs_directory, self.processed_csvs_directory)


class ProcessCodesFlow:
    def __init__(self):
        self.config = ProcessCodesConfig()

    def execute(self):
        engine = get_dbengine(**db_dict)

        codes_processor = ProcessCodes.from_dataclass_config(
            self.config,
        )

        codes_processor.execute(engine)


class ProcessTransactionsFlow:
    def __init__(self):
        self.config = ProcessTransactionsConfig()

    def execute(self):
        engine = get_dbengine(**db_dict)

        transactions_processor = ProcessTransactions.from_dataclass_config(
            self.config,
        )

        transactions_processor.execute(engine)


class ProcessAdmissionsFlow:
    def __init__(self):
        self.config = ProcessAdmissionsConfig()

    def execute(self):
        engine = get_dbengine(**db_dict)

        admissions_processor = ProcessAdmissions.from_dataclass_config(
            self.config,
        )

        admissions_processor.execute(engine)


class BaselinesFlow:
    def __init__(self):
        self.config = BaselineConfig()

    def execute(self, distance_metric):
        for metric in list(distance_metric):
            logging.info(f"Generating baseline for {metric} distance")
            if metric == "jaro":
                return BaselineJaro(metric)
            if metric == "nearest_text":
                return BaselineNearestText(metric)
            if metric == "desc_overlap":
                return BaselineDescOverlap(metric)


class TimeSplitterFlow:
    def __init__(self):
        self.config = TimeSplitterConfig()

    def execute(
        self,
    ):
        return TimeSplitter.from_dataclass_config(
            self.config,
        )


class RetrainSplitterFlow:
    def __init__(self):
        self.config = RetrainSplitterConfig()

    def execute(
        self,
    ):
        return RetrainSplitter.from_dataclass_config(
            self.config,
        )


class CohortBuilderFlow:
    def __init__(self):
        self.config = CohortBuilderConfig()

    def execute(self):
        return CohortBuilder.from_dataclass_config(
            self.config,
        )


class BuildFeaturesFlow:
    def __init__(self):
        self.config = MatrixGeneratorConfig()

    def execute(self, text_features_path, features_path, labels_path):
        return MatrixGenerator.from_dataclass_config(
            self.config,
            text_features_path,
            features_path,
            labels_path,
        )


class LabelGeneratorFlow:
    def __init__(self):
        self.config = LabelGeneratorConfig()

    def execute(self, labels_path):
        return LabelGenerator.from_dataclass_config(self.config, labels_path)


class FeatureImportanceFlow:
    def __init__(self):
        self.config = FeatureImportanceConfig()

    def execute(
        self,
        model_name,
        X_valid,
        models_directory,
        model_id,
        start_datetime,
        schema_type,
    ):
        feat_importance_sklearn = FeatureImportanceSklearn.from_dataclass_config(
            self.config,
        )

        return feat_importance_sklearn.execute(
            model_name,
            X_valid,
            models_directory,
            model_id,
            start_datetime,
            schema_type,
        )


class ModelTrainerFlow:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def execute(self):
        return ModelTrainer.from_dataclass_config(self.config)


class ModelScorerFlow:
    def __init__(self):
        self.config = ModelScorerConfig()

    def execute(self, labels_directory, features_directory):
        return ModelScorer.from_dataclass_config(
            self.config, labels_directory, features_directory
        )


class ModelRankerFlow:
    def __init__(self):
        self.config = ModelRankerConfig()

    def execute(self):
        return ModelRanker.from_dataclass_config(
            self.config,
        )


class ModelPredictorFlow:
    def __init__(self):
        self.config = RetrainingConfig()

    def execute(self):
        return ModelPredictor.from_dataclass_config(
            self.config,
        )


class ModelEvaluatorFlow:
    def __init__(self):
        self.config = ModelEvaluatorConfig()

    def execute(self):
        return ModelEvaluator.from_dataclass_config(
            self.config,
        )


class ModelVisualizerFlow:
    def __init__(self, plots_directory):
        self.config = ModelVisualizerConfig()
        self.eval_config = ModelEvaluatorConfig()
        self.plots_directory = plots_directory

    def execute(self, run_date, schema_type):
        model_visualizer = ModelVisualizer.from_dataclass_config(
            self.config, self.eval_config
        )

        model_visualizer.execute(
            schema_type, path=self.plots_directory, run_date=run_date
        )


@click.command("process-xlsx", help="Process xlsx")
@click.argument("xlsx_directory")
@click.argument("csvs_directory")
def process_xlsx(xlsx_directory, csvs_directory):
    ProcessXlsxFlow(xlsx_directory, csvs_directory).execute()


@click.command("process-csv", help="Process csv")
@click.argument("raw_csvs_directory")
@click.argument("processed_csvs_directory")
def process_csv(raw_csvs_directory, processed_csvs_directory):
    ProcessCsvFlow(raw_csvs_directory, processed_csvs_directory).execute()


@click.command("process-official-codes", help="Process official codes textfiles")
@click.argument("raw_text_directory")
@click.argument("raw_csvs_directory")
def process_official_codes(raw_text_directory, raw_csvs_directory):
    ProcessOfficialCodesFlow(raw_text_directory, raw_csvs_directory).execute()


@click.command("process-codes", help="Generate processed codes table")
def process_codes():
    ProcessCodesFlow().execute()


@click.command("process-transactions", help="Generate processed transactions table")
def process_transactions():
    ProcessTransactionsFlow().execute()


@click.command("process-admissions", help="Generate processed admissions table")
def process_admissions():
    ProcessAdmissionsFlow().execute()


@click.command("run-baselines", help="Generate baseline models")
@click.option(
    "--distance_metric",
    "-",
    type=click.Choice(["jaro", "desc_overlap", "nearest_text"]),
    multiple=True,
)
def run_baselines(distance_metric):
    engine = get_dbengine(**db_dict)

    baseline = BaselinesFlow().execute(distance_metric)
    baseline.execute(engine)


@click.command("time-splitter", help="Generate time splitting table")
@click.argument("schema_type")
def time_splitter(schema_type):
    time_splitter = TimeSplitterFlow().execute()
    time_splitter.execute(schema_type)


@click.command("retrain-splitter", help="Generate retraining splits")
@click.argument("schema_type")
def retrain_splitter(schema_type):
    retrain_splitter = RetrainSplitterFlow().execute()
    retrain_splitter.execute(schema_type)


@click.command("cohort-builder", help="Generate cohorts for time splits")
@click.argument("schema_type")
def cohort_builder(schema_type):
    engine = get_dbengine(**db_dict)

    time_splitter = TimeSplitterFlow().execute()
    train_validation_list = time_splitter.execute(schema_type)

    cohort_builder = CohortBuilderFlow().execute()
    cohort_builder.execute(train_validation_list, schema_type, engine)


@click.command("label-generator", help="Generate labels")
@click.argument("schema_type")
@click.argument("labels_directory", envvar="LABELPATH")
def label_generator(schema_type, labels_directory):
    label_generator = LabelGeneratorFlow().execute(labels_directory)
    start_datetime = dt.datetime.now()
    label_generator.execute(start_datetime, schema_type)


@click.command("model-visualizer", help="Visualize results of model")
@click.argument("plots_directory", envvar="PLOTPATH")
def model_visualizer(plots_directory):
    ModelVisualizerFlow(plots_directory).execute()


@click.command("build-features", help="Generate features")
@click.argument("schema_type")
@click.argument("text_features_directory", envvar="TEXTFEATUREPATH")
@click.argument("features_directory", envvar="FEATUREPATH")
@click.argument("labels_directory", envvar="LABELPATH")
def build_features(
    schema_type, text_features_directory, features_directory, labels_directory
):
    matrix_generator = BuildFeaturesFlow().execute(
        text_features_directory, features_directory, labels_directory
    )
    matrix_generator.execute_train_valid_set(schema_type)


@click.command("train-model", help="Train one model")
def train_model():
    (
        train_model,
        valid_df,
        valid_y,
        model_id,
        model_name,
        mlb,
    ) = ModelTrainerFlow().execute()
    ModelEvaluatorFlow().execute(
        train_model, valid_df, valid_y, model_id, model_name, mlb
    )


@click.command("run-pipeline", help="Run full pipeline")
@click.option(
    "--schema_type",
    "-",
    type=click.Choice(["dev", "prod"]),
    multiple=False,
)
@click.argument("plots_directory", envvar="PLOTPATH")
@click.argument("models_directory", envvar="MODELPATH")
@click.argument("text_features_directory", envvar="TEXTFEATUREPATH")
@click.argument("features_directory", envvar="FEATUREPATH")
@click.argument("labels_directory", envvar="LABELPATH")
def run_pipeline(
    schema_type,
    plots_directory,
    models_directory,
    text_features_directory,
    features_directory,
    labels_directory,
):
    start_datetime = dt.datetime.now()
    logging.info(f"Starting pipeline at {start_datetime}")

    engine = get_dbengine(**db_dict)

    time_splitter = TimeSplitterFlow().execute()
    train_validation_list = time_splitter.execute(schema_type, engine)
    cohort_builder = CohortBuilderFlow().execute()
    cohort_builder.execute(train_validation_list, schema_type, engine)

    label_generator = LabelGeneratorFlow().execute(labels_directory)
    label_generator.execute(start_datetime, schema_type)

    # this is the matrix generator
    build_features = BuildFeaturesFlow().execute(
        text_features_directory, features_directory, labels_directory
    )
    train_validation_set = build_features.execute_train_valid_set(schema_type)

    # loop for time splits
    model_output = []
    for i in train_validation_set:
        start_model_datetime = dt.datetime.now()
        (
            validation_csr,
            full_features_csr,
            valid_labels,
            train_labels,
        ) = build_features.execute(i, start_datetime, schema_type)
        logging.info(f"Starting pipeline for model {i} {start_model_datetime}")

        model_trainer = ModelTrainerFlow().execute()
        model_output += model_trainer.train_all_models(
            i,
            full_features_csr,
            train_labels.drop(
                ["unique_id", "cohort", "cohort_type", "train_validation_set"], axis=1
            ),
            models_directory,
            start_datetime,
            schema_type,
        )

        del full_features_csr  # noqa F821
        del validation_csr  # noqa F821
        del train_labels  # noqa F821
        del valid_labels  # noqa F821
        gc.collect()

    logging.info("Getting model output")
    for model_id, model_name, i in model_output:
        logging.info(f"Training and evaluating model {model_id}")

        feature_importance = FeatureImportanceFlow()
        feature_importance.execute(
            model_name,
            models_directory,
            model_id,
            start_datetime,
            i,
            schema_type,
        )

        model_scorer = ModelScorerFlow().execute(labels_directory, features_directory)
        valid_labels = model_scorer.execute(
            model_id,
            model_name,
            start_datetime,
            models_directory,
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

    ModelVisualizerFlow(plots_directory).execute(start_datetime, schema_type)

    end_datetime = dt.datetime.now()
    logging.info(f"Ending pipeline at {end_datetime}")
    logging.info(f"Total time ellapsed: {end_datetime - start_datetime}")


@click.command("run-retraining", help="Retrain pipeline")
@click.option(
    "--schema_type",
    "-",
    type=click.Choice(["dev", "prod"]),
    multiple=False,
)
@click.argument("models_directory", envvar="MODELPATH")
@click.argument("text_features_directory", envvar="TEXTFEATUREPATH")
@click.argument("features_directory", envvar="FEATUREPATH")
@click.argument("labels_directory", envvar="LABELPATH")
@click.argument("output_directory", envvar="OUTPUTPATH")
def run_retraining(
    schema_type,
    models_directory,
    text_features_directory,
    features_directory,
    labels_directory,
    output_directory,
):
    start_datetime = dt.datetime.now()
    logging.info(f"Starting retraining at {start_datetime}")

    engine = get_dbengine(**db_dict)

    retrain_splitter = RetrainSplitterFlow().execute()
    train_validation_list, retrain_table_name = retrain_splitter.execute(
        schema_type, engine
    )
    cohort_builder = CohortBuilderFlow().execute()
    cohort_builder.execute(
        train_validation_list, schema_type, engine, retrain_table_name
    )

    label_generator = LabelGeneratorFlow().execute(labels_directory)
    label_generator.execute(start_datetime, schema_type, retrain_table_name)

    # this is the matrix generator
    build_features = BuildFeaturesFlow().execute(
        text_features_directory, features_directory, labels_directory
    )
    train_validation_set = build_features.execute_train_valid_set(schema_type)

    # loop for time splits
    for i in train_validation_set:
        (
            validation_csr,
            full_features_csr,
            valid_labels,
            train_labels,
        ) = build_features.execute(i, start_datetime, schema_type, retrain_table_name)

        model_trainer = ModelTrainerFlow().execute()
        model_id, model_name, train_valid_split = model_trainer.train_best_model(
            i,
            full_features_csr,
            train_labels.drop(
                ["unique_id", "cohort", "cohort_type", "train_validation_set"], axis=1
            ),
            models_directory,
            start_datetime,
            schema_type,
        )

        model_scorer = ModelScorerFlow().execute(labels_directory, features_directory)
        model_scorer.execute(
            model_id,
            model_name,
            start_datetime,
            models_directory,
            schema_type,
            train_valid_split,
        )

        model_ranker = ModelRankerFlow().execute()
        model_ranker.execute(model_id, start_datetime, schema_type)

        model_predictor = ModelPredictorFlow().execute()
        model_predictor.execute(model_id, start_datetime, schema_type, output_directory)

    end_datetime = dt.datetime.now()
    logging.info(f"Ending pipeline at {end_datetime}")
    logging.info(f"Total time ellapsed: {end_datetime - start_datetime}")


@click.group("pakistan-ihhn", help="Library for the pakistan-ihhn DSSG project")
@click.pass_context
def cli(ctx):
    ...


cli.add_command(process_xlsx)
cli.add_command(process_csv)
cli.add_command(process_official_codes)
cli.add_command(process_codes)
cli.add_command(process_transactions)
cli.add_command(process_admissions)
cli.add_command(train_model)
cli.add_command(time_splitter)
cli.add_command(retrain_splitter)
cli.add_command(cohort_builder)
cli.add_command(build_features)
cli.add_command(train_model)
cli.add_command(run_pipeline)
cli.add_command(run_retraining)
cli.add_command(model_visualizer)
cli.add_command(label_generator)
cli.add_command(run_baselines)

if __name__ == "__main__":
    cli()
