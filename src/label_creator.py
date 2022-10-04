import os
import pandas as pd
import logging
from joblib import dump

from src.utils.utils import get_data, write_to_db
from setup_environment import db_dict, get_dbengine
from sklearn.preprocessing import MultiLabelBinarizer

from config.model_settings import LabelGeneratorConfig


class LabelGenerator:
    def __init__(
        self,
        schema_name: str,
        table_name: str,
        labels_path: str,
        train_table_name: str,
        target_col: str,
    ) -> None:
        self.schema_name = schema_name
        self.table_name = table_name
        self.labels_path = labels_path
        self.train_table_name = train_table_name
        self.target_col = target_col

        engine = get_dbengine(**db_dict)
        self.engine = engine

    @classmethod
    def from_dataclass_config(
        cls, config: LabelGeneratorConfig, labels_path: str
    ) -> "LabelGenerator":
        return cls(
            schema_name=config.SCHEMA_NAME,
            table_name=config.TABLE_NAME,
            labels_path=labels_path,
            train_table_name=config.TRAIN_TABLE_NAME,
            target_col=config.TARGET_COL,
        )

    def execute(self, run_date, schema_type, train_table_name=None):
        self.get_schema_name(schema_type)

        if train_table_name is not None:
            self.train_table_name = train_table_name

        labels_data = self.labels_from_db()

        self.convert_label_to_mlb_classes(labels_data, run_date)

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def labels_from_db(self):
        label_query = """select c.unique_id, c.cohort, c.cohort_type,
            c.train_validation_set,
            features_and_labels.triage_datetime, features_and_labels.ed_dx,
            features_and_labels.hopi,
            features_and_labels.code, features_and_labels.category
        from {schema}.cohorts c
        left join {schema}.{table} as features_and_labels
        on c.unique_id = features_and_labels.unique_id;""".format(
            schema=self.schema_name, table=self.train_table_name
        )

        data = get_data(label_query)

        return data

    def convert_label_to_mlb_classes(self, data, run_date):
        mlb = MultiLabelBinarizer(sparse_output=True)
        data[f"{self.target_col}"] = data[f"{self.target_col}"].apply(
            lambda x: x[1:-1].replace("'", "").split(", ")
        )
        list_of_cat_combinations = data[f"{self.target_col}"].tolist()
        df_fit = mlb.fit_transform(list_of_cat_combinations)

        y = pd.DataFrame(df_fit.todense(), columns=mlb.classes_)
        y = pd.concat(
            [
                data.loc[
                    :, ["unique_id", "train_validation_set", "cohort", "cohort_type"]
                ],
                y,
            ],
            axis=1,
        )
        logging.info(f"Shape of mlb classes {len(mlb.classes_)}")
        self._write_labels_as_pickle(y, run_date)
        self._write_labels_to_db(y, run_date, mlb.classes_)

    def _write_labels_to_db(self, df, run_date, mlb_categories):
        df = df.drop(mlb_categories, axis=1)
        df["run_date"] = run_date
        df["categories"] = ",".join(mlb_categories)

        write_to_db(df, self.engine, "labels", self.schema_name, "append", index=False)

    def _write_labels_as_pickle(self, y, run_date):
        # save as pickle
        filename = "_".join(["labels", run_date.strftime("%Y%m%d_%H%M%S%f")])
        dump(
            y,
            os.path.join(self.labels_path, filename + ".joblib"),
        )
