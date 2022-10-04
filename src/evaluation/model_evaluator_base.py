import logging
from sqlalchemy import text
from abc import ABC

from src.utils.utils import write_to_db, get_data
from setup_environment import db_dict, get_dbengine


class ModelEvaluatorBase(ABC):
    def __init__(self, schema_name, id_var):
        self.schema_name = schema_name
        self.id_var = id_var

        # initialize engine
        engine = get_dbengine(**db_dict)
        self.engine = engine

    def get_schema_name(self, schema_type):
        if schema_type == "dev":
            self.schema_name = f"{self.schema_name}_{schema_type}"
        else:
            self.schema_name

    def _output_to_db(self, df, model_id, run_date, table_name, index=True):
        """Write output (scores, rankings, pred) to a database table"""
        df = self._add_model_information(df, model_id)

        # print scores to one table
        if index:
            write_to_db(
                df,
                self.engine,
                table_name,
                self.schema_name,
                "append",
                index=True,
                index_label=self.id_var,
            )
        else:
            write_to_db(
                df, self.engine, table_name, self.schema_name, "append", index=False
            )

        return table_name

    def _add_model_information(self, df, model_id):
        """Add model run date and model id"""
        df.loc[:, "model_id"] = model_id
        return df

    def _get_sql_cols(self, table_name):
        return get_data(
            """SELECT * FROM information_schema.columns
                                                WHERE table_schema = '{}'
                                                AND table_name   = '{}' AND
                                                column_name != '{}' """.format(
                self.schema_name, table_name, self.id_var
            )
        )["column_name"].tolist()

    def _mismatch_columns(self, df_cols, sql_cols):
        """Check columns of dataframe and db table match"""

        if len(sql_cols) == 0:
            return False
        elif (len(sql_cols) != len(df_cols)) & (len(sql_cols) > 1):
            return True
        elif (df_cols != sql_cols).any() & (len(sql_cols) > 1):
            return True
        elif (df_cols == sql_cols).all():
            return False
        else:
            logging.warning("Check dataframe and db match.")

    def _results_to_db(
        self,
        results,
        table_name,
        run_date,
        index_label="summary",
    ):
        """Write model results to the database for all metrics and constraints"""

        # add today's date
        results["run_date"] = run_date

        columns_to_add = [
            x + " numeric"
            # hardcoded as they should not change
            # even if self.metrics change, the columns should just
            # be blank to prevent issues when appending data later
            for x in ["precision", "recall", "accuracy"]
        ]

        with self.engine.begin() as connection:
            connection.execute(text("""SET ROLE "pakistan-ihhn-role" """))

            connection.execute(
                text(
                    """CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        model_id text,
                        constraint_val integer,
                        run_date timestamp,
                        {index_label} text,
                        {extra_cols},
                        CONSTRAINT {table}_pk PRIMARY KEY
                        (model_id, constraint_val, run_date, {index_label}))
                    """.format(
                        schema=self.schema_name,
                        table=table_name,
                        index_label=index_label,
                        extra_cols=",".join(columns_to_add),
                    )
                )
            )

        write_to_db(
            results,
            self.engine,
            table_name,
            self.schema_name,
            "append",
        )

    def _validation_to_db(self, valid_data, run_date, model_id):
        """Write validation data to database"""
        valid_data_exp = valid_data.copy()
        valid_data_exp["run_date"] = run_date
        self._output_to_db(
            valid_data_exp, model_id, run_date, self.validation_table_name
        )
