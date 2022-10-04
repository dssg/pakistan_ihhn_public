#!/usr/bin/env python
"""
Setup Enviroment

Tools for connecting to the
database.

"""

from contextlib import contextmanager

import pandas as pd
import psycopg2
import yaml
from pkg_resources import resource_filename
from sqlalchemy import create_engine

db_setup_file = resource_filename(__name__, "/config/secret_default_profile.yaml")
example_db_setup_file = resource_filename(
    __name__, "./config/example_default_profile.yaml"
)

try:
    db_dict = yaml.safe_load(open(db_setup_file))
except IOError:
    print("Cannot find file")
    db_dict = yaml.safe_load(open(example_db_setup_file))


def get_dbengine(
    PGDATABASE="", PGHOST="", PGPORT=5432, PGPASSWORD="", PGUSER="", DBTYPE="postgresql"
):
    """
    Returns a sql engine

    Input
    -----
    PGDATABASE: str
    DB Name
    PGHOST: str
    hostname
    PGPASSWORD: str
    DB password
    DBTYPE: str
    type of database, default is posgresql

    Output
    ------
    engine: SQLalchemy engine
    """
    str_conn = "{dbtype}://{username}@{host}:{port}/{db}".format(
        dbtype=DBTYPE, username=PGUSER, db=PGDATABASE, host=PGHOST, port=PGPORT
    )

    return create_engine(str_conn)


@contextmanager
def connect_to_db(PGDATABASE="", PGHOST="", PGPORT=5432, PGUSER="", PGPASSWORD=""):
    """
    Connects to database
    Output
    ------
    conn: object
       Database connection.
    """
    try:
        engine = get_dbengine(
            PGDATABASE=PGDATABASE,
            PGHOST=PGHOST,
            PGPORT=PGPORT,
            PGUSER=PGUSER,
            PGPASSWORD=PGPASSWORD,
        )
        conn = engine.connect()

        yield conn
    except psycopg2.Error:
        raise SystemExit("Cannot Connect to DB")
    else:
        conn.close()


def run_query(query):
    """
    Runs a query on the database and returns
    the result in a dataframe.
    """
    with connect_to_db(**db_dict) as conn:
        data = pd.read_sql(query, conn)
    return data


def test_database_connect():
    """
    test database connection
    """
    with connect_to_db(**db_dict) as conn:
        query = "select * from raw.codes limit 10"
        data = pd.read_sql_query(query, conn)
        assert len(data) > 1
