import logging
import re
from datetime import timedelta
from typing import Any
import ohio.ext.pandas  # noqa: F401
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sqlalchemy import text
from setup_environment import connect_to_db, db_dict

logging.basicConfig(level=logging.INFO)


def date_range(start, end):
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days


def read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Read csv ensuring that nan's are not parsed
    """

    return pd.read_csv(
        path, sep=",", low_memory=False, encoding="utf-8", na_filter=False, **kwargs
    )


def write_csv(df: pd.DataFrame, path: str, **kwargs: Any) -> None:
    """
    Write csv to provided path ensuring that the correct encoding and escape
    characters are applied.

    Needed when csv's have text with html tags in it and lists inside cells.
    """
    df.to_csv(
        path,
        index=False,
        na_rep="",
        sep=",",
        line_terminator="\n",
        encoding="utf-8",
        escapechar="\r",
        **kwargs,
    )


def read_excel(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Read excel ensuring that nan's are not parsed
    """
    return pd.read_excel(
        path,
        header=0,
        **kwargs,
    )


def get_data(query):
    """
    Pulls data from the db based on the query
    Input
    -----
    query: str
       SQL query from the database
    Output
    ------
    data: DataFrame
       Dump of Query into a DataFrame
    """

    with connect_to_db(**db_dict) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def write_to_db(
    df, engine, table_name, schema_name, table_behaviour, index=False, **kwargs
):

    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text("""SET ROLE "pakistan-ihhn-role" """))
            df.pg_copy_to(
                name=table_name,
                schema=schema_name,
                con=conn,
                if_exists=table_behaviour,
                index=index,
                **kwargs,
            )


def gen_even_slices(n, n_packs, *, n_samples=None):
    """Generator to create n_packs slices going up to n.
    Parameters
    ----------
    n : int
    n_packs : int
        Number of slices to generate.
    n_samples : int, default=None
        Number of samples. Pass n_samples when the slices are to be used for
        sparse matrix indexing; slicing off-the-end raises an exception, while
        it works for NumPy arrays.
    Yields
    ------
    slice
    See Also
    --------
    gen_batches: Generator to create slices containing batch_size elements
        from 0 to n.
    Examples
    --------
    >>> from sklearn.utils import gen_even_slices
    >>> list(gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(gen_even_slices(10, 10))
    [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
    >>> list(gen_even_slices(10, 5))
    [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
    >>> list(gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def parallel_apply(df, func, n_jobs=-1, **kwargs):
    """Pandas apply in parallel using joblib.
    Args:
        df: Pandas DataFrame, Series, or any object that supports slicing and apply.
        func: Callable to apply
        n_jobs: Desired number of workers. Default value -1 uses all available cores.
        **kwargs: Any additional parameters will be supplied to the apply function
    Returns:
        Same as for normal Pandas DataFrame.apply()
    """

    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df.iloc[s], func, **kwargs)
            for s in gen_even_slices(len(df), effective_n_jobs(n_jobs))
        )
        return pd.concat(ret)


def execute_hash(df, colname):
    """returns a hashed column for each columns with type: str"""
    df[colname + "_hash"] = parallel_apply(df[colname], hash_row, n_jobs=-1)

    return df


def hash_row(row):
    """
    Creates a hash for a specific row and column
    Args:
        row: a row of the dataframe
    Returns:
        row with hash complete
    """
    try:
        row_hash = hash(row)
    except TypeError:
        row_hash = 0
    return row_hash


def iterate_categories(code_list):
    """Generates a set of categories for given codes"""

    # check if missing prior to any additional steps
    if code_list is None or code_list == [None]:
        return None

    # convert to set for iterating
    if type(code_list) == list:
        code_list = set(code_list)
    else:
        code_list = set(list(code_list.split(",")))

    # iterate through set and get categories
    return [get_category(individual_code) for individual_code in code_list]


def get_category(code):
    """Takes in a code and creates a category."""
    if re.search(r"\.", code):
        return re.sub(r"\..*", "", code)
    elif len(code) == 3:
        return code
    else:
        return None


def get_official_codes():
    official_code_query = """
    select icd_10_cm, description_long
    from raw.icd10cm_order_2023 where LENGTH(icd_10_cm) = 3;"""

    official_code_data = get_data(official_code_query)
    return official_code_data
