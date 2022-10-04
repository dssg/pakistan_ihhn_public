from itertools import chain
import pandas as pd
from src.utils.utils import get_data


class Filter:
    @staticmethod
    def filter_pregnant_patients(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows which are pregnant

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with `Pregnancies` removed
        """

        return (
            df.assign(
                Pregnancies=(
                    df.triagecomplaint.apply(
                        lambda complaint: "pregnancy" in str(complaint).lower()
                    )
                )
            )
            .query("Pregnancies == False")
            .drop(["Pregnancies"], axis=1)
        )

    @staticmethod
    def filter_child_patients(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows which are children

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with `Child` removed
        """

        return (
            df.assign(Child=(df.age_years.apply(lambda age: float(age) <= 15)))
            .query("Child == False")
            .drop(["Child"], axis=1)
        )

    @staticmethod
    def filter_non_standard_codes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows which contain non-standard categories
        """

        def _get_non_standard_categories(labels_data):
            cats = labels_data["category"].apply(
                lambda x: x[1:-1].replace("'", "").split(",")
            )
            unique_cats = pd.DataFrame(
                data=chain.from_iterable(cats), columns=["category"]
            )["category"].unique()
            return [i for i in unique_cats if len(i) != 3]

        non_standard_cats = _get_non_standard_categories(df)
        return (
            df.assign(
                non_standard_codes=(
                    df.category.apply(
                        lambda cat: any(
                            str_ in cat[1:-1].split(",") for str_ in non_standard_cats
                        )
                    )
                )
            )
            .query("non_standard_codes == False")
            .drop(["non_standard_codes"], axis=1)
        )

    def filter_priority_categories(df: pd.DataFrame, no_of_occurrences) -> pd.DataFrame:
        """
        Filter out rows which contain non-priority categories
        """

        def _list_priority_categories(no_of_occurrences):
            """Getting priority codes as a list"""
            priority_cat_query = f"""select icd_10_cm from raw.priority_codes
            where count >= {no_of_occurrences};"""
            priority_cats = get_data(priority_cat_query)
            return list(priority_cats["icd_10_cm"]) + ["999"]

        priority_cats = _list_priority_categories(no_of_occurrences)
        return df.assign(
            priority_cats=(
                df.category.apply(
                    lambda cat: any(
                        str_.lower() in cat[1:-1].replace("'", "").split(",")
                        for str_ in priority_cats
                    )
                )
            )
        ).query("priority_cats == True")
