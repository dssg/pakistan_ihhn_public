from typing import Tuple, List
from abc import ABC, abstractmethod
from heapq import nlargest


class BaseFlow(ABC):
    def __init__(
        self,
        constraint: Tuple[int],
        category_col: str,
        schema_name: str,
        text_cols: List[str],
    ):
        self.constraint = constraint
        self.category_col = category_col
        self.schema_name = schema_name
        self.text_cols = text_cols

    @abstractmethod
    def execute_for_constraint(self):
        """Execute flow when inside of a constraint"""
        ...

    @abstractmethod
    def execute(self):
        """Execute flow starting a run"""
        ...

    @abstractmethod
    def filter_distance(self):
        """Filter text based on a distance measure"""
        ...

    def _get_n_highest_similarity(self, dist_dict, constraint):
        return nlargest(constraint, dist_dict, key=dist_dict.get)

    def _precision_recall_from_lists(self, top_k_codes, actual_codes):
        code_intersect = len(set(actual_codes) & set(top_k_codes))
        recall = code_intersect / len(actual_codes)
        precision = code_intersect / len(top_k_codes)
        return precision, recall
