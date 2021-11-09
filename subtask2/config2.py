"""
Overrides default hyperparameters for subtask 2.
"""

from config import NcgConfig

from subtask2.model2 import Model2


class Config2(NcgConfig):
    """
    Hyperparameters for subtask 2
    """

    MODEL = Model2
