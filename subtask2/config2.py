"""
Overrides default hyperparameters for subtask 2.
"""
import sys

from subtask2.model2 import Model2

sys.path.append("../")

from config import NcgConfig


class Config2(NcgConfig):
    """
    Hyperparameters for subtask 2
    """

    MODEL = Model2
