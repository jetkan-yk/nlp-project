"""
Overrides default hyperparameters for subtask 2.
"""
import sys

from subtask2.model2 import Model2

sys.path.append("../")

from config import Config


class Config2(Config):
    """
    Hyperparameters for subtask 2
    """

    MODEL = Model2
