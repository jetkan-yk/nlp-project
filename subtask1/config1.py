"""
Overrides default hyperparameters for subtask 1.
"""
import sys

from subtask1.scibert import SciBert

sys.path.append("../")

from config import NcgConfig, Optimizer, Pipeline, Sampling


class Config1(NcgConfig):
    """
    Hyperparameters for subtask 1
    """

    EPOCHS = 2
    LEARNING_RATE = 5e-5
    MODEL = SciBert
    OPTIMIZER = Optimizer.ADAM
    PIPELINE = Pipeline.CLASSIFICATION
    SAMPLING = Sampling.OVERSAMPLING
