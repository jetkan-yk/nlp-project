"""
Overrides default hyperparameters for subtask 1.
"""

from config import NcgConfig, Optimizer, Pipeline, Sampling, Model

from subtask1.scibert import SciBert


class Config1(NcgConfig):
    """
    Hyperparameters for subtask 1
    """

    EPOCHS = 2
    LEARNING_RATE = 2e-5
    MODEL = Model.NAIVEBAYES
    OPTIMIZER = Optimizer.ADAMW
    PIPELINE = Pipeline.CLASSIFICATION
    SAMPLING = Sampling.OVERSAMPLING
