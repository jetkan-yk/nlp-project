"""
Overrides default hyperparameters for subtask 1.
"""
import sys

sys.path.append("../")

from config import Config, Optimizer, Pipeline, Sampling


class Config1(Config):
    """
    Hyperparameters for subtask 1
    """

    EPOCHS = 2
    LEARNING_RATE = 5e-5
    OPTIMIZER = Optimizer.ADAM
    PIPELINE = Pipeline.CLASSIFICATION
    SAMPLING = Sampling.OVERSAMPLING
