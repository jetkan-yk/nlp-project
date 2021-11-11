"""
Overrides default hyperparameters for subtask 1.
"""

from config import NcgConfig, Optimizer, Pipeline, Sampling

from subtask1.scibert import SciBert
from subtask1.Longformer import Longformer

class Config1(NcgConfig):
    """
    Hyperparameters for subtask 1
    """

    EPOCHS = 2
    LEARNING_RATE = 5e-5
    MODEL = Longformer #SciBert
    OPTIMIZER = Optimizer.ADAM
    PIPELINE = Pipeline.SUMMARISATION #Pipeline.CLASSIFICATION
    SAMPLING = Sampling.OVERSAMPLING
