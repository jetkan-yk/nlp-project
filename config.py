"""
Hyperparameter configs
"""
from enum import Enum, auto


class Pipeline(Enum):
    CLASSIFICATION = auto()
    SUMMARISATION = auto()
    DEFAULT = auto()


class Sampling(Enum):
    OVERSAMPLING = auto()
    SHUFFLE = auto()


class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()


class NcgConfig:
    """
    Default hyperparameter configs
    """

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.1
    MODEL = None
    MOMENTUM = 0.9
    OPTIMIZER = Optimizer.SGD
    PIPELINE = Pipeline.DEFAULT
    SAMPLING = Sampling.SHUFFLE
    TRAIN_RATIO = 0.8
