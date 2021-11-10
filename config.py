"""
Hyperparameter configs
"""
from enum import Enum, auto

from subtask1.scibert import SciBert


class Pipeline(Enum):
    CLASSIFICATION = auto()
    DEFAULT = auto()


class Sampling(Enum):
    OVERSAMPLING = auto()
    SHUFFLE = auto()


class Optimizer(Enum):
    ADAM = auto()
    ADAMW = auto()
    SGD = auto()


class Model(Enum):
    NAIVEBAYES = None  # TODO
    SCIBERT = SciBert


DEFAULT = dict(
    SUBTASK=None,
    BATCH_SIZE=32,
    EPOCHS=10,
    LEARNING_RATE=0.1,
    MODEL=None,
    MOMENTUM=0.9,
    OPTIMIZER=Optimizer.SGD,
    PIPELINE=Pipeline.DEFAULT,
    SAMPLING=Sampling.SHUFFLE,
    TRAIN_RATIO=0.8,
)

SBERT_ADAM_OSMP_1 = DEFAULT | dict(
    SUBTASK=1,
    EPOCHS=2,
    LEARNING_RATE=2e-5,
    MODEL=Model.SCIBERT,
    OPTIMIZER=Optimizer.ADAM,
    PIPELINE=Pipeline.CLASSIFICATION,
    SAMPLING=Sampling.OVERSAMPLING,
)

NB_ADAMW_OSMP_1 = DEFAULT | dict(
    SUBTASK=1,
    EPOCHS=2,
    LEARNING_RATE=2e-5,
    MODEL=Model.NAIVEBAYES,
    OPTIMIZER=Optimizer.ADAMW,
    PIPELINE=Pipeline.CLASSIFICATION,
    SAMPLING=Sampling.OVERSAMPLING,
)

"""
TODO: Copy and overwrite this sample new config, then add the new config into NcgConfigs below

NEW_CONFIG = DEFAULT | dict(
    SUBTASK=None,
    BATCH_SIZE=32,
    EPOCHS=10,
    LEARNING_RATE=0.1,
    MODEL=None,
    MOMENTUM=0.9,
    OPTIMIZER=Optimizer.SGD,
    PIPELINE=Pipeline.DEFAULT,
    SAMPLING=Sampling.SHUFFLE,
    TRAIN_RATIO=0.8,
)
"""

NcgConfigs = [DEFAULT, SBERT_ADAM_OSMP_1, NB_ADAMW_OSMP_1]
