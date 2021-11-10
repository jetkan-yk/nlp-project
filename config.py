"""
Hyperparameter configs
"""
from enum import Enum, auto

from sklearn.naive_bayes import MultinomialNB

from subtask1.scibert import SciBert


class Pipeline(Enum):
    CLASSIFICATION = auto()


class Sampling(Enum):
    OVERSAMPLING = auto()
    SHUFFLE = auto()


class Optimizer(Enum):
    ADAM = auto()
    ADAMW = auto()
    SGD = auto()


class Model(Enum):
    NAIVE_BAYES = MultinomialNB
    SCIBERT = SciBert


DEFAULT = dict(
    SUBTASK=None,
    BATCH_SIZE=32,
    EPOCHS=10,
    LEARNING_RATE=0.1,
    MODEL=None,
    MOMENTUM=0.9,
    OPTIMIZER=Optimizer.SGD,
    PIPELINE=Pipeline.CLASSIFICATION,
    SAMPLING=Sampling.SHUFFLE,
    TRAIN_RATIO=0.8,
)

SBERT_ADAM_OSMP_1 = dict(
    **DEFAULT,
    **dict(
        SUBTASK=1,
        EPOCHS=2,
        LEARNING_RATE=2e-5,
        MODEL=Model.SCIBERT,
        OPTIMIZER=Optimizer.ADAM,
        PIPELINE=Pipeline.CLASSIFICATION,
        SAMPLING=Sampling.OVERSAMPLING,
    )
)

NB_ADAMW_OSMP_1 = dict(
    **SBERT_ADAM_OSMP_1, **dict(MODEL=Model.NAIVE_BAYES, OPTIMIZER=Optimizer.ADAMW)
)

NcgConfigs = [DEFAULT, SBERT_ADAM_OSMP_1, NB_ADAMW_OSMP_1]
