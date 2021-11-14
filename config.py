"""
Hyperparameter configs
"""
from enum import Enum, auto

from sklearn.naive_bayes import MultinomialNB

from subtask1.scibert import SciBert

from subtask1.sentencebert import SentenceBertClass

from subtask1.scibertbilstm import SCIBERTBILSTMClass


class Pipeline(Enum):
    CLASSIFICATION = auto()
    EXTRACTIVE = auto()
    SBERTEXTRACTIVE = auto()


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
    SENTBERT = SentenceBertClass
    SCIBERTBILSTM = SCIBERTBILSTMClass


class Criterion(Enum):
    CELOSS = auto()
    BCELOSS = auto()


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
    CRITERION=Criterion.CELOSS,
)

SBERT_ADAMW_OSMP_1 = {
    **DEFAULT,
    **dict(
        SUBTASK=1,
        EPOCHS=2,
        LEARNING_RATE=2e-5,
        MODEL=Model.SCIBERT,
        OPTIMIZER=Optimizer.ADAMW,
        PIPELINE=Pipeline.CLASSIFICATION,
        SAMPLING=Sampling.OVERSAMPLING,
    ),
}

SBERTBILSTM_ADAMW_OSMP_1 = {
    **SBERT_ADAMW_OSMP_1,
    **dict(
        SUBTASK=1,
        EPOCHS=2,
        LEARNING_RATE=2e-5,
        MODEL=Model.SCIBERTBILSTM,
        OPTIMIZER=Optimizer.ADAMW,
        PIPELINE=Pipeline.CLASSIFICATION,
        SAMPLING=Sampling.OVERSAMPLING,
    ),
}

NB_OSMP_1 = {
    **DEFAULT,
    **dict(
        MODEL=Model.NAIVE_BAYES,
        SUBTASK=1,
        PIPELINE=Pipeline.CLASSIFICATION,
        SAMPLING=Sampling.OVERSAMPLING,
    ),
}

SENTB_ADAMW_OSMP_1 = {
    **DEFAULT,
    **dict(
        SUBTASK=1,
        MODEL=Model.SENTBERT,
        OPTIMIZER=Optimizer.ADAMW,
        PIPELINE=Pipeline.SBERTEXTRACTIVE,
        SAMPLING=Sampling.OVERSAMPLING,
        BATCH_SIZE=16,
        EPOCHS=1,
        LEARNING_RATE=1e-05,
        CRITERION=Criterion.BCELOSS,
    ),
}

NcgConfigs = [
    DEFAULT,
    SBERT_ADAMW_OSMP_1,
    NB_OSMP_1,
    SENTB_ADAMW_OSMP_1,
    SBERTBILSTM_ADAMW_OSMP_1,
]
