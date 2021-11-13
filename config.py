"""
Hyperparameter configs
"""
from enum import Enum, auto

from sklearn.naive_bayes import MultinomialNB

from subtask1.scibert import SciBert
from subtask2.SciBert_BiLSTM_CRF import SciBert_BiLSTM_CRF


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
    SciBert_BiLSTM_CRF = SciBert_BiLSTM_CRF


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
    WEIGHT_DECAY=0,
)

SBERT_ADAM_OSMP_1 = {
    **DEFAULT,
    **dict(
        SUBTASK=1,
        EPOCHS=2,
        LEARNING_RATE=2e-5,
        MODEL=Model.SCIBERT,
        OPTIMIZER=Optimizer.ADAM,
        PIPELINE=Pipeline.CLASSIFICATION,
        SAMPLING=Sampling.OVERSAMPLING,
    ),
}

NB_ADAMW_OSMP_1 = {
    **SBERT_ADAM_OSMP_1,
    **dict(MODEL=Model.NAIVE_BAYES, OPTIMIZER=Optimizer.ADAMW),
}


#### Maps BIO labels to index numbers and vice-versa.
tag_to_ix = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

SciBert_BiLSTM_CRF = {
    **DEFAULT,
    **dict(
        SUBTASK=2,
        EMBEDDING_DIM=768,
        EPOCHS=1,
        HIDDEN_DIM=200,
        IX_TO_TAG=ix_to_tag,
        LEARNING_RATE=0.01,
        MODEL=Model.SciBert_BiLSTM_CRF,
        OPTIMIZER=Optimizer.SGD,
        TAG_TO_IX=tag_to_ix,
        WEIGHT_DECAY=1e-4,
    ),
}


NcgConfigs = [DEFAULT, SBERT_ADAM_OSMP_1, NB_ADAMW_OSMP_1, SciBert_BiLSTM_CRF]
