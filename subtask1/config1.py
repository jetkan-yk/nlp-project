"""
Contains hyperparameters for subtask 1.
"""


class Config1:
    PIPELINE = "classification"
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    MOMENTUM = 0.9
    EPOCHS = 10
    OPTIMIZER = "sgd"
    SAMPLING_STRAT = "oversampling"
