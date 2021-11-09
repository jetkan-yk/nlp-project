"""
Contains hyperparameters for subtask 1.
"""


class Config1:
    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 5e-5
    OPTIMIZER = "adam"
    PIPELINE = "classification"
    SAMPLING_STRAT = "oversampling"
