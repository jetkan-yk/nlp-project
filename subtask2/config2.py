"""
Contains hyperparameters for subtask 2.
"""


class Config2:
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    OPTIMIZER = "sgd"
    PIPELINE = "default"  # problem formulation determines data processing eg. "classification", "extractive summarization"
    SAMPLING_STRAT = "default"
