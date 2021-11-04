"""
Contains hyperparameters for subtask 2.
"""
class Config2:
    PIPELINE = "default" # problem formulation determines data processing eg. "classification", "extractive summarization"
    BATCH_SIZE = 20
    LEARNING_RATE = 0.3
    MOMENTUM = 0.8
    EPOCHS = 10
    OPTIMIZER = "sgd"
    SAMPLING_STRAT = "default"
