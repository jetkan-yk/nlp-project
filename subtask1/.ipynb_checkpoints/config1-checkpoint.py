"""
Contains hyperparameters for subtask 1.
"""
class Config1:
    PIPELINE = "classification"
    BATCH_SIZE = 20
    LEARNING_RATE = 0.3
    MOMENTUM = 0.8
    EPOCHS = 10
    OPTIMIZER = "sgd"
    SAMPLING_STRAT = "oversampling"
