"""
Implements the Model class for subtask 1
"""

from torch import nn


class Model1(nn.Module):
    """
    A `PyTorch nn.Module` subclass for subtask 1.
    """

    def __init__(self):
        super().__init__()
        # TODO: Implement model here

    def forward(self, x):
        # TODO: Implement forward here
        raise NotImplementedError


def collator1(batch):
    """
    Collate function for DataLoader
    """
    # TODO: Implement collator here
    raise NotImplementedError


def evaluator1(preds, labels):
    """
    Evaluates the predicted results against the expected labels and
    returns a score for the result batch
    """
    # TODO: Implement evaluator here
    raise NotImplementedError
