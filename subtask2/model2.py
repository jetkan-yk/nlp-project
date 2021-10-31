"""
Implements the Model class for subtask 2
"""

from torch import nn


class Model2(nn.Module):
    """
    A `PyTorch nn.Module` subclass for subtask 2.
    """

    def __init__(self):
        super().__init__()
        # TODO: Implement model here

    def forward(self, x):
        # TODO: Implement forward here
        raise NotImplementedError


def collator2(batch):
    """
    Collate function for DataLoader
    """
    raise NotImplementedError


def evaluator2(preds, labels):
    """
    Evaluates the predicted results against the expected labels
    """
    raise NotImplementedError
