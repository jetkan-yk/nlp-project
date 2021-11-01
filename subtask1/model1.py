"""
Implements the `Model1` class and other subtask 1 helper functions
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


def collate1(batch):
    """
    Collate function for DataLoader
    """
    # TODO: Implement collate here
    raise NotImplementedError

def predict1(outputs):
    """
    Given a batch output, predict the final result
    """
    # TODO: Implement predict here
    raise NotImplementedError