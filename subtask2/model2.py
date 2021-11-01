"""
Implements the `Model2` class and other subtask 2 helper functions
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


def collate2(batch):
    """
    Collate function for DataLoader
    """
    # TODO: Implement collate here
    raise NotImplementedError

def predict2(outputs):
    """
    Given a batch output, predict the final result
    """
    # TODO: Implement predict here
    raise NotImplementedError