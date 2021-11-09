"""
Implements the `Model2` model for subtask 2
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

    def collate(self, batch):
        """
        Collate function for DataLoader
        """
        # TODO: Implement collate here
        raise NotImplementedError

    def predict(self, outputs):
        """
        Given a batch output, predict the final result
        """
        # TODO: Implement predict here
        raise NotImplementedError
