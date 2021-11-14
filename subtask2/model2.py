"""
Implements the `Model2` model for subtask 2
"""

from torch import nn


class Model2(nn.Module):
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

    def evaluate(self, preds, labels):
        """
        Evaluates the predicted results against the expected labels and
        returns the tp, fp, tn, fn values for the result batch
        """
        tp = fp = tn = fn = 0

        # TODO: Implement evaluate here

        return tp, fp, tn, fn
