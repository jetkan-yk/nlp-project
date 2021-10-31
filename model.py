"""
Implements the NcgModel class
"""

from subtask1.model1 import Model1
from subtask2.model2 import Model2


class NcgModel:
    """
    A model interface that is powered by a `PyTorch nn.Module` subclass.
    """

    def __init__(self, subtask, device):
        if subtask == 1:
            self.model = Model1().to(device)
        elif subtask == 2:
            self.model = Model2().to(device)
        else:
            raise KeyError

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and stores the model in `model_name.pkl`
        """
        print(f"Training {self.model}")

    def test(self, test_data, summary_name):
        """
        Tests the model using `test_data` and generates a summary file `summary_name.txt`
        """
        print(f"Testing {self.model}")
