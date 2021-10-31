"""
Implements the NcgModel class
"""

from subtask1.model1 import Model1
from subtask2.model2 import Model2


def NcgModel(subtask):
    """
    Returns the subtask's `PyTorch nn.Module` subclass.
    """
    if subtask == 1:
        return Model1()
    elif subtask == 2:
        return Model2()
    else:
        raise KeyError
