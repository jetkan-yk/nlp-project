"""
Implements the `Model1` class and other subtask 1 helper functions
"""
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class Model1(nn.Module):
    """
    A `PyTorch nn.Module` subclass for subtask 1.
    """

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.collator = collator()

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def collate(self, batch):
        return self.collator(batch)

    def predict(self, outputs):
        """
        Maps predicted batch outputs to labels
        """
        return torch.argmax(outputs, dim=1)


class collator:
    """
    Collate function class for DataLoader
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )

    def __call__(self, batch):
        """
        Receives batch data (text, labels) and returns a pair of tensors
        """
        texts, labels = zip(*batch)
        return self.tokenizer(list(texts), return_tensors="pt"), torch.LongTensor(
            labels
        )
