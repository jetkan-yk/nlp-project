"""
Implements the Model class for demo
"""

import torch
import torch.nn.functional as F
from torch import nn


class Model0(nn.Module):
    """
    A `PyTorch nn.Module` subclass for demo.
    """

    def __init__(self, num_vocab, num_class):
        super().__init__()
        embedding_dim = 256
        hidden_dim = 128
        self.embedding = nn.Embedding(num_vocab + 1, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        # represent each bigram using embedding
        x = self.embedding(x)
        # average the embedding layer to obtain a single vector h0
        h0 = torch.sum(x, dim=1) / torch.sum(x != 0, dim=1).clamp(min=1)

        # project h0 through a hidden linear layer with ReLU to obtain h1
        h1 = F.relu(self.hidden(h0))
        # apply dropout to h1
        h1 = self.dropout(h1)

        # project h1 through another linear layer to obtain output layer
        output = F.relu(self.output(h1))
        return output


def collator0(batch):
    """
    Collate function for DataLoader
    """
    texts, labels = zip(*batch)

    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)

    return texts, labels


def evaluator0(preds, labels):
    """
    Evaluates the predicted results against the expected labels
    """
    y_pred = torch.argmax(preds, dim=1)
    y_true = torch.argmax(labels, dim=1)

    return y_pred == y_true
