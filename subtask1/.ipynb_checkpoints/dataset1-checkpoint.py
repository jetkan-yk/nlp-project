import os
from collections import defaultdict

from torch.utils.data import Dataset

class Dataset1(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    For subtask 1:
        x = a full article plaintext (a list of `string`) \\
        y = a contributing sentence (a `string`)
    """

    def __init__(self, names, articles, sents, phrases, pipeline):
        """
        Initializes the dataset for subtask 1
        """
        self.names = names
        self.articles = articles
        self.sents = sents
        self.phrases = phrases
        self.pipeline = pipeline

        self.x = []
        self.y = []
        
        # formats data for classification task (sent, label), 
        # where label == 1 for contributing sentences for label == 0 otherwise
        if self.pipeline == "classification":
            for idx, sent_list in enumerate(self.sents):
                # get list of sentences in paper
                paper_sents = self._stringify(idx)

                # separate into contributing and non-contributing sentences
                contributing_sents = [self._stringify((idx, sent)) for sent in sent_list]
                non_contributing_sents = [x for x in paper_sents if x not in contributing_sents]

                # appends appropriate labels
                for sent in contributing_sents:
                    self.x.append(sent)
                    self.y.append(1)

                for sent in non_contributing_sents:
                    self.x.append(sent)
                    self.y.append(0)
                
        else:
            # formats data into (paper, contributing sentence)
            for idx, sent_list in enumerate(self.sents):
                for sent in sent_list:
                    self.x.append(self._stringify(idx))
                    self.y.append(self._stringify((idx, sent)))

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple
        """
        return self.x[i], self.y[i]

    def _stringify(self, data):
        """
        Converts the article or sentence index into `string`
        """
        if isinstance(data, list):
            return data
        elif isinstance(data, tuple):
            idx, sent = data
            return self.articles[idx][sent]
        elif isinstance(data, int):
            return self.articles[data]
        else:
            raise KeyError