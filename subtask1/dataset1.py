from config import Pipeline
from torch.utils.data import Dataset

from .config1 import Config1


class Dataset1(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    For subtask 1:
        x = a full article plaintext (a list of `string`) \\
        y = a contributing sentence (a `string`)
    """

    def __init__(self, names, articles, sents, phrases):
        """
        Initializes the dataset for subtask 1
        """
        self.names = names
        self.articles = articles
        self.sents = sents
        self.phrases = phrases

        self.x = []
        self.y = []

        # formats data for classification task (sent, label),
        # where label == 1 for contributing sentences and label == 0 otherwise
        if Config1.PIPELINE is Pipeline.CLASSIFICATION:
            for idx, sent_list in enumerate(self.sents):
                article = self._stringify(idx)
                for sent_id, sent in enumerate(article):
                    label = int(sent_id in sent_list)
                    self.x.append(sent)
                    self.y.append(label)
        else:
            # formats data into (article, contributing sentences)
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
