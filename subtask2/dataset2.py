from torch.utils.data import Dataset


class Dataset2(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    For subtask 2:
        x = a contributing sentence (a `string`) \\
        y = a list of phrases (a list of `string`)
    """

    def __init__(self, names, articles, sents, phrases, config):
        """
        Initializes the dataset for subtask 2
        """
        self.names = names
        self.articles = articles
        self.sents = sents
        self.phrases = phrases

        self.x = []
        self.y = []
        for idx, phrase_dict in enumerate(self.phrases):
            for sent, phrase in phrase_dict.items():
                self.x.append((idx, sent))
                self.y.append(phrase)

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple
        """
        return self._stringify(self.x[i]), self._stringify(self.y[i])

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
