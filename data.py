"""
Loads dataset, pre-processes data and implements the NcgDataset class
"""

import os
import pprint
from collections import defaultdict

from torch.utils.data import Dataset


def load_data(data_dir):
    """
    Loads data from directory `data_dir` and returns a tuple of `(names, articles, sents, phrases)`.

    - `names` the sample's folder name, e.g. `data/constituency_parsing/0`
    - `articles` the NLP scholarly article plaintext, each article is a list of `string`
    - `sents` a list of contributing sentence ids (0-indexed)
    - `phrases` the scientific terms and relational cue phrases extracted from the contributing
    sentences, each entry is a `dict` of `{sents: phrases}` (each phrase is a list of `string`),
    """
    names = []
    articles = []
    sents = []
    phrases = []

    for dirpath, dirnames, filenames in os.walk(data_dir):
        if dirnames != ["info-units", "triples"]:
            continue

        # Currently located in task folder
        names.append(dirpath)

        for filename in filenames:
            # Load article
            if filename.endswith("-Stanza-out.txt"):
                with open(os.path.join(dirpath, filename)) as f:
                    article = f.read().splitlines()
                    articles.append(article)

            # Load contributing sentence id
            if filename == "sentences.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    sent_str_list = f.read().splitlines()
                    sent = parse_sent(sent_str_list)
                    sents.append(sent)

            # Load phrase
            if filename == "entities.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    phrase_str_list = f.read().splitlines()
                    phrase = parse_phrase(phrase_str_list)
                    phrases.append(phrase)

    return names, articles, sents, phrases


def parse_sent(sent_str_list: list[str]):
    """
    Parses a list of `string` contributing sentence ids to a list of `int`, 0-indexed.
    """
    return sorted(map(lambda sent_str: int(sent_str) - 1, sent_str_list))


def parse_phrase(phrase_str_list: list[str]):
    """
    Parses a list of `string` phrase list into a `dict` that maps the contributing sentence id
    to a list of `string` phrase.
    """
    phrase_list = map(lambda phrase_str: phrase_str.split("\t", 4), phrase_str_list)

    phrase_dict = defaultdict(list)
    for row in phrase_list:
        sent = int(row[0]) - 1
        phrase = row[-1]
        phrase_dict[sent].append(phrase)
    return phrase_dict


class NcgDataset(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    For subtask 1:
        x = a full article plaintext (a list of strings) \\
        y = a contributing sentence (a string)

    For subtask 2:
        x = a contributing sentence (a string) \\
        y = a list of phrases (a list of strings)
    """

    def __init__(self, subtask, data_dir):
        self.subtask = subtask
        self.names, self.articles, self.sents, self.phrases = load_data(data_dir)

        if self.subtask == 1:
            self.init_subtask1()
        elif self.subtask == 2:
            self.init_subtask2()
        else:
            raise KeyError

    def init_subtask1(self):
        """
        Initializes the dataset for subtask 1
        """
        self.x = []
        self.y = []
        for idx, sent_list in enumerate(self.sents):
            for sent in sent_list:
                self.x.append(idx)
                self.y.append((idx, sent))

    def init_subtask2(self):
        """
        Initializes the dataset for subtask 2
        """
        self.x = []
        self.y = []
        for idx, phrase_dict in enumerate(self.phrases):
            for sent, phrase in phrase_dict.items():
                self.x.append((idx, sent))
                self.y.append(phrase)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple.
        """
        return self.stringify(self.x[i]), self.stringify(self.y[i])

    def stringify(self, data):
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


if __name__ == "__main__":
    # Example use case
    dataset1 = NcgDataset(subtask=1, data_dir="data-mini")
    for d in dataset1:
        pprint.pp(d)

    print("=============")
    dataset2 = NcgDataset(subtask=2, data_dir="data-mini")
    for d in dataset2:
        pprint.pp(d)
