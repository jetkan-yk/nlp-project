"""
Loads dataset, preprocesses data and implements the NcgDataset class
"""

import os

from torch import Tensor
from torch.utils.data import Dataset


def load_data(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a tuple `(text, label)`. \\
    Both `text` and `label` are a `list` of `N` items, where `N` is the number of samples in the 
    dataset.

    For each sample:
    - a `text` item is a `list` of sentences, the preprocessed NLP scholarly article plaintext
    - a `label` item is a `list` of 0-indexed sentence ids, the contributing sentence ids
    """
    texts = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(data_dir):
        if dirnames != ["info-units", "triples"]:
            continue

        # Currently located in task folder
        for filename in filenames:
            # Load text
            if filename.endswith("-Stanza-out.txt"):
                with open(os.path.join(dirpath, filename)) as f:
                    text = f.read().splitlines()
                    texts.append(text)

            # Load label
            if filename == "sentences.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    label_str = f.readlines()
                    label = sorted(map(int, label_str))
                    labels.append(label)

    return texts, labels


def load_task_names(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a list of task names.
    """
    tasks = []

    for dirpath, dirnames, _ in os.walk(data_dir):
        if dirnames != ["info-units", "triples"]:
            continue

        # Currently located in task folder
        tasks.append(dirpath)

    return tasks


def tokenize(sentence):
    """
    Tokenizes a sentence string into a list of word string
    """
    raise NotImplementedError


def parse(word):
    """
    Parses a word string into a torch.int type
    """
    raise NotImplementedError


class NcgDataset(Dataset):
    """
    A `PyTorch Dataset` class that accepts a data directory.

    - `self.task_names`: maps all `N` dataset sample ids to their corresponding task names, e.g. `data/constituency_parsing/0`
    - `self.texts`: `N` lists of preprocessed NLP scholarly article plaintext
    - `self.labels`: `N` lists of contributing sentence ids

    For each sample:
    - a `text` item is a `list` of sentences, the preprocessed NLP scholarly article plaintext
    - a `label` item is a `list` of 0-indexed sentence ids, the contributing sentence ids
    """

    def __init__(self, data_dir):
        self.task_names = load_task_names(data_dir)
        self.texts, self.labels = load_data(data_dir)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(text, label)` tuple.
        """
        text = self.texts[i]
        label = self.labels[i]

        return text, label


if __name__ == "__main__":
    dataset = NcgDataset("data-small")
    for text, label in dataset:
        for t in text:
            print(t)
        print(label)
