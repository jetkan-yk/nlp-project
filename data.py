"""
Loads dataset, preprocesses data and implements the NcgDataset class
"""

import os

import torch
from torch.utils.data import Dataset


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


def load_data(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a tuple `(text, label)`. \\
    Both `text` and `label` are a list of `N` items, where `N` is the number of samples in the 
    dataset.

    For each sample:
    - a `text` item is a list of sentences (one sentence is a list of `string` words), the 
    preprocessed NLP scholarly article plaintext
    - a `label` item is a list of sentence ids in `torch.int`, the contributing sentence ids
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
                    text_str_list = f.read().splitlines()
                    text = parse_text(text_str_list)
                    texts.append(text)

            # Load label
            if filename == "sentences.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    label_str_list = f.read().splitlines()
                    label = parse_label(label_str_list)
                    labels.append(label)

    return texts, labels


def parse_text(text_str_list: list[str]):
    """
    Parses a list of `string` sentences into a list of list of `string` words.
    """
    # TODO: perform extra word token preprocessing here if required
    return list(map(lambda text_str: text_str.split(" "), text_str_list))


def parse_label(label_str_list: list[str]):
    """
    Parses a list of `string` labels into a list of `torch.int`.

    Note: labels are 0-indexed
    """
    return torch.tensor(
        sorted(map(lambda label_str: int(label_str) - 1, label_str_list))
    )


class NcgDataset(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.

    - `self.task_names`: maps all `N` dataset sample ids to their corresponding task names, e.g.
    `data/constituency_parsing/0`
    - `self.texts`: `N` lists of preprocessed NLP scholarly article plaintext
    - `self.labels`: `N` lists of contributing sentence ids
    """

    def __init__(self, subtask, data_dir):
        self.subtask = subtask
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

        - `text` is a list of `string`
        - `label` is a `torch.int`
        """
        if self.subtask == 1:
            text = self.texts[i]
            label = self.labels[i]
        elif self.subtask == 2:
            text = None
            label = None
        else:
            raise KeyError

        return text, label


if __name__ == "__main__":
    # Example use case
    dataset = NcgDataset(subtask=1, data_dir="data-mini")

    for text, label in dataset:
        for i in label:
            # print all contributing sentences
            print(text[i])
