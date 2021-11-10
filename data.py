"""
Loads, pre-processes data and implements the `NcgDataset` class
"""

import os
from collections import defaultdict

from torch.utils.data import Dataset

from subtask1.dataset1 import Dataset1
from subtask2.dataset2 import Dataset2


def load_data(data_dir):
    """
    Loads data from directory `data_dir` and returns a tuple of `(names, articles, sents, phrases)`.

    - `names` the sample's folder name, e.g. `data/constituency_parsing/0`
    - `articles` the NLP scholarly article plaintext, each article is a list of `string`
    - `sents` a list of contributing sentence ids (0-indexed)
    - `phrases` the scientific terms and relational cue phrases extracted from the contributing
    sentences, each entry is a `dict` of `{sents: phrases}` (each phrase is a list of `string`)
    """
    names = []
    articles = []
    sents = []
    phrases = []

    for dirpath, dirnames, filenames in os.walk(data_dir):
        if dirnames != ["info-units", "triples"]:
            continue

        # currently located in task folder
        names.append(dirpath)

        for filename in filenames:
            # load article
            if filename.endswith("-Stanza-out.txt"):
                with open(os.path.join(dirpath, filename)) as f:
                    article = f.read().splitlines()
                    articles.append(article)

            # load contributing sentence id
            if filename == "sentences.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    sent_str_list = f.read().splitlines()
                    sent = parse_sent(sent_str_list)
                    sents.append(sent)

            # load phrase
            if filename == "entities.txt":
                with open(os.path.join(dirpath, filename)) as f:
                    phrase_str_list = f.read().splitlines()
                    phrase = parse_phrase(phrase_str_list)
                    phrases.append(phrase)

    return names, articles, sents, phrases


def parse_sent(sent_str_list):
    """
    Parses a list of `string` contributing sentence ids to a list of `int`, 0-indexed
    """
    return sorted(map(lambda sent_str: int(sent_str) - 1, sent_str_list))


def parse_phrase(phrase_str_list):
    """
    Parses a list of `string` phrase list into a `dict` that maps the contributing sentence id
    to a list of `string` phrase
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
        x = a full article plaintext (a list of `string`) \\
        y = a contributing sentence (a `string`)

    For subtask 2:
        x = a contributing sentence (a `string`) \\
        y = a list of phrases (a list of `string`)
    """

    def __init__(self, subtask, data_dir):
        self.subtask = subtask
        names, articles, sents, phrases = load_data(data_dir)

        print(f"Loaded data from {data_dir}\n")

        if self.subtask == 1:
            self.dataset = Dataset1(names, articles, sents, phrases)
        elif self.subtask == 2:
            self.dataset = Dataset2(names, articles, sents, phrases)
        else:
            raise KeyError(f"Invalid subtask number: {self.subtask}")

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple
        """
        return self.dataset[i]
