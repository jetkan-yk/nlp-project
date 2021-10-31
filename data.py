"""
Loads dataset, preprocesses data and implements the NcgDataset class
"""

import os
import pprint
from collections import defaultdict

from torch.utils.data import Dataset


def load_data(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a tuple of
    `(names, articles, sents, phrases)`.

    - `names` maps the sample id to the sample's folder name, e.g. `data/constituency_parsing/0`
    - `articles` maps the sample id to a list of sentences, the NLP scholarly article plaintext
    - `sents` maps the sample id to a list of contributing sentence ids (0-indexed)
    - `phrases` maps the sample id to a `dict` of `{sents: phrases}` (each phrase is a list of
    `string`), the scientific terms and relational cue phrases extracted from the contributing
    sentences
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
    Parses a list of `string` contributing sentence ids into a list of `int`.
    """
    return sorted(map(lambda sent_str: int(sent_str) - 1, sent_str_list))


def parse_phrase(phrase_str_list: list[str]):
    """
    Parses a list of `string` phrases into a `dict` that maps the contributing sentence id
    to a list of `string` phrase.
    """
    phrase_list = map(lambda phrase_str: phrase_str.split("\t", 4), phrase_str_list)

    phrases = defaultdict(list)
    for row in phrase_list:
        sent = int(row[0])
        phrase = row[3]
        phrases[sent].append(phrase)
    return phrases


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

    def init_subtask1(self):
        raise NotImplementedError

    def init_subtask2(self):
        raise NotImplementedError

    def __init__(self, subtask, data_dir):
        self.subtask = subtask
        self.names, self.articles, sents, phrases = load_data(data_dir)

        # helper for y of subtask 1
        self.y1 = []  # (sample, sent_id)
        for sample, sent_ids in enumerate(sents):
            for sent_id in sent_ids:
                self.y1.append((sample, sent_id))

        # helper for x, y of subtask 2
        self.x2 = []  # (sample, sent_id)
        self.y2 = []  # phrase
        for sample, phrase_dict in enumerate(phrases):
            for sent_id, phrase in phrase_dict.items():
                self.x2.append((sample, sent_id))
                self.y2.append(phrase)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        if self.subtask == 1:
            return len(self.y1)
        elif self.subtask == 2:
            return len(self.y2)
        else:
            raise KeyError

    def get_text(self, sample, sent=None):
        raise NotImplementedError

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple.
        """
        if self.subtask == 1:
            sample, sent_id = self.y1[i]
            x = self.articles[sample]
            y = self.articles[sample][sent_id]

        elif self.subtask == 2:
            sample, sent_id = self.x2[i]
            x = self.articles[sample][sent_id]
            y = self.y2[i]

        else:
            raise KeyError

        return x, y


if __name__ == "__main__":
    # Example use case
    dataset1 = NcgDataset(subtask=1, data_dir="data-mini")
    for d in dataset1:
        pprint.pp(d)

    print("=============")
    dataset2 = NcgDataset(subtask=2, data_dir="data-mini")
    for d in dataset2:
        pprint.pp(d)
