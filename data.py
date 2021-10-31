"""
Loads dataset, preprocesses data and implements the NcgDataset class
"""

from collections import defaultdict
import os

from torch.utils.data import Dataset


def load_data(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a tuple of
    `(names, articles, sents, phrases)`.

    - `names` maps the sample id to the sample's folder name, e.g. `data/constituency_parsing/0`
    - `articles` maps the sample id to a list of sentences (each sentence is tokenized into a list
    of `string` words), the preprocessed NLP scholarly article plaintext
    - `sents` maps the sample id to a list of contributing sentence ids (0-indexed)
    - `phrases` maps the sample id to a `dict` of `{sents: phrases}` (each phrase is a list of `string`
    words), the scientific terms and relational cue phrases extracted from the contributing sentences
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
                    art_str_list = f.read().splitlines()
                    article = parse_article(art_str_list)
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


def parse_article(art_str_list: list[str]):
    """
    Parses a list of `string` from article plaintext into a list of list of `string` words.
    """
    # TODO: perform extra word token preprocessing here if required
    # return list(map(lambda art_str: art_str.split(" "), art_str_list))
    return art_str_list


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
        sent = row[0]
        phrase = row[3]
        phrases[sent].append(phrase)
    return phrases


class NcgDataset(Dataset):
    """
    A `PyTorch Dataset` class that accepts a subtask number and a data directory.
    """

    def __init__(self, subtask, data_dir):
        self.subtask = subtask
        self.names, self.articles, self.sents, self.phrases = load_data(data_dir)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.names)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple.
        """
        if self.subtask == 1:
            return self.articles[i], self.sents[i]
        elif self.subtask == 2:
            return self.sents[i], self.phrases[i]
        else:
            raise KeyError


if __name__ == "__main__":
    # Example use case
    dataset1 = NcgDataset(subtask=1, data_dir="data-mini")
    for d in dataset1:
        print(f"Input: {d[0]}\nOutput: {d[1]}")

    dataset2 = NcgDataset(subtask=1, data_dir="data-mini")
    for d in dataset2:
        print(f"Input: {d[0]}\nOutput: {d[1]}")
