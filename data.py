"""
Dataset class only for demo purpose.
"""
import os
import string

import torch
from torch.utils.data import Dataset


def load_texts(data_dir):
    """
    Read the x.txt content from data_dir and parse each sentence into bigram_id
    """
    text_path = os.path.join(data_dir, "x.txt")

    with open(text_path, "r", encoding="utf8") as f:
        data = f.readlines()

    texts = []
    for row in data:
        bigrams = []
        sentence = row.strip().lower()
        for i in range(len(sentence) - 1):
            s1, s2 = parse_char(sentence[i]), parse_char(sentence[i + 1])
            bigrams.append(NcgDatasetDemo.VOCAB[s1, s2])
        texts.append(torch.tensor(bigrams))
    return texts


def parse_char(c: str):
    """
    Parse character c into their correct symbol type
    """
    if c in string.ascii_lowercase:
        return c
    elif c.isdigit():
        return "*"
    elif c.isspace():
        return "_"
    else:
        return "?"


def load_labels(data_dir):
    """
    Read the y.txt content from data_dir and parse each label into lang_id
    """
    label_path = os.path.join(data_dir, "y.txt")

    with open(label_path, "r") as f:
        data = f.readlines()

    labels = []
    for row in data:
        label = row.strip()
        labels.append(NcgDatasetDemo.LANG[label])
    return labels


class NcgDatasetDemo(Dataset):
    """
    Dataset class only for demo purpose.
    """

    SYMBOLS = [c for c in string.ascii_lowercase] + [
        "_",  # space
        "*",  # digits
        "?",  # other characters
    ]
    # every combinations of (symbol1, symbol2)
    BIGRAMS = ((s1, s2) for s1 in SYMBOLS for s2 in SYMBOLS)
    # maps a tuple of symbols to their bigram_id
    VOCAB = {bigram: idx for idx, bigram in enumerate(BIGRAMS, start=1)}

    LANGUAGES = ["eng", "deu", "fra", "ita", "spa"]
    # maps a language name to their lang_id
    LANG = {language: idx for idx, language in enumerate(LANGUAGES)}

    def __init__(self, subtask, data_dir):
        self.subtask = subtask

        if self.subtask == 0:
            self.x = load_texts(data_dir)
            self.y = load_labels(data_dir)
        else:
            raise KeyError

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).
        both text and label are recommended to be in pytorch tensor type.

        DO NOT pad the tensor here, do it at the collator function.
        """
        return self.x[i], self.y[i]

    def vocab_size():
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(NcgDatasetDemo.VOCAB)
        num_class = len(NcgDatasetDemo.LANG)

        return num_vocab, num_class
