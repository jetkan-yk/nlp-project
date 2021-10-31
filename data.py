import os

from torch.utils.data import Dataset


def load_data(data_dir):
    """
    Loads dataset from directory `data_dir` and returns a tuple `(text, label)`. \\
    Both `text` and `label` are a `list` of `N` items, where `N` is the number of samples in the 
    dataset.

    - Each `text` item is a `list` of sentences, the preprocessed NLP scholarly article plaintext
    - Each `label` item is a `list` of 0-indexed sentence ids, the contributing sentence ids
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


class NcgDataset(Dataset):
    """
    A PyTorch Dataset class that accepts a data directory.

    - task_names: maps the dataset sample id to its corresponding task name, e.g. `data/constituency_parsing/0`
    - texts: preprocessed NLP scholarly article plaintext
    - labels: contributing sentence ids
    """

    def __init__(self, data_dir):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __get_item__(self, i):
        raise NotImplementedError
