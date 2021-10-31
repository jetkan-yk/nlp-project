from collections import Counter
import os


def load_data(folder_dir):
    """
    Loads dataset from directory `folder_dir` and returns a tuple `(text, label)`. \\
    Both `text` and `label` are a `list` of `N` items, where `N` is the number of samples in the 
    dataset.

    - Each `text` item is a `list` of sentences, the preprocessed NLP scholarly article plaintext
    - Each `label` item is a `list` of 0-indexed sentence ids, the contributing sentence ids
    """
    texts = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(folder_dir):
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


def load_task_names(folder_dir):
    """
    Loads dataset from directory `folder_dir` and returns a list of task names.
    """
    tasks = []

    for dirpath, dirnames, _ in os.walk(folder_dir):
        if dirnames != ["info-units", "triples"]:
            continue

        # Currently located in task folder
        tasks.append(dirpath)

    return tasks


if __name__ == "__main__":
    load_data("data-mini")
    load_task_names("data-small")
