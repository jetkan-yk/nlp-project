from torch.utils.data import Dataset


def phrase_pos(sentence, start_idx, end_idx):
    """
    Convert the phrase's char-level index to word-level index
    """
    word_pos = phrase_size = start = end = 0
    for idx in range(len(sentence)):
        if sentence[idx] == " ":
            word_pos += 1
            if idx > start_idx:
                phrase_size += 1
        if idx == start_idx:
            start = word_pos
        elif idx == end_idx:
            end = start + phrase_size
    return start, end


def tag_sent(tags, start, end):
    """
    Given a sentence tagging, the idx where a phrase starts, the
    idx where a phrase ends, updates the sentence's "B", "I", "O" tags
    """
    tags[start] = "B"
    for idx in range(start + 1, end):
        tags[idx] = "I"
    return tags


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
                sentence = self._stringify((idx, sent))
                self.x.append(sentence)

                # initiaize sentence taggings with all "O"s
                tags = ["O"] * len(sentence.split(" "))
                for start_idx, end_idx in phrase:
                    # convert the phrase's char-level index to word-level index
                    start, end = phrase_pos(sentence, start_idx, end_idx)
                    # tag sentence with "B"s and "I"s
                    tags = tag_sent(tags, start, end)
                self.y.append(tags)

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.x)

    def __getitem__(self, i):
        """
        Returns the i-th sample's `(x, y)` tuple
        """
        return self.x[i], self.y[i]

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
