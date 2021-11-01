"""
My assignment 2 file for demo mode reference
"""

import argparse
import datetime
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


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


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """

    def __init__(self, text_path, label_path=None):
        """
        Read the content of vocab and text_file
        Args:
            text_path (string): Path to the text file.
            label_path (string, optional): Path to the label file.
        """
        self.texts = self.load_texts(text_path)
        self.labels = (
            self.load_labels(label_path) if label_path else [-1] * len(self.texts)
        )

    def load_texts(self, text_path):
        """
        Read the content from text_path and parse each sentence into bigram_id
        """
        with open(text_path, "r", encoding="utf8") as f:
            data = f.readlines()

        texts = []
        for row in data:
            bigrams = []
            sentence = row.strip().lower()
            for i in range(len(sentence) - 1):
                s1, s2 = parse_char(sentence[i]), parse_char(sentence[i + 1])
                bigrams.append(VOCAB[s1, s2])
            texts.append(torch.tensor(bigrams))
        return texts

    def load_labels(self, label_path):
        """
        Read the content from label_path and parse each label into lang_id
        """
        with open(label_path, "r") as f:
            data = f.readlines()

        labels = []
        for row in data:
            label = row.strip()
            labels.append(LANG[label])
        return labels

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(VOCAB)
        num_class = len(LANG)

        return num_vocab, num_class

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).
        both text and label are recommended to be in pytorch tensor type.

        DO NOT pad the tensor here, do it at the collator function.
        """
        text = self.texts[i]
        label = self.labels[i]

        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer, a hidden
    feed-forward layer, a dropout layer, and a feed-forward
    layer that reduces the dimension to num_class
    """

    def __init__(self, num_vocab, num_class, dropout=0.5):
        super().__init__()
        embedding_dim = 256
        hidden_dim = 128
        self.embedding = nn.Embedding(num_vocab + 1, embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        # represent each bigram using embedding
        x = self.embedding(x)
        # average the embedding layer to obtain a single vector h0
        h0 = torch.sum(x, dim=1) / torch.sum(x != 0, dim=1).clamp(min=1)

        # project h0 through a hidden linear layer with ReLU to obtain h1
        h1 = F.relu(self.hidden(h0))
        # apply dropout to h1
        h1 = self.dropout(h1)

        # project h1 through another linear layer to obtain output layer
        output = F.relu(self.output(h1))
        return output


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    texts, labels = zip(*batch)

    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)

    return texts, labels


def train(
    model,
    dataset,
    batch_size,
    learning_rate,
    momentum,
    num_epoch,
    device="cpu",
    model_path=None,
):
    """
    Complete the training procedure below by specifying the loss function
    and the optimizer with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, shuffle=True
    )

    # assign these variables
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation
            y_pred = model(texts)

            # do loss calculation
            loss = criterion(y_pred, labels)

            # do backward propagation
            loss.backward()

            # do parameter optimization step
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100)
                )
                running_loss = 0.0

    end = datetime.datetime.now()

    # save the model weight in the checkpoint variable
    # and dump it to system on the model_path
    # tip: the checkpoint can contain more than just the model
    checkpoint = model.state_dict()
    torch.save(checkpoint, model_path)

    print("Model saved in ", model_path)
    print("Training finished in {} minutes.".format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device="cpu"):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions
            y_pred = torch.argmax(outputs, dim=1)
            labels += [class_map[y] for y in y_pred]
    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = "cuda:{}".format(0)
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert (
            args.label_path is not None
        ), "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)

        # you may change these hyper-parameters
        learning_rate = 0.3
        momentum = 0.8
        batch_size = 20
        num_epochs = 30

        train(
            model,
            dataset,
            batch_size,
            learning_rate,
            momentum,
            num_epochs,
            device,
            args.model_path,
        )
    if args.test:
        assert (
            args.model_path is not None
        ), "Please provide the model to test using --model_path argument"
        # the lang map should map the class index to the language id (e.g. eng, fra, etc.)
        lang_map = LANGUAGES

        # create the test dataset object using LangDataset class
        dataset = LangDataset(args.text_path)
        num_vocab, num_class = dataset.vocab_size()

        # initialize and load the model
        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(torch.load(args.model_path))

        # run the prediction
        preds = test(model, dataset, lang_map, device)

        # write the output
        with open(args.output_path, "w", encoding="utf-8") as out:
            out.write("\n".join(preds))
    print("\n==== A2 Part 2 Done ====")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", help="path to the text file")
    parser.add_argument("--label_path", default=None, help="path to the label file")
    parser.add_argument(
        "--train", default=False, action="store_true", help="train the model"
    )
    parser.add_argument(
        "--test", default=False, action="store_true", help="test the model"
    )
    parser.add_argument(
        "--model_path", required=True, help="path to the output file during testing"
    )
    parser.add_argument(
        "--output_path",
        default="out.txt",
        help="path to the output file during testing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
