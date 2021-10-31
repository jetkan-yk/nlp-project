"""
Usage: python3 main.py [subtask] [-d data_dir] [-m model_name] [-s summary_name]
"""

import argparse

import torch

from data import NcgDataset
from model import NcgModel


def main(args):
    if torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    dataset = NcgDataset(args.subtask, args.d)
    # TODO: train test split
    train_data, test_data = train_test_split(dataset)

    model = NcgModel(args.subtask).to(device)
    model.train(train_data, device, args.m)
    model.test(test_data, device)
    model.eval(args.s)


def train_test_split(dataset):
    raise NotImplementedError


def parse_args():
    """
    Parses the `main.py` inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("subtask", choices=[1, 2], type=int, help="choose subtask")
    parser.add_argument("-d", default="data", type=str, help="specify data directory")
    parser.add_argument("-m", default="model", type=str, help="specify model name")
    parser.add_argument("-s", default="summary", type=str, help="specify summary name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
