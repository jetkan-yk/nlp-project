"""
Usage: python3 main.py [subtask] [-d data_dir] [-m model_name] [-s summary_name] [--train | --test]
"""

import argparse

import torch
from torch.utils.data.dataset import random_split

from data import NcgDataset
from model import NcgModel

TRAIN_RATIO = 0.8


def main(args):
    dataset = NcgDataset(args.subtask, args.d)
    train_data, test_data = train_test_split(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NcgModel(args.subtask, device)

    if args.train or args.train == args.test:
        model.train(train_data, args.m)
    if args.test or args.train == args.test:
        model.test(test_data, args.s)


def train_test_split(dataset):
    """
    Returns 2 randomly split training and testing dataset
    """
    train_size = int(len(dataset) * TRAIN_RATIO)
    test_size = len(dataset) - train_size

    return random_split(dataset, [train_size, test_size])


def parse_args():
    """
    Parses the `main.py` inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("subtask", choices=[1, 2], type=int, help="choose subtask")
    parser.add_argument("-d", default="data", type=str, help="specify data directory")
    parser.add_argument("-m", default="model", type=str, help="specify model name")
    parser.add_argument("-s", default="summary", type=str, help="specify summary name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="train model only")
    group.add_argument("--test", action="store_true", help="test model only")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
