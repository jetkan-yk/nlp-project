"""
Usage: python3 main.py {1 | 2} [--train | --test] [--summary] [-d data_dir] [-m model_name] [-s summary_name]
"""

import argparse

import torch
from torch.utils.data.dataset import random_split

import wandb
from config import NcgConfig
from data import NcgDataset
from model import NcgModel


def main(args):
    if args.summary:
        wandb.init(project="ncg", entity="jetkan-yk", name=args.s)

    dataset = NcgDataset(args.subtask, args.d)
    train_data, test_data = train_test_split(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NcgModel(args.subtask, device)

    if args.train or args.train == args.test:
        model.train(train_data, args.m)
    if args.test or args.train == args.test:
        model.test(test_data, args.m)


def train_test_split(dataset):
    """
    Returns 2 randomly split training and testing dataset
    """
    train_size = int(len(dataset) * NcgConfig.TRAIN_RATIO)
    test_size = len(dataset) - train_size

    return random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )


def parse_args():
    """
    Parses the `main.py` inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("subtask", choices=[1, 2], type=int, help="choose subtask")

    parser.add_argument("-d", default="data", type=str, help="specify data directory")
    parser.add_argument("-m", default="model", type=str, help="specify model name")
    parser.add_argument("-s", default=None, type=str, help="specify summary name")

    parser.add_argument("--summary", action="store_true", help="enable summary mode")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="train model only")
    group.add_argument("--test", action="store_true", help="test model only")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
