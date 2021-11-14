"""
Usage: python3 main.py {config} [--train | --test] [--summary] [-d data_dir] [-m model_name] [-s summary_name]
"""

import argparse
import pprint as pp

import torch
import wandb
from torch.utils.data.dataset import random_split

from config import NcgConfigs
from dataset import NcgDataset
from model import NcgModel


def main(args):
    if args.c not in range(1, len(NcgConfigs)):
        options = {id: config for id, config in enumerate(NcgConfigs) if id > 0}
        raise ValueError(f"Please select config number from\n{pp.pformat(options)}")

    config = NcgConfigs[args.c]
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.summary:
        config["SUMMARY_MODE"] = True
        wandb.init(project="ncg", entity="cs4248-g17", name=args.s, config=config)
    else:
        config["SUMMARY_MODE"] = False

    dataset = NcgDataset(config, args.d)
    train_data, test_data = train_test_split(dataset, config["TRAIN_RATIO"])

    model = NcgModel(config)

    if args.train or args.train == args.test:
        model.train(train_data, args.m)
    if args.test or args.train == args.test:
        model.test(test_data, args.m)


def train_test_split(dataset, train_ratio):
    """
    Returns 2 randomly split training and testing dataset
    """
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    return random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )


def parse_args():
    """
    Parses the `main.py` inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "c",
        choices=range(len(NcgConfigs)),
        default=0,
        type=int,
        help="select config, choose 0 to show all configs",
    )

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
