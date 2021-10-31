"""
Usage: python3 main.py [subtask] [-d data_dir] [-m model_name] [-s summary_name]
"""

import argparse
import torch

from data import NcgDataset


def main(args):
    if torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    dataset = NcgDataset(args.subtask, args.data)
    # model = Model(args.subtask).to(device)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("subtask", choices=[1, 2], type=int, help="choose subtask")
    parser.add_argument("-d", default="data", type=str, help="specify data directory")
    parser.add_argument("-m", default="model", type=str, help="specify model name")
    parser.add_argument("-s", default="summary", type=str, help="specify summary name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
