"""
Usage: python3 main.py [subtask] [-d dataset_path] [-m model_name] [-s summary_name]
"""

import argparse

subtask = None


def load_data(dataset_path):
    raise NotImplementedError


def train_model(model_name):
    raise NotImplementedError


def test_model(model_name):
    raise NotImplementedError


def evaluate(summary_name):
    raise NotImplementedError


def main(args):
    global subtask
    subtask = args.subtask


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("subtask", choices=[1, 2], type=int, help="choose subtask")
    parser.add_argument("-d", default="data", type=str, help="specify dataset path")
    parser.add_argument("-m", default="model", type=str, help="specify model name")
    parser.add_argument("-s", default="summary", type=str, help="specify summary name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
