"""
Loads, saves model and implements the `NcgModel` class
"""

import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from subtask1.config1 import Config1
from subtask1.model1 import Model1
from subtask2.config2 import Config2
from subtask2.model2 import Model2


class NcgModel:
    """
    A model class that is powered by a `PyTorch nn.Module` subclass.
    """

    def __init__(self, subtask, device):
        self.subtask = subtask
        self.device = device

        # determines hyperparameters, model for each subtask
        if self.subtask == 1:
            self.config = Config1
            self.model = Model1().to(self.device)
        elif self.subtask == 2:
            self.config = Config2
            self.model = Model2().to(self.device)
        else:
            raise KeyError

        print(f"{self.model}\n")

    def _dataloader(self, data):
        if self.config.SAMPLING_STRAT == "oversampling":
            if self.config.PIPELINE == "classification":
                # oversampling with replacement, used to correct class imbalance for classification
                # data is of format (feature, int_label)
                assert (
                    len(data[0]) == 2
                    and type(data[0][0]) == str
                    and type(data[0][1]) == int
                )

                features, labels = zip(*data)
                labels = list(labels)
                classes = np.unique(labels)

                # gets distribution of class labels
                count_dict = defaultdict(lambda: 0, dict())
                for label in labels:
                    count_dict[label] += 1

                # gets class weights for sampling by taking reciprocal of class counts
                class_count = [count_dict[i] for i in classes]
                class_weights = (
                    1.0 / torch.tensor(class_count, dtype=torch.float)
                ).tolist()

                # list of weights denoting probability of sample of corresponding indice being sampled
                sample_weights = [class_weights[i] for i in labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )

                if False:
                    # checks if sampling was correctly performed
                    loader = DataLoader(
                        data,
                        batch_size=len(data),
                        sampler=sampler,
                    )

                    sampled_labels = next(iter(loader))[1].tolist()
                    sampled_count_dict = defaultdict(lambda: 0, {})

                    for label in sampled_labels:
                        sampled_count_dict[label] += 1

                    print(sampled_count_dict)
                    exit()

                return DataLoader(
                    data,
                    batch_size=self.config.BATCH_SIZE,
                    sampler=sampler,
                    collate_fn=self.model.collate,
                )

        else:
            # default sampling method: shuffling of data
            return DataLoader(
                data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                collate_fn=self.model.collate,
            )

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        if self.config.OPTIMIZER == "adam":
            return optim.Adam(self.model.parameters(), self.config.LEARNING_RATE)
        else:
            return optim.SGD(
                self.model.parameters(), self.config.LEARNING_RATE, self.config.MOMENTUM
            )

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        data_loader = self._dataloader(train_data)
        criterion = self._criterion()
        optimizer = self._optimizer()

        start = datetime.now()
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            running_loss = 0.0

            for step, data in enumerate(data_loader):
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # do forward propagation
                preds = self.model(features)

                # do loss calculation
                loss = criterion(preds, labels)

                # do backward propagation
                loss.backward()

                # do parameter optimization step
                optimizer.step()

                # calculate running loss value
                running_loss += loss.item()

                # print loss value every 100 steps and reset the running loss
                if step % 100 == 99:
                    print(
                        f"[{epoch + 1}, {step + 1:{4}}] loss: {running_loss / 100:.{3}}"
                    )
                    running_loss = 0.0
        end = datetime.now()
        print(f"\nTraining finished in {(end - start).seconds / 60.0} minutes.\n")

        save_model(self.subtask, self.model, model_name)

    def test(self, test_data, model_name):
        """
        Tests the `model_name` model in the subtask folder using `test_data`
        """
        self.model = load_model(self.subtask, self.model, model_name)

        data_loader = self._dataloader(test_data)
        batch_score = 0.0

        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(features)
                preds = self.model.predict(outputs)
                batch_score += evaluate(preds, labels)

        print(f"Accuracy: {batch_score / len(data_loader):.{3}}\n")


def save_model(subtask, model: nn.Module, model_name):
    """
    Saves the model as `model_name` in the subtask folder
    """
    checkpoint = model.state_dict()
    model_path = os.path.join(f"subtask{subtask}", model_name)

    torch.save(checkpoint, model_path)
    print(f"Model saved in {model_path}\n")


def load_model(subtask, model: nn.Module, model_name):
    """
    Loads the model from `model_name` in the subtask folder
    """
    model_path = os.path.join(f"subtask{subtask}", model_name)

    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}\n")

    return model


def evaluate(preds, labels):
    """
    Evaluates the predicted results against the expected labels and
    returns a f1-score for the result batch
    """
    tp = fp = fn = tn = 0

    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        if pred == 1 and label == 0:
            fp += 1
        if pred == 0 and label == 1:
            fn += 1
        if pred == 0 and label == 0:
            tn += 1

    return f1_score(tp, fp, fn)


def f1_score(tp, fp, fn):
    """
    Computes the fscore using the tp, fp, fn
    When true positive + false positive == 0, precision is undefined.
    When true positive + false negative == 0, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score
    """

    if (tp + 0.5 * (fp + fn)) == 0:
        return 0

    return tp / (tp + 0.5 * (fp + fn))
