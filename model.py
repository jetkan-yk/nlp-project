"""
Loads, saves model and implements the `NcgModel` class
"""

import os
from collections import Counter
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb

from config import Optimizer, Pipeline, Sampling
from subtask1.config1 import Config1
from subtask2.config2 import Config2


class NcgModel:
    """
    A model class that is powered by a `PyTorch nn.Module` subclass.
    """

    def __init__(self, subtask, device):
        self.subtask = subtask
        self.device = device

        if self.subtask == 1:
            self.config = Config1
        elif self.subtask == 2:
            self.config = Config2
        else:
            raise KeyError(f"Invalid subtask number: {self.subtask}")

        self.model = self.config.MODEL().to(self.device)

        wandb.watch(self.model)

        print(f"{self.model}\n")

    def _dataloader(self, dataset):
        if self.config.SAMPLING is Sampling.OVERSAMPLING:
            if self.config.PIPELINE is not Pipeline.CLASSIFICATION:
                raise TypeError("Cannot oversampling non-classification problem")

            _, labels = zip(*dataset)

            class_count = list(Counter(labels).values())
            class_weights = 1.0 / torch.tensor(class_count)

            # list of weights denoting probability of sample of corresponding indice being sampled
            sample_weights = [class_weights[i] for i in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

            return DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                sampler=sampler,
                collate_fn=self.model.collate,
            )

        elif self.config.SAMPLING is Sampling.SHUFFLE:
            return DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                collate_fn=self.model.collate,
            )

        else:
            raise NotImplementedError

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        if self.config.OPTIMIZER is Optimizer.ADAM:
            return optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        elif self.config.OPTIMIZER is Optimizer.SGD:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
            )

        else:
            raise NotImplementedError

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        data_loader = self._dataloader(train_data)
        criterion = self._criterion()
        optimizer = self._optimizer()

        print(f"Begin training...")
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
        # Use default samping method
        self.config.SAMPLING = Sampling.SHUFFLE

        data_loader = self._dataloader(test_data)
        batch_score = 0.0

        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(features)
                preds = self.model.predict(outputs)
                tp, fp, _, fn = self.model.evaluate(preds, labels)
                batch_score += f1_score(tp, fp, fn)

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
