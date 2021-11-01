"""
Loads, saves model and implements the `NcgModel` class
"""

import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from subtask1.model1 import Model1, collator1
from subtask2.model2 import Model2, collator2


class NcgModel:
    """
    A model class that is powered by a `PyTorch nn.Module` subclass.
    """

    # Hyperparameters
    BATCH_SIZE = 20
    LEARNING_RATE = 0.3
    MOMENTUM = 0.8
    EPOCHS = 10

    def __init__(self, subtask, device):
        self.subtask = subtask
        self.device = device

        if self.subtask == 1:
            self.model = Model1().to(self.device)
            self.collator = collator1
        elif self.subtask == 2:
            self.model = Model2().to(self.device)
            self.collator = collator2
        else:
            raise KeyError

        print(f"{self.model}\n")

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        data_loader = DataLoader(
            train_data,
            NcgModel.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collator,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(), NcgModel.LEARNING_RATE, NcgModel.MOMENTUM
        )

        start = datetime.now()
        for epoch in range(NcgModel.EPOCHS):
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

        data_loader = DataLoader(
            test_data, NcgModel.BATCH_SIZE, collate_fn=self.collator
        )
        batch_score = 0.0

        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                preds = self.model(features)
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
    returns a fscore for the result batch
    """
    tp = fp = fn = 0

    for pred, label in zip(preds, labels):
        tp_data = [i for i in pred if i in label]
        tp = tp + len(tp_data)

        fp_data = [i for i in pred if i not in label]
        fp = fp + len(fp_data)

        fn_data = [i for i in label if i not in pred]
        fn = fn + len(fn_data)

    return fscore(tp, fp, fn)


def fscore(tp, fp, fn):
    """
    Computes the fscore using the tp, fp, fn
    """
    return tp / (tp + 0.5 * (fp + fn))
