"""
Implements the NcgModelDemo class
"""

import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from subtask0.model0 import Model0, collator0, evaluator0


class NcgModelDemo:
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

        if self.subtask == 0:
            self.model = Model0().to(self.device)
            self.collator = collator0
            self.evaluator = evaluator0
        else:
            raise KeyError

        print(f"Initialized model {self.model}")

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        data_loader = DataLoader(
            train_data,
            NcgModelDemo.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collator,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(), NcgModelDemo.LEARNING_RATE, NcgModelDemo.MOMENTUM
        )

        start = datetime.now()
        for epoch in range(NcgModelDemo.EPOCHS):
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
        print(f"Training finished in {(end - start).seconds / 60.0} minutes.")

        save_model(self.subtask, self.model, model_name)

    def test(self, test_data, model_name):
        """
        Tests the model using `test_data`
        """
        self.model = load_model(self.subtask, self.model, model_name)

        data_loader = DataLoader(
            test_data, NcgModelDemo.BATCH_SIZE, collate_fn=self.collator
        )
        score = 0.0

        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                preds = self.model(features)
                score += self.evaluator(preds, labels)

        print(f"Accuracy: {score / len(data_loader):.{3}}")


def save_model(subtask, model: nn.Module, model_name):
    """
    Saves the model as `model_name` in the subtask folder.
    """
    checkpoint = model.state_dict()
    model_path = os.path.join(f"subtask{subtask}", model_name)

    torch.save(checkpoint, model_path)
    print(f"Model saved in {model_path}")


def load_model(subtask, model: nn.Module, model_name):
    """
    Loads the model state from `model_name`in the subtask folder.
    """
    model_path = os.path.join(f"subtask{subtask}", model_name)

    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model {model_path}")

    return model
