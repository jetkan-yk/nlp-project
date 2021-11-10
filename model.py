"""
Loads, saves model and implements the `NcgModel` class
"""

import os
import pickle
from collections import Counter
from datetime import datetime

import sklearn.metrics as metrics
import torch
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import Model, Optimizer, Pipeline, Sampling


class NcgModel:
    """
    A model class that is powered by a `PyTorch nn.Module` subclass.
    """

    def __init__(self, config):
        self.subtask = config["SUBTASK"]
        self.device = config["DEVICE"]
        self.model_type = config["MODEL"]
        try:
            self.model = config["MODEL"].value().to(self.device)
        except AttributeError:
            self.model = config["MODEL"].value()
        self.batch_size = config["BATCH_SIZE"]
        self.epochs = config["EPOCHS"]
        self.lr = config["LEARNING_RATE"]
        self.momentum = config["MOMENTUM"]
        self.optimizer = config["OPTIMIZER"]
        self.pipeline = config["PIPELINE"]
        self.sampling = config["SAMPLING"]
        self.summary_mode = config["SUMMARY_MODE"]

        print(f"Using model: {self.model_type.name}\n")

    def _dataloader(self, dataset):
        if self.sampling is Sampling.OVERSAMPLING:
            if self.pipeline is not Pipeline.CLASSIFICATION:
                raise TypeError("Cannot oversampling non-classification problem")

            _, labels = zip(*dataset)

            class_count = list(Counter(labels).values())
            class_weights = 1.0 / torch.tensor(class_count)

            # list of weights denoting probability of sample of corresponding indice being sampled
            sample_weights = [class_weights[i] for i in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.model.collate,
            )

        elif self.sampling is Sampling.SHUFFLE:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.model.collate,
            )

        else:
            raise NotImplementedError

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        if self.optimizer is Optimizer.ADAM:
            return optim.Adam(self.model.parameters(), lr=self.lr)

        elif self.optimizer is Optimizer.ADAMW:
            return optim.AdamW(self.model.parameters(), lr=self.lr)

        elif self.optimizer is Optimizer.SGD:
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
            )

        else:
            raise NotImplementedError

    def train(self, train_data, model_name):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        # training of naive bayes classifier
        if self.model_type is Model.NAIVE_BAYES:
            # get [features], [labels]
            loader = DataLoader(train_data, batch_size=len(train_data))

            train_x, train_y = next(iter(loader))

            # encode features with tf-idf
            tfidf_vect = TfidfVectorizer(max_features=5000)
            tfidf_vect.fit(train_x)

            train_x = tfidf_vect.transform(train_x)

            # train classifier
            classifier = MultinomialNB().fit(train_x, train_y)

            # save classifier
            model_path = os.path.join(f"subtask{self.subtask}", model_name)
            with open(model_path, "wb") as outfile:
                pickle.dump(
                    {"vectorizer": tfidf_vect, "classifier": classifier}, outfile
                )
                outfile.close()
            return

        # training of neural models
        data_loader = self._dataloader(train_data)
        criterion = self._criterion()
        optimizer = self._optimizer()

        print(f"Begin training...")
        start = datetime.now()
        for epoch in range(self.epochs):
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

                # log loss
                if self.summary_mode:
                    wandb.log({"loss": loss})

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
        # testing of naive bayes classifier
        if self.model_type is Model.NAIVE_BAYES:
            # load checkpoint
            model_path = os.path.join(f"subtask{self.subtask}", model_name)
            f = open(model_path, "rb")
            checkpoint = pickle.load(f)
            tfidf_vect = checkpoint["vectorizer"]
            classifier = checkpoint["classifier"]

            # get [features], [labels]
            loader = DataLoader(test_data, batch_size=len(test_data))

            test_x, test_y = next(iter(loader))

            # encode features with tf-idf
            test_x = tfidf_vect.transform(test_x)

            print(f"Begin testing...")
            # predict labels
            y_score = classifier.predict(test_x)
            preds = y_score
            labels = test_y

            # calculate f1 score
            score = metrics.f1_score(labels, preds)
            print(f"F1 score: {score:.{3}}\n")

            # calculate accuracy
            #             n_right = 0
            #             for i in range(len(y_score)):
            #                 if y_score[i] == test_y[i]:
            #                     n_right += 1

            #             print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

            return

        # testing of neural models
        self.model = load_model(self.subtask, self.model, model_name)
        # Use default samping method
        self.sampling = Sampling.SHUFFLE

        data_loader = self._dataloader(test_data)
        batch_score = 0.0

        print(f"Begin testing...")
        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                features = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(features)
                preds = self.model.predict(outputs)
                tp, fp, _, fn = self.model.evaluate(preds, labels)
                batch_score += f1_score(tp, fp, fn)

        print(f"F1 score: {batch_score / len(data_loader):.{3}}\n")


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
