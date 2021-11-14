"""
Loads, saves model and implements the `NcgModel` class
"""

import os
import pickle

# fix seed
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import sklearn.metrics as metrics
import torch
import wandb
from imblearn.over_sampling import RandomOverSampler
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from config import Criterion, Model, Optimizer, Pipeline, Sampling


class NcgModel:
    """
    A model class that is powered by a `PyTorch nn.Module` subclass.
    """

    def __init__(self, config):
        self.subtask = config["SUBTASK"]
        self.device = config["DEVICE"]
        self.model_type = config["MODEL"]
        if self.model_type == Model.NAIVE_BAYES:
            self.model = config["MODEL"].value()
        elif self.model_type == Model.SciBert_BiLSTM_CRF:
            tag_to_ix = config["TAG_TO_IX"]
            embedding_dim = config["EMBEDDING_DIM"]
            hidden_dim = config["HIDDEN_DIM"]
            self.model = (
                config["MODEL"]
                .value(tag_to_ix, embedding_dim, hidden_dim)
                .to(self.device)
            )
        else:
            self.model = config["MODEL"].value().to(self.device)
        self.batch_size = config["BATCH_SIZE"]
        self.epochs = config["EPOCHS"]
        self.lr = config["LEARNING_RATE"]
        self.momentum = config["MOMENTUM"]
        self.optimizer = config["OPTIMIZER"]
        self.pipeline = config["PIPELINE"]
        self.sampling = config["SAMPLING"]
        self.summary_mode = config["SUMMARY_MODE"]
        self.weight_decay = config["WEIGHT_DECAY"]
        self.criterion = config["CRITERION"]

        print(f"Using model: {self.model_type.name}\n")

    def _dataloader(self, dataset):
        # TODO: remove
        print(self.sampling)
        print(self.pipeline)

        if self.sampling is Sampling.OVERSAMPLING:
            if self.pipeline not in [Pipeline.CLASSIFICATION, Pipeline.SBERTEXTRACTIVE]:
                raise TypeError("Cannot oversample non-classification problem")

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
        if self.criterion == Criterion.CELOSS:
            return nn.CrossEntropyLoss()
        elif self.criterion == Criterion.BCELOSS:
            return nn.BCELoss()
        else:
            raise NotImplementedError

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
                weight_decay=self.weight_decay,
            )

        else:
            raise NotImplementedError

    def train(self, train_data, model_name, test_data):
        """
        Trains the model using `train_data` and saves the model in `model_name`
        """
        # training of naive bayes classifier
        if self.model_type is Model.NAIVE_BAYES:
            # get [features], [labels]
            loader = DataLoader(train_data, batch_size=len(train_data))

            train_x, train_y = next(iter(loader))

            if self.sampling is Sampling.OVERSAMPLING:
                train_x = np.array(train_x).reshape(-1, 1)

                ros = RandomOverSampler(random_state=0)
                train_x, train_y = ros.fit_resample(train_x, train_y)

                train_x = train_x.flatten()

            # encode features with tf-idf, reduce to lowercase, remove stopwords
            tfidf_vect = TfidfVectorizer(
                max_features=5000, lowercase=True, stop_words="english"
            )
            tfidf_vect.fit(train_x)

            train_x = tfidf_vect.transform(train_x)

            # train classifier
            classifier = naive_bayes.MultinomialNB().fit(train_x, train_y)

            # save classifier
            model_path = os.path.join(f"subtask{self.subtask}", model_name)
            with open(model_path, "wb") as outfile:
                pickle.dump(
                    {"vectorizer": tfidf_vect, "classifier": classifier}, outfile
                )
                outfile.close()

        elif self.model_type is Model.SciBert_BiLSTM_CRF:
            optimizer = self._optimizer()
            print(f"Begin training...")
            start = datetime.now()
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0

                for step, (x, y) in enumerate(train_data):
                    sentence_in = self.model.prepare_sequence(x.split()).to(self.device)
                    targets = torch.tensor(
                        [2] + [self.model.tag_to_ix[tag] for tag in y] + [2],
                        dtype=torch.long,
                    ).to(self.device)
                    # print(sentence_in.input_ids.size(), targets.size())

                    self.model.zero_grad()
                    loss = self.model.neg_log_likelihood(sentence_in, targets)
                    loss.backward()
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

                save_model(self.subtask, self.model, model_name)
                self.test(test_data, model_name)

            end = datetime.now()
            print(f"\nTraining finished in {(end - start).seconds / 60.0} minutes.\n")
            save_model(self.subtask, self.model, model_name)
            # test(test_data, model_name)
        else:
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

        elif self.model_type is Model.SciBert_BiLSTM_CRF:
            self.model = load_model(self.subtask, self.model, model_name)
            total_score = 0.0

            print(f"Begin testing...")
            self.model.eval()
            Tp = Fp = Tn = Fn = 0
            with torch.no_grad():
                for (x, y) in test_data:
                    sentence_in = self.model.prepare_sequence(x.split()).to(self.device)
                    targets = torch.tensor(
                        [2] + [self.model.tag_to_ix[tag] for tag in y] + [2],
                        dtype=torch.long,
                    ).to(self.device)

                    # model output is in the format [score, list of tags]
                    outputs = self.model(sentence_in)
                    output_phrases, target_phrases = self.model.predict(
                        outputs[1], targets
                    )

                    tp, fp, tn, fn = self.model.evaluate(output_phrases, target_phrases)
                    batch_score = f1_score(tp, fp, fn)
                    total_score += batch_score

                    Tp += tp
                    Fp += fp
                    Tn += tn
                    Fn += fn

                    if self.summary_mode:
                        wandb.log({"batch_score": batch_score})

            avg_score = total_score / len(test_data)
            score = f1_score(Tp, Fp, Fn)
            if self.summary_mode:
                wandb.log({"f1_score": avg_score})
            print(
                f"F1 score (each sentence): {avg_score:.{3}}, F1 score (all phrases): {score:.{3}}\n"
            )

        else:
            # testing of neural models
            self.model = load_model(self.subtask, self.model, model_name)
            # Use default samping method
            self.sampling = Sampling.SHUFFLE

            data_loader = self._dataloader(test_data)
            total_score = 0.0

            print(f"Begin testing...")
            self.model.eval()
            with torch.no_grad():
                for data in data_loader:
                    features = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    outputs = self.model(features)
                    preds = self.model.predict(outputs)
                    tp, fp, _, fn = self.model.evaluate(preds, labels)
                    batch_score = f1_score(tp, fp, fn)
                    total_score += batch_score

                    if self.summary_mode:
                        wandb.log({"batch_score": batch_score})

            avg_score = total_score / len(data_loader)
            if self.summary_mode:
                wandb.log({"f1_score": avg_score})
            print(f"F1 score: {avg_score:.{3}}\n")


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
