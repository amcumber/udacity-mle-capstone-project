from abc import ABC
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score

from capstone_tools.data_splitting import DataLabelType


class ModelProtocol(ABC):
    def fit(self):
        ...

    def fit_predict(self):
        ...

    def predict(self):
        ...


class ModelTrainer(ABC):
    """Protocol Class for ModelTrainer"""

    def __init__(self, *args, **kwargs):
        self.clf = ...

    def train(self, epochs, datasets: Dict[str, DataLabelType]):
        """Run main training loop"""

    def eval(self, sample):
        """Evaluate model"""


class SKLearnModelTrainer:
    def __init__(self, clf):
        self.clf = clf

    def train(self, datasets: Dict[str, DataLabelType]):
        accuracy = {}
        train_data, train_labels = datasets["train"]
        self.clf.fit(train_data, train_labels)
        train_preds = self.clf.predict(train_data)
        accuracy["train"] = accuracy_score(train_labels, train_preds)

        # valid_data, valid_labels = datasets["validation"]
        # valid_preds = self.clf.predict(valid_data)
        # acc_train = accuracy_score(valid_labels, valid_preds)

        test_data, test_labels = datasets["test"]
        test_preds = self.clf.predict(test_data)
        accuracy["test"] = accuracy_score(test_labels, test_preds)
        return accuracy

    def eval(self, sample):
        return self.clf.predict(sample)
