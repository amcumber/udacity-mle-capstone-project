import sys
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
import pathlib
import pickle
from typing import Any, Dict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

    def train(
        self, datasets: Dict[str, DataLabelType], *args, **kwargs
    ) -> Any:
        """Run main training loop"""

    def predict(self, sample):
        """Predict outputs given model"""


class SKLearnModelTrainer:
    def __init__(self, clf):
        self.clf = clf

    def train(
        self,
        datasets: Dict[str, DataLabelType],
        save_path: pathlib.Path = None,
    ) -> Dict[str, float]:
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
        if save_path:
            self.save(save_path)
        return accuracy

    def save(self, path: pathlib.Path) -> None:
        """Save Classifier at location"""
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.clf, fh)

    def predict(self, sample):
        return self.clf.predict(sample)


@dataclass
class TorchModelData:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    critereon: nn.modules.loss._Loss


def dataset_to_dataloader(
    dataset: DataLabelType,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Convert custom DataLabelType to a torch DataLoader"""
    data, labels = dataset
    tensor_x = torch.tensor(data, dtype=torch.float)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)


def dataset_dict_to_data_loader_dict(
    datasets: Dict[str, DataLabelType],
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Convert dict of custom DataLabelTypes to a dict of torch DataLoaders"""
    loader = {}
    for key, dataset in datasets.items():  # data, labels = subset
        loader[key] = dataset_to_dataloader(dataset, batch_size)
    return loader


class TorchTrainer:
    def __init__(self, model_data: TorchModelData, use_cuda: bool = None):
        self.clf = model_data.model
        self.optimizer = model_data.optimizer
        self.critereon = model_data.critereon
        self.use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.use_cuda = use_cuda

    def train(
        self,
        datasets: Dict[str, DataLabelType],
        n_epochs: int,
        batch_size: int,
        save_path: pathlib.Path,
        patience: int = None,
    ) -> Dict[str, float]:
        """Train the classifier given hyperparameters

        Parameters
        ----------
        datasets : Dict[str, DataLabelType]
            Dictionary of DataLabelTypes keys = ('train', 'test', 'validation')
        n_epochs : int
            Number of epochs to train
        batch_size : int
            Size of batches
        save_path : pathlib.Path
            filename for model to save
        patience : int, optional
            number of epochs to allow no progress made, enable to enact early
            stopping after this parameter's epochs, by default None

        Returns
        -------
        Dict[str, float]
            losses with keys = ('train', 'validation')
        """
        valid_loss_min = np.Inf
        data_loaders = dataset_dict_to_data_loader_dict(datasets, batch_size)
        stop_counter = 0
        epic_losses = defaultdict(list)
        if not patience:
            patience = np.Inf

        for epoch in range(n_epochs):
            train_loss, valid_loss = 0.0, 0.0
            # Train
            train_loss = self.train_epoch(data_loaders["train"], train_loss)
            epic_losses["train"].append(train_loss)

            # Validation
            valid_loss, _ = self.eval_epoch(
                data_loaders["validation"], valid_loss
            )
            epic_losses["validation"].append(valid_loss)

            print(
                f"Epoch: {epoch+1}\tTraining Loss:{train_loss:.4f}\tValid"
                f" Loss:{valid_loss:.4f}",
                end="\r",
            )
            sys.stdout.flush()
            if valid_loss <= valid_loss_min:
                print(
                    f"\nValidation Loss decreased({valid_loss_min:.4f} ->"
                    f" {valid_loss:.4f}) Saving Model...",
                )
                sys.stdout.flush()
                valid_loss_min = valid_loss
                self.save(save_path)
                stop_counter = 0
            else:
                stop_counter += 1

            if stop_counter > patience:
                print("Early stopping triggered, stopping...")
                return epic_losses
        return epic_losses

    def train_epoch(self, data_loader, train_loss):
        """Train the classifier through a single epoch"""
        # Train
        self.clf.train()
        for batch, (data, labels) in enumerate(data_loader):
            if self.use_cuda:
                data, labels = data.cuda(), labels.cuda()
                self.clf = self.clf.cuda()
            self.optimizer.zero_grad()
            preds = self.clf(data)
            loss = self.critereon(preds, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += 1 / (batch + 1) * (loss.data - train_loss)
        train_loss = train_loss.item()
        return train_loss

    def eval_epoch(self, data_loader, test_loss=0.0):
        """
        Test the classifier - can be used to evaluate validation or test
        data
        """
        # Validation / Test
        self.clf.eval()
        correct = 0
        total = 0
        for batch, (data, labels) in enumerate(data_loader):
            if self.use_cuda:
                data, labels = data.cuda(), labels.cuda()
                self.clf = self.clf.cuda()
            preds = self.clf(data)
            loss = self.critereon(preds, labels)
            test_loss += 1 / (batch + 1) * (loss.data - test_loss)
            total += data.size(0)
            pred_args = torch.argmax(preds, axis=1)
            correct += torch.sum(pred_args == labels.data)
        test_loss, correct = (ten.item() for ten in (test_loss, correct))
        return test_loss, (correct, total)

    def eval(self, dataset: DataLabelType, batch_size: int):
        data_loader = dataset_to_dataloader(dataset, batch_size)
        test_loss, (correct, total) = self.eval_epoch(
            data_loader, test_loss=0.0
        )
        print(
            f"Test Loss: {test_loss:.4f}\tAccuracy:"
            f" {correct/total*100:.2f} ({correct}/{total})"
        )

    def save(self, path: pathlib.Path) -> None:
        """Save Classifier at location"""
        path.parent.mkdir(exist_ok=True)
        torch.save(self.clf.state_dict(), path)

    def predict(self, sample):
        return self.clf.predict(sample)
