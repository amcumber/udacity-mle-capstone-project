from abc import ABC

import numpy as np

from capstone_tools.data_splitting import DataSet, DataSetType


class ModelTrainer(ABC):
    """Protocol Class for ModelTrainer"""

    def __init__(self, *args, **kwargs):
        self.clf = ...

    def train(self, epochs, data: DataSetType, labels: DataSetType):
        """Run main training loop"""

    def eval(self, sample):
        """Evaluate model"""


class SKLearnModelTrainer:
    def __init__(self, clf):
        ...
