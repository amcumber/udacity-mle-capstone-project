import pathlib
import pickle
from collections import namedtuple
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from capstone_tools.enums import ViewedAndRedeemedCols as VRCols
from capstone_tools.data_splitting import DataLabel, DataLabelType

TARGET = VRCols.offer_success


def load_dataset(location: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and labels within a folder location, requires the location to
    have data.pkl and label.pkl within
    """
    data_file = location.joinpath("data.pkl")
    label_file = location.joinpath("label.pkl")
    for file in (data_file, label_file):
        with open(file, "rb") as fh:
            data = pickle.load(fh)
        yield data


def prune_labels(labels: pd.DataFrame) -> np.ndarray:
    """Prune data label to the 'y' values - removing ID information"""
    return labels[TARGET].values


def prune_data(data: pd.DataFrame) -> np.ndarray:
    """Prune data to the 'X' values - removing headers and indices"""
    return data.values


def package_data(data_arr: np.ndarray, label_arr: np.ndarray) -> DataLabelType:
    """Package data and label into a DataSet"""
    return DataLabel(data_arr, label_arr)


def load_data_corpus(location: pathlib.Path) -> Dict[str, DataLabelType]:
    """Primary entrypoint to load all data from folder structure"""
    folders = ["train", "test", "validation"]
    datasets = {}
    for folder in folders:
        loc = location.joinpath(folder)
        assert loc.exists(), f"bad file structure - <root>/{folder}/data.pkl"
        data, labels = load_dataset(loc)
        X, y = prune_data(data), prune_labels(labels)
        dataset = package_data(X, y)
        datasets[folder] = dataset
    return datasets
