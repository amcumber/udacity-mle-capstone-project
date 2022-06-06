from collections import namedtuple
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DataLabelTuple = Tuple[np.ndarray, np.ndarray]
DataLabel = namedtuple("DataLabel", ["X", "y"])


class BadValueError(AttributeError):
    """Used for invalid fractions"""

    pass


class OverflowFractionError(ValueError):
    """Used to indicate fractions that sum over 1"""

    pass


@dataclass
class PreSplitData:
    """Dataclass to hold scaled numpy data and labels"""

    data: pd.DataFrame
    labels: pd.DataFrame
    val_splits: float
    test_splits: float


def train_test_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed_base: int = 42,
) -> Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]:
    """Split a dataset between train, test, and validation

    Parameters
    ----------
    X : np.ndarray
        data oriented where records are in axis=0
    y : np.ndarray
        labels oriented where records are in axis=0
    test_size : float, optional
        size of test set in percentage (0,1]
    val_size : float, optional
        size of validation set [0,1], by default 0.2
    seed_base : int, optional
        seed base for random number generators, by default 42

    Returns
    -------
    Tuple[DataLabelTuple, DataLabelTuple, DataLabelTuple]
        _description_
    """

    def validate_sizes():
        sizes = (test_size, val_size)
        for size in sizes:
            if size > 1 or size < 0:
                raise BadValueError(
                    "test_size and val_size must be between 0 and 1"
                )

        if sum(sizes) > 1:
            raise OverflowFractionError(
                "test_size and val_size must be less than 1"
            )

    validate_sizes()

    test_seed = seed_base + 42
    val_seed = seed_base + 420

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=test_seed
    )
    if val_size == 0:
        return (
            DataLabel(X_train, y_train),
            DataLabel(X_test, y_test),
            DataLabel(None, None),
        )

    corrected_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=corrected_val_size, random_state=val_seed
    )
    return (
        DataLabel(X_train, y_train),
        DataLabel(X_test, y_test),
        DataLabel(X_val, y_val),
    )


def write_dataset(
    dataset: DataLabelTuple, location: pathlib.Path, subdir: str
):
    """write pickles for each type of data  with each dataset"""
    folder = location.joinpath(subdir)
    folder.mkdir(exist_ok=True, parents=True)
    data_file, label_file = (
        folder / f"{item}.pkl" for item in ("data", "label")
    )
    for file, contents in zip((data_file, label_file), dataset):
        with open(file, "wb") as fh:
            pickle.dump(contents, fh)


def process_scaled_data(
    scaled_data: PreSplitData,
    location: pathlib.Path,
) -> None:
    """
    Splits scaled data into test sets, train sets, and validation sets and
    saves them as pickles in a data subfolder
    """
    X, y = scaled_data.data, scaled_data.labels
    test_split, val_split = scaled_data.test_splits, scaled_data.val_splits
    datasets = train_test_val_split(
        X=X, y=y, test_size=test_split, val_size=val_split
    )
    for dataset, split_name in zip(datasets, ("train", "test", "validation")):
        if type(dataset[0]) == type(None):
            continue
        write_dataset(dataset, location, split_name)
