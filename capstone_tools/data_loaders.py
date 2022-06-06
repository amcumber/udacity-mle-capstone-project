from typing import Tuple
import pathlib
import pickle

import pandas as pd


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
