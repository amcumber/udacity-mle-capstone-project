from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from capstone_tools.validators import RegistrationError, validate_cols
from capstone_tools.enums import PortfolioCols

# _registered_cleaners: dict[str, CleanerBase] = {}
_registered_cleaners = {}
PCols = PortfolioCols()


def clean(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Clean a dataframe provided a key to a registered CleanerBase class"""
    global _registered_cleaners
    cleaner = _registered_cleaners.get(key)

    if not cleaner:
        raise RegistrationError(
            f"Cleaner Class not registered, check key: {key}"
        )
    return cleaner(df).clean()


def _assign_empty(ser: pd.Series, assign_to: Dict[Any, Any]) -> pd.Series:
    """
    Convert a value in a series to a new value - intended for assigning nan to
    coded values
    """
    ser = ser.copy()
    for key, val in assign_to.items():
        ser[ser == key] = val
    return ser


def _assign_categories(
    df: pd.DataFrame, category_cols: List[Any]
) -> pd.DataFrame:
    """
    Assign columns as categories specified in category_cols

    Parameters
    ----------
    df : DataFrame
        dataframe to assign new columns
    category_cols : list[str]
        list of column names to assign as categories
    is_ordered : bool
        sets columns as an ordered category
    """
    new_df = df.copy()
    for col in category_cols:
        new_df = new_df.assign(**{col: (lambda df: pd.Categorical(df[col]))})

    return new_df


def _assign_string_cols(
    df: pd.DataFrame, string_cols: List[Any]
) -> pd.DataFrame:
    """
    Assign columns as categories specified in category_cols

    Parameters
    ----------
    df : DataFrame
        dataframe to assign new columns
    category_cols : list[str]
        list of column names to assign as categories
    is_ordered : bool
        sets columns as an ordered category
    """
    # Note: assigning is broken if you try to pass multiple evaluation values
    new_df = df.copy()
    for col in string_cols:
        new_df = new_df.assign(
            **{col: (lambda df_: df_[col].astype("string"))}
        )

    return new_df


@dataclass
class CleanerBase(ABC):
    """ABC for Cleaners"""

    df: pd.DataFrame

    @abstractmethod
    def clean(self) -> pd.DataFrame:
        raise NotImplementedError("Implement clean method")

    @classmethod
    def register(cls, key: str = None) -> None:
        """Register the Cleaner"""
        global _registered_cleaners
        if not key:
            key = cls.__name__
        _registered_cleaners[key] = cls

    @classmethod
    def unregister(cls, key: str = None) -> None:
        """Unregister a Transformer"""
        global _registered_cleaners
        if not key:
            key = cls.__name__
        _registered_cleaners.pop(key)


class TranscriptCleaner(CleanerBase):
    """Cleaner for Transcripts / Event Log DataFrames"""

    def clean(self) -> pd.DataFrame:
        """
        Clean the Event Log / Transcript DataFrame
        """
        STR_COLS = ("offer_id", "person")
        CAT_COLS = ("event",)
        for col in CAT_COLS:
            assert col in self.df.columns, f"Required column missing: {col}"

        return (
            self.df.pipe(self._expand_value_col)
            .pipe(lambda df_: _assign_categories(df_, CAT_COLS))
            .pipe(lambda df_: _assign_string_cols(df_, STR_COLS))
            .drop("value", axis=1)
        )

    @classmethod
    def _expand_value_col(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add expand categories within the "value" column within an Event Log
        DataFrame.

        ~~Pipeable~~
        """
        value_map = cls._get_value_map(df)  # Performs Validation
        for event, subdf in df.groupby("event"):
            idx = subdf.index
            cols = value_map.get(event, ())

            for col in cols:
                clean_col = col.strip().replace(" ", "_")
                new_col_vals = subdf["value"].apply(lambda x: x[col])
                if clean_col not in df.columns:
                    df = df.assign(**{clean_col: np.nan})
                df.loc[idx, clean_col] = new_col_vals
        return df

    @staticmethod
    def _get_value_map(df: pd.DataFrame) -> Dict[str, Tuple[str]]:
        """
        Generate a dictionary a key map to contents in the "value" column
        depending on event type from "event" column

        Parameters
        ----------
        df : DataFrame containing columns "event" and "value" (event log)
            Event Log DataFrame

        Returns
        -------
        dict[str, tuple[str, ...]]
            Map of event categories containing keys within the "value" column
        """
        REQ_COLS = ("event", "value")
        # Validation
        for col in REQ_COLS:
            assert col in df.columns, f"DataFrame needs: {REQ_COLS!r}"

        value_map = {}
        for event_type, subdf in df.groupby("event"):
            value_map[event_type] = tuple(subdf["value"].iloc[0].keys())
        return value_map


class PortfolioCleaner(CleanerBase):
    """Cleaner for Portfolio DataFrames"""

    def clean(self) -> pd.DataFrame:
        """Clean Portfolio DataFrame"""
        STR_COLS = ("id",)
        CAT_COLS = ("offer_type",)
        RENAME_COLS = {"duration": "duration_days"}

        for col in chain(STR_COLS, CAT_COLS, RENAME_COLS.keys()):
            assert col in self.df.columns, f"Required column missing: {col}"

        return (
            self.df.pipe(lambda df_: _assign_categories(df_, CAT_COLS))
            .pipe(lambda df_: _assign_string_cols(df_, STR_COLS))
            .pipe(lambda df_: df_.rename(columns=RENAME_COLS))
            .pipe(self.make_one_hot_from_channel)
            .pipe(self.make_one_hot_from_offer_type)
            .pipe(self.convert_duration_days_to_hours)
        )

    @staticmethod
    def make_one_hot_from_channel(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the channels column into a series of one-hot columns for each
        category for the portfolio DataFrame
        """
        assert "channels" in df.columns, f"DataFrame needs 'channels' column"

        df = df.copy()
        # Note: assumes one list has all categories - this is true by
        #       inspection, but not generally true
        idx = df.channels.str.len().argmax()
        max_list = tuple(df.loc[idx, "channels"])
        new_df = df.copy()
        for cat in max_list:
            new_df = new_df.assign(
                **{
                    cat: (
                        lambda df_: (
                            df_.channels.transform(lambda x: cat in x)
                        ).astype(int)
                    )
                }
            )
        return new_df.drop("channels", axis=1)

    @staticmethod
    @validate_cols((PCols.offer_type,))
    def make_one_hot_from_offer_type(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the channels column into a series of one-hot columns for each
        category for the portfolio DataFrame
        """
        dummies = pd.get_dummies(df[[PCols.offer_type]]).rename(
            columns=lambda x: x.split(f"{PCols.offer_type}_")[1]
        )
        return pd.concat(objs=(df, dummies), axis=1)

    @staticmethod
    @validate_cols((PCols.duration_days,))
    def convert_duration_days_to_hours(df: pd.DataFrame) -> pd.DataFrame:
        old_col = PCols.duration_days
        col_name = PCols.duration_hours
        DAY2HOUR = 24.0

        return df.assign(
            **{
                col_name: (
                    lambda df_: df_[old_col].apply(lambda x: x * DAY2HOUR)
                )
            }
        )


class ProfileCleaner(CleanerBase):
    """Cleaner for Profile DataFrame"""

    def clean(self) -> pd.DataFrame:
        """Clean Profile DataFrame"""
        STR_COLS = ("id",)
        CAT_COLS = ("gender",)
        DATE_COLS = ("became_member_on",)
        NAN_INPUTS = {"age": {118: np.nan}}
        for col in chain(STR_COLS, CAT_COLS, DATE_COLS, NAN_INPUTS.keys()):
            assert col in self.df.columns, f"Required column missing: {col}"

        return (
            self.df.pipe(lambda df_: _assign_categories(df_, CAT_COLS))
            .pipe(lambda df_: _assign_string_cols(df_, STR_COLS))
            .assign(
                **{
                    col: (
                        lambda df_: pd.to_datetime(
                            df_["became_member_on"], format="%Y%m%d"
                        )
                    )
                    for col in DATE_COLS
                }
            )
            .assign(
                **{
                    col: (lambda df_: _assign_empty(df_[col], na_map))
                    for col, na_map in NAN_INPUTS.items()
                }
            )
        )


def __register_cleaners():
    """Register Base Cleaners"""
    cleaners = {
        "transcript": TranscriptCleaner,
        "portfolio": PortfolioCleaner,
        "profile": ProfileCleaner,
    }
    for key, cleaner in cleaners.items():
        cleaner.register(key)


__register_cleaners()
