from abc import ABC, abstractmethod
from dataclasses import dataclass
from glob import glob
from typing import List

import numpy as np
import pandas as pd

from capstone_tools.enums import Event
from capstone_tools.validators import validate_cols, RegistrationError


# _registered_transformers: dict[str, TransformerBase] = {}
_registered_transformers = {}


def transform(df: pd.DataFrame, key: str, **kwargs) -> pd.DataFrame:
    """
    Transform a dataframe provided a key and a registered TransformerBase
    Class
    """
    global _registered_transformers
    transformer = _registered_transformers.get(key)
    if not transformer:
        raise RegistrationError(
            f"Transformer Class not registered, check key: {key}"
        )
    return transformer(df, **kwargs).transform()


@dataclass
class TransformerBase(ABC):
    """Abstract Base Class for Transformers"""

    df: pd.DataFrame

    @abstractmethod
    def transform(self):
        """Transform data in self.df"""
        raise NotImplementedError("Implement this function")

    @classmethod
    def register(cls, key: str = None) -> None:
        """Register the Transformer"""
        global _registered_transformers
        if not key:
            key = cls.__name__
        _registered_transformers[key] = cls

    @classmethod
    def unregister(cls, key: str = None) -> None:
        """Unregister a Transformer"""
        global _registered_transformers
        if not key:
            key = cls.__name__
        _registered_transformers.pop(key)


@dataclass
class TranscriptTransformer(TransformerBase):
    """Transformer for Event Log / Transcript dataframe"""

    portfolio: pd.DataFrame

    def transform(self) -> pd.DataFrame:
        """
        Transform Event Log / Transcript dataframe to add new features

        New Features
        ------------
        start : float
            time latest offer started - for each person ID
        elapsed_time : float
            time since last offer that has elapsed
        duration : float
            duration of offer
        is_valid : bool
            if offer is valid
        is_viewed : bool
            if offer has been viewed
        """
        return (
            self._merge_portfolio()
            .pipe(self._assign_offer_start)
            .pipe(self._assign_offer_duration)
            .pipe(self._assign_elapsed_time)
            .pipe(self._assign_is_valid)
            .pipe(self._assign_is_viewed)
            .pipe(self._assign_offer_success)
        )
        # dfs = []
        # for _, person_df in df.groupby("person"):
        #     formatted_df = (
        #         person_df.pipe(self._assign_offer_start)
        #         .pipe(self._assign_offer_duration)
        #         .pipe(self._assign_elapsed_time)
        #         .pipe(self._assign_is_valid)
        #         .pipe(self._assign_is_viewed)
        #         .pipe(self._assign_offer_success)
        #     )
        #     dfs.append(formatted_df)
        # return pd.concat(dfs).sort_values("time")

    def _merge_portfolio(
        self, portfolio_keys: List[str] = ["id", "duration"]
    ) -> pd.DataFrame:
        REQ_KEYS = ("id", "duration")
        for key in REQ_KEYS:
            err_txt = f"key: '{key}' is required in `portfolio_keys`"
            assert key in portfolio_keys, err_txt

        return pd.merge(
            self.df,
            self.portfolio[portfolio_keys],
            how="left",
            left_on="offer_id",
            right_on="id",
            suffixes=("", "_offer"),
        ).drop("id", axis=1)

    @staticmethod
    @validate_cols(("time",))
    def _assign_offer_start(person_df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Start times to a dataframe filtered by person"""
        new_df = person_df.assign(start=np.nan)

        event_starts = new_df.query(f"event == '{Event.received}'")["time"]
        new_df.loc[event_starts.index, "start"] = event_starts.values

        return new_df.assign(start=lambda df_: df_["start"].ffill())

    @staticmethod
    @validate_cols(("duration",))
    def _assign_offer_duration(person_df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Durations to dataframe filtered by person"""
        new_df = person_df.assign(elapsed_time=np.nan)
        return new_df.assign(duration=lambda df_: df_["duration"].ffill())

    @staticmethod
    @validate_cols(("time", "start"))
    def _assign_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
        """Assign elapsed time of offer"""
        return df.assign(elapsed_time=lambda df_: df_["time"] - df_["start"])

    @staticmethod
    @validate_cols(("elapsed_time", "duration"))
    def _assign_is_valid(df: pd.DataFrame) -> pd.DataFrame:
        """Assign valid Boolean for valid offers"""
        return df.assign(
            is_valid=lambda df_: df_["elapsed_time"] <= df_["duration"]
        )

    @staticmethod
    def _assign_is_viewed(person_df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        new_df = person_df.assign(is_viewed=False)
        offer_views = new_df.query(f"event == '{Event.viewed}'").index
        new_df.loc[offer_views, "is_viewed"] = True
        return new_df.assign(is_viewed=lambda df_: df_["is_viewed"].ffill())

    @staticmethod
    def _assign_offer_success(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign transaction events successful if is_viewed and is_valid are
        both true
        """
        new_df = df.assign(offer_success=False)
        transactions = new_df.query(f"event == '{Event.transaction}'").index
        new_df.loc[transactions, "offer_success"] = (
            new_df.loc[transactions, "is_viewed"]
            & new_df.loc[transactions, "is_valid"]
        )
        return new_df


class PortfolioTransformer(TransformerBase):
    def transform(self):
        return self.df


class ProfileTransformer(TransformerBase):
    def transform(self):
        return self.df


def __register_transformers():
    """Register Core Transformers"""
    transformers = {
        "portfolio": PortfolioTransformer,
        "transcript": TranscriptTransformer,
        "profile": ProfileTransformer,
    }
    for key, transformer in transformers.items():
        transformer.register(key)


__register_transformers()
