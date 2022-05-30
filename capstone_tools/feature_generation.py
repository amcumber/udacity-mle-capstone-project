from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from uuid import uuid4
from attr import field

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

    df: pd.DataFrame = field(repr=False)

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

    portfolio: pd.DataFrame = field(repr=False)

    def transform(self) -> pd.DataFrame:
        """
        Transform Event Log / Transcript dataframe to add new features

        New Features
        ------------
        start : float
            time latest offer started - for each person ID
        elapsed_time : float
            time since last offer that has elapsed
        offer_duration : float
            duration of offer
        offer_valid : bool
            if offer is valid
        offer_viewed : bool
            if offer has been viewed
        """
        MERGE_KEYS = (
            "id",
            "duration_hours",
            "reward",
            "difficulty",
            "offer_type",
        )
        df = self.merge_portfolio(MERGE_KEYS)
        sort_on = ["event_id", "time"]
        index_name = df.index.name
        if index_name is None:
            index_name = "index"

        return (
            df.pipe(self.assign_event_id)
            .pipe(lambda df: self.sort_by(df, sort_on))
            .pipe(self.assign_offer_start)
            .pipe(self.assign_offer_duration)
            .pipe(self.ffill_offer_id)
            .pipe(self.assign_elapsed_time)
            .pipe(self.assign_offer_valid)
            .pipe(self.assign_offer_viewed)
            .pipe(self.assign_reward_redeemed)
            # .pipe(self.assign_offer_success)
            .pipe(self.calculate_cumulative_transactions)
            .pipe(self.calculate_costs)
            .pipe(self.calculate_profit)
            .pipe(lambda df: self.reset(df, index_name))
        )

    def merge_portfolio(self, portfolio_keys: List[str]) -> pd.DataFrame:
        REQ_KEYS = ("id", "duration_hours")
        for key in REQ_KEYS:
            err_txt = f"key: '{key}' is required in `portfolio_keys`"
            assert key in portfolio_keys, err_txt

        return pd.merge(
            self.df,
            self.portfolio[list(portfolio_keys)],
            how="left",
            left_on="offer_id",
            right_on="id",
            suffixes=("", "_offer"),
        ).drop("id", axis=1)

    @staticmethod
    def sort_by(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        for key in keys:
            if key not in df.columns:
                raise AttributeError(f"'{key}' must be present in DataFrame")

        # index_name = df.index.name
        # if not index_name:
        #     index_name = "index"
        # return df.sort_values(key).reset_index(drop=False), index_name
        return df.sort_values(keys).reset_index(drop=False)

    @staticmethod
    def reset(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if key not in df.columns:
            raise AttributeError(f"'{key}' must be present in DataFrame")
        return df.set_index(key).sort_index()

    @staticmethod
    @validate_cols(("time", "event_id", "event"))
    def assign_offer_start(df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Start times to a dataframe filtered by event id"""
        EVENT_COL = "event"
        col_name = "offer_start"
        new_df = df.assign(**{col_name: np.nan})

        event_starts = new_df.query(f"{EVENT_COL} == '{Event.received}'")[
            "time"
        ]
        new_df.loc[event_starts.index, col_name] = event_starts.values

        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @staticmethod
    @validate_cols(("duration_hours",))
    def assign_offer_duration(df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Durations to dataframe filtered by event id"""
        col_name = "offer_duration"
        parent_col = "duration_hours"
        return df.assign(**{col_name: (lambda df_: df_[parent_col].ffill())})

    @staticmethod
    @validate_cols(("time", "offer_start"))
    def assign_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:

        """Assign elapsed time of offer"""
        col_name = "elapsed_time"
        return df.assign(
            **{col_name: (lambda df_: df_["time"] - df_["offer_start"])}
        )

    @staticmethod
    @validate_cols(("elapsed_time", "offer_duration"))
    def assign_offer_valid(df: pd.DataFrame) -> pd.DataFrame:
        """Assign valid Boolean for valid offers"""
        col_name = "offer_valid"
        return df.assign(
            **{
                col_name: (
                    lambda df_: df_["elapsed_time"] <= df_["offer_duration"]
                )
            }
        )

    @staticmethod
    @validate_cols(("event",))
    def assign_offer_viewed(df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        EVENT_COL = "event"
        col_name = "offer_viewed"
        new_df = df.assign(**{col_name: np.nan})
        offer_views = new_df.query(f"{EVENT_COL} == '{Event.viewed}'").index
        offer_starts = new_df.query(f"{EVENT_COL} == '{Event.received}'").index
        new_df.loc[offer_views, col_name] = True
        new_df.loc[offer_starts, col_name] = False
        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @staticmethod
    @validate_cols(("event",))
    def assign_reward_redeemed(df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        EVENT_COL = "event"
        col_name = "reward_redeemed"
        new_df = df.assign(**{col_name: np.nan})
        offer_starts = new_df.query(f"{EVENT_COL} == '{Event.received}'").index
        offer_rewarded = new_df.query(
            f"{EVENT_COL} == '{Event.completed}'"
        ).index
        new_df.loc[offer_starts, col_name] = False
        new_df.loc[offer_rewarded, col_name] = True
        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    # # [ACM] Removing since I expect the model to assess if the offer was
    # #       successful
    #
    # @staticmethod
    # @validate_cols(("offer_viewed", "offer_valid"))
    # def assign_offer_success(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Assign transaction events successful if offer_viewed and offer_valid are
    #     both true
    #     """
    #     new_df = df.assign(offer_success=False)
    #     transactions = new_df.query(f"event == '{Event.transaction}'").index
    #     new_df.loc[transactions, "offer_success"] = (
    #         new_df.loc[transactions, "offer_viewed"]
    #         & new_df.loc[transactions, "offer_valid"]
    #     )
    #     return new_df

    @staticmethod
    def assign_event_id(
        df: pd.DataFrame, sorted_key: List[str] = ["person", "time"]
    ) -> pd.DataFrame:
        """
        Generate an event ID column for each offer event - ends when a new
        offer is sent
        """
        col_name = "event_id"
        for ele in sorted_key:
            if ele not in df.columns:
                raise AttributeError(f"'{ele}' must be present in DataFrame")
        index_name = df.index.name
        if not index_name:
            index_name = "index"
        new_df = (
            df.sort_values(sorted_key)
            .reset_index(drop=False)
            .assign(**{col_name: np.nan})
        )

        sub_df = new_df.query(f'event == "{Event.received}"').assign(
            **{
                col_name: (
                    lambda df: (df.apply(lambda _: uuid4().hex, axis=1))
                )
            }
        )

        new_df.loc[sub_df.index, col_name] = sub_df[col_name]

        return (
            new_df.assign(
                **{
                    col_name: (
                        lambda df_: df_[col_name].ffill().astype("string")
                    )
                }
            )
            .set_index(index_name)
            .sort_index()
        )

    def ffill_offer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME implement
        return df

    def calculate_cumulative_transactions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        # FIXME implement
        return df

    def calculate_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME implement
        return df

    def calculate_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Defining profit as transactions minus costs"""
        # FIXME implement
        return df


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
