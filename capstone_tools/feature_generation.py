from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from uuid import uuid4
from attr import field

import numpy as np
import pandas as pd

from capstone_tools.enums import (
    Event,
    EventCols,
    PortfolioCols,
    ProfileTransformedCols,
    TranscriptCols,
    TranscriptTransformedCols,
)
from capstone_tools.validators import validate_cols, RegistrationError


# _registered_transformers: dict[str, TransformerBase] = {}
_registered_transformers = {}

PCols = PortfolioCols()
TCols = TranscriptCols()
TTCols = TranscriptTransformedCols()
ECols = EventCols()
PTCols = ProfileTransformedCols()


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
    MERGE_KEYS: Tuple[str] = field(repr=False, init=False)

    def __post_init__(self):
        self.MERGE_KEYS = (
            PCols.id,
            PCols.duration_hours,
            PCols.reward,
            PCols.difficulty,
            PCols.offer_type,
            PCols.web,
            PCols.email,
            PCols.mobile,
            PCols.social,
            PCols.bogo,
            PCols.discount,
            PCols.info,
        )

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
        df = self.merge_portfolio(self.MERGE_KEYS)
        sort_on = [TTCols.event_id, TCols.time]
        index_name = df.index.name
        if index_name is None:
            index_name = TTCols.index

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
            .pipe(self.calculate_sales_total)
            .pipe(self.calculate_costs)
            .pipe(self.calculate_profit)
            .pipe(lambda df: self.reset(df, index_name))
        )

    def merge_portfolio(self, portfolio_keys: List[str]) -> pd.DataFrame:
        REQ_KEYS = (PCols.id, PCols.duration_hours)
        for key in REQ_KEYS:
            err_txt = f"key: '{key}' is required in `portfolio_keys`"
            assert key in portfolio_keys, err_txt

        return pd.merge(
            self.df,
            self.portfolio[list(portfolio_keys)],
            how="left",
            left_on=TCols.offer_id,
            right_on=PCols.id,
            suffixes=("", "_offer"),
        ).drop(PCols.id, axis=1)

    @staticmethod
    def sort_by(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        """
        Sort by a given key and reset the index in order to preserve the
        previous order
        """
        for key in keys:
            if key not in df.columns:
                raise AttributeError(f"'{key}' must be present in DataFrame")
        return df.sort_values(keys).reset_index(drop=False)

    @staticmethod
    def reset(df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Reset index to given key name and sort the dataframe on the new index.
        """
        if key not in df.columns:
            raise AttributeError(f"'{key}' must be present in DataFrame")
        return df.set_index(key).sort_index()

    @classmethod
    @validate_cols((TTCols.time, TTCols.event_id, TTCols.event))
    def assign_offer_start(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Start times to a dataframe filtered by event id"""
        col_name = TTCols.offer_start
        new_df = df.assign(**{col_name: np.nan})

        event_starts = new_df.query(f"{TCols.event} == '{Event.received}'")[
            TCols.time
        ]
        new_df.loc[event_starts.index, col_name] = event_starts.values

        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @staticmethod
    @validate_cols((TTCols.duration_hours,))
    def assign_offer_duration(df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Durations to dataframe filtered by event id"""
        col_name = TTCols.offer_duration
        parent_col = PCols.duration_hours
        return df.assign(**{col_name: (lambda df_: df_[parent_col].ffill())})

    @staticmethod
    @validate_cols((TTCols.time, TTCols.offer_start))
    def assign_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
        """Assign elapsed time of offer"""
        col_name = TTCols.elapsed_time
        return df.assign(
            **{
                col_name: (
                    lambda df_: df_[TCols.time] - df_[TTCols.offer_start]
                )
            }
        )

    @staticmethod
    @validate_cols((TTCols.elapsed_time, TTCols.offer_duration))
    def assign_offer_valid(df: pd.DataFrame) -> pd.DataFrame:
        """Assign valid Boolean for valid offers"""
        col_name = TTCols.offer_valid
        return df.assign(
            **{
                col_name: (
                    lambda df_: df_[TTCols.elapsed_time]
                    <= df_[TTCols.offer_duration]
                )
            }
        )

    @classmethod
    @validate_cols((TTCols.event,))
    def assign_offer_viewed(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        col_name = TTCols.offer_viewed
        new_df = df.assign(**{col_name: np.nan})
        offer_views = new_df.query(f"{TCols.event} == '{Event.viewed}'").index
        offer_starts = new_df.query(
            f"{TCols.event} == '{Event.received}'"
        ).index
        new_df.loc[offer_views, col_name] = True
        new_df.loc[offer_starts, col_name] = False
        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @classmethod
    @validate_cols((TTCols.event,))
    def assign_reward_redeemed(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        col_name = TTCols.offer_redeemed
        new_df = df.assign(**{col_name: np.nan})
        offer_starts = new_df.query(
            f"{TCols.event} == '{Event.received}'"
        ).index
        offer_rewarded = new_df.query(
            f"{TCols.event} == '{Event.completed}'"
        ).index
        new_df.loc[offer_starts, col_name] = False
        new_df.loc[offer_rewarded, col_name] = True
        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    # # [ACM] Removing since I expect the model to assess if the offer was
    # #       successful
    #
    # @staticmethod
    # @validate_cols((FCols.offer_viewed, FCols.offer_valid))
    # def assign_offer_success(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Assign transaction events successful if offer_viewed and offer_valid are
    #     both true
    #     """
    #     new_df = df.assign(offer_success=False)
    #     transactions = new_df.query(f"event == '{Event.transaction}'").index
    #     new_df.loc[transactions, "offer_success"] = (
    #         new_df.loc[transactions, FCols.offer_viewed]
    #         & new_df.loc[transactions, FCols.offer_valid]
    #     )
    #     return new_df

    @staticmethod
    def assign_event_id(
        df: pd.DataFrame, sorted_key: List[str] = [TCols.person, TCols.time]
    ) -> pd.DataFrame:
        """
        Generate an event ID column for each offer event - ends when a new
        offer is sent. Each offer event is separated by sending a new offer to
        a customer since only one offer is sent at a time.
        """
        col_name = TTCols.event_id
        for ele in sorted_key:
            if ele not in df.columns:
                raise AttributeError(f"'{ele}' must be present in DataFrame")
        index_name = df.index.name
        if not index_name:
            index_name = TTCols.index
        new_df = (
            df.sort_values(sorted_key)
            .reset_index(drop=False)
            .assign(**{col_name: np.nan})
        )

        sub_df = new_df.query(f'{TCols.event} == "{Event.received}"').assign(
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

    @classmethod
    @validate_cols((TTCols.offer_id,))
    def ffill_offer_id(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill offer_id with a sorted df"""
        col_name = TCols.offer_id
        return df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @classmethod
    @validate_cols((TTCols.amount,))
    def calculate_sales_total(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total sales (cumulative) of for each event id"""
        col_name = TTCols.sales
        old_col = TCols.amount
        new_df = df.assign(**{col_name: (lambda df_: df_[old_col])})
        new_df[col_name] = new_df[col_name].fillna(0.0)
        return new_df.assign(
            **{
                col_name: (
                    lambda df_: df_[[col_name, TTCols.event_id]]
                    .groupby(TTCols.event_id)
                    .cumsum()[col_name]
                )
            }
        )

    @classmethod
    def calculate_costs(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total costs (cumulative) of for each event id"""
        col_name = TTCols.costs
        old_col = TCols.reward
        new_df = df.assign(**{col_name: (lambda df_: df_[old_col])})
        new_df[col_name] = new_df[col_name].fillna(0.0)
        return new_df.assign(
            **{
                col_name: (
                    lambda df_: df_[[col_name, TTCols.event_id]]
                    .groupby(TTCols.event_id)
                    .cumsum()[col_name]
                )
            }
        )

    @classmethod
    @validate_cols((TTCols.costs, TTCols.sales))
    def calculate_profit(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Defining profit as transactions minus costs"""
        col_name = TTCols.profit
        return df.assign(
            **{col_name: (lambda df_: df_[TTCols.sales] - df_[TTCols.costs])}
        )


class PortfolioTransformer(TransformerBase):
    def transform(self):
        return self.df


@dataclass
class ProfileTransformer(TransformerBase):
    # events: pd.DataFrame = field(repr=False)

    def transform(self):
        return (
            self.df.rename(columns={PTCols.id: PTCols.person})
            .dropna()
            .pipe(self.get_one_hot_gender)
            .pipe(self.membership_to_int)
        )

    @staticmethod
    @validate_cols((PTCols.age,))
    def get_one_hot_gender(df: pd.DataFrame) -> pd.DataFrame:
        """Make Age column a one-hot set"""
        dummies = pd.get_dummies(df[[PTCols.gender]])
        return (
            pd.concat(objs=(df, dummies), axis=1)
            .drop([PTCols.gender], axis=1)
            .rename(columns=(lambda x: x.lower()))
        )

    @staticmethod
    @validate_cols((PTCols.became_member_on,))
    def membership_to_int(df: pd.DataFrame) -> pd.DataFrame:
        """
        Change membership column to an int equal to the number of days from
        year 2010.
        """
        timedelta = df[PTCols.became_member_on].transform(
            lambda x: x - pd.Timestamp("20100101")
        )
        return df.assign(**{PTCols.membership: timedelta.dt.days}).drop(
            [PTCols.became_member_on], axis=1
        )


class EventTransformer(TransformerBase):
    """
    Transformer for Event Specific Elements - used as output from
    TranscriptTransformer
    """

    def transform(self) -> pd.DataFrame:
        max_events = self.get_max_event_data()
        last_events = self.get_last_event_data()
        df = pd.merge(
            max_events,
            last_events,
            left_index=True,
            right_index=True,
        )
        return self.bool_cols_to_float(df)

    # def get_n_elements(self) -> pd.Series:
    #     return self.df.groupby(ECols.event_id)[FCols.person].count()

    def get_max_event_data(self) -> pd.DataFrame:
        cols = [
            ECols.event_id,
            ECols.profit,
            ECols.time,
            ECols.offer_start,
            ECols.difficulty,
            ECols.web,
            ECols.email,
            ECols.mobile,
            ECols.social,
            ECols.bogo,
            ECols.discount,
            ECols.info,
        ]
        return self.df[cols].groupby(ECols.event_id).max()

    def get_last_event_data(self) -> pd.DataFrame:
        cols = [
            ECols.event_id,
            ECols.offer_viewed,
            ECols.offer_redeemed,
            ECols.person,
        ]
        return self.df[cols].groupby(ECols.event_id).last()

    @staticmethod
    @validate_cols((ECols.offer_viewed, ECols.offer_redeemed))
    def bool_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
        bool_cols = [
            ECols.offer_viewed,
            ECols.offer_redeemed,
        ]
        return df.assign(
            **{
                col: (lambda df_: df_[col].astype("float"))
                for col in bool_cols
            }
        )


def register_transformers():
    """Register Core Transformers"""
    transformers = {
        "portfolio": PortfolioTransformer,
        "transcript": TranscriptTransformer,
        "event": EventTransformer,
        "profile": ProfileTransformer,
    }
    for key, transformer in transformers.items():
        transformer.register(key)


register_transformers()
