from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from attr import field

from capstone_tools.enums import Event, Offer


class RegistrationError(KeyError):
    """Registration Error for Registered Classes"""


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
class EventTransformer(TransformerBase):
    """Transformer for Event Log / Transcript dataframe"""

    n_cats: int = 5
    # na_cat_val: int = 0

    def __post_init__(self):
        self.categories = {}

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
        df = self.df
        sort_on = ["event_id", "time"]
        index_name = df.index.name
        if index_name is None:
            index_name = "index"

        df = (
            df.pipe(self.assign_event_id)
            .sort_values(sort_on)
            .pipe(self.assign_offer_start)
            .pipe(self.assign_offer_duration)
            .pipe(self.ffill_offer_id)
            .pipe(self.assign_elapsed_time)
            .pipe(self.assign_offer_valid)
            .pipe(self.assign_offer_viewed)
            .pipe(self.assign_offer_redeemed)
            .pipe(self.assign_offer_success)
            .pipe(self.calculate_sales_total)
            .pipe(self.calculate_costs)
            .pipe(self.calculate_profit)
            .assign(gender=lambda df_: self.convert_cat_to_enum(df_["gender"]))
            # .pipe(self.get_one_hot_gender)
        )

        for col in ("income", "became_member_on", "age"):
            new_col, cats = self.categorize_cont_ser(df[col])
            self.categories[col] = cats
            df = (
                df.assign(**{col: new_col})
                # .pipe(lambda df: self.merge_cats_to_one_hots(df, col))
                .sort_values(["event_id", "time"])
            )
        return df

    @classmethod
    # @validate_cols(("time", "event_id", "event"))
    def assign_offer_start(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Start times to a dataframe filtered by event id"""
        col_name = "offer_start"
        new_df = df.assign(**{col_name: np.nan})

        event_starts = new_df.query(f"event == '{Event.received}'")["time"]
        new_df.loc[event_starts.index, col_name] = event_starts.values

        return new_df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @staticmethod
    def assign_offer_duration(df: pd.DataFrame) -> pd.DataFrame:
        """Assign Event Durations to dataframe filtered by event id"""
        col_name = "offer_duration"
        parent_col = "offer_duration"
        return df.assign(**{col_name: (lambda df_: df_[parent_col].ffill())})

    @staticmethod
    def assign_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
        """Assign elapsed time of offer"""
        col_name = "elapsed_time"
        return df.assign(
            **{col_name: (lambda df_: df_["time"] - df_["offer_start"])}
        )

    @staticmethod
    # @validate_cols(("elapsed_time", "offer_duration"))
    def assign_offer_valid(df: pd.DataFrame) -> pd.DataFrame:
        """Assign valid Boolean for valid offers"""
        col_name = "offer_valid"
        time = "elapsed_time"
        duration = "offer_duration"
        new_df = df.assign(
            **{col_name: (lambda df_: df_[time] <= df_[duration])}
        )
        new_df[col_name] = new_df[col_name].astype(int)
        return new_df

    @classmethod
    def assign_offer_viewed(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed"""
        col_name = "offer_viewed"
        new_df = df.assign(**{col_name: np.nan})
        offer_views = new_df.query(f"event == '{Event.viewed}'").index
        offer_starts = new_df.query(f"event == '{Event.received}'").index
        new_df.loc[offer_views, col_name] = True
        new_df.loc[offer_starts, col_name] = False
        return new_df.assign(
            **{col_name: (lambda df_: df_[col_name].ffill().astype(int))}
        )

    @classmethod
    def assign_offer_redeemed(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Assign viewed boolean for events after an offer is viewed
        ACM Update: new algo - take groupby offer_id and take max value"""
        col_name = "offer_redeemed"
        new_df = df.assign(**{col_name: np.NaN})
        offer_starts = new_df.query(f"event == '{Event.received}'").index
        offer_rewarded = new_df.query(f"event == '{Event.completed}'").index
        new_df.loc[offer_rewarded, col_name] = True
        new_df.loc[offer_starts, col_name] = False
        return new_df.assign(
            **{col_name: (lambda df_: df_[col_name].ffill().astype(int))}
        )

    @classmethod
    def assign_offer_success(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign offer success based on a combination of offer_viewed,
        offer_redeemed, offer_completed
        """
        col = "offer_success"
        viewed = "offer_viewed"
        valid = "offer_valid"
        complete = "offer_redeemed"
        return df.assign(
            **{col: (lambda df_: df_[viewed] * df_[valid] * df_[complete])}
        )

    @staticmethod
    def assign_event_id(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an event ID column for each offer event - ends when a new
        offer is sent. Each offer event is separated by sending a new offer to
        a customer since only one offer is sent at a time.
        """
        sorted_key: List[str] = ["person", "time"]

        def get_uuid(_: Any):
            return uuid4().hex

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
        sub_df = new_df.query(f'event == "{Event.received}"')
        sub_df = sub_df.assign(
            **{col_name: (lambda df: (df.apply(get_uuid, axis=1)))}
        )

        new_df.loc[sub_df.index, col_name] = sub_df[col_name]
        new_df = new_df.assign(
            **{col_name: (lambda df_: df_[col_name].ffill().astype("string"))}
        )
        new_df = new_df.set_index(index_name).sort_index()
        return new_df

    @classmethod
    def ffill_offer_id(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill offer_id with a sorted df"""
        col_name = "offer_id"
        return df.assign(**{col_name: (lambda df_: df_[col_name].ffill())})

    @classmethod
    def calculate_sales_total(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total sales (cumulative) of for each event id"""
        col_name = "sales"
        old_col = "amount"
        new_df = df.assign(**{col_name: (lambda df_: df_[old_col])})
        new_df[col_name] = new_df[col_name].fillna(0.0)
        return new_df.assign(
            **{
                col_name: (
                    lambda df_: df_[[col_name, "event_id"]]
                    .groupby("event_id")
                    .cumsum()[col_name]
                )
            }
        )

    @classmethod
    def calculate_costs(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the total costs (cumulative) of for each event id"""
        col_name = "costs"
        old_col = "reward"
        new_df = df.assign(**{col_name: (lambda df_: df_[old_col])})
        new_df[col_name] = new_df[col_name].fillna(0.0)
        return new_df.assign(
            **{
                col_name: (
                    lambda df_: df_[[col_name, "event_id"]]
                    .groupby("event_id")
                    .cumsum()[col_name]
                )
            }
        )

    @classmethod
    # @validate_cols(("costs", "sales"))
    def calculate_profit(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Defining profit as transactions minus costs"""
        col_name = "profit"
        return df.assign(
            **{col_name: (lambda df_: df_["sales"] - df_["costs"])}
        )

    def categorize_cont_ser(
        self, ser: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[int, pd.CategoricalIndex]]:
        """
        Categorize series into index categories, returning the new categorized
        series along with a dict of the categories
        """
        qcut_names = pd.qcut(ser, self.n_cats)

        cats = {i: cat for i, cat in enumerate(qcut_names.cat.categories)}
        qcut_index = pd.qcut(ser, self.n_cats, labels=range(self.n_cats))
        return qcut_index, cats

    @staticmethod
    def get_one_hot_gender(df: pd.DataFrame) -> pd.DataFrame:
        """Make Age column a one-hot set"""
        # dummy_na = False
        dummy_na = True
        dummies = pd.get_dummies(df[["gender"]], dummy_na=dummy_na)
        return (
            pd.concat(objs=(df, dummies), axis=1)
            .drop(["gender"], axis=1)
            .rename(columns=(lambda x: x.lower()))
        )

    @staticmethod
    def convert_cat_to_enum(ser: pd.Series) -> pd.Series:
        """convert column to enumerations"""
        return ser.cat.codes

    @staticmethod
    def merge_cats_to_one_hots(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Convert a categorical column to a one-hot set of columns"""
        one_hots = pd.get_dummies(df[col], prefix=col)
        new_df = pd.merge(
            df.drop([col], axis=1),
            one_hots,
            how="left",
            left_index=True,
            right_index=True,
        )
        return new_df


@dataclass
class OutcomeTransformer(TransformerBase):
    """
    Transformer for Outcome Specific Elements - used as output from
    EventTransformer
    """

    offer: Offer = None

    def transform(self) -> pd.DataFrame:
        # offer = Offer.bogo
        labels = self.get_max_event_data(self.offer)
        input_data = self.get_input_data(self.offer)
        df = pd.merge(
            labels,
            input_data,
            on="event_id",
            how="outer",
        )
        return df

    # def get_n_elements(self) -> pd.Series:
    #     return self.df.groupby("event_id")["person"].count()

    def get_max_event_data(self, offer: Offer = None) -> pd.DataFrame:
        select_cols = [
            "event_id",
            "time",
            "offer_type",
            "offer_viewed",
            "offer_redeemed",
            "offer_success",
            "sales",
            "costs",
            "profit",
        ]
        df = self.df[select_cols]
        if offer:
            df = df.query(f'offer_type == "{offer}"')
        df = (
            df.drop("offer_type", axis=1)
            .groupby("event_id")
            .max()
            .reset_index()
        )
        return df

    def get_input_data(self, offer) -> pd.DataFrame:
        select_cols = [
            "event_id",
            "person",
            "offer_id",
            "offer_reward",
            "offer_difficulty",
            "gender",
            "age",
            "became_member_on",
            "income",
            "offer_type",
        ]

        df = self.df[select_cols]
        if offer:
            df = df.query(f'offer_type == "{offer}"')
        df = df.groupby("event_id").last().reset_index()
        return df


def register_transformers():
    """Register Core Transformers"""
    transformers = {
        "events": EventTransformer,
        "outcomes": OutcomeTransformer,
    }
    for key, transformer in transformers.items():
        transformer.register(key)


register_transformers()
