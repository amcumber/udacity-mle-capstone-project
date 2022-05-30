from cmath import exp
import pathlib
import unittest
import numpy as np
import pandas as pd
from itertools import chain

from test import make_test_assets, tear_down_test_assets
from capstone_tools import feature_generation
from capstone_tools.data_cleaners import clean
from capstone_tools.enums import Event


# NOTE: data_cleaners must pass tests - otherwise this will fail as well


class TestTranscriptTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.KEYS = (
            "id",
            "duration_hours",
            "duration_days",
            "difficulty",
        )
        make_test_assets.main()
        root = pathlib.Path(__file__).parent.parent / "assets"
        df_names = ("portfolio", "profile", "transcript")
        filenames = (root.joinpath(f"test_{f}.json") for f in df_names)
        for name, file in zip(df_names, filenames):
            df = pd.read_json(file, orient="records", lines=True)
            clean_df = clean(df, name)
            if name == "transcript":
                # Correct for bad enumerations
                clean_df = clean_df.assign(
                    reward=np.nan,
                    amount=np.nan,
                    event=lambda df_: df.event.astype("string"),
                )
                clean_df.loc[1, ["event", "offer_id", "person", "amount"]] = [
                    Event.transaction,
                    clean_df.loc[0, "offer_id"],
                    clean_df.loc[0, "person"],
                    10.0,
                ]
                clean_df.loc[2, ["event", "offer_id", "person", "reward"]] = [
                    Event.completed,
                    clean_df.loc[0, "offer_id"],
                    clean_df.loc[0, "person"],
                    5.0,
                ]
                clean_df.event = clean_df.event.astype("category")
            setattr(self, name, clean_df)
        self.transformer = feature_generation.TranscriptTransformer(
            self.transcript,
            self.portfolio,
        )

    def tearDown(self) -> None:
        tear_down_test_assets.main()

    def test_init(self):
        transformer = feature_generation.TranscriptTransformer(
            self.transcript,
            self.portfolio,
        )
        self.assertTrue(
            isinstance(transformer, feature_generation.TranscriptTransformer)
        )

    def test_merge_portfolio(self):
        df = self.transformer.merge_portfolio(self.KEYS)
        expected_cols = (
            "person",
            "event",
            "time",
            "offer_id",
        )
        for col in chain(self.KEYS, expected_cols):
            if "id" == col:
                continue
            self.assertTrue(col in df.columns, f"{col} not in DataFrame")

    def test_assign_event_id(self):
        df = self.transformer.merge_portfolio(self.KEYS).pipe(
            self.transformer.assign_event_id
        )
        self.assertTrue("event_id" in df.columns)

    def test_sort_by(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
        )
        expected = list(df["event_id"])
        expected.sort()
        self.assertListEqual(df["event_id"].to_list(), expected)

    def test_assign_offer_start(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
        )
        expected = "offer_start"
        self.assertTrue(expected in df.columns)

    def test_assign_offer_duration(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
        )
        expected = "offer_duration"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_assign_elapsed_time(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
        )
        expected = "elapsed_time"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_assign_offer_valid(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_offer_valid)
        )
        expected = "offer_valid"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_assign_offer_viewed(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_offer_valid)
            .pipe(self.transformer.assign_offer_viewed)
        )
        expected = "offer_viewed"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    # def test_assign_offer_success(self):
    #     sort_on = ["event_id", "time"]
    #     df = (
    #         self.transformer.merge_portfolio(self.KEYS)
    #         .pipe(self.transformer.assign_event_id)
    #         .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
    #         .pipe(self.transformer.assign_offer_start)
    #         .pipe(self.transformer.assign_offer_duration)
    #         .pipe(self.transformer.assign_elapsed_time)
    #         .pipe(self.transformer.assign_offer_valid)
    #         .pipe(self.transformer.assign_offer_viewed)
    #         .pipe(self.transformer.assign_offer_success)
    #     )
    #     expected = "offer_success"
    #     self.assertTrue(expected in df.columns)
    #     self.assertFalse(df[expected].hasnans)

    def test_transform(self):
        df = self.transformer.transform()
        expected_cols = (
            "offer_valid",
            "offer_viewed",
            "offer_redeemed",
            "sales",
            "costs",
            "profit",
        )
        for col in expected_cols:
            self.assertTrue(col in df.columns)

    def test_reward_redeemed(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_reward_redeemed)
        )
        expected = "offer_redeemed"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_ffill_offer_id(self):
        expected = "offer_id"
        event_col = "event"
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
        )
        df[event_col] = df[event_col].astype("string")
        df = df.pipe(self.transformer.ffill_offer_id)
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_calculate_cumulative_sales(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_reward_redeemed)
        )
        df = df.pipe(self.transformer.calculate_sales_total)
        expected = "sales"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)
        max_val = df[expected].max()
        max_loc = df[expected].argmax()
        if max_loc + 2 < len(df):
            self.assertTrue(df.loc[max_loc + 2, expected] != max_val)
        if max_loc - 2 > 0:
            self.assertTrue(df.loc[max_loc - 2, expected] != max_val)

    def test_calculate_costs(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_reward_redeemed)
        )
        df = df.pipe(self.transformer.calculate_costs)

        expected = "costs"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)
        max_val = df[expected].max()
        max_loc = df[expected].argmax()
        if max_loc + 2 < len(df):
            self.assertTrue(df.loc[max_loc + 2, expected] != max_val)
        if max_loc - 2 > 0:
            self.assertTrue(df.loc[max_loc - 2, expected] != max_val)

    def test_calculate_profit(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_reward_redeemed)
            .pipe(self.transformer.calculate_sales_total)
            .pipe(self.transformer.calculate_costs)
            .pipe(self.transformer.calculate_profit)
        )
        expected = "profit"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)
        costs = "costs"
        sales = "sales"
        cost_loc = df[costs].argmax()
        self.assertTrue(
            df.loc[cost_loc, expected]
            == (df.loc[cost_loc, sales] - df.loc[cost_loc, costs])
        )


class TestEventTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tc = TestTranscriptTransformer()
        self.tc.setUp()
        self.input = self.tc.transformer.transform()
        self.transformer = feature_generation.EventTransformer(self.input)

    def tearDown(self) -> None:
        return self.tc.tearDown()

    def test_init(self):
        transformer = feature_generation.EventTransformer(
            self.input,
        )
        self.assertTrue(
            isinstance(transformer, feature_generation.EventTransformer)
        )

    def test_transform(self):
        df = self.transformer.transform()
        expected = len(self.input["event_id"].unique())
        self.assertTrue(len(df) == expected)
        self.assertTrue(len(df) == len(df.dropna()))

    def test_get_max_event_data(self):
        df = self.transformer.get_max_event_data()
        expected = len(self.input["event_id"].unique())
        self.assertTrue(len(df) == expected)

    def test_bool_cols_to_int(self):
        df = self.transformer.get_last_event_data()
        returned = self.transformer.bool_cols_to_float(df)
        cols = ["offer_viewed", "offer_redeemed"]
        for col in cols:
            self.assertTrue(col in returned.columns)
            self.assertTrue(returned[col].dtype == "float")


class TestProfileTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tc = TestTranscriptTransformer()
        self.tc.setUp()
        self.input = self.tc.profile
        self.input["gender"] = self.input["gender"].astype("string")
        self.input.loc[0, ["gender", "age", "income"]] = ["O", 22, 100000.0]
        self.input["gender"] = self.input["gender"].astype("category")
        self.transformer = feature_generation.ProfileTransformer(self.input)

    def tearDown(self) -> None:
        return self.tc.tearDown()

    def test_init(self):
        transformer = feature_generation.ProfileTransformer(
            self.input,
        )
        self.assertTrue(
            isinstance(transformer, feature_generation.ProfileTransformer)
        )

    def test_get_one_hot_gender(self):
        df = self.transformer.get_one_hot_gender(self.input.dropna())
        cols = [
            "gender_m",
            "gender_f",
            "gender_o",
        ]
        for col in cols:
            self.assertTrue(col in df.columns)

    def test_membership_to_int(self):
        df = self.transformer.membership_to_int(self.input.dropna())
        col = "membership"
        self.assertTrue(col in df.columns)
        self.assertTrue(df[col].dtype == "int64")


if __name__ == "__main__":
    unittest.main()
