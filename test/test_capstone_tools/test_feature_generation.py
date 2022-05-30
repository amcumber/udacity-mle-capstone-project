import pathlib
import unittest
import pandas as pd
from itertools import chain

from test import make_test_assets, tear_down_test_assets
from capstone_tools import feature_generation
from capstone_tools.data_cleaners import clean


# NOTE: data_cleaners must pass tests - otherwise this will fail as well


class TestTranscriptTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.KEYS = (
            "id",
            "duration_hours",
            "duration_days",
            "reward",
            "difficulty",
        )
        make_test_assets.main()
        root = pathlib.Path(__file__).parent.parent / "assets"
        keys = ("portfolio", "profile", "transcript")
        filenames = (root.joinpath(f"test_{f}.json") for f in keys)
        for key, file in zip(keys, filenames):
            df = pd.read_json(file, orient="records", lines=True)
            clean_df = clean(df, key)
            setattr(self, key, clean_df)
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
            if "id" is col:
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
        expected_cols = ("offer_valid", "offer_viewed", "reward_redeemed")
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
        expected = "reward_redeemed"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

    def test_ffill_offer_id(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.ffill_offer_id)
        )
        expected = "offer_id"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)
        self.assertFalse(True, "this test is not working as expected")

    def test_calculate_cumulative_transaction(self):
        sort_on = ["event_id", "time"]
        df = (
            self.transformer.merge_portfolio(self.KEYS)
            .pipe(self.transformer.assign_event_id)
            .pipe(lambda df_: self.transformer.sort_by(df_, sort_on))
            .pipe(self.transformer.assign_offer_start)
            .pipe(self.transformer.assign_offer_duration)
            .pipe(self.transformer.assign_elapsed_time)
            .pipe(self.transformer.assign_reward_redeemed)
            .pipe(self.transformer.calculate_cumulative_transactions)
        )
        expected = "TBR"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

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
            .pipe(self.transformer.calculate_costs)
        )
        expected = "TBR"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)

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
            .pipe(self.transformer.calculate_cumulative_transactions)
            .pipe(self.transformer.calculate_costs)
            .pipe(self.transformer.calculate_profit)
        )
        expected = "TBR"
        self.assertTrue(expected in df.columns)
        self.assertFalse(df[expected].hasnans)


if __name__ == "__main__":
    unittest.main()
