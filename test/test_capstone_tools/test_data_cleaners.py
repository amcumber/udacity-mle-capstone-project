import pathlib
import unittest
import pandas as pd

from test import make_test_assets, tear_down_test_assets

# from capstone_tools.data_cleaners import clean
from capstone_tools.data_cleaners import PortfolioCleaner


class TestPortfolioCleaner(unittest.TestCase):
    def setUp(self) -> None:
        make_test_assets.main()
        root = pathlib.Path(__file__).parent.parent / "assets"
        keys = ("portfolio", "profile", "transcript")
        filenames = (root.joinpath(f"test_{f}.json") for f in keys)
        for key, file in zip(keys, filenames):
            df = pd.read_json(file, orient="records", lines=True)
            setattr(self, key, df)
        self.transformer = PortfolioCleaner(
            self.portfolio,
        )

    def tearDown(self) -> None:
        tear_down_test_assets.main()

    def test_init(self):
        transformer = PortfolioCleaner(
            self.portfolio,
        )
        self.assertTrue(isinstance(transformer, PortfolioCleaner))

    def test_clean(self):
        df = self.transformer.clean()
        expected_cols = (
            "reward",
            "difficulty",
            "duration_days",
            "offer_type",
            "id",
            "web",
            "email",
            "social",
            "duration_hours",
        )
        for col in expected_cols:
            self.assertTrue(col in df.columns)

    def test_convert_duration_days_to_hours(self):
        DAYS_COLS = "duration_days"
        HOURS_COL = "duration_hours"
        DAY2HOUR = 24.0
        rename_cols = {"duration": DAYS_COLS}
        df = self.transformer.df.rename(columns=rename_cols)
        returned = self.transformer.convert_duration_days_to_hours(df)
        expected_col = "duration_hours"
        self.assertTrue(expected_col in returned.columns)

        expected = df.loc[0, DAYS_COLS] * DAY2HOUR
        self.assertAlmostEqual(returned.loc[0, HOURS_COL], expected)


if __name__ == "__main__":
    unittest.main()
