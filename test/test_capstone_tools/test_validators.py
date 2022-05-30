import unittest
import pandas as pd
import numpy as np

from capstone_tools import validators


class MockClass:
    @staticmethod
    @validators.validate_cols(("TestCol",))
    def mock_static_method(df: pd.DataFrame) -> bool:
        return True

    @validators.validate_cols(("TestCol",))
    def mock_method(self, df: pd.DataFrame) -> bool:
        return True


class TestValidateCols(unittest.TestCase):
    def setUp(self) -> None:
        self.good_df = pd.DataFrame(
            np.random.rand(10, 2), columns=["TestCol", "b"]
        )
        self.bad_df = pd.DataFrame(
            np.random.rand(10, 2), columns=["NotTestCol", "b"]
        )

    def test_static_method_happy_path(self):
        self.assertTrue(MockClass.mock_static_method(self.good_df))

    def test_method_happy_path(self):
        self.assertTrue(MockClass().mock_method(self.good_df))

    def test_static_method_key_error(self):
        with self.assertRaises(KeyError):
            MockClass.mock_static_method(self.bad_df)

    def test_method_key_error(self):
        with self.assertRaises(KeyError):
            MockClass().mock_method(self.bad_df)

    def test_no_args(self):
        with self.assertRaises(AttributeError):

            @validators.validate_cols
            def bad_func():
                return True

            bad_func()


if __name__ == "__main__":
    unittest.main()
