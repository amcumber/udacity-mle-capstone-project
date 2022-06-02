import numpy as np
import pytest

from capstone_tools.data_splitting import (
    BadValueError,
    OverflowFractionError,
    train_test_val_split,
)


@pytest.mark.parametrize(
    "test_size, val_size", [(0.2, 0.2), (0.2, 0.3), (0.1, 0)]
)
def test_split_normal_operation(test_size, val_size):
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 2)
    expected_test = round(len(X) * test_size)
    expected_val = round(len(X) * val_size)
    expected_train = round(len(X) * (1 - test_size - val_size))
    train, test, val = train_test_val_split(
        X, y, test_size=test_size, val_size=val_size
    )

    returned_X, returned_y = train
    assert len(returned_X) == expected_train, "training data are not equal"
    assert len(returned_y) == expected_train, "training labels are not equal"

    returned_X, returned_y = test
    assert len(returned_X) == expected_test, "test data are not equal"
    assert len(returned_y) == expected_test, "test labels are not equal"

    returned_X, returned_y = val
    if val_size == 0:
        assert (
            returned_X is None
        ), "validation data is not None with 0 val_size"
        assert (
            returned_y is None
        ), "validation label is not None with 0 val_size"
    else:
        assert len(returned_X) == expected_val, "val data are not equal"
        assert len(returned_y) == expected_val, "val labels are not equal"


@pytest.mark.parametrize(
    "test_size, val_size", [(-0.2, 0.3), (0.1, -0.1), (2, 4)]
)
def test_split_error(test_size, val_size):
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 2)
    with pytest.raises(BadValueError):
        train_test_val_split(X, y, test_size=test_size, val_size=val_size)
        assert False, "Error not Raised"


@pytest.mark.parametrize(
    "test_size, val_size",
    [
        (0.5, 0.6),
    ],
)
def test_split_overflow_error(test_size, val_size):
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 2)
    with pytest.raises(OverflowFractionError):
        train_test_val_split(X, y, test_size=test_size, val_size=val_size)
        assert False, "Error not Raised"
