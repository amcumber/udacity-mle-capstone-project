from functools import wraps
from typing import Iterable

import pandas as pd


class RegistrationError(KeyError):
    """Registration Error for Registered Classes"""


def validate_cols(required_columns):
    """
    Validates required columns within input DataFrame to a wrapped function
    """
    if not isinstance(required_columns, Iterable):
        raise AttributeError("Needs `required_columns` as input to wrapper")

    def decorator(func):
        """Decorates Function"""

        @wraps(func)
        def wrapped(*args, **kwargs):
            """Wraps the function"""
            for arg in args:
                if not isinstance(arg, pd.DataFrame):
                    continue
                for col in required_columns:
                    if col not in arg.columns:
                        raise KeyError(f"Missing col: {col} in DataFrame")
                return func(*args, **kwargs)
            raise AttributeError("Expected DataFrame as argument")

        return wrapped

    return decorator
