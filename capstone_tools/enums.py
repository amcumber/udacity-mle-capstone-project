from abc import ABC
from dataclasses import dataclass


class CategoryABC(ABC):
    """
    ABC to category (Column and Enums) dataclasses - providing a cols function
    """

    @classmethod
    def to_list(cls):
        """Return Enumerations as a list"""
        cols = [col for col in cls.__dict__ if not col.startswith("_")]
        return [col for col in cols if col != "to_list"]


@dataclass
class Event(CategoryABC):
    """Enumerations for Events Column in Transcript / Event Log"""

    received: str = "offer received"
    viewed: str = "offer viewed"
    transaction: str = "transaction"
    completed: str = "offer completed"


@dataclass
class Offer(CategoryABC):
    """Enumerations for Offer Types in Portfolio"""

    bogo: str = "bogo"
    discount: str = "discount"
    info: str = "informational"


@dataclass
class Channel(CategoryABC):
    """Enumerations for Offer Channel Types in Portfolio"""

    web: str = "web"
    email: str = "email"
    mobile: str = "mobile"
    social: str = "social"


@dataclass
class Gender(CategoryABC):
    """Enumerations for Gender in Profile"""

    m: str = "M"
    f: str = "F"
    o: str = "O"

    na: str = "na"


final_model_cols = [
    "index",
    "person",
    "offer_id",
    "event_id",
    "gender_f",
    "gender_m",
    "gender_o",
    # "gender_nan",
    # "became_member_on_0",
    "became_member_on_1",
    "became_member_on_2",
    "became_member_on_3",
    "became_member_on_4",
    "became_member_on_5",
    # "income_0",
    "income_1",
    "income_2",
    "income_3",
    "income_4",
    "income_5",
    # "age_0",
    "age_1",
    "age_2",
    "age_3",
    "age_4",
    "age_5",
    # "viewed_and_redeemed",
    "offer_success",
]
