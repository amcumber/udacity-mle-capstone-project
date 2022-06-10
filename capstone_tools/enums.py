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


@dataclass
class PortfolioCols(CategoryABC):
    """Column Names that are within the json file for portfolio.json"""

    channels: str = "channels"
    duration: str = "duration"
    reward: str = "reward"
    difficulty: str = "difficulty"
    offer_type: str = "offer_type"
    id: str = "id"

    duration_hours: str = "duration_hours"
    duration_days: str = "duration_days"
    web: str = "web"
    email: str = "email"
    mobile: str = "mobile"
    social: str = "social"
    info: str = "informational"
    bogo: str = "bogo"
    discount: str = "discount"


@dataclass
class ProfileCols(CategoryABC):
    """Column Names that are within the json file for profile.json"""

    gender: str = "gender"
    age: str = "age"
    id: str = "id"
    became_member_on: str = "became_member_on"
    income: str = "income"


@dataclass
class TranscriptCols(CategoryABC):
    """Column Names that are within the json file for profile.json"""

    value: str = "value"
    person: str = "person"
    event: str = "event"
    time: str = "time"

    offer_id: str = "offer_id"
    amount: str = "amount"
    reward: str = "reward"


@dataclass
class TranscriptTransformedCols(CategoryABC):
    """
    Column names for generated Transcript dataframe with implemented feature
    calculations.
    """

    index: str = "index"
    person: str = "person"
    event: str = "event"
    time: str = "time"
    offer_id: str = "offer_id"
    reward: str = "reward"
    amount: str = "amount"
    duration_hours: str = "duration_hours"
    reward_offer: str = "reward_offer"
    difficulty: str = "difficulty"
    offer_type: str = "offer_type"
    web: str = "web"
    email: str = "email"
    social: str = "social"
    bogo: str = "bogo"
    discount: str = "discount"
    informational: str = "informational"
    event_id: str = "event_id"
    offer_start: str = "offer_start"
    offer_duration: str = "offer_duration"
    elapsed_time: str = "elapsed_time"
    offer_viewed: str = "offer_viewed"
    offer_valid: str = "offer_valid"
    offer_redeemed: str = "offer_redeemed"
    sales: str = "sales"
    costs: str = "costs"
    profit: str = "profit"


@dataclass
class EventCols(CategoryABC):
    index: str = "index"
    event_id: str = "event_id"
    profit: str = "profit"
    time: str = "time"
    offer_start: str = "offer_start"
    difficulty: str = "difficulty"

    web: str = "web"
    email: str = "email"
    mobile: str = "mobile"
    social: str = "social"
    bogo: str = "bogo"
    discount: str = "discount"
    info: str = "informational"

    offer_viewed: str = "offer_viewed"
    offer_redeemed: str = "offer_redeemed"

    person: str = "person"
    offer_id: str = "offer_id"


@dataclass
class ProfileTransformedCols(ProfileCols):
    index: str = "index"
    gender_m: str = "gender_m"
    gender_f: str = "gender_f"
    gender_o: str = "gender_o"
    gender_nan: str = "gender_nan"
    person: str = "person"
    became_member_on_0: str = "became_member_on_0"
    became_member_on_1: str = "became_member_on_1"
    became_member_on_2: str = "became_member_on_2"
    became_member_on_3: str = "became_member_on_3"
    became_member_on_4: str = "became_member_on_4"
    became_member_on_5: str = "became_member_on_5"
    income_0: str = "income_0"
    income_1: str = "income_1"
    income_2: str = "income_2"
    income_3: str = "income_3"
    income_4: str = "income_4"
    income_5: str = "income_5"
    age_0: str = "age_0"
    age_1: str = "age_1"
    age_2: str = "age_2"
    age_3: str = "age_3"
    age_4: str = "age_4"
    age_5: str = "age_5"


@dataclass
class BestOfferCols(CategoryABC):
    index: str = "index"
    person: str = "person"
    gender_f: str = "gender_f"
    gender_m: str = "gender_m"
    gender_o: str = "gender_o"
    gender_nan: str = "gender_nan"
    became_member_on_0: str = "became_member_on_0"
    became_member_on_1: str = "became_member_on_1"
    became_member_on_2: str = "became_member_on_2"
    became_member_on_3: str = "became_member_on_3"
    became_member_on_4: str = "became_member_on_4"
    became_member_on_5: str = "became_member_on_5"
    income_0: str = "income_0"
    income_1: str = "income_1"
    income_2: str = "income_2"
    income_3: str = "income_3"
    income_4: str = "income_4"
    income_5: str = "income_5"
    age_0: str = "age_0"
    age_1: str = "age_1"
    age_2: str = "age_2"
    age_3: str = "age_3"
    age_4: str = "age_4"
    age_5: str = "age_5"
    best_offer: str = "best_offer"


@dataclass
class ViewedAndRedeemedCols(CategoryABC):
    index: str = "index"
    person: str = "person"
    offer_id: str = "offer_id"
    event_id: str = "event_id"
    gender_f: str = "gender_f"
    gender_m: str = "gender_m"
    gender_o: str = "gender_o"
    # gender_nan: str = "gender_nan"
    # became_member_on_0: str = "became_member_on_0"
    became_member_on_1: str = "became_member_on_1"
    became_member_on_2: str = "became_member_on_2"
    became_member_on_3: str = "became_member_on_3"
    became_member_on_4: str = "became_member_on_4"
    became_member_on_5: str = "became_member_on_5"
    # income_0: str = "income_0"
    income_1: str = "income_1"
    income_2: str = "income_2"
    income_3: str = "income_3"
    income_4: str = "income_4"
    income_5: str = "income_5"
    # age_0: str = "age_0"
    age_1: str = "age_1"
    age_2: str = "age_2"
    age_3: str = "age_3"
    age_4: str = "age_4"
    age_5: str = "age_5"

    viewed_and_redeemed: str = "viewed_and_redeemed"
