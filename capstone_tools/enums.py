from dataclasses import dataclass, field


@dataclass
class PortfolioCols:
    """Column Names that are within the json file for portfolio.json"""

    channels: str = "channels"
    duration: str = "duration"
    reward: str = "reward"
    difficulty: str = "difficulty"
    offer_type: str = "offer_type"
    id: str = "id"

    @property
    def duration_hours(self):
        """derived column duration_hours"""
        return self.get_name(self.duration, "hours")

    @property
    def duration_days(self):
        """derived column duration_days"""
        return self.get_name(self.duration, "days")

    @property
    def web(self):
        """derived column web"""
        return "web"

    @property
    def email(self):
        """derived column email"""
        return "email"

    @property
    def mobile(self):
        """derived column mobile"""
        return "mobile"

    @property
    def social(self):
        """derived column social"""
        return "social"

    @property
    def info(self):
        """derived column info"""
        return "informational"

    @property
    def bogo(self):
        """derived column bogo"""
        return "bogo"

    @property
    def discount(self):
        """derived column discount"""
        return "discount"

    @staticmethod
    def get_name(root: str, suffix: str) -> str:
        """Get a name with root followed by suffix"""
        return f"{root}_{suffix}"


@dataclass
class ProfileCols:
    """Column Names that are within the json file for profile.json"""

    gender: str = "gender"
    age: str = "age"
    id: str = "id"
    became_member_on: str = "became_member_on"
    income: str = "income"


@dataclass
class TranscriptCols:
    """Column Names that are within the json file for profile.json"""

    value: str = "value"
    person: str = "person"
    event: str = "event"
    time: str = "time"

    @property
    def offer_id(self) -> str:
        """derived column offer_id"""
        return "offer_id"

    @property
    def amount(self) -> str:
        """derived column amount"""
        return "amount"

    @property
    def reward(self) -> str:
        """derived column reward"""
        return "reward"


@dataclass
class Event:
    """Enumerations for Events Column in Transcript / Event Log"""

    received: str = "offer received"
    viewed: str = "offer viewed"
    transaction: str = "transaction"
    completed: str = "offer completed"


@dataclass
class Offer:
    """Enumerations for Offer Types in Portfolio"""

    bogo: str = "bogo"
    discount: str = "discount"
    info: str = "informational"


@dataclass
class Channel:
    """Enumerations for Offer Channel Types in Portfolio"""

    web: str = "web"
    email: str = "email"
    mobile: str = "mobile"
    social: str = "social"


@dataclass
class Gender:
    """Enumerations for Gender in Profile"""

    m: str = "M"
    f: str = "F"
    o: str = "O"

    @property
    def na(self) -> str:
        """derived enumeration na"""
        return "Not Assigned"


@dataclass
class TranscriptTransformedCols(PortfolioCols, TranscriptCols):
    """
    Column names for generated Transcript dataframe with implemented feature
    calculations.
    """

    index: str = "index"
    event_id: str = "event_id"
    offer_start: str = "offer_start"
    elapsed_time: str = "elapsed_time"
    offer_viewed: str = "offer_viewed"
    offer_valid: str = "offer_valid"
    offer_duration: str = "offer_duration"
    sales: str = "sales"
    costs: str = "costs"
    profit: str = "profit"
    offer_redeemed: str = "offer_redeemed"


@dataclass
class EventCols:
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
    membership: str = "membership"


@dataclass
class ModelDataCols:
    index: str = "index"
    age: str = "age"
    person: str = "person"
    income: str = "income"
    gender_f: str = "gender_f"
    gender_m: str = "gender_m"
    gender_o: str = "gender_o"
    gender_nan: str = "gender_nan"
    membership: str = "membership"
    best_offer: str = "best_offer"
