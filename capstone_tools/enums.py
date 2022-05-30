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
    base_enums: TranscriptTransformedCols = field(
        repr=False, default_factory=TranscriptTransformedCols
    )
    index: str = field(init=False)
    event_id: str = field(init=False)
    profit: str = field(init=False)
    time: str = field(init=False)
    offer_start: str = field(init=False)
    difficulty: str = field(init=False)

    web: str = field(init=False)
    email: str = field(init=False)
    mobile: str = field(init=False)
    social: str = field(init=False)
    bogo: str = field(init=False)
    discount: str = field(init=False)
    info: str = field(init=False)

    offer_viewed: str = field(init=False)
    offer_redeemed: str = field(init=False)

    person: str = field(init=False)

    def __post_init__(self):
        self.index = self.base_enums.index
        self.event_id = self.base_enums.event_id
        self.profit = self.base_enums.profit
        self.time = self.base_enums.time
        self.offer_start = self.base_enums.offer_start
        self.difficulty = self.base_enums.difficulty

        self.web = self.base_enums.web
        self.email = self.base_enums.email
        self.mobile = self.base_enums.mobile
        self.social = self.base_enums.social
        self.bogo = self.base_enums.bogo
        self.discount = self.base_enums.discount
        self.info = self.base_enums.info

        self.offer_viewed = self.base_enums.offer_viewed
        self.offer_redeemed = self.base_enums.offer_redeemed

        self.person = self.base_enums.person


@dataclass
class ProfileTransformedCols(ProfileCols):
    index: str = "index"
    gender_m: str = "m"
    gender_f: str = "f"
    gender_o: str = "o"
    person: str = "person"
    membership: str = "membership"
