from dataclasses import dataclass


@dataclass
class Event:
    """Enumerations for Events Column in Transcript / Event Log"""

    received = "offer received"
    viewed = "offer viewed"
    transaction = "transaction"
    completed = "offer completed"


@dataclass
class Offer:
    """Enumerations for Offer Types in Portfolio"""

    bogo = "bogo"
    discount = "discount"
    info = "informational"


@dataclass
class Channel:
    """Enumerations for Offer Channel Types in Portfolio"""

    web = "web"
    email = "email"
    mobile = "mobile"
    social = "social"


@dataclass
class Gender:
    """Enumerations for Gender in Profile"""

    m = "M"
    f = "F"
    o = "O"
