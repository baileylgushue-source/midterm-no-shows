"""One booking, as the front desk enters it."""

from dataclasses import asdict, dataclass
from numbers import Real

import pandas as pd

from .pipeline import FEATURES

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]
BOOKINGS = ["online", "phone", "front_desk"]
INSURANCES = ["private", "provincial", "self_pay"]
TREATMENTS = ["physio", "massage", "acupuncture"]

# The one-hot encoder ignores categories it has not seen, so a misspelled value
# would be scored quietly. Anything outside these lists is refused instead.
CHOICES = {
    "weekday": WEEKDAYS,
    "booking": BOOKINGS,
    "insurance": INSURANCES,
    "treatment": TREATMENTS,
}

# A little wider than the training data, which the model has no basis to go beyond.
NUMBER_LIMITS = {
    "lead_days": (0, 120),
    "age": (12, 110),
    "prior_no_shows": (0, 30),
    "hour": (8, 17),
}
DISTANCE_LIMITS = (0.1, 150)


def _is_number(value) -> bool:
    """True for int and float, but not for True/False, text or None."""
    return isinstance(value, Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class Appointment:
    """One booking. It checks itself when created and cannot change afterwards."""

    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    first_visit: int
    distance_km: float | None  # None means not recorded; the model uses the median
    weekday: str
    booking: str
    insurance: str
    treatment: str

    def __post_init__(self) -> None:
        """Raise ValueError, naming the field, for anything the model cannot score."""
        for name, (low, high) in NUMBER_LIMITS.items():
            value = getattr(self, name)
            if not _is_number(value) or not low <= value <= high:
                raise ValueError(f"{name} must be a number from {low} to {high}.")
        if not _is_number(self.first_visit) or self.first_visit not in (0, 1):
            raise ValueError("first_visit must be 0 or 1.")
        low, high = DISTANCE_LIMITS
        if self.distance_km is not None and (
            not _is_number(self.distance_km) or not low <= self.distance_km <= high
        ):
            raise ValueError(
                f"distance_km must be a number from {low} to {high}, or left empty."
            )
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of: {', '.join(allowed)}.")

    @property
    def distance_missing(self) -> bool:
        """True when distance_km was left empty, the one gap we accept."""
        return self.distance_km is None

    def to_frame(self) -> pd.DataFrame:
        """Return a one-row table with the training column names."""
        return pd.DataFrame([asdict(self)], columns=FEATURES)
