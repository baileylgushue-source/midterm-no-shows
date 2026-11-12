from dataclasses import FrozenInstanceError

import pytest

from noshows.appointment import CHOICES, NUMBER_LIMITS, Appointment
from noshows.data import make_appointments
from noshows.pipeline import FEATURES, NUMERIC_FEATURES

GOOD = {
    "lead_days": 10,
    "age": 40,
    "prior_no_shows": 1,
    "hour": 9,
    "first_visit": 0,
    "distance_km": 5.5,
    "weekday": "mon",
    "booking": "online",
    "insurance": "private",
    "treatment": "physio",
}


def make(**changes):
    """A valid booking, with some fields changed."""
    return Appointment(**{**GOOD, **changes})


def test_a_valid_booking_is_accepted():
    appointment = make()
    assert appointment.age == 40
    assert not appointment.distance_missing


def test_a_missing_distance_is_allowed():
    appointment = make(distance_km=None)
    assert appointment.distance_missing
    assert appointment.to_frame()["distance_km"].isna().all()


@pytest.mark.parametrize(
    "field, value",
    [
        ("lead_days", 0),
        ("lead_days", 120),
        ("age", 12),
        ("age", 110),
        ("prior_no_shows", 30),
        ("hour", 8),
        ("hour", 17),
        ("first_visit", 1),
        ("distance_km", 0.1),
        ("distance_km", 150),
    ],
)
def test_values_at_the_limits_are_accepted(field, value):
    assert getattr(make(**{field: value}), field) == value


@pytest.mark.parametrize(
    "field, bad",
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("lead_days", None),
        ("age", 11),
        ("age", 111),
        ("age", "40"),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("hour", float("nan")),
        ("first_visit", 2),
        ("first_visit", 0.5),
        ("first_visit", True),
        ("distance_km", 0),
        ("distance_km", 151),
        ("distance_km", "far"),
    ],
)
def test_an_impossible_number_is_rejected_with_the_field_named(field, bad):
    with pytest.raises(ValueError, match=field):
        make(**{field: bad})


@pytest.mark.parametrize(
    "field, bad",
    [
        ("weekday", "sat"),
        ("weekday", "Mon"),
        ("booking", "email"),
        ("insurance", "none"),
        ("treatment", "yoga"),
    ],
)
def test_an_unseen_category_is_rejected_with_the_field_named(field, bad):
    with pytest.raises(ValueError, match=f"{field} must be one of"):
        make(**{field: bad})


def test_a_booking_cannot_change_after_creation():
    appointment = make()
    with pytest.raises(FrozenInstanceError):
        appointment.age = 99


def test_to_frame_is_one_row_with_the_training_columns():
    frame = make().to_frame()
    assert len(frame) == 1
    assert list(frame.columns) == FEATURES


def test_choices_are_exactly_the_categories_in_the_training_data():
    frame = make_appointments()
    for field, allowed in CHOICES.items():
        assert set(frame[field]) == set(allowed)


def test_every_numeric_feature_is_checked():
    checked = set(NUMBER_LIMITS) | {"first_visit", "distance_km"}
    assert checked == set(NUMERIC_FEATURES)
