import json

import joblib
import pytest

from noshows.appointment import Appointment
from noshows.predictor import INFO_PATH, NoShowPredictor, Prediction

SAFE = {
    "lead_days": 1,
    "age": 70,
    "prior_no_shows": 0,
    "hour": 12,
    "first_visit": 0,
    "distance_km": 2.0,
    "weekday": "wed",
    "booking": "front_desk",
    "insurance": "provincial",
    "treatment": "acupuncture",
}

RISKY = {
    "lead_days": 60,
    "age": 20,
    "prior_no_shows": 5,
    "hour": 8,
    "first_visit": 1,
    "distance_km": 40.0,
    "weekday": "mon",
    "booking": "online",
    "insurance": "self_pay",
    "treatment": "physio",
}


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor()


def test_probability_is_between_zero_and_one(predictor):
    assert 0 <= predictor.probability(Appointment(**SAFE)) <= 1
    assert 0 <= predictor.probability(Appointment(**RISKY)) <= 1


def test_a_riskier_booking_scores_higher_than_a_safer_one(predictor):
    risky = predictor.probability(Appointment(**RISKY))
    safe = predictor.probability(Appointment(**SAFE))
    assert risky > safe


def test_a_missing_distance_still_scores(predictor):
    booking = Appointment(**{**SAFE, "distance_km": None})
    assert 0 <= predictor.probability(booking) <= 1


def test_the_threshold_comes_from_model_info(predictor):
    info = json.loads(INFO_PATH.read_text())
    assert predictor.threshold == info["threshold"]


@pytest.mark.parametrize(
    "offset, expected",
    [
        (0.0, True),  # exactly at the threshold: call
        (-0.001, False),  # just below: no call
        (0.001, True),  # just above: call
    ],
)
def test_the_decision_around_the_threshold(predictor, offset, expected):
    assert predictor.should_call(predictor.threshold + offset) is expected


def test_predict_returns_the_probability_and_the_decision_together(predictor):
    result = predictor.predict(Appointment(**RISKY))
    assert isinstance(result, Prediction)
    assert result.probability == predictor.probability(Appointment(**RISKY))
    assert result.call == (result.probability >= predictor.threshold)


def test_the_model_file_is_read_once(monkeypatch):
    calls = []
    real_load = joblib.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    predictor = NoShowPredictor()
    for _ in range(3):
        predictor.predict(Appointment(**SAFE))
    assert len(calls) == 1
