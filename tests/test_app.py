import gradio as gr
import pytest

import app
from app import check_booking

GOOD = {
    "lead_days": 10,
    "age": 40,
    "prior_no_shows": 1,
    "hour": 9,
    "first_visit": False,
    "distance_km": 5.5,
    "weekday": "mon",
    "booking": "online",
    "insurance": "private",
    "treatment": "physio",
}


def run(**changes):
    """Call the page function with a valid booking, some fields changed."""
    return check_booking(**{**GOOD, **changes})


def test_a_valid_booking_gives_a_chance_and_an_action():
    chance, action, rule = run()
    assert chance.endswith("%")
    assert action in {"Call the patient the day before to confirm.", "No call needed."}
    assert rule.startswith("Call when the chance is")


def test_the_first_visit_checkbox_is_turned_into_0_or_1(monkeypatch):
    seen = []
    real_predict = app.predictor.predict

    def spy(appointment):
        seen.append(appointment.first_visit)
        return real_predict(appointment)

    monkeypatch.setattr(app.predictor, "predict", spy)
    run(first_visit=True)
    run(first_visit=False)
    assert seen == [1, 0]
    assert all(type(value) is int for value in seen)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("lead_days", 500),
        ("age", None),
        ("hour", 3),
        ("distance_km", 400),
        ("weekday", "sat"),
        ("treatment", None),
    ],
)
def test_bad_input_becomes_a_page_error_naming_the_field(field, bad):
    with pytest.raises(gr.Error, match=field):
        run(**{field: bad})


def test_an_empty_distance_still_scores_and_warns():
    with pytest.warns(UserWarning, match="Distance was left empty"):
        chance, _, _ = run(distance_km=None)
    assert chance.endswith("%")