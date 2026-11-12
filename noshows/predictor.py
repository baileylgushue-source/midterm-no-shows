"""Turn a booking into a chance of a no-show and an action."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from .appointment import Appointment

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "no_show.joblib"
INFO_PATH = MODELS_DIR / "model_info.json"


@dataclass(frozen=True)
class Prediction:
    """The chance of a no-show and whether to call, returned together."""

    probability: float
    call: bool

    @property
    def action(self) -> str:
        """The plain instruction for the front desk."""
        if self.call:
            return "Call the patient the day before to confirm."
        return "No call needed."


class NoShowPredictor:
    """Loads the saved pipeline once and applies the threshold from model_info.json."""

    def __init__(self) -> None:
        self._pipeline = joblib.load(MODEL_PATH)  # once, when the predictor is made
        self._info = json.loads(INFO_PATH.read_text())

    @property
    def threshold(self) -> float:
        """The threshold chosen by train.py."""
        return float(self._info["threshold"])

    def probability(self, appointment: Appointment) -> float:
        """Return the chance, from 0 to 1, that this booking is a no-show."""
        return float(self._pipeline.predict_proba(appointment.to_frame())[0, 1])

    def should_call(self, probability: float) -> bool:
        """Flag a booking when its chance reaches the threshold."""
        return probability >= self.threshold

    def predict(self, appointment: Appointment) -> Prediction:
        """Return the probability and the call decision together."""
        chance = self.probability(appointment)
        return Prediction(probability=chance, call=self.should_call(chance))

    def __repr__(self) -> str:
        return (
            f"NoShowPredictor(threshold={self.threshold}, "
            f"scikit_learn={self._info['scikit_learn']!r})"
        )
