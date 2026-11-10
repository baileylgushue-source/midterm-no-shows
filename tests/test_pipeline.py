import pytest
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import (
    FEATURES,
    build_pipeline,
    build_preprocessor,
    split_data,
)


def test_excluded_columns():
    assert "appointment_id" not in FEATURES
    assert "fee_charged" not in FEATURES
    assert "no_show" not in FEATURES


def test_split_size():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)

    assert len(X_train) == 3000
    assert len(X_test) == 1000
    assert len(y_train) == 3000
    assert len(y_test) == 1000


def test_stratified_split():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)

    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(train_rate - test_rate) < 0.01


def test_missing_distance():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape[0] == len(X_train)


@pytest.mark.parametrize("seed", [11, 12])
def test_reproducible_data(seed):
    frame1 = make_appointments(seed=seed)
    frame2 = make_appointments(seed=seed)

    assert frame1.equals(frame2)


def test_build_pipeline():
    model = LogisticRegression()
    pipeline = build_pipeline(model)

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert pipeline.named_steps["model"] is model
