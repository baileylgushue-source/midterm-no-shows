"""Train, compare and save the no-show model: python -m noshows.train"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from .data import make_appointments
from .evaluation import choose_threshold, compare_costs
from .pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Recorded after the visit, so it is not known at booking time.
LEAKY_COLUMN = "fee_charged"

# zero_division=0: the baseline never flags anyone, so its precision is 0.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
TUNING_SCORING = "roc_auc"
PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0],
    "model__class_weight": [None, "balanced"],
}


def make_folds():
    """Five stratified folds, shared by every comparison and search."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def candidate_models():
    """A baseline and three candidates, one with class_weight balanced."""
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LogisticRegression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED
        ),
    }


def compare_models(X_train, y_train):
    """Mean and spread of every metric across five folds, training data only."""
    rows = []
    for name, model in candidate_models().items():
        scores = cross_validate(
            build_pipeline(model), X_train, y_train, cv=make_folds(), scoring=SCORING
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 3)
            row[f"{metric}_std"] = round(float(values.std()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def tune_model(X_train, y_train):
    """Grid search over C and class_weight (six combinations)."""
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        PARAM_GRID,
        scoring=TUNING_SCORING,
        cv=make_folds(),
    )
    search.fit(X_train, y_train)
    best = search.best_index_
    return search, {
        "scoring": TUNING_SCORING,
        "best_params": search.best_params_,
        "best_mean": round(float(search.cv_results_["mean_test_score"][best]), 3),
        "best_std": round(float(search.cv_results_["std_test_score"][best]), 3),
    }


def leakage_check(frame, X_train, y_train, model):
    """CV ROC-AUC of the same pipeline with the after-the-visit column added."""
    honest = build_pipeline(clone(model))
    leaky = build_pipeline(clone(model))
    leaky.set_params(
        preprocessor__transformers=[
            (name, transformer, columns + [LEAKY_COLUMN])
            if name == "numeric"
            else (name, transformer, columns)
            for name, transformer, columns in leaky.named_steps[
                "preprocessor"
            ].transformers
        ]
    )
    X_leaky = frame.loc[X_train.index, X_train.columns.tolist() + [LEAKY_COLUMN]]

    def mean_auc(pipeline, X):
        scores = cross_validate(
            pipeline, X, y_train, cv=make_folds(), scoring="roc_auc"
        )
        return round(float(scores["test_score"].mean()), 3)

    return {
        "column": LEAKY_COLUMN,
        "honest_roc_auc": mean_auc(honest, X_train),
        "with_leaky_column_roc_auc": mean_auc(leaky, X_leaky),
    }


def test_scores(y_test, probabilities, threshold):
    """Score the test set once at the chosen threshold."""
    flagged = probabilities >= threshold
    return {
        "confusion_matrix": confusion_matrix(y_test, flagged).tolist(),
        "precision": round(float(precision_score(y_test, flagged, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, flagged)), 3),
        "f1": round(float(f1_score(y_test, flagged)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
    }


def main():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Compare a baseline and three candidates.
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    print("Five-fold cross-validation on the training data:")
    print(comparison.to_string(index=False))

    # 2. Tune the chosen model.
    search, tuning = tune_model(X_train, y_train)
    untuned = comparison.set_index("model").loc["LogisticRegression"]
    tuning["untuned_mean"] = float(untuned[f"{TUNING_SCORING}_mean"])
    tuning["untuned_std"] = float(untuned[f"{TUNING_SCORING}_std"])
    tuning["improvement"] = round(tuning["best_mean"] - tuning["untuned_mean"], 3)
    tuning["beats_spread"] = bool(tuning["improvement"] > tuning["untuned_std"])
    print("\nTuning:", tuning)

    # 3. Leakage check with the after-the-visit column.
    leakage = leakage_check(frame, X_train, y_train, search.best_estimator_["model"])
    print("Leakage check:", leakage)

    # 4. Threshold from out-of-fold training probabilities.
    best_pipeline = search.best_estimator_
    oof = cross_val_predict(
        clone(best_pipeline), X_train, y_train, cv=make_folds(), method="predict_proba"
    )[:, 1]
    threshold = round(float(choose_threshold(y_train, oof)), 2)
    print("Chosen threshold:", threshold)

    # 5. Score the test set once.
    final = best_pipeline.fit(X_train, y_train)
    probabilities = final.predict_proba(X_test)[:, 1]
    scores = test_scores(y_test, probabilities, threshold)
    costs = compare_costs(y_test, probabilities, threshold)
    print("Test scores:", scores)
    print("Test costs:", costs)

    # 6. Save the model and every report.
    joblib.dump(final, MODELS_DIR / "no_show.joblib")
    info = {
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": tuning["best_params"],
        "test": scores,
        "costs": costs,
    }
    metrics = {
        "tuning": tuning,
        "leakage": leakage,
        "threshold": threshold,
        "test": scores,
        "costs": costs,
        "test_rows": int(len(y_test)),
        "test_no_show_rate": round(float(np.mean(y_test)), 3),
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Saved model, model_info.json and reports.")


if __name__ == "__main__":
    main()
