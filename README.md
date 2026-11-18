---
title: Maple Physio No-Show Check
emoji: 📅
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Maple Physio no-show check

## What it does

Predicts the chance that a patient will miss a physiotherapy appointment
and recommends whether the front desk should call to confirm.

A confirmation call costs $8. An unflagged no-show costs $35.
The model chooses a calling threshold using out-of-fold training predictions.

Public Space: [Open Maple Physio no-show check](https://huggingface.co/spaces/heli2062/midterm-no-shows)

## How to run it locally

Use Anaconda or Miniconda. From the project root:

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
```

Recreate the saved pipeline and reports:

```bash
python -m noshows.train
```

Run the page:

```bash
gradio app.py
```

The local environment uses Python 3.11. The Space settings specify
Python 3.12. Package versions are pinned in requirements.txt and
requirements-dev.txt.

## Team

| Member | Assigned responsibilities |
|---|---|
| Bailey | Steps 1–2: Repository and environment setup; unchanged data.py; pipeline.py and evaluation.py with tests and the hand-worked cost example. |
| Ankitha | Step 3: Training, model comparison, tuning, saved outputs and leakage check. |
| Yule | Steps 4 and 7: Appointment and NoShowPredictor classes and their tests, and public Hugging Face Space deployment. |
| Gurleen | Step 5: Gradio app, and input checks. |
| Janmesh Bhavsar | Step 6: Deployment tests, README report and verification of formatting, lint and tests. |
| Yoh | Step 8: Fresh-clone verification, training reproducibility, final checks, demo rehearsal and submission coordination. |

The table records assigned responsibilities. All members are expected
to review changes and participate in the demo.

## Decisions

The dataset contains 4,000 synthetic appointments. We use a stratified
75/25 split: 3,000 training rows and 1,000 test rows, with seed 11.

Model comparison, hyperparameter tuning and threshold selection use
only training data. The held-out test set is evaluated after these choices.

### Features

| Column | Decision and reason |
|---|---|
| lead_days | Use: booking lead time is known before the visit. |
| age | Use: patient age is available at booking. |
| prior_no_shows | Use: previous missed appointments are known. |
| hour | Use: the scheduled appointment hour is known. |
| weekday | Use: the scheduled weekday is known. |
| booking | Use: the booking channel is known. |
| insurance | Use: the insurance category is available at booking. |
| treatment | Use: the booked treatment is known. |
| first_visit | Use: first-visit status is known. |
| distance_km | Use: distance may affect attendance; missing values are imputed. |
| appointment_id | Exclude: it identifies a row rather than a useful booking characteristic. |
| fee_charged | Exclude: it is recorded after the visit and leaks the outcome. |
| no_show | Target only: this is the outcome being predicted. |

### Input checks

| Field | Accepted values |
|---|---|
| lead_days | 0–120 |
| age | 12–110 |
| prior_no_shows | 0–30 |
| hour | 8–17 |
| first_visit | 0 or 1 |
| distance_km | 0.1–150, or empty |
| weekday | mon, tue, wed, thu, fri |
| booking | online, phone, front_desk |
| insurance | private, provincial, self_pay |
| treatment | physio, massage, acupuncture |

The numeric limits follow the project specification and are slightly
wider than the training ranges to accommodate plausible bookings.
They still reject values far outside the data the model learned from.
The exception is hour: 8–17 exactly matches the clinic's operating
hours and the training range.

Appointment validates values when created and cannot be changed afterwards.
Only missing distance is accepted; the pipeline replaces it with the median
learned from training data. The page must warn when this happens.

Numeric features use median imputation and standard scaling. Categorical
features use most-frequent imputation and one-hot encoding.

The encoder uses handle_unknown="ignore", so an unseen category would
be silently encoded without matching any known category. Appointment
therefore rejects unknown values explicitly.

Dropdowns help prevent mistakes on the page, but the Space can also
be called through its API without using the page. Validation belongs
in Appointment so the same rules apply to both entry points.

## Model comparison

We compared a DummyClassifier baseline and three candidates using
five-fold stratified cross-validation on the training set only.
Values below are mean ± standard deviation across folds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy: most frequent | 0.738 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.758 ± 0.017 | 0.592 ± 0.078 | 0.229 ± 0.050 | 0.330 ± 0.064 | 0.727 ± 0.012 |
| Logistic regression: balanced | 0.675 ± 0.009 | 0.422 ± 0.011 | 0.660 ± 0.031 | 0.515 ± 0.015 | 0.727 ± 0.012 |
| Random forest | 0.751 ± 0.013 | 0.600 ± 0.110 | 0.138 ± 0.037 | 0.223 ± 0.057 | 0.714 ± 0.005 |

Accuracy is misleading here: the dummy model achieves 73.8% by predicting
attendance for everyone, while identifying no no-shows.

We selected logistic regression because it achieved the highest mean
ROC-AUC among the candidates and provides a relatively simple model.
The balanced candidate also identified more no-shows at the default
classification threshold. The final calling threshold was selected
separately using the business cost rule.

### Hyperparameter tuning

GridSearchCV evaluated six combinations on training data:

- C: 0.01, 0.1 and 1.0.
- class_weight: None and "balanced".
- Scoring: ROC-AUC.
- Cross-validation: five stratified folds.

The selected settings were C=0.1 and class_weight="balanced".

The best cross-validated ROC-AUC was 0.727 ± 0.012, compared with
0.727 ± 0.012 for untuned logistic regression. At the reported precision,
the improvement was 0.000, so tuning did not beat the fold-to-fold spread.

## Result

The calling threshold was selected from 0.05 to 0.95 in increments of
0.01 using out-of-fold training probabilities. The selected threshold
was 0.48. A booking is flagged when its model score is at least 0.48.

The held-out test set contained 1,000 appointments, including 262 no-shows.
It was evaluated after model and threshold selection.

### Test metrics

| Metric | Value |
|---|---:|
| Precision | 0.417 |
| Recall | 0.695 |
| F1 | 0.521 |
| ROC-AUC | 0.728 |

| Actual outcome | No call | Call |
|---|---:|---:|
| Attended | 484 | 254 |
| No-show | 80 | 182 |

### Business costs

Total cost = $8 × patients called + $35 × unflagged no-shows.

| Strategy | Cost on 1,000 test appointments |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| Use the model | $6,288 |

The model flags 436 appointments, giving a cost of
436 × $8 + 80 × $35 = $6,288.

This saves $2,882 compared with calling nobody and $1,712 compared
with calling everybody. It beats both baselines on this test set.

These are estimates from one synthetic test set of 1,000 appointments,
not guaranteed future savings. We have not calculated a confidence
interval; results could change with a different sample or real clinic data.

### Leakage check

| Training cross-validation experiment | Mean ROC-AUC |
|---|---:|
| Honest features | 0.727 |
| Including fee_charged | 0.923 |

The larger score cannot be trusted because fee_charged is recorded
after the visit and reveals information about the outcome.
It is excluded from the deployed pipeline.

Because the selected classifier uses balanced class weights, its
predict_proba output has not been established as a calibrated real-world
probability. The calling threshold is therefore chosen empirically
from training predictions using the stated cost rule.



## Tests

The test suite covers:

- Reproducible data generation and exclusion of leakage columns.
- Split sizes and stratification.
- Preprocessing with missing distance values.
- The cost rule against a hand-calculated example.
- Threshold selection and comparison of calling strategies.
- Valid bookings, missing distance, input limits and unknown categories.
- Appointment immutability and training-column compatibility.
- Prediction ranges and safer-versus-riskier bookings.
- Decisions exactly at, below and above the saved threshold.
- Loading the saved pipeline once per predictor.
- Agreement between installed packages, requirements and model metadata.
- Gradio Space settings and the existence of deployment files.

Run the required checks from the project root:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

With app.py present, Ruff formatting and lint checks passed,
and all 80 test cases passed in the working clone.
Fresh-clone verification remains a separate final submission check.

Before submission, repeat the checks from a fresh clone and run
python -m noshows.train to verify that the model and reports can
be recreated.

## Limitations

- The data is synthetic. Performance on real clinic bookings is unknown.
- The scenario assumes that every patient who receives a confirmation
  call attends. Actual calls may not prevent every no-show.
- The cost calculation assumes fixed costs of $8 per call and $35 per
  unflagged no-show.
- The model does not account for staff calling capacity or patients
  who cannot be reached.
- Missing distance is replaced with a training-data median, which may
  not represent that individual patient.
- Accepted input limits extend beyond parts of the training range.
  Passing validation does not guarantee reliable predictions.
- Predictions do not explain why a particular patient might miss a visit.
- Age and insurance are model inputs. Fairness across patient groups
  has not been evaluated.
- The model score has not been calibrated or validated on real clinic
  data and should support, rather than replace, staff judgment.
- The page checks one booking at a time and stores no booking data.
  It cannot see rescheduling history or changes across earlier bookings.

  ## Presentation Recording Link

  https://humberital-my.sharepoint.com/:v:/g/personal/n10041639_humber_ca/IQAOGql4vhd2TL6GSaqEf5RtAU_00tMmCcapo40GztYzKYo?e=BHXGW9&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D
