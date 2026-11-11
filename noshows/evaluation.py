import numpy as np

CALL_COST = 8
MISSED_NO_SHOW_COST = 35


def total_cost(y_true, flagged):
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)

    calls = np.sum(flagged)
    missed_without_call = np.sum((y_true == 1) & (~flagged))

    return int(CALL_COST * calls + MISSED_NO_SHOW_COST * missed_without_call)


def choose_threshold(y_true, probabilities):
    best_threshold = 0.0
    best_cost = float("inf")

    for threshold in np.linspace(0.05, 0.95, 91):
        flagged = np.asarray(probabilities) >= threshold
        cost = total_cost(y_true, flagged)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


def compare_costs(y_true, probabilities, threshold):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    nobody = np.zeros(len(y_true), dtype=bool)
    everybody = np.ones(len(y_true), dtype=bool)
    model_flags = probabilities >= threshold

    return {
        "calling_nobody": total_cost(y_true, nobody),
        "calling_everybody": total_cost(y_true, everybody),
        "using_model": total_cost(y_true, model_flags),
    }
