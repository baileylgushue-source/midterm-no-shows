import pytest

from noshows.evaluation import (
    choose_threshold,
    compare_costs,
    total_cost,
)


def test_hand_worked_cost():
    y_true = [1, 0, 1, 0, 1]
    flagged = [True, True, False, False, False]

    # 2 calls at $8 = $16
    # 2 unflagged no-shows at $35 = $70
    # Total = $86
    assert total_cost(y_true, flagged) == 86


def test_calling_nobody():
    y_true = [1, 0, 1, 0]
    flagged = [False, False, False, False]

    assert total_cost(y_true, flagged) == 70


def test_calling_everybody():
    y_true = [1, 0, 1, 0]
    flagged = [True, True, True, True]

    assert total_cost(y_true, flagged) == 32


def test_choose_threshold():
    y_true = [1, 0, 1, 0]
    probabilities = [0.9, 0.1, 0.8, 0.2]

    threshold = choose_threshold(y_true, probabilities)

    assert (
        total_cost(
            y_true,
            [p >= threshold for p in probabilities],
        )
        == 16
    )


def test_compare_costs():
    y_true = [1, 0, 1, 0]
    probabilities = [0.9, 0.1, 0.8, 0.2]

    results = compare_costs(y_true, probabilities, 0.5)

    assert results["calling_nobody"] == 70
    assert results["calling_everybody"] == 32
    assert results["using_model"] == 16


@pytest.mark.parametrize(
    "y_true, flagged, expected_cost",
    [
        ([1, 0], [False, False], 35),
        ([1, 0], [True, False], 8),
        ([1, 1], [True, False], 43),
    ],
)
def test_total_cost_parametrized(y_true, flagged, expected_cost):
    assert total_cost(y_true, flagged) == expected_cost
