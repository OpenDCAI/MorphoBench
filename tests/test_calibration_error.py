import math

import numpy as np
import pytest

from eval_agent.evaluators.correctness import CorrectnessEvaluator
from scripts.evaluate_judge import calib_err, dump_metrics


@pytest.fixture(params=[calib_err, CorrectnessEvaluator._calib_err], ids=["legacy", "advanced"])
def calibration(request):
    return request.param


def assert_calibration(calibration, confidence, correct, *, l1, l2, maximum, beta=100):
    assert calibration(confidence, correct, p="1", beta=beta) == pytest.approx(l1)
    assert calibration(confidence, correct, p="2", beta=beta) == pytest.approx(l2)
    for norm in ("infty", "infinity", "max"):
        assert calibration(confidence, correct, p=norm, beta=beta) == pytest.approx(maximum)


@pytest.mark.parametrize("size", [1, 20, 99, 100, 101, 200, 250, 300])
def test_confidently_wrong_at_bin_boundaries(calibration, size):
    confidence = np.ones(size)
    correct = np.zeros(size, dtype=bool)

    assert_calibration(calibration, confidence, correct, l1=1, l2=1, maximum=1)


def test_final_bin_contributes_its_share_of_scored_examples(calibration):
    # The first 100 predictions are calibrated. The remaining 150 have a gap of 1.
    confidence = np.repeat([0.0, 1.0], [100, 150])
    correct = np.zeros(250, dtype=bool)

    assert_calibration(calibration, confidence, correct, l1=0.6, l2=math.sqrt(0.6), maximum=1)


@pytest.mark.parametrize("shuffled", [False, True])
def test_remainder_stays_in_final_bin(calibration, shuffled):
    confidence = np.array([0.125] * 3 + [0.5] * 3 + [0.75, 0.75, 0.875, 0.875])
    correct = np.array([False] * 3 + [True, False, False] + [False, False, True, True])
    if shuffled:
        order = [8, 2, 0, 6, 4, 7, 1, 9, 5, 3]
        confidence, correct = confidence[order], correct[order]
    original_confidence, original_correct = confidence.copy(), correct.copy()

    # With beta=3, the three bins have 3, 3, and 4 examples.
    # Their confidence/accuracy gaps are 1/8, 1/6, and 5/16, respectively.
    assert_calibration(
        calibration,
        confidence,
        correct,
        l1=(3 / 8 + 3 / 6 + 4 * 5 / 16) / 10,
        l2=math.sqrt((3 / 8**2 + 3 / 6**2 + 4 * (5 / 16)**2) / 10),
        maximum=5 / 16,
        beta=3,
    )
    np.testing.assert_array_equal(confidence, original_confidence)
    np.testing.assert_array_equal(correct, original_correct)


def test_perfect_calibration_with_tied_confidences(calibration):
    confidence = np.repeat([0.0, 0.5, 1.0], 4)
    correct = np.array([False] * 4 + [False, True] * 2 + [True] * 4)

    assert_calibration(calibration, confidence, correct, l1=0, l2=0, maximum=0, beta=4)


def test_empty_calibration_is_zero(calibration):
    assert_calibration(
        calibration, np.array([]), np.array([], dtype=bool), l1=0, l2=0, maximum=0
    )


def test_reports_agree_on_calibration_and_keep_other_metrics(tmp_path):
    predictions = {
        "correct": {
            "judge_response": {"correct": "yes", "confidence": 100},
            "response": "abcdefgh",
        },
        "incorrect": {
            "judge_response": {"correct": "no", "confidence": 20},
            "response": "abc",
        },
        "unscored": {"response": "unscored response"},
    }
    legacy_path = tmp_path / "legacy.txt"
    advanced_path = tmp_path / "advanced.txt"

    dump_metrics(predictions, n=4, stats_txt_path=legacy_path)
    stats = CorrectnessEvaluator.compute_metrics(predictions, total_n=4)
    CorrectnessEvaluator.dump_stats(stats, advanced_path)

    # In the single bin, confidence is 0.6 and accuracy is 0.5. Reports scale by 100.
    assert stats == {
        "accuracy": 50.0,
        "ci_95": 49.0,
        "calibration_error": 10.0,
        "avg_response_length": 5.5,
        "n_total": 4,
        "n_scored": 2,
    }
    assert legacy_path.read_text().splitlines() == [
        "Accuracy: 50.0% ± 49.0% | n=4",
        "Calibration Error: 10.0",
        "Average model response length: 5.5",
    ]
    assert advanced_path.read_text().splitlines() == [
        "Accuracy: 50.0% ± 49.0% | n=4 (scored=2)",
        "Calibration Error: 10.0",
        "Average response length: 5.5",
    ]


def test_empty_scored_cohort_preserves_report_behavior(tmp_path):
    predictions = {"unscored": {"response": "unscored response"}}
    legacy_path = tmp_path / "legacy.txt"

    with pytest.warns(RuntimeWarning):
        dump_metrics(predictions, n=1, stats_txt_path=legacy_path)

    assert CorrectnessEvaluator.compute_metrics(predictions, total_n=1) == {}
    assert legacy_path.read_text().splitlines() == [
        "Accuracy: nan% ± nan% | n=1",
        "Calibration Error: 0.0",
        "Average model response length: 0.0",
    ]
