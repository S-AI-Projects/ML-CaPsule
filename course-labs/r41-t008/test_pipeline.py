"""Unit tests for Task R41-T008: Air passengers – moving-average forecast."""

import unittest
from typing import Any, Dict, List

from fixture import (
    DISCLAIMER,
    ID_FIELD,
    SPLIT_MANIFEST,
    SYNTHETIC_RECORDS,
    TIMESTAMP_FIELD,
    VALUE_FIELD,
    get_split_records,
)
from pipeline import (
    MovingAverageForecaster,
    SchemaValidationError,
    create_predictions_with_injected_error,
    generate_error_analysis_report,
    validate_time_series_records,
)


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties, ordering, and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not real-world Air Passengers airline passenger data", DISCLAIMER)

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(SYNTHETIC_RECORDS), 10)
        expected_keys = {ID_FIELD, TIMESTAMP_FIELD, VALUE_FIELD}
        for rec in SYNTHETIC_RECORDS:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_chronological_values_and_ids(self) -> None:
        expected_values = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
        expected_ids = [f"d{i}" for i in range(1, 11)]

        for idx, rec in enumerate(SYNTHETIC_RECORDS):
            self.assertEqual(rec["record_id"], expected_ids[idx])
            self.assertEqual(rec["value"], expected_values[idx])

    def test_split_manifest_no_shuffling(self) -> None:
        self.assertEqual(SPLIT_MANIFEST["train"], ["d1", "d2", "d3", "d4", "d5", "d6"])
        self.assertEqual(SPLIT_MANIFEST["validation"], ["d7", "d8"])
        self.assertEqual(SPLIT_MANIFEST["test"], ["d9", "d10"])

        # Check disjointness and ordering
        all_ids = SPLIT_MANIFEST["train"] + SPLIT_MANIFEST["validation"] + SPLIT_MANIFEST["test"]
        self.assertEqual(all_ids, [f"d{i}" for i in range(1, 11)])


class TestErrorAnalysisReport(unittest.TestCase):
    """Verify error analysis report behavior with injected wrong prediction."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.held_out_records = splits["validation"] + splits["test"]  # d7, d8, d9, d10

    def test_injected_wrong_prediction_appears_and_correct_do_not(self) -> None:
        # Deliberately inject a wrong prediction on record 'd10' (expected: 19, injected: 99.0)
        preds = create_predictions_with_injected_error(
            self.held_out_records, injected_wrong_id="d10", injected_wrong_value=99.0
        )

        report = generate_error_analysis_report(preds, self.held_out_records, tolerance=2.0)

        # 1. Total evaluated is 4 (d7, d8, d9, d10)
        self.assertEqual(report["total_evaluated"], 4)

        # 2. Injected wrong prediction appears with correct ID
        self.assertEqual(len(report["error_rows"]), 1)
        err_entry = report["error_rows"][0]
        self.assertEqual(err_entry["record_id"], "d10")
        self.assertEqual(err_entry["expected"], 19.0)
        self.assertEqual(err_entry["predicted"], 99.0)
        self.assertEqual(err_entry["residual"], 80.0)
        self.assertEqual(err_entry["reason_category"], "INJECTED_ANOMALY")

        # 3. Correct predictions (d7, d8, d9) do NOT appear in error_rows
        error_ids = {row["record_id"] for row in report["error_rows"]}
        self.assertNotIn("d7", error_ids)
        self.assertNotIn("d8", error_ids)
        self.assertNotIn("d9", error_ids)

        # 4. Aggregate error counts match detailed rows
        self.assertEqual(report["error_count"], len(report["error_rows"]))
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["correct_count"], 3)
        self.assertEqual(report["correct_count"] + report["error_count"], report["total_evaluated"])

    def test_moving_average_forecaster_execution(self) -> None:
        splits = get_split_records()
        train_records = splits["train"]  # [10, 12, 11, 13, 15, 14]
        forecaster = MovingAverageForecaster(window_size=3)
        forecaster.fit(train_records)

        # Last 3 train values are 13, 15, 14 -> mean = 42 / 3 = 14.0
        next_val = forecaster.forecast_next()
        self.assertAlmostEqual(next_val, 14.0)


class TestValidationAndNegativeRegression(unittest.TestCase):
    """Verify input validation and test suite failure on altered expectations."""

    def test_boolean_values_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_time_series_records([{"record_id": "d1", "timestamp": "2026-01-01", "value": True}])

    def test_duplicate_record_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_time_series_records([
                {"record_id": "d1", "timestamp": "2026-01-01", "value": 10},
                {"record_id": "d1", "timestamp": "2026-01-02", "value": 12},
            ])

    def test_negative_regression_check_fails_when_expected_altered(self) -> None:
        """Verify that altering an expected metric causes assertion failure."""
        splits = get_split_records()
        held_out = splits["validation"] + splits["test"]
        preds = create_predictions_with_injected_error(held_out, "d10", 99.0)
        report = generate_error_analysis_report(preds, held_out, tolerance=2.0)

        actual_error_count = report["error_count"]
        deliberate_wrong_count = 999
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_error_count, deliberate_wrong_count)


if __name__ == "__main__":
    unittest.main()
