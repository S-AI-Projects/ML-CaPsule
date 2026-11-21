"""Offline tests for error analysis on the R41-T048 formula fixture."""

import copy
import json
from pathlib import Path
import unittest

from pipeline import FixtureValidationError, analyze_errors, load_fixture


HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
EXPECTED = {
    "data_kind": "simulated_pipeline_verification",
    "fixture_kind": "deterministic_formula",
    "records": [
        {
            "record_id": f"r{index}",
            "split": split,
            "expected_value": float(2 * (index - 1) + 1),
            "predicted_value": float(2 * (index - 1) + 1) + (1.0 if index == 6 else 0.0),
            "absolute_error": 1.0 if index == 6 else 0.0,
            "reason_category": "overprediction" if index == 6 else "no_error",
        }
        for index, split in enumerate(
            ["train", "train", "train", "train", "validation", "test"], start=1
        )
    ],
    "aggregates": {
        "train": {"record_count": 4, "error_count": 0, "no_error_count": 4},
        "validation": {"record_count": 1, "error_count": 0, "no_error_count": 1},
        "test": {"record_count": 1, "error_count": 1, "no_error_count": 0},
    },
    "total_record_count": 6,
    "total_error_count": 1,
}


class ErrorAnalysisTests(unittest.TestCase):
    def test_literal_formula_fixture_and_explicit_splits(self):
        self.assertEqual(FIXTURE["data_kind"], "simulated")
        self.assertEqual(FIXTURE["fixture_kind"], "deterministic_formula")
        self.assertEqual([row["x1"] for row in FIXTURE["records"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([row["x2"] for row in FIXTURE["records"]], [1] * 6)
        self.assertEqual([row["target"] for row in FIXTURE["records"]], [1, 3, 5, 7, 9, 11])
        self.assertEqual(
            SCHEMA["required"], ["record_id", "x1", "x2", "target"]
        )
        self.assertEqual(
            FIXTURE["split_ids"],
            {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
        )

    def test_injected_wrong_prediction_is_reported_but_correct_predictions_are_not(self):
        records, splits, predictions = load_fixture()
        report = analyze_errors(records, predictions, splits)
        self.assertEqual(report, EXPECTED)
        by_id = {entry["record_id"]: entry for entry in report["records"]}
        self.assertEqual(by_id["r6"]["reason_category"], "overprediction")
        self.assertEqual(by_id["r6"]["expected_value"], 11.0)
        self.assertEqual(by_id["r6"]["predicted_value"], 12.0)
        self.assertTrue(
            all(by_id[f"r{index}"]["reason_category"] == "no_error" for index in range(1, 6))
        )

    def test_aggregate_counts_match_detailed_report(self):
        records, splits, predictions = load_fixture()
        report = analyze_errors(records, predictions, splits)
        detailed_error_count = sum(
            entry["reason_category"] != "no_error" for entry in report["records"]
        )
        self.assertEqual(report["total_error_count"], detailed_error_count)
        for split_name, aggregate in report["aggregates"].items():
            detailed = [row for row in report["records"] if row["split"] == split_name]
            self.assertEqual(aggregate["record_count"], len(detailed))
            self.assertEqual(
                aggregate["error_count"],
                sum(row["reason_category"] != "no_error" for row in detailed),
            )

    def test_incomplete_predictions_are_rejected(self):
        records, splits, predictions = load_fixture()
        predictions.pop("r6")
        with self.assertRaisesRegex(FixtureValidationError, "exactly"):
            analyze_errors(records, predictions, splits)

    def test_overlapping_split_ids_are_rejected(self):
        records, splits, predictions = load_fixture()
        bad_splits = copy.deepcopy(splits)
        bad_splits["test"] = ["r5"]
        with self.assertRaisesRegex(FixtureValidationError, "partition"):
            analyze_errors(records, predictions, bad_splits)

    def test_non_finite_prediction_is_rejected(self):
        records, splits, predictions = load_fixture()
        predictions["r6"] = float("nan")
        with self.assertRaisesRegex(FixtureValidationError, "finite"):
            analyze_errors(records, predictions, splits)


if __name__ == "__main__":
    unittest.main()
