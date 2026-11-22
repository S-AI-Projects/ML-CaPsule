"""Offline unit checks for R41-T049 numeric-regression robustness."""

import copy
import json
from pathlib import Path
import unittest

from pipeline import (
    RegressionValidationError,
    load_fixture,
    predict,
    run_pipeline,
    validate_records,
)

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))


class RegressionRobustnessTests(unittest.TestCase):
    def test_literal_synthetic_formula_fixture_and_splits(self):
        self.assertEqual(FIXTURE["data_kind"], "simulated")
        self.assertEqual(FIXTURE["fixture_kind"], "deterministic_formula")
        self.assertEqual(FIXTURE["seed"], 49)
        self.assertEqual([row["record_id"] for row in FIXTURE["records"]], ["r1", "r2", "r3", "r4", "r5", "r6"])
        self.assertEqual([row["x1"] for row in FIXTURE["records"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([row["x2"] for row in FIXTURE["records"]], [1] * 6)
        self.assertEqual([row["target"] for row in FIXTURE["records"]], [1, 3, 5, 7, 9, 11])
        self.assertEqual(SCHEMA["required"], ["record_id", "x1", "x2", "target"])
        self.assertEqual(FIXTURE["split_ids"], {
            "train": ["r1", "r2", "r3", "r4"],
            "validation": ["r5"],
            "test": ["r6"],
        })

    def test_record_order_change_preserves_results_by_stable_id(self):
        records, splits = load_fixture()
        original = run_pipeline(records, splits)
        reordered = list(reversed(copy.deepcopy(records)))
        shuffled = run_pipeline(reordered, splits)
        self.assertEqual(shuffled, original)
        self.assertEqual(shuffled["predictions"]["r5"], {
            "target": 9.0,
            "predicted": 9.0,
            "absolute_error": 0.0,
        })
        self.assertEqual(shuffled["predictions"]["r6"], {
            "target": 11.0,
            "predicted": 11.0,
            "absolute_error": 0.0,
        })

    def test_missing_field_is_rejected(self):
        records, _ = load_fixture()
        bad = copy.deepcopy(records)
        bad[0].pop("x2")
        with self.assertRaisesRegex(RegressionValidationError, "exactly"):
            validate_records(bad)

    def test_unseen_record_id_in_split_is_rejected(self):
        records, splits = load_fixture()
        bad = copy.deepcopy(splits)
        bad["test"] = ["r-unseen"]
        with self.assertRaisesRegex(RegressionValidationError, "unknown record_id"):
            run_pipeline(records, bad)

    def test_feature_shape_mismatch_is_rejected(self):
        records, _ = load_fixture()
        coefficients = (1.0, 2.0)
        bad = copy.deepcopy(records[0])
        bad.pop("x2")
        with self.assertRaisesRegex(RegressionValidationError, "exactly"):
            predict(bad, coefficients)

    def test_validation_and_test_remain_explicitly_separate(self):
        records, splits = load_fixture()
        result = run_pipeline(records, splits)
        self.assertEqual(result["train_record_ids"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(set(result["predictions"]), {"r5", "r6"})


if __name__ == "__main__":
    unittest.main()
