"""Unit tests for Task R41-T009: Kaggle house prices – feature engineering."""

import unittest
from typing import Any, Dict, List

from fixture import (
    DISCLAIMER,
    FEATURE_ORDER,
    ID_FIELD,
    SPLIT_MANIFEST,
    SYNTHETIC_RECORDS,
    TARGET_FIELD,
    get_split_records,
)
from pipeline import (
    SCOPE_LIMITATION_DISCLAIMER,
    IncompatibleInputError,
    RobustRegressionPipeline,
    permute_records,
)


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not real-world Kaggle house prices data", DISCLAIMER)

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(SYNTHETIC_RECORDS), 6)
        expected_keys = {ID_FIELD, "x1", "x2", TARGET_FIELD}
        for rec in SYNTHETIC_RECORDS:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_deterministic_formula_and_features(self) -> None:
        expected_x1 = [0, 1, 2, 3, 4, 5]
        for idx, rec in enumerate(SYNTHETIC_RECORDS):
            self.assertEqual(rec["x1"], expected_x1[idx])
            self.assertEqual(rec["x2"], 1)
            self.assertEqual(rec["target"], 2 * rec["x1"] + 1)

    def test_split_manifest_disjoint(self) -> None:
        self.assertEqual(SPLIT_MANIFEST["train"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(SPLIT_MANIFEST["validation"], ["r5"])
        self.assertEqual(SPLIT_MANIFEST["test"], ["r6"])


class TestRobustnessPerturbationAndFallback(unittest.TestCase):
    """Test permutation invariance with ID alignment and unsupported input handling."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.train_records = splits["train"]
        self.val_records = splits["validation"]
        self.test_records = splits["test"]
        self.pipeline = RobustRegressionPipeline(fallback_prediction=4.0)
        self.pipeline.fit(self.train_records)

    def test_scope_disclaimer_present(self) -> None:
        self.assertIn("does NOT claim broad adversarial robustness", SCOPE_LIMITATION_DISCLAIMER)

    def test_record_order_perturbation_preserves_result_after_id_alignment(self) -> None:
        held_out = self.val_records + self.test_records  # [r5, r6]
        # Original order
        orig_preds = self.pipeline.predict_by_id(held_out)
        self.assertEqual(orig_preds["r5"], 9.0)
        self.assertEqual(orig_preds["r6"], 11.0)

        # Perturbed order: reverse [r6, r5]
        reversed_records = permute_records(held_out, [1, 0])
        self.assertEqual([r["record_id"] for r in reversed_records], ["r6", "r5"])

        perturbed_preds = self.pipeline.predict_by_id(reversed_records)

        # Assert ID-aligned invariance: results are identical after alignment
        self.assertEqual(orig_preds["r5"], perturbed_preds["r5"])
        self.assertEqual(orig_preds["r6"], perturbed_preds["r6"])
        self.assertEqual(orig_preds, perturbed_preds)

    def test_unsupported_input_rejected_without_fallback(self) -> None:
        # 1. Missing required field 'x1'
        bad_rec_missing = {"record_id": "r_bad", "x2": 1}
        with self.assertRaises(IncompatibleInputError):
            self.pipeline.predict_single(bad_rec_missing, allow_fallback=False)

        # 2. Boolean feature type rejected
        bad_rec_bool = {"record_id": "r_bad", "x1": True, "x2": 1}
        with self.assertRaises(IncompatibleInputError):
            self.pipeline.predict_single(bad_rec_bool, allow_fallback=False)

        # 3. Non-numeric category/string rejected
        bad_rec_str = {"record_id": "r_bad", "x1": "unseen_token", "x2": 1}
        with self.assertRaises(IncompatibleInputError):
            self.pipeline.predict_single(bad_rec_str, allow_fallback=False)

    def test_unsupported_input_mapped_to_explicit_fallback(self) -> None:
        bad_rec = {"record_id": "r_bad", "x2": 1}  # Missing x1
        fallback_pred = self.pipeline.predict_single(bad_rec, allow_fallback=True)
        # Explicit fallback value is 4.0
        self.assertEqual(fallback_pred, 4.0)

    def test_negative_regression_check_fails_when_expected_altered(self) -> None:
        """Verify that altering an expected prediction causes assertion failure."""
        preds = self.pipeline.predict_by_id(self.test_records)
        actual_r6_pred = preds["r6"]
        deliberate_wrong_val = 999.0
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_r6_pred, deliberate_wrong_val)


if __name__ == "__main__":
    unittest.main()
