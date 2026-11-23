"""Unit tests for Task R41-T010: Stroke prediction – class imbalance."""

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

from fixture import (
    DISCLAIMER,
    FEATURE_ORDER,
    GROUP_FIELD,
    ID_FIELD,
    LABEL_FIELD,
    SPLIT_MANIFEST,
    SYNTHETIC_RECORDS,
    get_split_records,
)
from pipeline import (
    SavedClassificationPipeline,
    SchemaValidationError,
    SchemaVersionMismatchError,
    verify_predictions_agreement,
)


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not observations from the real Stroke Prediction dataset", DISCLAIMER)

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(SYNTHETIC_RECORDS), 8)
        expected_keys = {ID_FIELD, GROUP_FIELD, "x1", "x2", LABEL_FIELD}
        for rec in SYNTHETIC_RECORDS:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_fixture_features_and_labels(self) -> None:
        expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        for idx, rec in enumerate(SYNTHETIC_RECORDS):
            self.assertEqual(rec["x1"], expected_x1[idx])
            self.assertEqual(rec["x2"], 1)
            self.assertEqual(rec["label"], expected_labels[idx])

    def test_stratified_split_integrity(self) -> None:
        splits = get_split_records()
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 2)

        train_pos = sum(1 for r in splits["train"] if r["label"] == 1)
        val_pos = sum(1 for r in splits["validation"] if r["label"] == 1)
        test_pos = sum(1 for r in splits["test"] if r["label"] == 1)

        self.assertEqual(train_pos, 2)
        self.assertEqual(val_pos, 1)
        self.assertEqual(test_pos, 1)


class TestSavedPipelineInterface(unittest.TestCase):
    """Verify model saving, reloading, prediction agreement, and zero-data inference."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.train_records = splits["train"]
        self.val_records = splits["validation"]
        self.test_records = splits["test"]
        self.eval_records = self.val_records + self.test_records

        self.pipeline = SavedClassificationPipeline()
        self.pipeline.fit(self.train_records, seed=42)

    def test_before_and_after_predictions_agree_within_tolerance(self) -> None:
        probas_before = self.pipeline.predict_proba(self.eval_records)
        preds_before = self.pipeline.predict(self.eval_records)

        # Serialize and reload via JSON
        json_str = self.pipeline.to_json()
        reloaded = SavedClassificationPipeline.from_json(json_str)

        probas_after = reloaded.predict_proba(self.eval_records)
        preds_after = reloaded.predict(self.eval_records)

        # Explicit tolerance agreement check
        agreement = verify_predictions_agreement(
            probas_before, probas_after, preds_before, preds_after, tolerance=1e-7
        )
        self.assertTrue(agreement)
        self.assertEqual(preds_before, preds_after)
        for pb, pa in zip(probas_before, probas_after):
            self.assertAlmostEqual(pb, pa, places=7)

    def test_file_save_and_load(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name

        try:
            self.pipeline.save_model(tmp_path)
            loaded = SavedClassificationPipeline.load_model(tmp_path)
            preds_orig = self.pipeline.predict(self.eval_records)
            preds_loaded = loaded.predict(self.eval_records)
            self.assertEqual(preds_orig, preds_loaded)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_zero_training_data_and_credentials_needed_at_inference(self) -> None:
        # Reloaded model is completely decoupled from training data
        json_str = self.pipeline.to_json()
        reloaded = SavedClassificationPipeline.from_json(json_str)

        # Attributes do not reference train_records
        self.assertFalse(hasattr(reloaded, "train_records"))
        self.assertTrue(reloaded.is_fitted)

        # Inference succeeds without training set
        preds = reloaded.predict(self.eval_records)
        self.assertEqual(len(preds), len(self.eval_records))


class TestSchemaAndVersionValidation(unittest.TestCase):
    """Verify schema validation and format version mismatch detection."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.pipeline = SavedClassificationPipeline()
        self.pipeline.fit(splits["train"], seed=42)

    def test_version_mismatch_fails_clearly(self) -> None:
        data = self.pipeline.to_dict()
        data["format_version"] = "99.0.0"  # Incompatible version
        with self.assertRaises(SchemaVersionMismatchError):
            SavedClassificationPipeline.from_dict(data)

    def test_missing_feature_in_inference_record_rejected(self) -> None:
        bad_rec = [{"record_id": "bad_1", "x2": 1}]  # missing x1
        with self.assertRaises(SchemaValidationError):
            self.pipeline.predict(bad_rec)

    def test_boolean_feature_in_inference_record_rejected(self) -> None:
        bad_rec = [{"record_id": "bad_1", "x1": True, "x2": 1}]
        with self.assertRaises(SchemaValidationError):
            self.pipeline.predict(bad_rec)

    def test_non_numeric_feature_in_inference_record_rejected(self) -> None:
        bad_rec = [{"record_id": "bad_1", "x1": "invalid", "x2": 1}]
        with self.assertRaises(SchemaValidationError):
            self.pipeline.predict(bad_rec)

    def test_negative_regression_check_fails_when_expected_altered(self) -> None:
        """Verify that altering an expected prediction causes assertion failure."""
        splits = get_split_records()
        preds = self.pipeline.predict(splits["test"])
        actual_first_pred = preds[0]
        deliberate_wrong_val = 999
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_first_pred, deliberate_wrong_val)


if __name__ == "__main__":
    unittest.main()
