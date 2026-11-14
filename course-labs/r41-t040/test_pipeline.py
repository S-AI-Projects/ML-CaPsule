"""Unit tests for R41-T040: Wine quality saved pipeline interface."""

import json
from pathlib import Path
import tempfile
import unittest

from pipeline import SafeSavedPipeline, load_fixture


class TestWineQualitySavedPipeline(unittest.TestCase):
    """Verifies safe JSON persistence, reload parity, and schema validation for Task R41-T040."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.train_records = [r for r in self.records if r["split"] == "train"]
        self.val_records = [r for r in self.records if r["split"] == "validation"]
        self.test_records = [r for r in self.records if r["split"] == "test"]

    def test_deterministic_formula_fixture_consistency(self) -> None:
        """Verify fixture conforms to target = 2 * x1 + 1 with stable IDs."""
        self.assertEqual(len(self.records), 6)
        for r in self.records:
            expected = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected, places=6)

    def test_before_and_after_reload_predictions_agree_within_tolerance(self) -> None:
        """Verify model before saving and after reloading yields identical predictions."""
        pipeline_before = SafeSavedPipeline()
        pipeline_before.fit(self.train_records)

        preds_before_val = pipeline_before.predict(self.val_records)
        preds_before_test = pipeline_before.predict(self.test_records)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            pipeline_before.save_json(tmp_path)
            pipeline_after = SafeSavedPipeline.load_json(tmp_path)

            preds_after_val = pipeline_after.predict(self.val_records)
            preds_after_test = pipeline_after.predict(self.test_records)

            self.assertEqual(len(preds_before_val), len(preds_after_val))
            for b, a in zip(preds_before_val, preds_after_val):
                self.assertAlmostEqual(b, a, places=10)

            for b, a in zip(preds_before_test, preds_after_test):
                self.assertAlmostEqual(b, a, places=10)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_schema_mismatch_fails_clearly(self) -> None:
        """Verify input missing expected feature or corrupted dtype raises ValueError."""
        pipeline = SafeSavedPipeline()
        pipeline.fit(self.train_records)

        corrupt_missing = [{"record_id": "bad", "x1": 1.0}]  # missing x2
        with self.assertRaises(ValueError) as ctx:
            pipeline.predict(corrupt_missing)
        self.assertIn("missing required feature 'x2'", str(ctx.exception))

        corrupt_type = [{"record_id": "bad", "x1": "invalid", "x2": 1.0}]
        with self.assertRaises(ValueError) as ctx:
            pipeline.predict(corrupt_type)
        self.assertIn("must be a finite float", str(ctx.exception))

    def test_version_mismatch_fails_clearly(self) -> None:
        """Verify loading an incompatible schema version is explicitly rejected."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            bad_state = {
                "schema_version": "99.0.0",
                "features": ["x1", "x2"],
                "weights": [2.0, 0.0],
                "bias": 1.0,
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(bad_state, f)

            with self.assertRaises(ValueError) as ctx:
                SafeSavedPipeline.load_json(tmp_path)
            self.assertIn("Schema version mismatch", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_zero_training_data_or_credentials_needed_at_inference(self) -> None:
        """Verify reloaded pipeline requires neither train data nor credentials to predict."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            p = SafeSavedPipeline()
            p.fit(self.train_records)
            p.save_json(tmp_path)

            # Fresh load from JSON
            loaded = SafeSavedPipeline.load_json(tmp_path)
            # Evaluate directly on test rows without train partition
            out = loaded.predict([{"record_id": "new_row", "x1": 5.0, "x2": 1.0}])
            self.assertAlmostEqual(out[0], 11.0, places=6)
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
