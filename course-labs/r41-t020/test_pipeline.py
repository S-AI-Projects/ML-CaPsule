"""Unit tests for saved-pipeline interface and schema validation (Task R41-T020)."""

import unittest
from pathlib import Path
from pipeline import SafeNaiveBayesModel, load_fixture, run_pipeline


class TestSavedPipelineInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.records = self.payload["records"]
        self.split = self.payload["metadata"]["stratified_split"]
        self.by_id = {r["record_id"]: r for r in self.records}

        self.train_records = [self.by_id[rid] for rid in self.split["train"]]
        self.test_records = [self.by_id[rid] for rid in self.split["test"]]

        self.model = SafeNaiveBayesModel()
        self.model.fit(self.train_records)

    def test_save_and_reload_prediction_parity(self) -> None:
        # Before saving
        preds_before = self.model.predict(self.test_records)

        # Export and reload state dictionary without training data or credentials
        state = self.model.export_state_dict()
        reloaded = SafeNaiveBayesModel()
        reloaded.load_state_dict(state)

        # After reloading: must agree identically
        preds_after = reloaded.predict(self.test_records)
        self.assertEqual(preds_before, preds_after)
        self.assertIn("r4", preds_after)
        self.assertIn("r8", preds_after)

    def test_schema_version_and_ordering_mismatch_fails(self) -> None:
        state = self.model.export_state_dict()

        # 1. Reject incompatible schema version
        bad_version_state = dict(state)
        bad_version_state["schema_version"] = "2.0.0"
        mismatched_model = SafeNaiveBayesModel()
        with self.assertRaisesRegex(ValueError, "Schema version mismatch"):
            mismatched_model.load_state_dict(bad_version_state)

        # 2. Reject feature ordering mismatch
        bad_order_state = dict(state)
        bad_order_state["feature_order"] = ["x2", "x1"]
        with self.assertRaisesRegex(ValueError, "Feature order mismatch"):
            mismatched_model.load_state_dict(bad_order_state)

    def test_inference_input_schema_validation(self) -> None:
        # 1. Missing required feature
        bad_record_missing = {"record_id": "err1", "x1": 2.0}
        with self.assertRaisesRegex(ValueError, "missing feature 'x2'"):
            self.model.predict_one(bad_record_missing)

        # 2. Non-numeric feature type
        bad_record_type = {"record_id": "err2", "x1": "invalid", "x2": 1.0}
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            self.model.predict_one(bad_record_type)

        # 3. Missing record_id
        bad_record_no_id = {"x1": 1.0, "x2": 1.0}
        with self.assertRaisesRegex(ValueError, "missing required field 'record_id'"):
            self.model.predict_one(bad_record_no_id)

    def test_deliberate_prediction_mismatch_fails_assertion(self) -> None:
        result = run_pipeline()
        # Deliberately assert false parity flag to verify failure reporting
        with self.assertRaises(AssertionError):
            self.assertFalse(result["predictions_agree"])


if __name__ == "__main__":
    unittest.main()
