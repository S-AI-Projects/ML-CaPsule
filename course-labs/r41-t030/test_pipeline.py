"""Unit tests for R41-T030 (Global CO2 emissions – trend forecast saved model).

Task: R41-T030 · 10 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python 3 standard library only):
    python3 -m unittest discover -s course-labs/r41-t030 -p "test_*.py" -v

All tests use ten synthetic consecutive daily values (d1..d10) strictly for
pipeline verification — NOT real Global CO2 emissions observations and NOT
evidence of real-world or clinical performance.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

# ── Exact analytical OLS trend expectations for d1..d6: [10,12,11,13,15,14] ─

EXPECTED_SLOPE = 31.0 / 35.0
EXPECTED_INTERCEPT = 72.0 / 7.0
EXPECTED_VAL_PREDS = (78.0 / 5.0, 577.0 / 35.0)
EXPECTED_TEST_PREDS = (608.0 / 35.0, 639.0 / 35.0)


class TestChronologicalFixtureContract(unittest.TestCase):
    """Verify ten consecutive daily synthetic observations and 6/2/2 split."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.records = self.fixture_doc["records"]

    def test_explicit_synthetic_labelling_and_no_shuffling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertFalse(self.fixture_doc["shuffling_allowed"])
        self.assertIn("not real global co2", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])
        self.assertFalse(self.schema_doc["chronological_split_policy"]["shuffling_allowed"])

    def test_literal_ten_daily_observations(self) -> None:
        self.assertEqual(len(self.records), 10)
        self.assertEqual([r["record_id"] for r in self.records], [f"d{i}" for i in range(1, 11)])
        self.assertEqual(
            [r["value"] for r in self.records],
            [10, 12, 11, 13, 15, 14, 16, 18, 17, 19],
        )

    def test_strict_chronological_6_2_2_split(self) -> None:
        splits = pipeline.split_chronological(self.records)
        self.assertEqual([r["record_id"] for r in splits.train], ["d1", "d2", "d3", "d4", "d5", "d6"])
        self.assertEqual([r["record_id"] for r in splits.validation], ["d7", "d8"])
        self.assertEqual([r["record_id"] for r in splits.test], ["d9", "d10"])


class TestSavedArtifactReloadAndInference(unittest.TestCase):
    """Verify safe JSON save/reload equivalence and standalone inference."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.splits = pipeline.split_chronological(self.fixture_doc["records"])
        self.artifact = pipeline.fit_trend_model(self.splits.train, seed=pipeline.FIXED_SEED)

    def test_fitted_ols_parameters_match_analytical_train_only_solution(self) -> None:
        self.assertEqual(self.artifact.n_train_observations, 6)
        self.assertEqual(self.artifact.origin_timestamp, "2026-01-01")
        self.assertEqual(self.artifact.train_end_timestamp, "2026-01-06")
        self.assertAlmostEqual(self.artifact.slope, EXPECTED_SLOPE, places=12)
        self.assertAlmostEqual(self.artifact.intercept, EXPECTED_INTERCEPT, places=12)

    def test_before_and_after_reload_predictions_agree_within_tolerance(self) -> None:
        val_inputs = [{"record_id": r["record_id"], "timestamp": r["timestamp"]} for r in self.splits.validation]
        test_inputs = [{"record_id": r["record_id"], "timestamp": r["timestamp"]} for r in self.splits.test]

        before_val = pipeline.predict_with_artifact(self.artifact, val_inputs)
        before_test = pipeline.predict_with_artifact(self.artifact, test_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = Path(tmpdir) / "trend_model.json"
            pipeline.save_artifact(self.artifact, art_path)
            raw_json = json.loads(art_path.read_text(encoding="utf-8"))
            # Confirm no training data or credentials are embedded in artifact
            self.assertNotIn("train_records", raw_json)
            self.assertNotIn("credentials", raw_json)
            reloaded = pipeline.load_artifact(art_path)

        self.assertEqual(reloaded, self.artifact)
        after_val = pipeline.predict_with_artifact(reloaded, val_inputs)
        after_test = pipeline.predict_with_artifact(reloaded, test_inputs)

        for b_p, a_p, exp_v in zip(before_val, after_val, EXPECTED_VAL_PREDS):
            self.assertLessEqual(
                abs(b_p.predicted_value - a_p.predicted_value),
                pipeline.RELOAD_TOLERANCE,
            )
            self.assertAlmostEqual(a_p.predicted_value, exp_v, places=12)

        for b_p, a_p, exp_v in zip(before_test, after_test, EXPECTED_TEST_PREDS):
            self.assertLessEqual(
                abs(b_p.predicted_value - a_p.predicted_value),
                pipeline.RELOAD_TOLERANCE,
            )
            self.assertAlmostEqual(a_p.predicted_value, exp_v, places=12)

    def test_inference_needs_no_training_data_or_target_values(self) -> None:
        # Predict future dates from reloaded artifact alone without any 'value' field
        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = Path(tmpdir) / "standalone.json"
            pipeline.save_artifact(self.artifact, art_path)
            reloaded = pipeline.load_artifact(art_path)

        future_only = [
            {"record_id": "d9", "timestamp": "2026-01-09"},
            {"record_id": "d10", "timestamp": "2026-01-10"},
        ]
        preds = pipeline.predict_with_artifact(reloaded, future_only)
        self.assertEqual(len(preds), 2)
        self.assertEqual([p.record_id for p in preds], ["d9", "d10"])
        self.assertAlmostEqual(preds[0].predicted_value, EXPECTED_TEST_PREDS[0], places=12)
        self.assertAlmostEqual(preds[1].predicted_value, EXPECTED_TEST_PREDS[1], places=12)


class TestUnsafePickleSchemaAndVersionFailures(unittest.TestCase):
    """Reject pickle formats, schema/version mismatches, shuffling, and bad inputs."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.splits = pipeline.split_chronological(self.fixture_doc["records"])
        self.artifact = pipeline.fit_trend_model(self.splits.train)

    def test_pickle_artifact_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "untrusted.pkl"
            pkl_path.write_bytes(b"\x80\x04\x95\x00")
            with self.assertRaises(pipeline.UnsafeArtifactFormatError):
                pipeline.load_artifact(pkl_path)
            with self.assertRaises(pipeline.UnsafeArtifactFormatError):
                pipeline.save_artifact(self.artifact, Path(tmpdir) / "model.pickle")

    def test_schema_version_mismatch_fails_clearly(self) -> None:
        bad_dict = self.artifact.to_dict()
        bad_dict["schema_version"] = "2.0.0"
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad_version.json"
            bad_path.write_text(json.dumps(bad_dict), encoding="utf-8")
            with self.assertRaises(pipeline.SchemaVersionMismatchError):
                pipeline.load_artifact(bad_path)

    def test_unknown_field_or_non_finite_parameter_in_artifact_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            extra_dict = self.artifact.to_dict()
            extra_dict["unexpected_key"] = "bad"
            p1 = Path(tmpdir) / "extra.json"
            p1.write_text(json.dumps(extra_dict), encoding="utf-8")
            with self.assertRaises(pipeline.ArtifactValidationError):
                pipeline.load_artifact(p1)

            non_json_fmt = self.artifact.to_dict()
            non_json_fmt["artifact_format"] = "pickle"
            p2 = Path(tmpdir) / "bad_fmt.json"
            p2.write_text(json.dumps(non_json_fmt), encoding="utf-8")
            with self.assertRaises(pipeline.UnsafeArtifactFormatError):
                pipeline.load_artifact(p2)

    def test_shuffled_or_non_consecutive_timestamps_rejected(self) -> None:
        shuffled = copy.deepcopy(self.fixture_doc["records"])
        shuffled[1], shuffled[2] = shuffled[2], shuffled[1]
        with self.assertRaises(pipeline.ChronologyError):
            pipeline.split_chronological(shuffled)

    def test_invalid_inference_schema_rejected(self) -> None:
        with self.assertRaises(pipeline.InferenceSchemaError):
            pipeline.predict_with_artifact(self.artifact, [])
        with self.assertRaises(pipeline.InferenceSchemaError):
            pipeline.predict_with_artifact(self.artifact, [{"record_id": "d9"}])
        with self.assertRaises(pipeline.InferenceSchemaError):
            pipeline.predict_with_artifact(
                self.artifact,
                [{"record_id": "d9", "timestamp": "09-01-2026"}],
            )
        with self.assertRaises(pipeline.ChronologyError):
            pipeline.predict_with_artifact(
                self.artifact,
                [
                    {"record_id": "d10", "timestamp": "2026-01-10"},
                    {"record_id": "d9", "timestamp": "2026-01-09"},
                ],
            )


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when an expected forecast value is deliberately changed."""

    def test_mutated_expected_forecast_fails_assertion(self) -> None:
        fixture_doc = pipeline.load_fixture()
        splits = pipeline.split_chronological(fixture_doc["records"])
        artifact = pipeline.fit_trend_model(splits.train)
        val_inputs = [{"record_id": r["record_id"], "timestamp": r["timestamp"]} for r in splits.validation]
        preds = pipeline.predict_with_artifact(artifact, val_inputs)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(preds[0].predicted_value, 99.0, places=6)


if __name__ == "__main__":
    unittest.main()
