"""Unit tests for R41-T027 (Banknote authentication – SVM config reproducibility).

Task: R41-T027 · 7 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python 3 standard library only):
    python3 -m unittest discover -s course-labs/r41-t027 -p "test_*.py" -v

All tests use the eight-row synthetic classification fixture for pipeline
verification only — NOT observations from the UCI Banknote Authentication
dataset and NOT evidence of real-world or clinical performance.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

# ── Exact expected outputs for config.json (seed=42, linear_svm, C=1.0) ─────

EXPECTED_TRAIN_PREDS = [0, 0, 1, 1]
EXPECTED_VAL_PREDS = [0, 1]
EXPECTED_TEST_PREDS = [1, 1]
EXPECTED_WEIGHTS = [0.5372837221280802, -0.0533138933872721]
EXPECTED_BIAS = -1.224999999999998


class TestSyntheticFixtureContract(unittest.TestCase):
    """Verify the literal eight-row synthetic fixture and schema."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.records = self.fixture_doc["records"]

    def test_explicit_synthetic_labelling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertIn("not observations", self.fixture_doc["disclaimer"].lower())
        self.assertIn("pipeline-verification", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])

    def test_literal_eight_rows_and_ids(self) -> None:
        self.assertEqual(len(self.records), 8)
        self.assertEqual([r["record_id"] for r in self.records], ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"])
        self.assertEqual([r["group_id"] for r in self.records], ["g1", "g1", "g2", "g3", "g4", "g4", "g5", "g6"])
        self.assertEqual([r["x1"] for r in self.records], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual([r["x2"] for r in self.records], [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([r["label"] for r in self.records], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_schema_matches_config_contract(self) -> None:
        self.assertEqual(self.schema_doc["schema_version"], pipeline.DEFAULT_SCHEMA_VERSION)
        self.assertEqual(
            set(self.schema_doc["config_required_fields"]),
            set(pipeline.ALLOWED_TOP_LEVEL_FIELDS),
        )
        self.assertEqual(
            set(self.schema_doc["algorithm_allowed_fields"]),
            set(pipeline.ALLOWED_ALGORITHM_FIELDS),
        )


class TestConfigSerializationAndReproducibility(unittest.TestCase):
    """Reloaded config must reconstruct the exact same splits and predictions."""

    def setUp(self) -> None:
        self.config = pipeline.ExperimentConfig.load()
        self.fixture_doc = pipeline.load_fixture()

    def test_roundtrip_json_file_reconstructs_identical_splits_and_predictions(self) -> None:
        first_run = pipeline.run_configured_experiment(self.config, self.fixture_doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = Path(tmpdir) / "roundtrip_config.json"
            self.config.save(saved_path)
            reloaded = pipeline.ExperimentConfig.load(saved_path)

        self.assertEqual(reloaded, self.config)
        second_run = pipeline.run_configured_experiment(reloaded, self.fixture_doc)

        self.assertEqual(first_run["config"], second_run["config"])
        self.assertEqual(first_run["model_parameters"], second_run["model_parameters"])
        self.assertEqual(first_run["splits"], second_run["splits"])

    def test_reconstructed_split_ids_and_predictions_match_expected(self) -> None:
        run = pipeline.run_configured_experiment(self.config, self.fixture_doc)
        self.assertTrue(run["verification_only"])
        self.assertEqual(run["splits"]["train"]["record_ids"], ["r1", "r2", "r5", "r6"])
        self.assertEqual(run["splits"]["validation"]["record_ids"], ["r3", "r7"])
        self.assertEqual(run["splits"]["test"]["record_ids"], ["r4", "r8"])

        self.assertEqual(run["splits"]["train"]["predictions"], EXPECTED_TRAIN_PREDS)
        self.assertEqual(run["splits"]["validation"]["predictions"], EXPECTED_VAL_PREDS)
        self.assertEqual(run["splits"]["test"]["predictions"], EXPECTED_TEST_PREDS)

        for a_w, e_w in zip(run["model_parameters"]["weights"], EXPECTED_WEIGHTS):
            self.assertAlmostEqual(a_w, e_w, places=12)
        self.assertAlmostEqual(run["model_parameters"]["bias"], EXPECTED_BIAS, places=12)

    def test_environment_versions_are_recorded(self) -> None:
        run = pipeline.run_configured_experiment(self.config, self.fixture_doc)
        env = run["environment"]
        self.assertTrue(env["python_version"])
        self.assertTrue(env["python_implementation"])
        self.assertTrue(env["platform"])
        self.assertTrue(env["stdlib_only"])
        self.assertEqual(env["schema_version"], "1.0.0")


class TestUnknownAndContradictoryConfigRejection(unittest.TestCase):
    """Reject unknown fields, unsupported versions, and contradictory settings."""

    def setUp(self) -> None:
        self.base_dict = pipeline.ExperimentConfig.load().to_dict()

    def test_unknown_top_level_field_rejected(self) -> None:
        bad = copy.deepcopy(self.base_dict)
        bad["unexpected_flag"] = True
        with self.assertRaises(pipeline.UnknownConfigFieldError):
            pipeline.ExperimentConfig.from_dict(bad)

    def test_unknown_split_and_algorithm_fields_rejected(self) -> None:
        bad_split = copy.deepcopy(self.base_dict)
        bad_split["split_ids"]["extra_split"] = ["r9"]
        with self.assertRaises(pipeline.UnknownConfigFieldError):
            pipeline.ExperimentConfig.from_dict(bad_split)

        bad_algo = copy.deepcopy(self.base_dict)
        bad_algo["algorithm"]["mystery_param"] = 123
        with self.assertRaises(pipeline.UnknownConfigFieldError):
            pipeline.ExperimentConfig.from_dict(bad_algo)

    def test_unsupported_schema_version_rejected(self) -> None:
        bad = copy.deepcopy(self.base_dict)
        bad["schema_version"] = "9.9.9"
        with self.assertRaises(pipeline.SchemaVersionError):
            pipeline.ExperimentConfig.from_dict(bad)

    def test_contradictory_linear_kernel_with_gamma_rejected(self) -> None:
        bad = copy.deepcopy(self.base_dict)
        bad["algorithm"]["kernel"] = "linear"
        bad["algorithm"]["gamma"] = 0.5
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad)

    def test_contradictory_rbf_kernel_without_gamma_or_with_linear_svm_name_rejected(self) -> None:
        bad_name = copy.deepcopy(self.base_dict)
        bad_name["algorithm"]["name"] = "linear_svm"
        bad_name["algorithm"]["kernel"] = "rbf"
        bad_name["algorithm"]["gamma"] = 0.5
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad_name)

        bad_rbf = copy.deepcopy(self.base_dict)
        bad_rbf["algorithm"]["name"] = "kernel_svm"
        bad_rbf["algorithm"]["kernel"] = "rbf"
        bad_rbf["algorithm"]["gamma"] = None
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad_rbf)

    def test_overlapping_split_ids_rejected(self) -> None:
        bad = copy.deepcopy(self.base_dict)
        bad["split_ids"]["validation"].append("r1")
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad)

    def test_invalid_hyperparameters_and_duplicate_features_rejected(self) -> None:
        bad_c = copy.deepcopy(self.base_dict)
        bad_c["algorithm"]["c"] = -1.0
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad_c)

        bad_seed = copy.deepcopy(self.base_dict)
        bad_seed["seed"] = -5
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad_seed)

        bad_feat = copy.deepcopy(self.base_dict)
        bad_feat["feature_ordering"] = ["x1", "x1"]
        with self.assertRaises(pipeline.ContradictoryConfigError):
            pipeline.ExperimentConfig.from_dict(bad_feat)


class TestIncompatibleInputShapeAndSplitIntegrity(unittest.TestCase):
    """Reject incompatible input shapes, wrong feature order, and split leaks."""

    def setUp(self) -> None:
        self.config = pipeline.ExperimentConfig.load()
        self.fixture_doc = pipeline.load_fixture()
        self.splits = pipeline.reconstruct_splits(self.fixture_doc["records"], self.config)
        self.clf = pipeline.LinearSVMClassifier(self.config).fit(self.splits.train)

    def test_incompatible_matrix_width_rejected(self) -> None:
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict([[1.0]])  # 1 feature instead of 2
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict([[1.0, 1.0, 1.0]])  # 3 features instead of 2
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict([])  # empty batch

    def test_swapped_missing_or_extra_record_features_rejected(self) -> None:
        swapped = [{"record_id": "r_bad", "group_id": "g_bad", "x2": 1, "x1": 2, "label": 0}]
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict(swapped)

        missing = [{"record_id": "r_bad", "group_id": "g_bad", "x1": 2, "label": 0}]
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict(missing)

        extra = [{"record_id": "r_bad", "group_id": "g_bad", "x1": 2, "x2": 1, "x3": 0, "label": 0}]
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict(extra)

        non_finite = [[float("nan"), 1.0]]
        with self.assertRaises(pipeline.IncompatibleInputShapeError):
            self.clf.predict(non_finite)

    def test_group_leakage_and_unknown_split_id_rejected(self) -> None:
        leaked_records = copy.deepcopy(self.fixture_doc["records"])
        # Assign validation record r3 to train group g1
        leaked_records[2]["group_id"] = "g1"
        with self.assertRaises(pipeline.SplitIntegrityError):
            pipeline.reconstruct_splits(leaked_records, self.config)

        bad_cfg_dict = self.config.to_dict()
        bad_cfg_dict["split_ids"]["test"] = ["r4", "r999"]
        bad_cfg = pipeline.ExperimentConfig.from_dict(bad_cfg_dict)
        with self.assertRaises(pipeline.SplitIntegrityError):
            pipeline.reconstruct_splits(self.fixture_doc["records"], bad_cfg)

    def test_predict_before_fit_raises(self) -> None:
        unfitted = pipeline.LinearSVMClassifier(self.config)
        with self.assertRaises(pipeline.ModelNotFittedError):
            unfitted.predict(self.splits.test)


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when an expected prediction or weight is mutated."""

    def test_mutated_expected_weight_fails_assertion(self) -> None:
        cfg = pipeline.ExperimentConfig.load()
        run = pipeline.run_configured_experiment(cfg)
        deliberately_wrong_w1 = 9.99
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(
                run["model_parameters"]["weights"][0],
                deliberately_wrong_w1,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
