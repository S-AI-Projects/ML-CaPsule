"""Unit tests for Task R41-T007: Spotify tracks – genre classification."""

import json
import unittest
from typing import Any, Dict, List

from fixture import (
    DISCLAIMER,
    FEATURE_ORDERING,
    GROUP_FIELD,
    ID_FIELD,
    LABEL_FIELD,
    SCHEMA_VERSION,
    SPLIT_MANIFEST,
    SYNTHETIC_RECORDS,
    get_split_records,
)
from pipeline import (
    ConfigurableClassifier,
    ConfigurationError,
    IncompatibleShapeError,
    PipelineConfig,
)


def get_default_config() -> PipelineConfig:
    return PipelineConfig(
        schema_version=SCHEMA_VERSION,
        seed=42,
        feature_ordering=list(FEATURE_ORDERING),
        split_ids=dict(SPLIT_MANIFEST),
        algorithm_settings={
            "algorithm": "logistic_regression",
            "learning_rate": 0.1,
            "max_iter": 50,
            "threshold": 0.5,
        },
    )


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not observations from the real Spotify tracks dataset", DISCLAIMER)

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

        # Balanced stratification: 50% positive in each partition
        train_pos = sum(1 for r in splits["train"] if r["label"] == 1)
        val_pos = sum(1 for r in splits["validation"] if r["label"] == 1)
        test_pos = sum(1 for r in splits["test"] if r["label"] == 1)

        self.assertEqual(train_pos, 2)
        self.assertEqual(val_pos, 1)
        self.assertEqual(test_pos, 1)


class TestConfigSerializationAndReproducibility(unittest.TestCase):
    """Verify serialization, deserialization, and reproducibility across reloads."""

    def test_reloaded_config_reconstructs_identical_splits_and_predictions(self) -> None:
        orig_config = get_default_config()
        json_serialized = orig_config.to_json()

        # Reload configuration from serialized JSON
        allowed_ids = {r[ID_FIELD] for r in SYNTHETIC_RECORDS}
        reloaded_config = PipelineConfig.from_json(json_serialized, allowed_record_ids=allowed_ids)

        # Verify configuration equivalence
        self.assertEqual(reloaded_config.schema_version, orig_config.schema_version)
        self.assertEqual(reloaded_config.seed, orig_config.seed)
        self.assertEqual(reloaded_config.feature_ordering, orig_config.feature_ordering)
        self.assertEqual(reloaded_config.split_ids, orig_config.split_ids)
        self.assertEqual(reloaded_config.algorithm_settings, orig_config.algorithm_settings)

        # 1. Fit classifier with original config
        orig_clf = ConfigurableClassifier(orig_config)
        orig_splits = orig_clf.partition_records(SYNTHETIC_RECORDS)
        orig_clf.fit(orig_splits["train"])
        orig_val_preds = orig_clf.predict(orig_splits["validation"])
        orig_test_preds = orig_clf.predict(orig_splits["test"])

        # 2. Fit classifier with reloaded config
        reloaded_clf = ConfigurableClassifier(reloaded_config)
        reloaded_splits = reloaded_clf.partition_records(SYNTHETIC_RECORDS)
        reloaded_clf.fit(reloaded_splits["train"])
        reloaded_val_preds = reloaded_clf.predict(reloaded_splits["validation"])
        reloaded_test_preds = reloaded_clf.predict(reloaded_splits["test"])

        # Assert identical split partitions and identical predictions
        self.assertEqual(reloaded_splits, orig_splits)
        self.assertEqual(reloaded_val_preds, orig_val_preds)
        self.assertEqual(reloaded_test_preds, orig_test_preds)
        self.assertEqual(reloaded_clf.weights_, orig_clf.weights_)
        self.assertEqual(reloaded_clf.bias_, orig_clf.bias_)


class TestConfigValidationAndRejection(unittest.TestCase):
    """Test rejection of unknown fields, contradictory settings, and incompatible shapes."""

    def test_unknown_top_level_field_rejected(self) -> None:
        raw = get_default_config().to_dict()
        raw["unknown_extra_param"] = "illegal_value"
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw)

    def test_unknown_algorithm_setting_rejected(self) -> None:
        raw = get_default_config().to_dict()
        raw["algorithm_settings"]["unsupported_algo_option"] = True
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw)

    def test_contradictory_overlapping_splits_rejected(self) -> None:
        raw = get_default_config().to_dict()
        # Duplicate rec_001 in both train and validation
        raw["split_ids"]["validation"].append("rec_001")
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw)

    def test_unknown_split_id_rejected_when_allowed_ids_given(self) -> None:
        raw = get_default_config().to_dict()
        raw["split_ids"]["train"].append("ghost_record_999")
        allowed_ids = {r[ID_FIELD] for r in SYNTHETIC_RECORDS}
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw, allowed_record_ids=allowed_ids)

    def test_invalid_learning_rate_or_max_iter_rejected(self) -> None:
        raw = get_default_config().to_dict()
        raw["algorithm_settings"]["learning_rate"] = -0.5
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw)

        raw = get_default_config().to_dict()
        raw["algorithm_settings"]["max_iter"] = 0
        with self.assertRaises(ConfigurationError):
            PipelineConfig.from_dict(raw)

    def test_incompatible_input_shape_rejected(self) -> None:
        clf = ConfigurableClassifier(get_default_config())
        # Record missing feature x2
        incompatible_rec = {"record_id": "bad_1", "x1": 5.0}
        with self.assertRaises(IncompatibleShapeError):
            clf.validate_record_shape(incompatible_rec)

    def test_negative_regression_check_fails_when_expected_altered(self) -> None:
        """Verify that altering an expected prediction causes assertion failure."""
        orig_config = get_default_config()
        clf = ConfigurableClassifier(orig_config)
        splits = clf.partition_records(SYNTHETIC_RECORDS)
        clf.fit(splits["train"])
        actual_preds = clf.predict(splits["test"])

        deliberate_wrong_preds = [999, 999]
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_preds, deliberate_wrong_preds)


if __name__ == "__main__":
    unittest.main()
