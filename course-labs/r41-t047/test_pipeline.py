"""Offline tests for the R41-T047 serialized feature-selection pipeline."""

import copy
import json
from pathlib import Path
import unittest

from pipeline import (
    ConfigurationError,
    load_fixture,
    predict,
    run_pipeline,
    validate_config,
)


HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RECORDS = FIXTURE["records"]
EXPECTED = {
    "selected_features": ["x1"],
    "train_record_ids": [
        "bcw-syn-001",
        "bcw-syn-004",
        "bcw-syn-005",
        "bcw-syn-008",
    ],
    "validation": {
        "record_ids": ["bcw-syn-002", "bcw-syn-006"],
        "y_true": [0, 1],
        "y_pred": [0, 1],
        "accuracy": 1.0,
    },
    "test": {
        "record_ids": ["bcw-syn-003", "bcw-syn-007"],
        "y_true": [0, 1],
        "y_pred": [0, 1],
        "accuracy": 1.0,
    },
}


class ClassificationConfigurationTests(unittest.TestCase):
    def test_literal_synthetic_fixture_and_schema(self):
        self.assertEqual(FIXTURE["data_kind"], "simulated")
        self.assertEqual(
            [row["x1"] for row in RECORDS], [0, 1, 2, 3, 4, 5, 6, 7]
        )
        self.assertEqual([row["x2"] for row in RECORDS], [1] * 8)
        self.assertEqual([row["label"] for row in RECORDS], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(
            SCHEMA["required"],
            ["record_id", "group_id", "x1", "x2", "label"],
        )
        self.assertEqual(len({row["record_id"] for row in RECORDS}), 8)
        self.assertEqual(len({row["group_id"] for row in RECORDS}), 8)

    def test_serialized_config_reloads_same_split_and_predictions(self):
        loaded_records, loaded_config = load_fixture()
        self.assertEqual(loaded_config, CONFIG)
        self.assertEqual(
            run_pipeline(loaded_records, loaded_config),
            run_pipeline(RECORDS, CONFIG),
        )
        actual = run_pipeline(RECORDS, CONFIG)
        self.assertEqual(actual["schema_version"], 1)
        self.assertEqual(actual["seed"], 47)
        self.assertEqual(actual["selected_features"], EXPECTED["selected_features"])
        self.assertEqual(actual["train_record_ids"], EXPECTED["train_record_ids"])
        self.assertEqual(actual["validation"], EXPECTED["validation"])
        self.assertEqual(actual["test"], EXPECTED["test"])
        self.assertEqual(actual["centroids"], {"0": [1.5], "1": [5.5]})

    def test_unknown_configuration_field_is_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["mystery_setting"] = True
        with self.assertRaisesRegex(ConfigurationError, "exactly"):
            validate_config(config, RECORDS)

    def test_incompatible_feature_shape_is_rejected(self):
        result = run_pipeline(RECORDS, CONFIG)
        with self.assertRaisesRegex(ConfigurationError, "expected 2"):
            predict(
                [1],
                feature_order=CONFIG["feature_order"],
                selected_features=result["selected_features"],
                centroids=result["centroids"],
            )

    def test_contradictory_algorithm_setting_is_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["algorithm"]["n_features"] = 3
        with self.assertRaisesRegex(ConfigurationError, "n_features"):
            validate_config(config, RECORDS)

    def test_cross_split_group_overlap_is_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["split_ids"]["test"][0] = config["split_ids"]["train"][0]
        with self.assertRaisesRegex(ConfigurationError, "group_id|partition"):
            validate_config(config, RECORDS)


if __name__ == "__main__":
    unittest.main()
