"""Unit tests for R41-T037: California housing reproducible configuration."""

import json
import unittest

from pipeline import (
    ConfigurableLinearRegression,
    load_config,
    load_fixture,
    validate_configuration,
)


class TestCaliforniaHousingReproducibleConfig(unittest.TestCase):
    """Verifies configuration reproducibility and validation for Task R41-T037."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.config = load_config()

    def test_deterministic_formula_fixture_consistency(self) -> None:
        """Verify fixture conforms to target = 2 * x1 + 1 with stable IDs."""
        self.assertEqual(len(self.records), 6)
        for r in self.records:
            expected = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected, places=6)

    def test_reloaded_configuration_reconstructs_identical_predictions(self) -> None:
        """Verify serializing and reloading config reconstructs identical split and predictions."""
        model1 = ConfigurableLinearRegression(self.config)
        model1.fit(self.records)
        preds1 = model1.predict(self.records)

        # Serialize config to JSON string and reload
        serialized = model1.to_json()
        model2 = ConfigurableLinearRegression.from_json(serialized)
        model2.fit(self.records)
        preds2 = model2.predict(self.records)

        self.assertEqual(model1.config["split_ids"], model2.config["split_ids"])
        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2, places=10)

    def test_unknown_field_is_rejected(self) -> None:
        """Verify configuration containing unexpected key raises ValueError."""
        bad_config = dict(self.config)
        bad_config["unauthorized_param"] = True

        with self.assertRaises(ValueError) as ctx:
            validate_configuration(bad_config)
        self.assertIn("Unknown configuration fields", str(ctx.exception))

    def test_incompatible_input_shape_rejected(self) -> None:
        """Verify input record missing a configured feature raises ValueError."""
        model = ConfigurableLinearRegression(self.config)
        model.fit(self.records)

        incompatible_records = [{"record_id": "bad", "x1": 1.0}]  # missing x2
        with self.assertRaises(ValueError) as ctx:
            model.predict(incompatible_records)
        self.assertIn("missing expected feature", str(ctx.exception))

    def test_contradictory_settings_rejected(self) -> None:
        """Verify conflicting algorithm or overlapping splits are rejected."""
        bad_algo_config = dict(self.config)
        bad_algo_config["algorithm_settings"] = {"algorithm": "unsupported_rf"}
        with self.assertRaises(ValueError):
            validate_configuration(bad_algo_config)

        overlapping_split_config = dict(self.config)
        overlapping_split_config["split_ids"] = {
            "train": ["r1", "r2"],
            "validation": ["r2"],  # r2 in both train and validation
            "test": ["r3"],
        }
        with self.assertRaises(ValueError):
            validate_configuration(overlapping_split_config)

    def test_environment_versions_recorded(self) -> None:
        """Verify environment versions are present in serialized configuration."""
        model = ConfigurableLinearRegression(self.config)
        cfg_out = json.loads(model.to_json())

        self.assertIn("environment", cfg_out)
        self.assertIn("python", cfg_out["environment"])
        self.assertIn("os", cfg_out["environment"])


if __name__ == "__main__":
    unittest.main()
