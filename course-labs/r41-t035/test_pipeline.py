"""Unit tests for R41-T035: World happiness report explicit preprocessing contract."""

import unittest

from pipeline import ExplicitPreprocessor, load_fixture


class TestWorldHappinessPreprocessing(unittest.TestCase):
    """Verifies explicit preprocessing contract and contamination rejection for Task R41-T035."""

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

    def test_preprocessor_output_values_dtype_and_shape(self) -> None:
        """Verify output shape (4,2), dtype float64, and strictly retained feature order."""
        prep = ExplicitPreprocessor()
        output = prep.fit_transform(self.train_records)

        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(output.dtype, "float64")
        self.assertEqual(output.feature_names, ["x1", "x2"])
        self.assertEqual(output.record_ids, ["r1", "r2", "r3", "r4"])

        # Check that x1 is standardized with mean=1.5
        self.assertAlmostEqual(prep.feature_means["x1"], 1.5, places=6)
        self.assertAlmostEqual(prep.feature_means["x2"], 1.0, places=6)

    def test_transform_before_fit_raises_runtime_error(self) -> None:
        """Verify calling transform prior to fitting is rejected."""
        prep = ExplicitPreprocessor()
        with self.assertRaises(RuntimeError):
            prep.transform(self.val_records)

    def test_fitting_on_held_out_input_is_impossible(self) -> None:
        """Verify transforming held-out rows preserves original training parameters without leakage."""
        prep = ExplicitPreprocessor()
        prep.fit(self.train_records)

        orig_x1_mean = prep.feature_means["x1"]
        orig_x1_std = prep.feature_stds["x1"]

        # Transform validation (x1=4.0) and test (x1=5.0)
        prep.transform(self.val_records)
        prep.transform(self.test_records)

        # Internal parameters must NOT mutate
        self.assertEqual(prep.feature_means["x1"], orig_x1_mean)
        self.assertEqual(prep.feature_stds["x1"], orig_x1_std)

    def test_deliberate_contamination_fixture_rejected(self) -> None:
        """Verify corrupted or non-numeric inputs raise clear ValueError."""
        prep = ExplicitPreprocessor()
        prep.fit(self.train_records)

        contaminated_record = [{"record_id": "bad_1", "x1": "invalid_string", "x2": 1.0}]
        with self.assertRaises(ValueError):
            prep.transform(contaminated_record)

        nan_record = [{"record_id": "bad_2", "x1": float("nan"), "x2": 1.0}]
        with self.assertRaises(ValueError):
            prep.transform(nan_record)


if __name__ == "__main__":
    unittest.main()
