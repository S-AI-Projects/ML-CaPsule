"""Unit tests for Task R41-T005: Student performance (UCI) – regression."""

import math
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
    ExplicitPreprocessor,
    PreprocessingContaminationError,
    SchemaValidationError,
    validate_records,
)


class TestSyntheticFixture(unittest.TestCase):
    """Verify synthetic fixture properties and disclaimers."""

    def test_fixture_disclaimer_present(self) -> None:
        self.assertIn("Synthetic pipeline-verification fixture", DISCLAIMER)
        self.assertIn("Not real-world or UCI student performance data", DISCLAIMER)

    def test_fixture_exact_count_and_keys(self) -> None:
        self.assertEqual(len(SYNTHETIC_RECORDS), 6)
        expected_keys = {ID_FIELD, "x1", "x2", TARGET_FIELD}
        for rec in SYNTHETIC_RECORDS:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_deterministic_formula_and_features(self) -> None:
        # Fixture: x1=[0,1,2,3,4,5], x2=1 and target=2*x1+1
        expected_x1 = [0, 1, 2, 3, 4, 5]
        for idx, rec in enumerate(SYNTHETIC_RECORDS):
            self.assertEqual(rec["x1"], expected_x1[idx])
            self.assertEqual(rec["x2"], 1)
            self.assertEqual(rec["target"], 2 * rec["x1"] + 1)

    def test_split_manifest_disjoint(self) -> None:
        self.assertEqual(SPLIT_MANIFEST["train"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(SPLIT_MANIFEST["validation"], ["r5"])
        self.assertEqual(SPLIT_MANIFEST["test"], ["r6"])

        all_ids = SPLIT_MANIFEST["train"] + SPLIT_MANIFEST["validation"] + SPLIT_MANIFEST["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 6)


class TestExplicitPreprocessor(unittest.TestCase):
    """Test preprocessing parameter fitting and transformation."""

    def setUp(self) -> None:
        splits = get_split_records()
        self.train_records = splits["train"]
        self.val_records = splits["validation"]
        self.test_records = splits["test"]
        self.preprocessor = ExplicitPreprocessor()

    def test_training_parameters_fitted_correctly(self) -> None:
        self.preprocessor.fit(self.train_records)
        self.assertTrue(self.preprocessor.is_fitted)

        # Training x1: [0, 1, 2, 3] -> mean = 1.5, std = sqrt(1.25)
        self.assertAlmostEqual(self.preprocessor.scale_means_["x1"], 1.5)
        expected_std_x1 = math.sqrt(1.25)
        self.assertAlmostEqual(self.preprocessor.scale_stds_["x1"], expected_std_x1)

        # Training x2: [1, 1, 1, 1] -> mean = 1.0, std = 0.0
        self.assertAlmostEqual(self.preprocessor.scale_means_["x2"], 1.0)
        self.assertAlmostEqual(self.preprocessor.scale_stds_["x2"], 0.0)

    def test_output_values_dtype_and_shape_asserted(self) -> None:
        self.preprocessor.fit(self.train_records)

        # 1. Transform training records
        train_matrix, train_shape, train_dtype = self.preprocessor.transform(self.train_records)
        self.assertEqual(train_shape, (4, 2))
        self.assertIs(train_dtype, float)

        expected_std_x1 = math.sqrt(1.25)
        # Verify row r1: x1=0 -> (0-1.5)/std, x2=1 -> 0.0
        self.assertAlmostEqual(train_matrix[0][0], (0 - 1.5) / expected_std_x1)
        self.assertAlmostEqual(train_matrix[0][1], 0.0)
        # Verify row r4: x1=3 -> (3-1.5)/std, x2=1 -> 0.0
        self.assertAlmostEqual(train_matrix[3][0], (3 - 1.5) / expected_std_x1)
        self.assertAlmostEqual(train_matrix[3][1], 0.0)

        # 2. Transform validation record (r5: x1=4, x2=1)
        val_matrix, val_shape, val_dtype = self.preprocessor.transform(self.val_records)
        self.assertEqual(val_shape, (1, 2))
        self.assertIs(val_dtype, float)
        self.assertAlmostEqual(val_matrix[0][0], (4 - 1.5) / expected_std_x1)
        self.assertAlmostEqual(val_matrix[0][1], 0.0)

        # 3. Transform test record (r6: x1=5, x2=1)
        test_matrix, test_shape, test_dtype = self.preprocessor.transform(self.test_records)
        self.assertEqual(test_shape, (1, 2))
        self.assertIs(test_dtype, float)
        self.assertAlmostEqual(test_matrix[0][0], (5 - 1.5) / expected_std_x1)
        self.assertAlmostEqual(test_matrix[0][1], 0.0)

    def test_feature_order_is_strictly_preserved(self) -> None:
        self.preprocessor.fit(self.train_records)
        # Even if dictionary keys are ordered differently
        reordered_rec = [{"record_id": "r1", "x2": 1, "x1": 0}]
        matrix, shape, dtype = self.preprocessor.transform(reordered_rec)
        self.assertEqual(shape, (1, 2))
        expected_std_x1 = math.sqrt(1.25)
        # Index 0 must correspond to x1, Index 1 to x2
        self.assertAlmostEqual(matrix[0][0], (0 - 1.5) / expected_std_x1)
        self.assertAlmostEqual(matrix[0][1], 0.0)

    def test_fitting_on_held_out_input_impossible_through_public_transform(self) -> None:
        self.preprocessor.fit(self.train_records)
        mean_x1_before = self.preprocessor.scale_means_["x1"]
        std_x1_before = self.preprocessor.scale_stds_["x1"]

        # Repeatedly transform held-out validation and test rows with extreme unseen values
        extreme_held_out = [
            {"record_id": "val_1", "x1": 1000.0, "x2": 50.0},
            {"record_id": "test_1", "x1": -500.0, "x2": -20.0},
        ]
        self.preprocessor.transform(extreme_held_out)

        # Parameters must remain strictly unchanged
        self.assertEqual(self.preprocessor.scale_means_["x1"], mean_x1_before)
        self.assertEqual(self.preprocessor.scale_stds_["x1"], std_x1_before)

    def test_missing_value_imputation_uses_training_means(self) -> None:
        self.preprocessor.fit(self.train_records)
        rec_with_missing = [{"record_id": "eval_miss", "x1": None, "x2": None}]
        matrix, shape, dtype = self.preprocessor.transform(rec_with_missing)
        # Both x1 and x2 should be imputed with training means (1.5 and 1.0)
        # Scaled values: (1.5 - 1.5) / std = 0.0, and constant x2 = 0.0
        self.assertAlmostEqual(matrix[0][0], 0.0)
        self.assertAlmostEqual(matrix[0][1], 0.0)


class TestContaminationAndValidation(unittest.TestCase):
    """Test rejection of contamination fixtures and invalid inputs."""

    def test_deliberate_contamination_fixture_rejected(self) -> None:
        preprocessor = ExplicitPreprocessor()
        # Deliberately include held-out test row r6 in training
        contaminated_training = [
            {"record_id": "r1", "x1": 0, "x2": 1},
            {"record_id": "r2", "x1": 1, "x2": 1},
            {"record_id": "r6", "x1": 5, "x2": 1},  # Contamination!
        ]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.fit(contaminated_training)

    def test_duplicate_record_ids_rejected(self) -> None:
        preprocessor = ExplicitPreprocessor()
        duplicates = [
            {"record_id": "r1", "x1": 0, "x2": 1},
            {"record_id": "r1", "x1": 1, "x2": 1},
        ]
        with self.assertRaises(PreprocessingContaminationError):
            preprocessor.fit(duplicates)

    def test_boolean_features_rejected(self) -> None:
        preprocessor = ExplicitPreprocessor()
        bool_rec = [{"record_id": "r1", "x1": True, "x2": 1}]
        with self.assertRaises(SchemaValidationError):
            preprocessor.fit(bool_rec)

    def test_non_numeric_features_rejected(self) -> None:
        preprocessor = ExplicitPreprocessor()
        str_rec = [{"record_id": "r1", "x1": "invalid", "x2": 1}]
        with self.assertRaises(SchemaValidationError):
            preprocessor.fit(str_rec)

    def test_transform_before_fit_raises_runtime_error(self) -> None:
        preprocessor = ExplicitPreprocessor()
        with self.assertRaises(RuntimeError):
            preprocessor.transform([{"record_id": "r1", "x1": 0, "x2": 1}])

    def test_regression_fails_when_expected_is_altered(self) -> None:
        """Verify that altering an expected value fails the test suite."""
        preprocessor = ExplicitPreprocessor()
        splits = get_split_records()
        preprocessor.fit(splits["train"])
        actual_mean = preprocessor.scale_means_["x1"]
        deliberate_wrong_mean = 999.0
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_mean, deliberate_wrong_mean)


if __name__ == "__main__":
    unittest.main()
