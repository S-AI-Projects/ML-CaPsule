"""Unit tests for explicit numeric preprocessing and validation (Task R41-T015)."""

import unittest
from pipeline import ExplicitNumericPreprocessor, load_fixture


class TestGlassPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.metadata = self.payload["metadata"]
        self.records = self.payload["records"]
        self.by_id = {r["record_id"]: r for r in self.records}

        self.train_records = [self.by_id[i] for i in self.metadata["stratified_split"]["train"]]
        self.val_records = [self.by_id[i] for i in self.metadata["stratified_split"]["validation"]]
        self.test_records = [self.by_id[i] for i in self.metadata["stratified_split"]["test"]]

        self.preprocessor = ExplicitNumericPreprocessor(
            feature_order=self.metadata["feature_order"],
            valid_labels=self.metadata["valid_labels"],
        )

    def test_output_values_dtype_and_shape(self) -> None:
        self.preprocessor.fit(self.train_records)
        train_scaled = self.preprocessor.transform(self.train_records)
        test_scaled = self.preprocessor.transform(self.test_records)

        # Shape assertions
        self.assertEqual(len(train_scaled), 4)
        self.assertEqual(len(train_scaled[0]), 2)
        self.assertEqual(len(test_scaled), 2)
        self.assertEqual(len(test_scaled[0]), 2)

        # Dtype assertion
        for row in train_scaled + test_scaled:
            for val in row:
                self.assertIsInstance(val, float)

        # Explicit training mean calculation check: x1 values in train are [0, 1, 4, 5] -> mean = 2.5
        self.assertAlmostEqual(self.preprocessor.means["x1"], 2.5, places=3)
        self.assertAlmostEqual(self.preprocessor.means["x2"], 1.0, places=3)

    def test_held_out_cannot_alter_fitted_parameters(self) -> None:
        self.preprocessor.fit(self.train_records)
        baseline_means = dict(self.preprocessor.means)
        baseline_stds = dict(self.preprocessor.stds)

        # Transform held-out inputs multiple times
        _ = self.preprocessor.transform(self.test_records)
        _ = self.preprocessor.transform(self.val_records)

        # Public transform interface must not alter or fit onto held-out data
        self.assertEqual(self.preprocessor.means, baseline_means)
        self.assertEqual(self.preprocessor.stds, baseline_stds)

    def test_unfitted_transform_raises_runtime_error(self) -> None:
        fresh_preprocessor = ExplicitNumericPreprocessor(
            feature_order=self.metadata["feature_order"],
            valid_labels=self.metadata["valid_labels"],
        )
        with self.assertRaisesRegex(RuntimeError, "must be fitted on training data"):
            fresh_preprocessor.transform(self.test_records)

    def test_deliberate_contamination_rejected(self) -> None:
        self.preprocessor.fit(self.train_records)

        # 1. Missing feature in record
        missing_feature_fixture = [{"record_id": "bad1", "x1": 2.0}]
        with self.assertRaisesRegex(ValueError, "Missing required feature"):
            self.preprocessor.transform(missing_feature_fixture)

        # 2. Non-numeric / missing value in feature
        none_feature_fixture = [{"record_id": "bad2", "x1": None, "x2": 1.0}]
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            self.preprocessor.transform(none_feature_fixture)

        # 3. Invalid label in training/validation set
        invalid_label_fixture = [{"record_id": "bad3", "x1": 2.0, "x2": 1.0, "label": 99}]
        with self.assertRaisesRegex(ValueError, "Invalid label"):
            self.preprocessor.fit(invalid_label_fixture)


if __name__ == "__main__":
    unittest.main()
