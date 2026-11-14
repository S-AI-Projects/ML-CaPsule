"""Unit tests for R41-T033: Palmer penguins regression training-mean baseline."""

import unittest

from pipeline import TrainingMeanBaseline, load_fixture


class TestPalmerPenguinsBaseline(unittest.TestCase):
    """Verifies training-mean baseline for Task R41-T033."""

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
            expected_target = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected_target, places=6)

    def test_training_mean_uses_only_training_data(self) -> None:
        """Verify baseline mean is strictly (1+3+5+7)/4 = 4.0."""
        model = TrainingMeanBaseline()
        model.fit(self.train_records)

        self.assertTrue(model.fitted)
        self.assertAlmostEqual(model.training_mean, 4.0, places=6)
        self.assertEqual(model.train_record_ids, ["r1", "r2", "r3", "r4"])

    def test_validation_prediction_and_metric(self) -> None:
        """Verify held-out validation prediction on r5 is 4.0, target is 9.0, MAE is 5.0."""
        model = TrainingMeanBaseline()
        model.fit(self.train_records)

        eval_val = model.evaluate(self.val_records, split_name="validation")
        self.assertEqual(eval_val.record_ids, ["r5"])
        self.assertEqual(eval_val.predicted, [4.0])
        self.assertEqual(eval_val.actual, [9.0])
        self.assertAlmostEqual(eval_val.mae, 5.0, places=6)

    def test_test_set_prediction_and_metric(self) -> None:
        """Verify held-out test prediction on r6 is 4.0, target is 11.0, MAE is 7.0."""
        model = TrainingMeanBaseline()
        model.fit(self.train_records)

        eval_test = model.evaluate(self.test_records, split_name="test")
        self.assertEqual(eval_test.record_ids, ["r6"])
        self.assertEqual(eval_test.predicted, [4.0])
        self.assertEqual(eval_test.actual, [11.0])
        self.assertAlmostEqual(eval_test.mae, 7.0, places=6)

    def test_unfitted_model_raises_runtime_error(self) -> None:
        """Verify predicting before fitting raises an error."""
        model = TrainingMeanBaseline()
        with self.assertRaises(RuntimeError):
            model.predict(self.val_records)


if __name__ == "__main__":
    unittest.main()
