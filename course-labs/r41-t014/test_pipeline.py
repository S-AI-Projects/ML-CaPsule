"""Unit tests for evaluation metric verification (Task R41-T014)."""

import unittest
from pipeline import compute_binary_classification_metrics, load_fixture, validate_stratified_split


class TestAdultIncomeEvaluationMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.records = self.payload["records"]
        self.split_manifest = self.payload["metadata"]["stratified_split"]

    def test_benchmark_metric_example(self) -> None:
        truth = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        res = compute_binary_classification_metrics(truth, prediction)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tn"], 1)
        self.assertAlmostEqual(res["precision"], 0.5, places=4)
        self.assertAlmostEqual(res["recall"], 0.5, places=4)
        self.assertAlmostEqual(res["accuracy"], 0.5, places=4)
        self.assertAlmostEqual(res["f1"], 0.5, places=4)

    def test_stratified_split_preserves_class_ratios(self) -> None:
        self.assertEqual(len(self.records), 8)
        check = validate_stratified_split(self.records, self.split_manifest)
        self.assertTrue(check["is_stratified"])
        self.assertEqual(check["counts"]["train"], 4)
        self.assertEqual(check["counts"]["validation"], 2)
        self.assertEqual(check["counts"]["test"], 2)
        self.assertEqual(check["distributions"]["train"], {0: 2, 1: 2})
        self.assertEqual(check["distributions"]["validation"], {0: 1, 1: 1})
        self.assertEqual(check["distributions"]["test"], {0: 1, 1: 1})

    def test_zero_denominator_handling(self) -> None:
        res = compute_binary_classification_metrics([0, 0], [0, 0], zero_division=0.0)
        self.assertEqual(res["precision"], 0.0)
        self.assertEqual(res["recall"], 0.0)

    def test_failure_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            compute_binary_classification_metrics([1, 0], [1])

        with self.assertRaisesRegex(ValueError, "Labels must be binary"):
            compute_binary_classification_metrics([1, 2], [1, 0])

        leaked_split = {
            "train": ["r1", "r2", "r5", "r6"],
            "validation": ["r1", "r7"],
            "test": ["r4", "r8"],
        }
        with self.assertRaisesRegex(ValueError, "ID leakage"):
            validate_stratified_split(self.records, leaked_split)


if __name__ == "__main__":
    unittest.main()
