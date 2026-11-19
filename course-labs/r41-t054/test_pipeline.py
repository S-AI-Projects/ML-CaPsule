import math
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_simulated_fixture(self):
        self.assertEqual(
            [record["record_id"] for record in self.records],
            ["t1", "t2", "t3", "t4", "t5", "t6"],
        )
        self.assertEqual(
            [record["text"] for record in self.records],
            ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"],
        )
        self.assertEqual(
            [record["label"] for record in self.records],
            [1, 1, 0, 0, 1, 0],
        )

    def test_fixed_seed_splits_are_disjoint_and_repeatable(self):
        first = pipeline.fixed_splits(self.records, seed=42)
        second = pipeline.fixed_splits(self.records, seed=42)
        ids = lambda rows: [record["record_id"] for record in rows]
        self.assertEqual(
            {name: ids(rows) for name, rows in first.items()},
            {name: ids(rows) for name, rows in second.items()},
        )
        self.assertEqual(
            {name: len(rows) for name, rows in first.items()},
            {"train": 4, "validation": 1, "test": 1},
        )
        self.assertEqual(
            {name: ids(rows) for name, rows in first.items()},
            {
                "train": ["t4", "t2", "t3", "t5"],
                "validation": ["t1"],
                "test": ["t6"],
            },
        )
        partitions = [set(ids(rows)) for rows in first.values()]
        self.assertTrue(partitions[0].isdisjoint(partitions[1]))
        self.assertTrue(partitions[0].isdisjoint(partitions[2]))
        self.assertTrue(partitions[1].isdisjoint(partitions[2]))
        self.assertEqual(
            set.union(*partitions),
            {record["record_id"] for record in self.records},
        )

    def test_binary_metrics_use_positive_class_one(self):
        y_true = [record["label"] for record in self.records[:4]]
        result = pipeline.classification_metrics(
            y_true,
            [1, 0, 1, 0],
            average="binary",
            positive_label=1,
        )
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["average"], "binary")
        self.assertEqual(result["positive_label"], 1)
        self.assertEqual(result["zero_division"], 0.0)

    def test_macro_average_is_unweighted_over_both_labels(self):
        result = pipeline.classification_metrics(
            [0, 0],
            [0, 0],
            average="macro",
        )
        self.assertEqual(result["labels"], [0, 1])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["zero_division"], 0.0)

    def test_binary_zero_denominators_return_zero(self):
        result = pipeline.classification_metrics(
            [0, 0],
            [0, 0],
            average="binary",
            positive_label=1,
        )
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_regression_mae_example(self):
        result = pipeline.mean_absolute_error([1, 2, 3], [1, 3, 2])
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result, 2 / 3)

    def test_mismatched_shapes_and_unknown_average_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            pipeline.classification_metrics([1, 0], [1])
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            pipeline.mean_absolute_error([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "average must be"):
            pipeline.classification_metrics([1], [1], average="weighted")

    def test_metric_inputs_are_bounded(self):
        labels = [index % 2 for index in range(33)]
        with self.assertRaisesRegex(ValueError, "at most 32 samples"):
            pipeline.classification_metrics(labels, labels)
        with self.assertRaisesRegex(ValueError, "at most 32 samples"):
            pipeline.mean_absolute_error([1] * 33, [1] * 33)


if __name__ == "__main__":
    unittest.main()
