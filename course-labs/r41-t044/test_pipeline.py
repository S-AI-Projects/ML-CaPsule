import random
import unittest

from pipeline import (
    binary_classification_metrics,
    load_fixture,
    mean_absolute_error,
    validate_records,
    validate_splits,
)

FIXED_SEED = 44044


class ImageFixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = load_fixture()
        self.records = self.fixture["records"]

    def test_six_literal_images_and_group_separated_splits(self):
        self.assertEqual(self.fixture["data_kind"], "simulated")
        self.assertEqual(
            self.fixture["schema_fields"],
            ["record_id", "group_id", "pixels", "label"],
        )
        self.assertEqual(len(self.records), 6)
        self.assertEqual(
            [(record["record_id"], record["label"]) for record in self.records],
            [
                ("zero-1", 0),
                ("diagonal-1", 1),
                ("zero-2", 0),
                ("diagonal-2", 1),
                ("zero-3", 0),
                ("diagonal-3", 1),
            ],
        )
        zeros = [[0] * 4 for _ in range(4)]
        diagonal = [
            [255, 0, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 255],
        ]
        self.assertEqual([record["pixels"] for record in self.records[::2]], [zeros] * 3)
        self.assertEqual([record["pixels"] for record in self.records[1::2]], [diagonal] * 3)
        splits = validate_splits(self.records, self.fixture["split_group_ids"])
        self.assertEqual(splits["train"], ["zero-1", "diagonal-1"])
        self.assertEqual(splits["validation"], ["zero-2", "diagonal-2"])
        self.assertEqual(splits["test"], ["zero-3", "diagonal-3"])
        self.assertEqual(len({record["record_id"] for record in self.records}), 6)

    def test_fixed_seed_shuffle_preserves_valid_records_and_split_membership(self):
        shuffled = list(self.records)
        random.Random(FIXED_SEED).shuffle(shuffled)
        validate_records(shuffled)
        splits = validate_splits(shuffled, self.fixture["split_group_ids"])
        self.assertEqual(set(splits["train"]), {"zero-1", "diagonal-1"})
        self.assertEqual(set(splits["validation"]), {"zero-2", "diagonal-2"})
        self.assertEqual(set(splits["test"]), {"zero-3", "diagonal-3"})

    def test_rejects_duplicate_ids_bad_pixels_and_boolean_labels(self):
        duplicate = [dict(record) for record in self.records]
        duplicate[1]["record_id"] = duplicate[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            validate_records(duplicate)

        bad_pixels = [dict(record) for record in self.records]
        bad_pixels[0]["pixels"] = [[0] * 4] * 3
        with self.assertRaisesRegex(ValueError, "4x4"):
            validate_records(bad_pixels)

        bad_label = [dict(record) for record in self.records]
        bad_label[0]["label"] = True
        with self.assertRaisesRegex(ValueError, "label must be integer"):
            validate_records(bad_label)

    def test_rejects_overlapping_or_missing_split_groups(self):
        splits = {
            "train": ["group-1"],
            "validation": ["group-1"],
            "test": ["group-3"],
        }
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_splits(self.records, splits)
        with self.assertRaisesRegex(ValueError, "unknown group IDs"):
            validate_splits(
                self.records,
                {
                    "train": ["group-1"],
                    "validation": ["group-2"],
                    "test": ["group-unknown"],
                },
            )


class MetricTests(unittest.TestCase):
    def test_macro_precision_recall_concrete_result(self):
        metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["averaging"], "macro over fixed labels [0, 1]")
        self.assertEqual(metrics["zero_denominator"], 0.0)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(
            metrics["per_class"],
            {
                "0": {"precision": 0.5, "recall": 0.5},
                "1": {"precision": 0.5, "recall": 0.5},
            },
        )

    def test_zero_denominators_are_zero_and_both_labels_remain_in_macro(self):
        metrics = binary_classification_metrics([0, 0], [0, 0])
        self.assertEqual(metrics["per_class"]["1"], {"precision": 0.0, "recall": 0.0})
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)

    def test_rejects_empty_mismatched_and_invalid_classification_vectors(self):
        for truth, prediction in (([], []), ([0], []), ([True], [1]), ([2], [1])):
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    binary_classification_metrics(truth, prediction)

    def test_regression_mae_concrete_result(self):
        self.assertAlmostEqual(mean_absolute_error([1, 2, 3], [1, 3, 2]), 2 / 3)

    def test_rejects_empty_mismatched_and_non_finite_regression_vectors(self):
        for truth, prediction in (
            ([], []),
            ([1], []),
            ([1, 2], [1]),
            ([1, float("nan")], [1, 2]),
            ([True], [1]),
        ):
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    mean_absolute_error(truth, prediction)

    def test_a_changed_expected_result_trips_an_assertion(self):
        metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        with self.assertRaises(AssertionError):
            self.assertEqual(metrics["precision"], 0.6)


if __name__ == "__main__":
    unittest.main()
