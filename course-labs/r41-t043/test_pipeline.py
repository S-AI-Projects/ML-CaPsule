import unittest

from pipeline import (
    evaluate_constant_label,
    fit_majority_baseline,
    run_pipeline,
    split_records,
    validate_records,
)
from pipeline import load_fixture


class MajorityBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()
        cls.records = cls.fixture["records"]
        cls.seed = cls.fixture["seed"]

    def test_seeded_split_and_held_out_result(self):
        result = run_pipeline(self.records, self.seed)
        self.assertEqual(result["train_record_ids"], ["t6", "t5", "t4", "t2"])
        self.assertEqual(result["validation"]["record_ids"], ["t3"])
        self.assertEqual(result["test"]["record_ids"], ["t1"])
        self.assertEqual(result["training_class_counts"], {"0": 2, "1": 2})
        self.assertEqual(result["predicted_label"], 0)
        self.assertEqual(result["validation"]["y_pred"], [0])
        self.assertEqual(result["validation"]["accuracy"], 1.0)
        self.assertEqual(result["test"]["y_true"], [1])
        self.assertEqual(result["test"]["y_pred"], [0])
        self.assertEqual(result["test"]["accuracy"], 0.0)

    def test_split_is_deterministic_for_fixed_seed(self):
        first = split_records(self.records, self.seed)
        second = split_records(self.records, self.seed)
        self.assertEqual(first, second)

    def test_fixture_has_the_literal_six_record_contract(self):
        self.assertEqual(
            [(row["record_id"], row["text"], row["label"]) for row in self.records],
            [
                ("t1", "red apple", 1),
                ("t2", "green apple", 1),
                ("t3", "blue sky", 0),
                ("t4", "dark sky", 0),
                ("t5", "red fruit", 1),
                ("t6", "blue cloud", 0),
            ],
        )

    def test_majority_is_fitted_only_from_training_rows(self):
        model = fit_majority_baseline(
            [self.records[0], self.records[1], self.records[4]]
        )
        self.assertEqual(model, 1)

    def test_held_out_label_change_does_not_change_fitted_prediction(self):
        changed = [dict(record) for record in self.records]
        changed[0]["label"] = 0
        self.assertEqual(
            run_pipeline(changed, self.seed)["predicted_label"],
            run_pipeline(self.records, self.seed)["predicted_label"],
        )

    def test_tied_training_classes_choose_lower_label(self):
        self.assertEqual(fit_majority_baseline(self.records[2:4]), 0)

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_records([])

    def test_rejects_duplicate_ids(self):
        duplicate = [dict(self.records[0]), dict(self.records[0])]
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            validate_records(duplicate)

    def test_rejects_invalid_labels_including_boolean(self):
        for invalid_label in (2, True, "1"):
            with self.subTest(label=invalid_label):
                invalid = [dict(self.records[0], label=invalid_label)]
                with self.assertRaisesRegex(ValueError, "label must be integer"):
                    validate_records(invalid)

    def test_rejects_train_evaluation_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap training"):
            evaluate_constant_label(1, [self.records[0]], {"t1"})

    def test_rejects_non_six_row_split_and_non_integer_seed(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            split_records(self.records[:5], self.seed)
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            split_records(self.records, "43")


if __name__ == "__main__":
    unittest.main()
