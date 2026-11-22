"""Offline unit tests for the concrete-strength input contract."""

import random
import unittest
from pathlib import Path

from pipeline import load_fixture, validate_records


FIXTURE_PATH = Path(__file__).with_name("fixture.json")
FIXED_SEED = 41041


def issue_tuples(report):
    return [
        (issue.record_id, issue.field, issue.reason) for issue in report.issues
    ]


class ValidateRecordsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = load_fixture(FIXTURE_PATH)
        self.records = self.fixture["records"]

    def test_literal_formula_fixture_is_valid_and_split_is_explicit(self):
        self.assertTrue(self.fixture["simulated"])
        self.assertEqual(self.fixture["fixture_name"], "deterministic formula fixture")
        self.assertEqual(len(self.records), 6)
        report = validate_records(self.records)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.record_count, 6)
        self.assertEqual(report.issues, ())

        x1_values = [row["x1"] for row in self.records]
        self.assertEqual(x1_values, [0, 1, 2, 3, 4, 5])
        self.assertTrue(all(row["x2"] == 1 for row in self.records))
        self.assertEqual(
            [row["target"] for row in self.records],
            [2 * row["x1"] + 1 for row in self.records],
        )

        splits = self.fixture["split_ids"]
        self.assertEqual(splits["train"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(splits["validation"], ["r5"])
        self.assertEqual(splits["test"], ["r6"])
        split_sets = [set(ids) for ids in splits.values()]
        self.assertEqual(sum(map(len, split_sets)), 6)
        fixture_ids = {row["record_id"] for row in self.records}
        self.assertEqual(set.union(*split_sets), fixture_ids)
        self.assertFalse(
            any(
                left & right
                for index, left in enumerate(split_sets)
                for right in split_sets[index + 1 :]
            )
        )

    def test_valid_fixture_remains_valid_after_fixed_seed_shuffle(self):
        shuffled = list(self.records)
        random.Random(FIXED_SEED).shuffle(shuffled)
        report = validate_records(shuffled)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.record_count, 6)

    def test_missing_id_reports_placeholder_field_and_reason(self):
        row = dict(self.records[0])
        del row["record_id"]
        report = validate_records([row])
        self.assertEqual(
            issue_tuples(report),
            [("<missing>", "record_id", "is required")],
        )
        self.assertEqual(report.record_count, 1)

    def test_numeric_text_and_tensor_like_value_are_rejected(self):
        row = dict(self.records[0], x1="2.0", x2=[1])
        report = validate_records([row])
        self.assertEqual(
            issue_tuples(report),
            [
                ("r1", "x1", "must be a real numeric scalar"),
                ("r1", "x2", "must be a real numeric scalar"),
            ],
        )

    def test_non_finite_and_boolean_numeric_values_are_rejected(self):
        row = dict(self.records[0], x1=float("nan"), target=True)
        report = validate_records([row])
        self.assertEqual(
            issue_tuples(report),
            [
                ("r1", "x1", "must be finite"),
                ("r1", "target", "must be a real numeric scalar"),
            ],
        )

    def test_duplicate_record_id_reports_duplicate_reason(self):
        duplicate = dict(self.records[0], x1=99)
        report = validate_records([self.records[0], duplicate])
        self.assertEqual(
            issue_tuples(report),
            [("r1", "record_id", "duplicates record_id first seen in row 1")],
        )
        self.assertEqual(report.record_count, 2)

    def test_all_missing_fields_are_reported_without_dropping_the_row(self):
        report = validate_records([{"record_id": "broken"}])
        self.assertEqual(
            issue_tuples(report),
            [
                ("broken", "x1", "is required"),
                ("broken", "x2", "is required"),
                ("broken", "target", "is required"),
            ],
        )
        self.assertEqual(report.record_count, 1)

    def test_malformed_row_and_non_iterable_input_are_reported(self):
        report = validate_records([self.records[0], ["tensor", "row"]])
        self.assertEqual(report.record_count, 2)
        self.assertEqual(
            issue_tuples(report),
            [("<unknown>", "<record>", "must be a mapping/object")],
        )

        invalid_input_report = validate_records(None)
        self.assertEqual(
            issue_tuples(invalid_input_report),
            [
                (
                    "<input>",
                    "records",
                    "must be an iterable of record objects",
                )
            ],
        )

    def test_unknown_field_is_rejected(self):
        row = dict(self.records[0], split="train")
        report = validate_records([row])
        self.assertEqual(
            issue_tuples(report),
            [("r1", "split", "is not part of the schema")],
        )

    def test_assertions_detect_a_deliberately_changed_expected_result(self):
        actual = validate_records(self.records).to_dict()
        changed_expected = dict(actual, record_count=actual["record_count"] + 1)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual, changed_expected)


if __name__ == "__main__":
    unittest.main()