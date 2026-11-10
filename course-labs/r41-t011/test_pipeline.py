"""Offline CPU-only unit tests for Task R41-T011 data contract validator."""

import copy
import random
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_fixture, validate_records, validate_stratified_split


class TestNetflixTitlesDataContract(unittest.TestCase):
    """Concrete verification and failure-mode tests using fixed seed and CPU-only data."""

    def setUp(self) -> None:
        random.seed(42)
        self.payload = load_fixture()
        self.records = copy.deepcopy(self.payload["records"])

    def test_valid_synthetic_fixture_and_stratified_split(self) -> None:
        self.assertEqual(
            self.payload["metadata"]["data_origin"],
            "SYNTHETIC_PIPELINE_VERIFICATION_ONLY",
        )
        self.assertEqual(len(self.records), 8)
        self.assertEqual([r["x1"] for r in self.records], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(all(r["x2"] == 1 for r in self.records))
        self.assertEqual(
            [r["label"] for r in self.records], [0, 0, 0, 0, 1, 1, 1, 1]
        )

        report = validate_records(self.records)
        self.assertTrue(report["valid"])
        self.assertEqual(report["total_records_checked"], 8)
        self.assertEqual(report["failure_count"], 0)

        split_report = validate_stratified_split(
            self.records, self.payload["metadata"]["stratified_split"]
        )
        self.assertTrue(split_report["disjoint"])
        self.assertTrue(split_report["complete_coverage"])
        self.assertEqual(
            split_report["counts"], {"train": 4, "validation": 2, "test": 2}
        )

    def test_missing_record_id_fails_without_dropping_records(self) -> None:
        bad_records = copy.deepcopy(self.records)
        del bad_records[0]["record_id"]
        bad_records[1]["record_id"] = "   "

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 8)
        self.assertEqual(report["failure_count"], 2)
        self.assertTrue(all(f["field"] == "record_id" for f in report["failures"]))

    def test_duplicate_record_id_fails_with_specific_reason(self) -> None:
        bad_records = copy.deepcopy(self.records)
        bad_records[3]["record_id"] = "rec_01"

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 8)
        self.assertEqual(report["failure_count"], 1)
        failure = report["failures"][0]
        self.assertEqual(failure["record_id"], "rec_01")
        self.assertEqual(failure["field"], "record_id")
        self.assertIn("duplicate", failure["reason"].lower())

    def test_malformed_numeric_text_and_tensor_data_fail_specifically(self) -> None:
        bad_records = copy.deepcopy(self.records)
        bad_records[0]["group_id"] = 999  # malformed text data
        bad_records[1]["x1"] = "not-a-number"  # malformed numeric data
        bad_records[2]["x2"] = [1.0, 2.0]  # malformed tensor/list data
        bad_records[4]["label"] = [0, 1]  # malformed tensor label
        bad_records[5]["x1"] = float("nan")  # non-finite numeric value

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 8)
        self.assertEqual(report["failure_count"], 5)

        by_id = {f["record_id"]: f for f in report["failures"]}
        self.assertEqual(by_id["rec_00"]["field"], "group_id")
        self.assertEqual(by_id["rec_01"]["field"], "x1")
        self.assertEqual(by_id["rec_02"]["field"], "x2")
        self.assertEqual(by_id["rec_04"]["field"], "label")
        self.assertEqual(by_id["rec_05"]["field"], "x1")


if __name__ == "__main__":
    unittest.main()
