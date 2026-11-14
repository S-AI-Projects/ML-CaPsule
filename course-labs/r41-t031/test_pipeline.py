"""Unit tests for R41-T031: sklearn digits - PCA + SVM data contract and projection."""

import copy
import math
from pathlib import Path
import unittest

from pipeline import load_fixture, project_and_reconstruct, validate_records


class TestDigitsPCADataContract(unittest.TestCase):
    """Verifies schema validation and PCA projection contract for Task R41-T031."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]

    def test_valid_synthetic_fixture_passes(self) -> None:
        report = validate_records(self.records)
        self.assertTrue(report["valid"], f"Expected valid fixture, got: {report['failures']}")
        self.assertEqual(report["total_records_checked"], 4)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(len(report["failures"]), 0)

    def test_missing_record_id_fails_without_dropping_records(self) -> None:
        bad_records = copy.deepcopy(self.records)
        del bad_records[0]["record_id"]
        bad_records[1]["record_id"] = None

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 4)
        self.assertEqual(report["failure_count"], 2)
        self.assertTrue(all(f["field"] == "record_id" for f in report["failures"]))

    def test_duplicate_record_id_fails_with_specific_reason(self) -> None:
        bad_records = copy.deepcopy(self.records)
        bad_records[2]["record_id"] = "p1"  # duplicate with p1

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 4)
        self.assertEqual(report["failure_count"], 1)
        failure = report["failures"][0]
        self.assertEqual(failure["record_id"], "p1")
        self.assertEqual(failure["field"], "record_id")
        self.assertIn("duplicate", failure["reason"].lower())

    def test_malformed_numeric_text_and_tensor_data_fail_specifically(self) -> None:
        bad_records = copy.deepcopy(self.records)
        bad_records[0]["record_id"] = "   "  # whitespace-only string
        bad_records[1]["x1"] = "not_numeric"  # string instead of float
        bad_records[2]["x2"] = [1.0, 2.0]  # tensor / list instead of scalar
        bad_records[3]["x1"] = float("nan")  # NaN numeric

        report = validate_records(bad_records)
        self.assertFalse(report["valid"])
        self.assertEqual(report["total_records_checked"], 4)
        self.assertEqual(report["failure_count"], 4)

        reasons_by_field = {f["field"]: f["reason"] for f in report["failures"]}
        self.assertIn("record_id", reasons_by_field)
        self.assertIn("x1", reasons_by_field)
        self.assertIn("x2", reasons_by_field)

    def test_collinear_points_projection_and_reconstruction_within_tolerance(self) -> None:
        points = [(r["x1"], r["x2"]) for r in self.records]
        results = project_and_reconstruct(points, tolerance=1e-6)

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r["recovered"], f"Point {r['original']} failed reconstruction with error {r['reconstruction_error']}")
            self.assertLessEqual(r["reconstruction_error"], 1e-6)

    def test_sign_reversal_of_learned_component_preserves_reconstruction(self) -> None:
        points = [(r["x1"], r["x2"]) for r in self.records]
        u = 1.0 / math.sqrt(2.0)
        # Reversed sign unit vector (-1/sqrt(2), -1/sqrt(2))
        reversed_u = (-u, -u)
        results = project_and_reconstruct(points, unit_vector=reversed_u, tolerance=1e-6)

        for r in results:
            self.assertTrue(r["recovered"])
            self.assertLessEqual(r["reconstruction_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
