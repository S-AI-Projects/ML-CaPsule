"""Unit tests for input perturbation and robustness checks (Task R41-T019)."""

import unittest
from pipeline import RobustLinearRegressionInterface, load_fixture, run_pipeline


class TestInputPerturbationRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.records = self.payload["records"]
        self.interface = RobustLinearRegressionInterface(w1=2.0, w2=1.0)

    def test_record_order_perturbation_preserves_id_aligned_predictions(self) -> None:
        result = run_pipeline()
        assessment = result["perturbation_assessment"]

        # Perturbation order changed
        self.assertNotEqual(assessment["canonical_order"], assessment["perturbed_order"])

        # Preserved under stable ID alignment
        self.assertTrue(assessment["is_preserved"])
        self.assertLessEqual(assessment["max_id_aligned_diff"], 1e-6)

    def test_formula_correctness_across_fixture(self) -> None:
        preds = self.interface.predict_records(self.records)
        for r in self.records:
            expected = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(preds[r["record_id"]], expected, places=4)

    def test_unsupported_inputs_raise_explicit_errors(self) -> None:
        # 1. Missing field 'x2'
        bad_record_missing = {"record_id": "bad1", "x1": 1.0}
        with self.assertRaisesRegex(ValueError, "missing field: x2"):
            self.interface.predict_one(bad_record_missing)

        # 2. Non-numeric type
        bad_record_type = {"record_id": "bad2", "x1": "invalid_string", "x2": 1.0}
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            self.interface.predict_one(bad_record_type)

        # 3. Missing record_id
        bad_record_no_id = {"x1": 1.0, "x2": 1.0}
        with self.assertRaisesRegex(ValueError, "missing required 'record_id'"):
            self.interface.predict_one(bad_record_no_id)

    def test_deliberate_perturbation_mismatch_fails_assertion(self) -> None:
        # Deliberately modifying a feature value must alter the aligned result
        corrupted_record = [{"record_id": "r1", "x1": 99.0, "x2": 1.0}]
        assessment = self.interface.predict_with_permutation_alignment(
            [self.records[0]], corrupted_record
        )
        self.assertFalse(assessment["is_preserved"])


if __name__ == "__main__":
    unittest.main()
