"""Unit tests for R41-T034: WHO life expectancy regression evaluation metric verification."""

import math
import unittest

from pipeline import (
    compute_mae,
    compute_mape,
    cross_check_mae,
    load_fixture,
)


class TestWHOLifeExpectancyMetric(unittest.TestCase):
    """Verifies evaluation metric correctness and zero-denominator behavior for Task R41-T034."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]

    def test_canonical_regression_mae_specification(self) -> None:
        """Verify the exact canonical case: truth=[1,2,3], prediction=[1,3,2] -> MAE=2/3."""
        truth = [1.0, 2.0, 3.0]
        prediction = [1.0, 3.0, 2.0]
        mae = compute_mae(truth, prediction)

        expected_mae = 2.0 / 3.0
        self.assertAlmostEqual(mae, expected_mae, places=10)
        self.assertAlmostEqual(mae, 0.6666666666666666, places=10)

    def test_deterministic_fixture_formula_and_splits(self) -> None:
        """Verify fixture conforms to target = 2 * x1 + 1 with stable IDs."""
        self.assertEqual(len(self.records), 6)
        for r in self.records:
            expected = 2.0 * r["x1"] + 1.0
            self.assertAlmostEqual(r["target"], expected, places=6)

    def test_independent_cross_check_consistency(self) -> None:
        """Verify independent implementation matches custom MAE within numerical tolerance."""
        truth = [r["target"] for r in self.records]
        # Simulate baseline predictions of 4.0
        prediction = [4.0 for _ in self.records]

        result = cross_check_mae(truth, prediction)
        self.assertTrue(result["matches_within_tolerance"])
        self.assertLess(result["discrepancy"], 1e-12)

    def test_zero_denominator_behavior_in_mape(self) -> None:
        """Verify explicit handling of zero denominator does not divide by zero or raise NaN."""
        truth_with_zero = [0.0, 2.0]
        prediction = [0.5, 2.0]

        # Using epsilon smoothing
        mape_eps = compute_mape(truth_with_zero, prediction, zero_denominator_strategy="epsilon")
        self.assertTrue(math.isfinite(mape_eps))

        # Using skip strategy
        mape_skip = compute_mape(truth_with_zero, prediction, zero_denominator_strategy="skip")
        self.assertAlmostEqual(mape_skip, 0.0, places=6)

    def test_invalid_shapes_raise_value_error(self) -> None:
        """Verify mismatched array lengths or empty inputs raise clear ValueError."""
        with self.assertRaises(ValueError):
            compute_mae([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            compute_mae([], [])


if __name__ == "__main__":
    unittest.main()
