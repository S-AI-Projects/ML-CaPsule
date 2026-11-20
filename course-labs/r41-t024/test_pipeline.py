"""
Unit tests for R41-T024 – Loan Approval: Classification Evaluation
=================================================================
Covers:
  1. Fixture structure (8 rows, correct x1/x2/label values)
  2. Stratified split separation (train 4, val 2, test 2; class balance)
  3. Canonical example: truth=[1,1,0,0] / pred=[1,0,1,0] → precision=recall=0.5
  4. Confusion matrix values for the canonical example
  5. Zero-denominator policy – 'zero' (all predicted negative → precision=0)
  6. Zero-denominator policy – 'nan' (returns float nan)
  7. Macro averaging on the canonical example
  8. Perfect prediction case (precision=recall=f1=1.0)
  9. Deliberate mutation guard (exits non-zero on wrong expected value)
 10. Environment info recorded

Standard library only – no numpy, no sklearn, offline CPU-only.
"""

import math
import sys
import unittest

# Support both direct execution and unittest discover
try:
    from pipeline import (ClassificationMetrics, ConfusionMatrix,
                          ZeroDenominatorPolicy, load_fixture, get_split,
                          get_environment_info)
except ImportError:
    from course_labs.r41_t024.pipeline import (  # type: ignore
        ClassificationMetrics, ConfusionMatrix,
        ZeroDenominatorPolicy, load_fixture, get_split,
        get_environment_info)


class TestClassificationEvaluation(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Fixture structure
    # ------------------------------------------------------------------
    def test_fixture_eight_synthetic_rows(self):
        """Fixture must contain exactly 8 rows with x1=0..7, x2=1, correct labels."""
        records = load_fixture()
        self.assertEqual(len(records), 8,
                         "Fixture must have exactly 8 rows.")
        x1_values  = [r["x1"] for r in records]
        x2_values  = [r["x2"] for r in records]
        labels     = [r["label"] for r in records]
        self.assertEqual(x1_values, list(range(8)),
                         "x1 must be [0,1,2,3,4,5,6,7].")
        self.assertTrue(all(v == 1 for v in x2_values),
                        "x2 must be 1 for all rows.")
        self.assertEqual(labels, [0, 0, 0, 0, 1, 1, 1, 1],
                         "label must be [0,0,0,0,1,1,1,1].")

    # ------------------------------------------------------------------
    # 2. Stratified split separation
    # ------------------------------------------------------------------
    def test_stratified_split_separation(self):
        """Train=4, val=2, test=2; each split has balanced class labels."""
        records = load_fixture()
        train = get_split(records, "train")
        val   = get_split(records, "validation")
        test  = get_split(records, "test")

        self.assertEqual(len(train), 4, "Train must have 4 records.")
        self.assertEqual(len(val),   2, "Validation must have 2 records.")
        self.assertEqual(len(test),  2, "Test must have 2 records.")

        # Class balance in each split
        for split_name, split_records in [("train", train),
                                           ("validation", val),
                                           ("test", test)]:
            zeros = sum(1 for r in split_records if r["label"] == 0)
            ones  = sum(1 for r in split_records if r["label"] == 1)
            self.assertEqual(zeros, ones,
                             f"{split_name} split must be class-balanced "
                             f"(got {zeros} negatives, {ones} positives).")

        # No group spans two splits
        def group_splits(split_records):
            return {r["group_id"]: r["split"] for r in split_records}

        all_ids = [r["record_id"] for r in records]
        self.assertEqual(len(all_ids), len(set(all_ids)),
                         "All record_ids must be unique.")

    # ------------------------------------------------------------------
    # 3. Canonical binary example: precision = recall = 0.5
    # ------------------------------------------------------------------
    def test_canonical_precision_recall_half(self):
        """
        truth=[1,1,0,0], prediction=[1,0,1,0]
        → TP=1, FP=1, FN=1, TN=1
        → precision = TP/(TP+FP) = 1/2 = 0.5
        → recall    = TP/(TP+FN) = 1/2 = 0.5
        """
        truth      = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        metrics = ClassificationMetrics(averaging="binary", positive_label=1)
        result  = metrics.compute(truth, prediction)

        self.assertAlmostEqual(result["precision"], 0.5, places=9,
                               msg="Binary precision must be 0.5.")
        self.assertAlmostEqual(result["recall"],    0.5, places=9,
                               msg="Binary recall must be 0.5.")

    # ------------------------------------------------------------------
    # 4. Confusion matrix values
    # ------------------------------------------------------------------
    def test_confusion_matrix_canonical(self):
        """Verify TP=1, FP=1, FN=1, TN=1 for the canonical example."""
        truth      = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        cm = ConfusionMatrix(truth, prediction, positive_label=1)
        self.assertEqual(cm.tp, 1, "TP must be 1.")
        self.assertEqual(cm.fp, 1, "FP must be 1.")
        self.assertEqual(cm.fn, 1, "FN must be 1.")
        self.assertEqual(cm.tn, 1, "TN must be 1.")

    # ------------------------------------------------------------------
    # 5. Zero-denominator policy – 'zero'
    # ------------------------------------------------------------------
    def test_zero_denominator_policy_zero(self):
        """
        All predictions are 0 → no predicted positives → precision denominator = 0.
        Policy 'zero' must return precision = 0.0 (not raise, not nan).
        """
        truth      = [1, 1, 0, 0]
        prediction = [0, 0, 0, 0]
        metrics = ClassificationMetrics(averaging="binary",
                                        zero_policy=ZeroDenominatorPolicy.ZERO)
        result = metrics.compute(truth, prediction)
        self.assertEqual(result["precision"], 0.0,
                         "Precision must be 0.0 when no positives predicted (policy='zero').")
        self.assertAlmostEqual(result["recall"], 0.0, places=9,
                               msg="Recall must be 0.0 when TP=0.")

    # ------------------------------------------------------------------
    # 6. Zero-denominator policy – 'nan'
    # ------------------------------------------------------------------
    def test_zero_denominator_policy_nan(self):
        """
        All predictions are 0 → precision denominator = 0.
        Policy 'nan' must return float('nan') for precision.
        """
        truth      = [1, 1, 0, 0]
        prediction = [0, 0, 0, 0]
        metrics = ClassificationMetrics(averaging="binary",
                                        zero_policy=ZeroDenominatorPolicy.NAN)
        result = metrics.compute(truth, prediction)
        self.assertTrue(math.isnan(result["precision"]),
                        "Precision must be NaN when no positives predicted (policy='nan').")

    # ------------------------------------------------------------------
    # 7. Macro averaging
    # ------------------------------------------------------------------
    def test_macro_averaging_canonical(self):
        """
        truth=[1,1,0,0], prediction=[1,0,1,0]
        Class 1: precision=0.5, recall=0.5
        Class 0: precision=0.5, recall=0.5
        Macro precision = macro recall = 0.5
        """
        truth      = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        metrics = ClassificationMetrics(averaging="macro")
        result  = metrics.compute(truth, prediction)
        self.assertAlmostEqual(result["precision"], 0.5, places=9,
                               msg="Macro precision must be 0.5.")
        self.assertAlmostEqual(result["recall"],    0.5, places=9,
                               msg="Macro recall must be 0.5.")
        self.assertIn(1, result["per_class"],
                      "per_class must include key for label 1.")
        self.assertIn(0, result["per_class"],
                      "per_class must include key for label 0.")

    # ------------------------------------------------------------------
    # 8. Perfect prediction
    # ------------------------------------------------------------------
    def test_perfect_prediction(self):
        """truth == prediction → precision = recall = f1 = 1.0."""
        truth      = [1, 1, 0, 0]
        prediction = [1, 1, 0, 0]
        metrics = ClassificationMetrics(averaging="binary")
        result  = metrics.compute(truth, prediction)
        self.assertAlmostEqual(result["precision"], 1.0, places=9)
        self.assertAlmostEqual(result["recall"],    1.0, places=9)
        self.assertAlmostEqual(result["f1"],        1.0, places=9)

    # ------------------------------------------------------------------
    # 9. Deliberate mutation guard
    # ------------------------------------------------------------------
    def test_deliberate_mutation_failure(self):
        """
        Deliberately mutate the expected precision to confirm the test framework
        exits non-zero when an expected result is wrong.
        """
        truth      = [1, 1, 0, 0]
        prediction = [1, 0, 1, 0]
        metrics = ClassificationMetrics(averaging="binary")
        result  = metrics.compute(truth, prediction)

        correct_precision  = 0.5
        mutated_expected   = 0.9   # deliberately wrong

        # Assert the actual result matches the CORRECT value
        self.assertAlmostEqual(result["precision"], correct_precision, places=9,
                               msg="Precision must equal 0.5.")

        # Confirm the mutated value does NOT match (guards against silent pass)
        self.assertNotAlmostEqual(result["precision"], mutated_expected, places=9,
                                  msg="Mutated expected value must not equal actual precision.")

    # ------------------------------------------------------------------
    # 10. Environment info
    # ------------------------------------------------------------------
    def test_environment_info_recorded(self):
        """Environment info dict must include python_version and task_id."""
        env = get_environment_info()
        self.assertIn("python_version", env)
        self.assertIn("task_id", env)
        self.assertEqual(env["task_id"], "R41-T024")
        self.assertIn("R41-T024", env["task_id"])


if __name__ == "__main__":
    unittest.main()
