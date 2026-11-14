"""Unit tests for R41-T038: Mobile price range error analysis."""

import unittest

from pipeline import (
    ClassificationErrorAnalyzer,
    build_testable_predictions_with_injected_error,
    load_fixture,
)


class TestMobilePriceErrorAnalysis(unittest.TestCase):
    """Verifies error reporting and injected misclassification detection for Task R41-T038."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.split = self.fixture["metadata"]["stratified_split"]
        self.analyzer = ClassificationErrorAnalyzer()

    def test_fixture_structure_and_stratified_split(self) -> None:
        """Verify 8 rows with x1 in [0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.records), 8)
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        for idx, r in enumerate(self.records):
            self.assertEqual(r["label"], expected_labels[idx])
            self.assertAlmostEqual(r["x1"], float(idx))
            self.assertAlmostEqual(r["x2"], 1.0)

        # Stratified balance check
        id_to_label = {r["record_id"]: r["label"] for r in self.records}
        for split_name, ids in self.split.items():
            labels = [id_to_label[i] for i in ids]
            # Each split must contain equal number of class 0 and class 1
            self.assertEqual(labels.count(0), labels.count(1))

    def test_injected_wrong_prediction_appears_in_report(self) -> None:
        """Verify injected wrong prediction (c7) appears with correct record ID and category."""
        predictions = build_testable_predictions_with_injected_error(self.records, injected_wrong_id="c7")
        report = self.analyzer.analyze(self.records, predictions)

        self.assertEqual(report.error_count, 1)
        self.assertEqual(len(report.error_records), 1)

        err = report.error_records[0]
        self.assertEqual(err.record_id, "c7")
        self.assertEqual(err.expected_value, 1)
        self.assertEqual(err.predicted_value, 0)
        self.assertIn("false_negative", err.reason_category)

    def test_correct_predictions_do_not_appear_in_error_list(self) -> None:
        """Verify correct predictions (c1..c6, c8) are excluded from detailed error records."""
        predictions = build_testable_predictions_with_injected_error(self.records, injected_wrong_id="c7")
        report = self.analyzer.analyze(self.records, predictions)

        error_ids = {e.record_id for e in report.error_records}
        self.assertNotIn("c1", error_ids)
        self.assertNotIn("c2", error_ids)
        self.assertNotIn("c3", error_ids)
        self.assertNotIn("c4", error_ids)
        self.assertNotIn("c5", error_ids)
        self.assertNotIn("c6", error_ids)
        self.assertNotIn("c8", error_ids)

    def test_aggregate_counts_match_detailed_rows(self) -> None:
        """Verify aggregate metrics exactly align with detailed error rows."""
        predictions = build_testable_predictions_with_injected_error(self.records, injected_wrong_id="c7")
        report = self.analyzer.analyze(self.records, predictions)

        self.assertEqual(report.total_records, 8)
        self.assertEqual(report.correct_count, 7)
        self.assertEqual(report.error_count, len(report.error_records))
        self.assertEqual(report.correct_count + report.error_count, report.total_records)
        self.assertAlmostEqual(report.error_rate, 1.0 / 8.0)
        self.assertEqual(sum(report.category_counts.values()), report.error_count)


if __name__ == "__main__":
    unittest.main()
