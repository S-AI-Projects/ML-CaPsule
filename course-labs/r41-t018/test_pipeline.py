"""Unit tests for error analysis report and injected failure detection (Task R41-T018)."""

import unittest
from pipeline import SimpleThresholdDecisionTree, build_error_report, load_fixture, run_pipeline


class TestTitanicErrorAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.records = self.payload["records"]
        self.by_id = {r["record_id"]: r for r in self.records}

        self.test_ids = self.payload["metadata"]["stratified_split"]["test"]
        self.test_records = [self.by_id[i] for i in self.test_ids]
        self.model = SimpleThresholdDecisionTree(threshold=3.5)

    def test_injected_wrong_prediction_appears_in_report(self) -> None:
        result = run_pipeline()
        report = result["error_analysis_report"]

        # Aggregate error count matches detailed rows count exactly
        self.assertEqual(report["error_count"], len(report["detailed_errors"]))
        self.assertEqual(report["error_count"], 1)

        # Injected record 'r4' must be present with expected values
        error_entry = report["detailed_errors"][0]
        self.assertEqual(error_entry["record_id"], "r4")
        self.assertEqual(error_entry["expected"], 0)
        self.assertEqual(error_entry["predicted"], 1)
        self.assertEqual(error_entry["reason_category"], "BOUNDARY_AMBIGUITY")

    def test_correct_prediction_does_not_appear_in_report(self) -> None:
        result = run_pipeline()
        report = result["error_analysis_report"]
        reported_ids = {entry["record_id"] for entry in report["detailed_errors"]}

        # Correctly classified instance 'r8' must NOT appear in errors
        self.assertNotIn("r8", reported_ids)

    def test_clean_predictions_produce_empty_error_report(self) -> None:
        clean_preds = self.model.predict(self.test_records)
        report = build_error_report(records=self.test_records, predictions=clean_preds)

        self.assertEqual(report["error_count"], 0)
        self.assertEqual(len(report["detailed_errors"]), 0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_deliberate_mismatch_fails_assertion(self) -> None:
        result = run_pipeline()
        report = result["error_analysis_report"]

        # Intentionally test that expecting 0 errors on the injected report raises AssertionError
        with self.assertRaises(AssertionError):
            self.assertEqual(report["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
