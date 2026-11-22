"""Unit tests for R41-T028 (Twitter airline sentiment – Naive Bayes error analysis).

Task: R41-T028 · 8 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

Run offline (CPU-only, Python 3 standard library only):
    python3 -m unittest discover -s course-labs/r41-t028 -p "test_*.py" -v

All tests use six fictional text records (t1..t6) strictly as a pipeline
verification fixture — NOT a real Twitter airline sentiment corpus and NOT
evidence of real-world or clinical performance.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pipeline  # noqa: E402

EXPECTED_CLEAN_PREDICTIONS = {
    "t1": 1,
    "t2": 1,
    "t3": 0,
    "t4": 0,
    "t5": 1,
    "t6": 0,
}
EXPECTED_INJECTED_ERROR_ID = "t5"
EXPECTED_INJECTED_ERROR_COUNT = 1


class TestSyntheticTextFixtureContract(unittest.TestCase):
    """Verify the literal six fictional records and schema contract."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.schema_doc = pipeline.load_schema()
        self.records = self.fixture_doc["records"]

    def test_explicit_synthetic_fixture_labelling(self) -> None:
        self.assertTrue(self.fixture_doc["is_synthetic"])
        self.assertIn("not a real", self.fixture_doc["disclaimer"].lower())
        self.assertIn("pipeline", self.fixture_doc["disclaimer"].lower())
        self.assertTrue(self.schema_doc["synthetic_only"])

    def test_literal_six_records_texts_and_labels(self) -> None:
        self.assertEqual(len(self.records), 6)
        self.assertEqual([r["record_id"] for r in self.records], ["t1", "t2", "t3", "t4", "t5", "t6"])
        self.assertEqual(
            [r["text"] for r in self.records],
            ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"],
        )
        self.assertEqual([r["label"] for r in self.records], [1, 1, 0, 0, 1, 0])

    def test_schema_declares_required_fields_and_reason_categories(self) -> None:
        self.assertEqual(
            tuple(self.schema_doc["required_fields"]),
            pipeline.REQUIRED_RECORD_FIELDS,
        )
        self.assertEqual(
            set(self.schema_doc["reason_categories"]),
            {cat.value for cat in pipeline.ErrorReasonCategory},
        )


class TestExplicitTrainValTestAndNaiveBayes(unittest.TestCase):
    """Verify explicit train/validation/test separation and Naive Bayes baseline."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.partitions = pipeline.validate_and_partition_fixture(self.fixture_doc["records"])
        self.model = pipeline.MultinomialNaiveBayes(alpha=1.0, seed=pipeline.FIXED_SEED).fit(
            self.partitions.train
        )

    def test_disjoint_train_val_test_splits(self) -> None:
        self.assertEqual([r["record_id"] for r in self.partitions.train], ["t1", "t2", "t3", "t4"])
        self.assertEqual([r["record_id"] for r in self.partitions.validation], ["t5"])
        self.assertEqual([r["record_id"] for r in self.partitions.test], ["t6"])

    def test_vocabulary_comes_from_train_only(self) -> None:
        self.assertEqual(
            self.model.vocabulary_,
            frozenset({"red", "apple", "green", "blue", "sky", "dark"}),
        )
        # Held-out tokens 'fruit' (t5) and 'cloud' (t6) must NOT leak into training vocab
        self.assertNotIn("fruit", self.model.vocabulary_)
        self.assertNotIn("cloud", self.model.vocabulary_)

    def test_clean_predictions_match_expected(self) -> None:
        all_records = self.partitions.train + self.partitions.validation + self.partitions.test
        preds = self.model.predict_records(all_records)
        self.assertEqual(preds, EXPECTED_CLEAN_PREDICTIONS)


class TestErrorAnalysisReportAndInjectedError(unittest.TestCase):
    """Verify injected wrong prediction appears, correct ones do not, and aggregates reconcile."""

    def setUp(self) -> None:
        self.result = pipeline.run_error_analysis_pipeline(
            injected_wrong_id=EXPECTED_INJECTED_ERROR_ID
        )

    def test_clean_held_out_report_has_zero_errors(self) -> None:
        clean = self.result["clean_held_out_report"]
        self.assertEqual(clean["total_evaluated"], 2)
        self.assertEqual(clean["correct_count"], 2)
        self.assertEqual(clean["error_count"], 0)
        self.assertEqual(clean["error_rows"], [])
        self.assertEqual(clean["counts_by_reason"], {})
        self.assertEqual(clean["counts_by_split"], {})

    def test_injected_wrong_prediction_appears_and_correct_prediction_does_not(self) -> None:
        held_out = self.result["injected_held_out_report"]
        self.assertEqual(held_out["total_evaluated"], 2)
        self.assertEqual(held_out["correct_count"], 1)
        self.assertEqual(held_out["error_count"], EXPECTED_INJECTED_ERROR_COUNT)
        self.assertEqual(len(held_out["error_rows"]), EXPECTED_INJECTED_ERROR_COUNT)

        row = held_out["error_rows"][0]
        self.assertEqual(row["record_id"], "t5")
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["expected_value"], 1)
        self.assertEqual(row["predicted_value"], 0)
        self.assertEqual(row["reason_category"], "injected_test_error")
        self.assertEqual(row["oov_tokens"], ["fruit"])

        # Correct prediction t6 must NOT appear in the error report
        reported_ids = {r["record_id"] for r in held_out["error_rows"]}
        self.assertIn("t5", reported_ids)
        self.assertNotIn("t6", reported_ids)

    def test_full_fixture_report_excludes_all_five_correct_predictions(self) -> None:
        full_rep = self.result["injected_all_records_report"]
        self.assertEqual(full_rep["total_evaluated"], 6)
        self.assertEqual(full_rep["correct_count"], 5)
        self.assertEqual(full_rep["error_count"], 1)
        self.assertEqual([r["record_id"] for r in full_rep["error_rows"]], ["t5"])
        for correct_id in ("t1", "t2", "t3", "t4", "t6"):
            self.assertNotIn(correct_id, [r["record_id"] for r in full_rep["error_rows"]])

    def test_aggregate_counts_match_detailed_rows(self) -> None:
        for report_key in ("injected_held_out_report", "injected_all_records_report"):
            rep = self.result[report_key]
            rows = rep["error_rows"]
            self.assertEqual(rep["error_count"], len(rows))
            self.assertEqual(rep["correct_count"] + rep["error_count"], rep["total_evaluated"])
            self.assertEqual(
                rep["false_positive_count"] + rep["false_negative_count"],
                rep["error_count"],
            )
            self.assertEqual(sum(rep["counts_by_reason"].values()), len(rows))
            self.assertEqual(sum(rep["counts_by_split"].values()), len(rows))
            self.assertEqual(rep["false_positive_count"], 0)
            self.assertEqual(rep["false_negative_count"], 1)
            self.assertEqual(rep["counts_by_reason"], {"injected_test_error": 1})
            self.assertEqual(rep["counts_by_split"], {"validation": 1})

    def test_natural_error_reason_categories_and_markdown_output(self) -> None:
        fixture_doc = pipeline.load_fixture()
        partitions = pipeline.validate_and_partition_fixture(fixture_doc["records"])
        model = pipeline.MultinomialNaiveBayes().fit(partitions.train)
        all_recs = partitions.train + partitions.validation + partitions.test

        # Simulate three distinct natural errors (no injected_error_ids):
        # t1 (in-vocab expected 1 -> 0: false_negative_lexical)
        # t3 (in-vocab expected 0 -> 1: false_positive_lexical)
        # t6 (has OOV 'cloud', expected 0 -> 1: oov_token_shift)
        preds = dict(EXPECTED_CLEAN_PREDICTIONS)
        preds["t1"] = 0
        preds["t3"] = 1
        preds["t6"] = 1
        rep = pipeline.build_error_analysis_report(
            all_recs,
            preds,
            train_vocab=model.vocabulary_,
        )
        self.assertEqual(rep.error_count, 3)
        self.assertEqual(rep.correct_count, 3)
        self.assertEqual(rep.false_positive_count, 2)
        self.assertEqual(rep.false_negative_count, 1)
        self.assertEqual(
            rep.counts_by_reason,
            {
                "false_negative_lexical": 1,
                "false_positive_lexical": 1,
                "oov_token_shift": 1,
            },
        )
        md = rep.to_markdown()
        self.assertIn("`t1`", md)
        self.assertIn("`t6`", md)
        self.assertNotIn("`t2`", md)


class TestFailureCasesAndGuards(unittest.TestCase):
    """Verify failure handling for invalid records, leakage, and report mismatches."""

    def setUp(self) -> None:
        self.fixture_doc = pipeline.load_fixture()
        self.records = self.fixture_doc["records"]
        self.partitions = pipeline.validate_and_partition_fixture(self.records)

    def test_non_train_record_passed_to_fit_rejected(self) -> None:
        leaked = list(self.partitions.train) + [dict(self.partitions.validation[0])]
        with self.assertRaises(pipeline.SplitContaminationError):
            pipeline.MultinomialNaiveBayes().fit(leaked)

    def test_overlapping_record_id_across_splits_rejected(self) -> None:
        contaminated = copy.deepcopy(self.records)
        contaminated.append({"record_id": "t1", "text": "blue sky", "label": 0, "split": "test"})
        with self.assertRaises(pipeline.SplitContaminationError):
            pipeline.validate_and_partition_fixture(contaminated)

    def test_missing_or_extra_prediction_ids_rejected(self) -> None:
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.build_error_analysis_report(self.partitions.validation, {})
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.build_error_analysis_report(
                self.partitions.validation,
                {"t5": 1, "t99": 0},
            )

    def test_invalid_labels_empty_text_and_unfitted_predict_rejected(self) -> None:
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.build_error_analysis_report(self.partitions.validation, {"t5": True})
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.tokenize("   ")
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.inject_deliberate_wrong_prediction(EXPECTED_CLEAN_PREDICTIONS, "t99")
        with self.assertRaises(pipeline.ModelNotFittedError):
            pipeline.MultinomialNaiveBayes().predict_records(self.partitions.test)

    def test_error_row_rejects_correct_prediction(self) -> None:
        with self.assertRaises(pipeline.SchemaValidationError):
            pipeline.ErrorRow(
                record_id="t6",
                split="test",
                text="blue cloud",
                expected_label=0,
                predicted_label=0,
                reason_category=pipeline.ErrorReasonCategory.INJECTED_TEST_ERROR,
                oov_tokens=("cloud",),
            )


class TestDeliberateMutationGuard(unittest.TestCase):
    """Prove assertions fail when an expected error count or ID is deliberately changed."""

    def test_mutated_expected_error_count_fails_assertion(self) -> None:
        res = pipeline.run_error_analysis_pipeline()
        actual_count = res["injected_held_out_report"]["error_count"]
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_count, 99)


if __name__ == "__main__":
    unittest.main()
