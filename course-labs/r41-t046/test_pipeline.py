"""Offline tests for the R41-T046 simulated classifier smoke test."""

import copy
import json
import math
from pathlib import Path
import unittest

from pipeline import (
    create_stratified_split,
    evaluate,
    fit_majority_baseline,
    load_fixture,
    run_pipeline,
    validate_records,
    validate_split,
)


HERE = Path(__file__).parent
FIXTURE = load_fixture()
RECORDS = FIXTURE["records"]
EXPECTED_SPLIT = {
    "train": ["credit-sim-003", "credit-sim-004", "credit-sim-008", "credit-sim-005"],
    "validation": ["credit-sim-002", "credit-sim-007"],
    "test": ["credit-sim-001", "credit-sim-006"],
}


class ClassificationSmokeTests(unittest.TestCase):
    def test_literal_simulated_fixture_and_schema(self):
        schema = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
        self.assertEqual(FIXTURE["data_kind"], "simulated_pipeline_fixture")
        self.assertEqual(
            [
                (row["record_id"], row["group_id"], row["x1"], row["x2"], row["label"])
                for row in RECORDS
            ],
            [
                (
                    f"credit-sim-{index:03}",
                    f"credit-group-{index:03}",
                    index - 1,
                    1,
                    0 if index <= 4 else 1,
                )
                for index in range(1, 9)
            ],
        )
        self.assertEqual(schema["required"], ["record_id", "group_id", "x1", "x2", "label"])

    def test_seeded_split_is_stratified_and_repeatable(self):
        self.assertEqual(create_stratified_split(RECORDS, seed=46), EXPECTED_SPLIT)
        self.assertEqual(
            create_stratified_split(RECORDS, seed=46),
            create_stratified_split(RECORDS, seed=46),
        )
        validate_split(RECORDS, EXPECTED_SPLIT)

    def test_pipeline_outputs_have_expected_shape_values_and_steps(self):
        result = run_pipeline(FIXTURE)
        self.assertEqual(result["model"], "training_majority_constant_baseline")
        self.assertEqual(result["update_steps"], 0)
        self.assertEqual(result["split_record_ids"], EXPECTED_SPLIT)
        for split_name, expected_count in (("train", 4), ("validation", 2), ("test", 2)):
            output = result["outputs"][split_name]
            self.assertEqual(output["predictions"], [0] * expected_count)
            self.assertEqual(output["probabilities"], [0.5] * expected_count)
            self.assertEqual(output["prediction_shape"], [expected_count])
            self.assertEqual(output["probability_shape"], [expected_count])
            self.assertAlmostEqual(output["loss"], 0.6931471805599453)
        self.assertEqual(result["outputs"]["validation"]["prediction_shape"], [2])
        self.assertEqual(result["outputs"]["test"]["probability_shape"], [2])
        self.assertTrue(
            all(
                isinstance(value, float) and math.isfinite(value)
                for split in result["outputs"].values()
                for value in split["probabilities"] + [split["loss"]]
            )
        )

    def test_training_majority_baseline_uses_only_training_rows(self):
        training_ids = {row["record_id"] for row in RECORDS}
        training = list(RECORDS)
        model = fit_majority_baseline(training)
        result = evaluate(model, [RECORDS[0]], training_ids - {"credit-sim-001"})
        self.assertEqual(model.predicted_label, 0)
        self.assertEqual(model.positive_probability, 0.5)
        self.assertEqual(model.update_steps, 0)
        self.assertEqual(result["predictions"], [0])
        self.assertAlmostEqual(result["loss"], 0.6931471805599453)

    def test_rejects_record_overlap_and_group_contamination(self):
        with self.assertRaisesRegex(ValueError, "overlap training"):
            evaluate(fit_majority_baseline(RECORDS), [RECORDS[0]], {"credit-sim-001"})

        contaminated_split = copy.deepcopy(EXPECTED_SPLIT)
        contaminated_split["test"][0] = contaminated_split["train"][0]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_split(RECORDS, contaminated_split)

        contaminated_records = copy.deepcopy(RECORDS)
        held_out = next(row for row in contaminated_records if row["record_id"] == EXPECTED_SPLIT["test"][0])
        training = next(row for row in contaminated_records if row["record_id"] == EXPECTED_SPLIT["train"][0])
        held_out["group_id"] = training["group_id"]
        with self.assertRaisesRegex(ValueError, "group_id appears in multiple splits"):
            validate_split(contaminated_records, EXPECTED_SPLIT)

    def test_rejects_invalid_labels_and_missing_features(self):
        invalid_label = copy.deepcopy(RECORDS)
        invalid_label[0]["label"] = True
        with self.assertRaisesRegex(ValueError, "label must be integer"):
            validate_records(invalid_label)

        missing = copy.deepcopy(RECORDS)
        del missing[0]["x1"]
        with self.assertRaisesRegex(ValueError, "contain exactly"):
            validate_records(missing)

        non_finite = copy.deepcopy(RECORDS)
        non_finite[0]["x2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid or missing x2"):
            validate_records(non_finite)

    def test_rejects_oversized_input_and_wrong_fixture_marker(self):
        with self.assertRaisesRegex(ValueError, "must not exceed 32"):
            validate_records(RECORDS * 5)
        invalid = copy.deepcopy(FIXTURE)
        invalid["data_kind"] = "real_taiwan_credit_data"
        with self.assertRaisesRegex(ValueError, "marked as simulated"):
            run_pipeline(invalid)

    def test_rejects_training_data_without_both_classes(self):
        one_class = [dict(row, label=0) for row in RECORDS[:3]]
        with self.assertRaisesRegex(ValueError, "contain both labels"):
            fit_majority_baseline(one_class)

if __name__ == "__main__":
    unittest.main()
