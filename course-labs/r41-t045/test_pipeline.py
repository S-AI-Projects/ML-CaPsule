"""Offline tests for the R41-T045 simulated preprocessing contract."""

import copy
import json
from pathlib import Path
import unittest

from pipeline import (
    FEATURE_ORDER,
    create_stratified_split,
    fit_preprocessor,
    load_fixture,
    run_pipeline,
    validate_feature_order,
    validate_records,
    validate_split,
)


HERE = Path(__file__).parent
FIXTURE = load_fixture()
RECORDS = FIXTURE["records"]
EXPECTED_SPLIT = {
    "train": ["iris-sim-001", "iris-sim-004", "iris-sim-008", "iris-sim-006"],
    "validation": ["iris-sim-002", "iris-sim-005"],
    "test": ["iris-sim-003", "iris-sim-007"],
}


class PreprocessingContractTests(unittest.TestCase):
    def test_literal_simulated_fixture_and_schema(self):
        schema = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
        self.assertEqual(FIXTURE["data_kind"], "simulated_pipeline_fixture")
        self.assertEqual(
            [
                (row["record_id"], row["group_id"], row["x1"], row["x2"], row["label"])
                for row in RECORDS
            ],
            [
                (f"iris-sim-{index:03}", f"iris-group-{index:03}", index - 1, 1, 0 if index <= 4 else 1)
                for index in range(1, 9)
            ],
        )
        self.assertEqual(
            schema["required"],
            ["record_id", "group_id", "x1", "x2", "label"],
        )

    def test_fixed_seed_is_stratified_and_repeatable(self):
        first = create_stratified_split(RECORDS, seed=45)
        self.assertEqual(first, EXPECTED_SPLIT)
        self.assertEqual(first, create_stratified_split(RECORDS, seed=45))
        validate_split(RECORDS, first)

    def test_fit_uses_only_training_rows_and_transform_is_fixed(self):
        training_ids = set(EXPECTED_SPLIT["train"])
        training_rows = [row for row in RECORDS if row["record_id"] in training_ids]
        held_out_rows = [row for row in RECORDS if row["record_id"] not in training_ids]
        preprocessor = fit_preprocessor(training_rows)
        original_mean = preprocessor.means
        self.assertFalse(hasattr(preprocessor, "fit"))
        transformed = preprocessor.transform(held_out_rows)

        self.assertEqual(preprocessor.means, original_mean)
        self.assertEqual(transformed["shape"], [4, 2])
        self.assertEqual(transformed["dtype"], "float64")
        self.assertTrue(all(type(value) is float for row in transformed["values"] for value in row))
        self.assertEqual(
            transformed["values"],
            [
                [-1.0634101379502299, 0.0],
                [-0.6767155423319645, 0.0],
                [0.09667364890456635, 0.0],
                [0.8700628401410971, 0.0],
            ],
        )

    def test_pipeline_output_values_shapes_and_dtype(self):
        result = run_pipeline(FIXTURE)
        self.assertEqual(result["split_record_ids"], EXPECTED_SPLIT)
        self.assertEqual(result["training_mean"], [3.75, 1.0])
        self.assertAlmostEqual(result["training_scale"][0], 2.5860201081971503)
        self.assertEqual(result["training_scale"][1], 1.0)
        self.assertEqual(result["transformed"]["train"]["shape"], [4, 2])
        self.assertEqual(result["transformed"]["validation"]["shape"], [2, 2])
        self.assertEqual(result["transformed"]["test"]["shape"], [2, 2])
        self.assertEqual(
            result["transformed"]["train"]["values"],
            [
                [-1.4501047335684953, 0.0],
                [-0.29002094671369905, 0.0],
                [1.2567574357593625, 0.0],
                [0.4833682445228318, 0.0],
            ],
        )
        self.assertEqual(
            result["transformed"]["validation"]["values"],
            [[-1.0634101379502299, 0.0], [0.09667364890456635, 0.0]],
        )
        self.assertEqual(
            result["transformed"]["test"]["values"],
            [[-0.6767155423319645, 0.0], [0.8700628401410971, 0.0]],
        )
        for matrix in result["transformed"].values():
            self.assertEqual(matrix["dtype"], "float64")
            self.assertEqual(matrix["shape"], [len(matrix["values"]), 2])
            self.assertTrue(all(type(value) is float for row in matrix["values"] for value in row))

    def test_rejects_split_with_record_contamination(self):
        contaminated = copy.deepcopy(EXPECTED_SPLIT)
        contaminated["test"][0] = contaminated["train"][0]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_split(RECORDS, contaminated)

    def test_rejects_split_with_group_contamination(self):
        contaminated_rows = copy.deepcopy(RECORDS)
        test_id = EXPECTED_SPLIT["test"][0]
        train_id = EXPECTED_SPLIT["train"][0]
        test_row = next(row for row in contaminated_rows if row["record_id"] == test_id)
        train_row = next(row for row in contaminated_rows if row["record_id"] == train_id)
        test_row["group_id"] = train_row["group_id"]
        with self.assertRaisesRegex(ValueError, "group_id appears in multiple splits"):
            validate_split(contaminated_rows, EXPECTED_SPLIT)

    def test_rejects_wrong_feature_order(self):
        with self.assertRaisesRegex(ValueError, "feature_order"):
            validate_feature_order(("x2", "x1"))
        preprocessor = fit_preprocessor(RECORDS[:4])
        with self.assertRaisesRegex(ValueError, "feature_order"):
            preprocessor.transform(RECORDS[4:], feature_order=("x2", "x1"))

    def test_rejects_invalid_label_set_for_fixture(self):
        invalid = copy.deepcopy(RECORDS)
        for row in invalid:
            row["label"] = 0
        with self.assertRaisesRegex(ValueError, "label set"):
            create_stratified_split(invalid)

    def test_rejects_missing_or_non_finite_features(self):
        for replacement in (None, float("nan"), float("inf")):
            with self.subTest(value=replacement):
                invalid = copy.deepcopy(RECORDS)
                invalid[0]["x1"] = replacement
                with self.assertRaisesRegex(ValueError, "invalid or missing x1"):
                    validate_records(invalid)

        missing = copy.deepcopy(RECORDS)
        del missing[0]["x2"]
        with self.assertRaisesRegex(ValueError, "contain exactly"):
            validate_records(missing)

    def test_rejects_boolean_numeric_and_label_values(self):
        invalid_feature = copy.deepcopy(RECORDS)
        invalid_feature[0]["x1"] = True
        with self.assertRaisesRegex(ValueError, "invalid or missing x1"):
            validate_records(invalid_feature)
        invalid_label = copy.deepcopy(RECORDS)
        invalid_label[0]["label"] = True
        with self.assertRaisesRegex(ValueError, "label must be integer"):
            validate_records(invalid_label)

    def test_rejects_non_simulated_fixture_marker(self):
        invalid = copy.deepcopy(FIXTURE)
        invalid["data_kind"] = "iris"
        with self.assertRaisesRegex(ValueError, "marked as simulated"):
            run_pipeline(invalid)

    def test_feature_contract_constant_is_ordered(self):
        self.assertEqual(FEATURE_ORDER, ("x1", "x2"))


if __name__ == "__main__":
    unittest.main()
