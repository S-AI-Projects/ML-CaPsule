import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_valid_fixture_passes_and_preserves_every_row(self):
        validated = pipeline.validate_records(self.records)
        self.assertEqual(len(validated), 6)
        self.assertEqual(validated, self.records)
        self.assertEqual([record["x1"] for record in validated], [0, 1, 2, 3, 4, 5])
        self.assertEqual([record["x2"] for record in validated], [1] * 6)
        self.assertEqual(
            [record["target"] for record in validated],
            [1, 3, 5, 7, 9, 11],
        )

    def test_fixed_train_validation_test_ids(self):
        manifest = pipeline.create_split_manifest(self.records)
        self.assertEqual(manifest, pipeline.create_split_manifest(self.records))
        self.assertEqual(
            manifest["split_ids"],
            {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
        )
        splits = pipeline.split_records(self.records, manifest)
        self.assertEqual({name: len(rows) for name, rows in splits.items()},
                         {"train": 4, "validation": 1, "test": 1})

    def test_missing_record_id_reports_row_field_and_reason(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["record_id"]
        with self.assertRaises(pipeline.DatasetValidationError) as context:
            pipeline.validate_records(malformed)
        error = context.exception.errors[0]
        self.assertEqual(error["record_id"], None)
        self.assertEqual(error["row_index"], 0)
        self.assertEqual(error["field"], "record_id")
        self.assertEqual(error["reason"], "missing_required_field")
        self.assertEqual(context.exception.input_count, 6)

    def test_malformed_numeric_and_tensor_values_report_specific_reasons(self):
        malformed = [dict(record) for record in self.records]
        malformed[1]["x1"] = "one"
        malformed[2]["x2"] = [0, 1]
        with self.assertRaises(pipeline.DatasetValidationError) as context:
            pipeline.validate_records(malformed)
        errors = {
            (error["record_id"], error["field"]): error["reason"]
            for error in context.exception.errors
        }
        self.assertEqual(errors[("r2", "x1")], "expected_numeric_scalar")
        self.assertEqual(errors[("r3", "x2")], "tensor_or_vector_not_supported")

    def test_duplicate_record_id_is_reported_without_dropping_rows(self):
        malformed = [dict(record) for record in self.records]
        malformed[5]["record_id"] = "r5"
        with self.assertRaises(pipeline.DatasetValidationError) as context:
            pipeline.validate_records(malformed)
        self.assertEqual(context.exception.input_count, 6)
        self.assertIn(
            {
                "record_id": "r5",
                "row_index": 5,
                "field": "record_id",
                "reason": "duplicate_record_id",
            },
            context.exception.errors,
        )

    def test_non_finite_numbers_are_rejected_with_field_reason(self):
        malformed = [dict(record) for record in self.records]
        malformed[0]["target"] = float("nan")
        with self.assertRaises(pipeline.DatasetValidationError) as context:
            pipeline.validate_records(malformed)
        self.assertIn(
            {
                "record_id": "r1",
                "row_index": 0,
                "field": "target",
                "reason": "non_finite_number",
            },
            context.exception.errors,
        )


if __name__ == "__main__":
    unittest.main()
