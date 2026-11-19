import json
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_fixture_and_nearest_centroid_assignments(self):
        self.assertEqual(
            [record["record_id"] for record in self.records],
            ["p1", "p2", "p3", "p4", "p5", "p6"],
        )
        self.assertEqual(
            [(record["x1"], record["x2"]) for record in self.records],
            [(0, 0), (0, 1), (1, 0), (5, 5), (5, 6), (6, 5)],
        )
        self.assertEqual(
            list(pipeline.assign_nearest_centroid(self.records).values()),
            [0, 0, 0, 1, 1, 1],
        )

    def test_fixed_seed_splits_are_disjoint_and_repeatable(self):
        first = pipeline.fixed_splits(self.records, seed=42)
        second = pipeline.fixed_splits(self.records, seed=42)
        ids = lambda split: [record["record_id"] for record in split]
        self.assertEqual([ids(split) for split in first], [ids(split) for split in second])
        self.assertEqual(
            [ids(split) for split in first],
            [["p4", "p2", "p3", "p5"], ["p1"], ["p6"]],
        )
        self.assertEqual([len(split) for split in first], [4, 1, 1])
        partitions = [set(ids(split)) for split in first]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual(set.union(*partitions), {record["record_id"] for record in self.records})

    def test_report_contains_only_the_injected_wrong_prediction(self):
        predictions = pipeline.load_predictions(HERE / "injected_predictions.json")
        report = pipeline.build_error_report(self.records, predictions)
        self.assertEqual(
            report["rows"],
            [
                {
                    "record_id": "p6",
                    "expected_value": 1,
                    "predicted_value": 0,
                    "reason_category": "cluster_assignment_mismatch",
                }
            ],
        )
        self.assertEqual(report["summary"]["error_count"], len(report["rows"]))
        self.assertEqual(report["summary"]["correct_count"], 5)
        self.assertEqual(
            report["summary"]["errors_by_reason"],
            {"cluster_assignment_mismatch": 1},
        )
        report_ids = {row["record_id"] for row in report["rows"]}
        correct_ids = {
            record_id
            for record_id, expected in pipeline.assign_nearest_centroid(
                self.records
            ).items()
            if predictions[record_id] == expected
        }
        self.assertTrue(report_ids.isdisjoint(correct_ids))

    def test_saved_example_report_matches_computed_report(self):
        predictions = pipeline.load_predictions(HERE / "injected_predictions.json")
        actual = pipeline.build_error_report(self.records, predictions)
        with (HERE / "error_analysis_report.json").open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
        self.assertEqual(actual, expected)

    def test_missing_coordinate_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["x2"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, x1, x2"):
            pipeline.validate_records(malformed)

    def test_prediction_ids_must_match_fixture_ids(self):
        predictions = pipeline.load_predictions(HERE / "injected_predictions.json")
        del predictions["p6"]
        with self.assertRaisesRegex(ValueError, "exactly match fixture record IDs"):
            pipeline.build_error_report(self.records, predictions)


if __name__ == "__main__":
    unittest.main()
