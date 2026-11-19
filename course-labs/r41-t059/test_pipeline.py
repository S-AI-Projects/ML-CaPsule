import pathlib
import random
import sys
import unittest
from datetime import date, timedelta


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_fixture_and_chronological_splits(self):
        self.assertEqual(
            [record["record_id"] for record in self.records],
            ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"],
        )
        self.assertEqual(
            [record["value"] for record in self.records],
            [10, 12, 11, 13, 15, 14, 16, 18, 17, 19],
        )

        train, validation, test = pipeline.fixed_splits(self.records)
        self.assertEqual(
            [record["record_id"] for record in train],
            ["d1", "d2", "d3", "d4", "d5", "d6"],
        )
        self.assertEqual([record["record_id"] for record in validation], ["d7", "d8"])
        self.assertEqual([record["record_id"] for record in test], ["d9", "d10"])

    def test_record_permutation_preserves_forecasts_after_id_alignment(self):
        train, validation, test = pipeline.fixed_splits(self.records)
        targets = validation + test
        expected = pipeline.forecast_last_value(train, targets)
        self.assertEqual(expected, {"d7": 14, "d8": 14, "d9": 14, "d10": 14})

        shuffled = list(self.records)
        random.Random(42).shuffle(shuffled)
        shuffled_train, shuffled_validation, shuffled_test = pipeline.fixed_splits(
            shuffled
        )
        actual = pipeline.forecast_last_value(
            shuffled_train, shuffled_validation + shuffled_test
        )
        self.assertEqual(actual, expected)

    def test_missing_field_is_rejected(self):
        malformed = [dict(record) for record in self.records]
        del malformed[0]["value"]
        with self.assertRaisesRegex(ValueError, "exactly record_id, timestamp, value"):
            pipeline.validate_records(malformed)

    def test_nonconsecutive_timestamps_are_rejected(self):
        malformed = [dict(record) for record in self.records]
        for record in malformed[3:]:
            shifted_date = date.fromisoformat(record["timestamp"]) + timedelta(days=1)
            record["timestamp"] = shifted_date.isoformat()
        with self.assertRaisesRegex(ValueError, "ten consecutive daily observations"):
            pipeline.fixed_splits(malformed)


if __name__ == "__main__":
    unittest.main()
