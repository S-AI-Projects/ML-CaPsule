import json
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline


class SplitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pipeline.load_fixture(HERE / "fixture.json")

    def test_literal_formula_fixture(self):
        self.assertEqual(
            [record["x1"] for record in self.records],
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual([record["x2"] for record in self.records], [1] * 6)
        self.assertEqual(
            [record["target"] for record in self.records],
            [1, 3, 5, 7, 9, 11],
        )

    def test_manifest_is_exact_and_repeatable(self):
        expected = {
            "schema_version": "r41-t052.split.v1",
            "seed": 42,
            "split_ids": {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
        }
        first = pipeline.create_manifest(self.records, seed=42)
        second = pipeline.create_manifest(self.records, seed=42)
        self.assertEqual(first, expected)
        self.assertEqual(second, first)

    def test_saved_manifest_loads_and_validates(self):
        loaded = pipeline.load_manifest(HERE / "split_manifest.json", self.records)
        self.assertEqual(loaded, pipeline.create_manifest(self.records))

    def test_overlapping_record_ids_are_rejected(self):
        with (HERE / "contaminated_manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            pipeline.validate_manifest(self.records, manifest)

    def test_group_leakage_is_rejected(self):
        grouped = [dict(record, group_id=record["record_id"]) for record in self.records]
        grouped[0]["group_id"] = "shared"
        grouped[4]["group_id"] = "shared"
        with self.assertRaisesRegex(ValueError, "group_id values must not cross"):
            pipeline.create_manifest(grouped)

    def test_time_splits_must_be_chronological(self):
        timed = [
            dict(record, timestamp="2024-01-%02d" % (index + 1))
            for index, record in enumerate(self.records)
        ]
        manifest = pipeline.create_manifest(timed)
        self.assertEqual(manifest["split_ids"], pipeline.FIXED_SPLIT_IDS)

        timed[4]["timestamp"] = "2023-12-31"
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            pipeline.create_manifest(timed)

    def test_duplicate_record_ids_are_rejected(self):
        duplicated = [dict(record) for record in self.records]
        duplicated[1]["record_id"] = duplicated[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            pipeline.validate_records(duplicated)


if __name__ == "__main__":
    unittest.main()
