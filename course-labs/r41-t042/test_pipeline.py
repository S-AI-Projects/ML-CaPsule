"""Offline unit tests for the R41-T042 synthetic split-manifest component."""

import copy
import json
from pathlib import Path
import unittest

from pipeline import (
    SplitValidationError,
    create_split_manifest,
    validate_manifest,
    validate_records,
)


HERE = Path(__file__).parent
with (HERE / "fixture.json").open(encoding="utf-8") as fixture_file:
    FIXTURE = json.load(fixture_file)
with (HERE / "split_manifest.json").open(encoding="utf-8") as manifest_file:
    CHECKED_IN_MANIFEST = json.load(manifest_file)
with (HERE / "schema.json").open(encoding="utf-8") as schema_file:
    SCHEMA = json.load(schema_file)
RECORDS = FIXTURE["records"]
EXPECTED_SPLITS = {
    "train": ["t5", "t6", "t1", "t2"],
    "validation": ["t3"],
    "test": ["t4"],
}


class SplitManifestTests(unittest.TestCase):
    def test_literal_fixture_and_exact_seeded_manifest(self):
        self.assertEqual(FIXTURE["data_kind"], "fictional_pipeline_fixture")
        self.assertEqual([row["record_id"] for row in RECORDS], ["t1", "t2", "t3", "t4", "t5", "t6"])
        self.assertEqual([row["text"] for row in RECORDS], [
            "red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"
        ])
        self.assertEqual([row["label"] for row in RECORDS], [1, 1, 0, 0, 1, 0])
        self.assertEqual(SCHEMA["required"], ["record_id", "text", "label"])
        actual = create_split_manifest(RECORDS, seed=41)
        self.assertEqual(actual, {"seed": 41, "splits": EXPECTED_SPLITS})
        self.assertEqual(CHECKED_IN_MANIFEST["seed"], actual["seed"])
        self.assertEqual(CHECKED_IN_MANIFEST["splits"], actual["splits"])
        validate_manifest(RECORDS, actual)

    def test_repeated_runs_are_identical_and_splits_disjoint(self):
        first = create_split_manifest(RECORDS, seed=41)
        second = create_split_manifest(RECORDS, seed=41)
        self.assertEqual(first, second)
        all_ids = [record_id for ids in first["splits"].values() for record_id in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {row["record_id"] for row in RECORDS})

    def test_duplicate_record_id_is_rejected(self):
        rows = copy.deepcopy(RECORDS)
        rows[-1]["record_id"] = rows[0]["record_id"]
        with self.assertRaisesRegex(SplitValidationError, r"record_id=t1, field=record_id, reason=duplicate"):
            validate_records(rows)

    def test_missing_id_is_rejected(self):
        rows = copy.deepcopy(RECORDS)
        del rows[0]["record_id"]
        with self.assertRaisesRegex(SplitValidationError, r"record_id=<missing>, field=record_id"):
            validate_records(rows)

    def test_invalid_text_and_label_are_rejected(self):
        rows = copy.deepcopy(RECORDS)
        rows[0]["text"] = 19
        with self.assertRaisesRegex(SplitValidationError, r"record_id=t1, field=text"):
            validate_records(rows)
        rows = copy.deepcopy(RECORDS)
        rows[1]["label"] = "positive"
        with self.assertRaisesRegex(SplitValidationError, r"record_id=t2, field=label"):
            validate_records(rows)

    def test_duplicate_manifest_id_and_missing_manifest_id_are_rejected(self):
        manifest = create_split_manifest(RECORDS, seed=41)
        duplicate = copy.deepcopy(manifest)
        duplicate["splits"]["test"].append(duplicate["splits"]["train"][0])
        with self.assertRaisesRegex(SplitValidationError, r"field=test, reason=ID already assigned"):
            validate_manifest(RECORDS, duplicate)
        missing = copy.deepcopy(manifest)
        missing["splits"]["test"].clear()
        with self.assertRaisesRegex(SplitValidationError, r"field=splits, reason=record missing"):
            validate_manifest(RECORDS, missing)

    def test_groups_are_kept_together_and_cross_split_group_is_rejected(self):
        rows = [
            {"record_id": f"g{i}", "text": "fixture text", "label": i % 2, "user_id": f"u{(i - 1) // 2}"}
            for i in range(1, 7)
        ]
        manifest = create_split_manifest(rows, seed=41)
        validate_manifest(rows, manifest)
        groups_to_splits = {}
        for split_name, ids in manifest["splits"].items():
            for record_id in ids:
                user_id = next(row["user_id"] for row in rows if row["record_id"] == record_id)
                groups_to_splits.setdefault(user_id, set()).add(split_name)
        self.assertTrue(all(len(split_names) == 1 for split_names in groups_to_splits.values()))

        overlapping = {"seed": 41, "splits": {"train": ["g1", "g3"], "validation": ["g2", "g4"], "test": ["g5", "g6"]}}
        with self.assertRaisesRegex(SplitValidationError, r"field=user_id, reason=group appears"):
            validate_manifest(rows, overlapping)

    def test_time_splits_are_chronological(self):
        rows = [
            {"record_id": f"d{i}", "text": "fixture text", "label": i % 2, "timestamp": f"2024-01-0{i}T00:00:00Z"}
            for i in range(1, 7)
        ]
        manifest = create_split_manifest(rows, seed=41, time_field="timestamp")
        split_times = [
            next(row["timestamp"] for row in rows if row["record_id"] == record_id)
            for split_name in ("train", "validation", "test")
            for record_id in manifest["splits"][split_name]
        ]
        self.assertEqual(split_times, sorted(split_times))
        bad_order = {"seed": 41, "splits": {"train": ["d4", "d5", "d6", "d1"], "validation": ["d2"], "test": ["d3"]}}
        with self.assertRaisesRegex(SplitValidationError, r"field=timestamp, reason=splits are not chronological"):
            validate_manifest(rows, bad_order, time_field="timestamp")


if __name__ == "__main__":
    unittest.main()