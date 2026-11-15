import os
import sys
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fixture import load_fixture
from pipeline import (
    SPLIT_SEED,
    build_split_manifest,
    validate_manifest,
)


class SplitManifestTests(unittest.TestCase):

    def setUp(self):
        self.records = load_fixture()

    def test_fixture_has_eight_records_and_balanced_labels(self):
        self.assertEqual(len(self.records), 8)
        labels = [record["label"] for record in self.records]
        self.assertEqual(labels.count(0), 4)
        self.assertEqual(labels.count(1), 4)

    def test_fixture_uses_required_feature_values(self):
        self.assertEqual(
            [record["x1"] for record in self.records],
            list(range(8)),
        )
        self.assertEqual(
            [record["x2"] for record in self.records],
            [1] * 8,
        )

    def test_split_is_deterministic(self):
        first = build_split_manifest(self.records, SPLIT_SEED)
        second = build_split_manifest(self.records, SPLIT_SEED)
        self.assertEqual(first, second)

    def test_expected_manifest(self):
        manifest = build_split_manifest(self.records, SPLIT_SEED)
        self.assertEqual(
            manifest["splits"],
            {
                "train": ["rec_002", "rec_003", "rec_006", "rec_007"],
                "validation": ["rec_004", "rec_008"],
                "test": ["rec_001", "rec_005"],
            },
        )

    def test_each_split_contains_both_classes(self):
        manifest = build_split_manifest(self.records)
        for split_ids in manifest["splits"].values():
            labels = {
                record["label"]
                for record in self.records
                if record["record_id"] in split_ids
            }
            self.assertEqual(labels, {0, 1})

    def test_record_and_group_isolation(self):
        manifest = build_split_manifest(self.records)
        validate_manifest(self.records, manifest)

    def test_duplicate_record_ids_are_rejected(self):
        invalid_records = self.records + [self.records[0].copy()]
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            build_split_manifest(invalid_records)

    def test_group_leakage_is_rejected(self):
        manifest = build_split_manifest(self.records)
        # Move rec_001 from test to train while leaving rec_005 in test.
        # Both rec_001 and rec_005 belong to grp_01.
        # This creates actual group leakage without any duplicate record IDs across splits.
        manifest["splits"]["test"].remove("rec_001")
        manifest["splits"]["train"].append("rec_001")
        with self.assertRaisesRegex(ValueError, "Groups overlap"):
            validate_manifest(self.records, manifest)

    def test_record_overlap_is_rejected(self):
        manifest = build_split_manifest(self.records)
        # Duplicate rec_001 into train without removing it from test.
        manifest["splits"]["train"].append("rec_001")
        with self.assertRaisesRegex(ValueError, "Record IDs overlap"):
            validate_manifest(self.records, manifest)

    def test_missing_record_is_rejected(self):
        manifest = build_split_manifest(self.records)
        manifest["splits"]["train"].remove("rec_002")
        with self.assertRaisesRegex(ValueError, "every record"):
            validate_manifest(self.records, manifest)

    def test_unknown_record_id_is_rejected(self):
        manifest = build_split_manifest(self.records)
        manifest["splits"]["train"].append("rec_999")
        with self.assertRaisesRegex(ValueError, "Unknown record IDs"):
            validate_manifest(self.records, manifest)

    def test_invalid_manifest_splits_rejected(self):
        manifest = {"splits": {"train": ["rec_001"], "validation": ["rec_002"]}}
        with self.assertRaisesRegex(ValueError, "Manifest must contain train, validation, and test"):
            validate_manifest(self.records, manifest)


if __name__ == "__main__":
    unittest.main()
