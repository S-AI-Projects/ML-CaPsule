"""Unit tests for Rain in Australia split leakage prevention (Task R41-T012)."""

import copy
import unittest

from pipeline import build_split_manifest, load_fixture


class TestRainInAustraliaSplitManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture()
        self.records = self.payload["records"]
        self.expected_split = self.payload["metadata"]["expected_split"]

    def test_deterministic_formula_fixture_and_manifest(self) -> None:
        self.assertEqual(len(self.records), 6)
        for idx, row in enumerate(self.records):
            self.assertEqual(row["record_id"], f"r{idx + 1}")
            self.assertEqual(row["x1"], idx)
            self.assertEqual(row["x2"], 1)
            self.assertEqual(row["target"], 2 * idx + 1)

        m1 = build_split_manifest(self.records, split_ids=self.expected_split, seed=42)
        m2 = build_split_manifest(self.records, split_ids=self.expected_split, seed=42)
        self.assertEqual(m1, m2)
        self.assertTrue(m1["leakage_free"])
        self.assertEqual(m1["splits"]["train"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(m1["splits"]["validation"], ["r5"])
        self.assertEqual(m1["splits"]["test"], ["r6"])

    def test_rejects_overlapping_record_ids(self) -> None:
        bad_split = {
            "train": ["r1", "r2", "r3", "r4"],
            "validation": ["r4", "r5"],
            "test": ["r6"],
        }
        with self.assertRaisesRegex(ValueError, "record_id leakage"):
            build_split_manifest(self.records, split_ids=bad_split, seed=42)

    def test_rejects_overlapping_groups_and_users(self) -> None:
        leaked_group_records = copy.deepcopy(self.records)
        leaked_group_records[4]["group_id"] = "grp_1"
        with self.assertRaisesRegex(ValueError, "group_id leakage"):
            build_split_manifest(leaked_group_records, split_ids=self.expected_split, seed=42)

        leaked_user_records = copy.deepcopy(self.records)
        for i, row in enumerate(leaked_user_records):
            row["user_id"] = f"u_{i}"
        leaked_user_records[5]["user_id"] = "u_0"
        with self.assertRaisesRegex(ValueError, "user_id leakage"):
            build_split_manifest(leaked_user_records, split_ids=self.expected_split, seed=42)

    def test_rejects_non_chronological_time_split(self) -> None:
        out_of_order_split = {
            "train": ["r1", "r2", "r3", "r6"],
            "validation": ["r5"],
            "test": ["r4"],
        }
        with self.assertRaisesRegex(ValueError, "chronological leakage"):
            build_split_manifest(self.records, split_ids=out_of_order_split, seed=42)


if __name__ == "__main__":
    unittest.main()
