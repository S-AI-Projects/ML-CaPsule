"""Unit tests for R41-T032: Seeds (UCI) k-means split leakage and centroid assignment."""

from pathlib import Path
import unittest

from pipeline import (
    SplitManifest,
    assign_nearest_centroids,
    create_deterministic_split,
    load_fixture,
    validate_manifest,
)


class TestSeedsKMeansLeakageFreeSplit(unittest.TestCase):
    """Verifies deterministic split manifest and centroid assignments for Task R41-T032."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.all_ids = {r["record_id"] for r in self.records}
        self.centroids = [tuple(c) for c in self.fixture["metadata"]["fixed_centroids"]]
        self.group_map = {r["record_id"]: r["group_id"] for r in self.records}

    def test_nearest_centroid_assignments_fixture(self) -> None:
        """Verify six points (0,0),(0,1),(1,0),(5,5),(5,6),(6,5) yield 0,0,0,1,1,1."""
        assignments = assign_nearest_centroids(self.records, self.centroids)
        expected = [r["expected_cluster"] for r in self.records]
        self.assertEqual(assignments, [0, 0, 0, 1, 1, 1])
        self.assertEqual(assignments, expected)

    def test_split_manifest_is_disjoint_and_complete(self) -> None:
        """Verify train/val/test splits are mutually disjoint and cover all records."""
        manifest = create_deterministic_split(self.records, train_ratio=0.5, val_ratio=0.25, seed=42)
        val_report = validate_manifest(manifest, self.all_ids)

        self.assertTrue(val_report["valid"], f"Validation failed: {val_report['errors']}")
        self.assertTrue(val_report["is_disjoint"])
        self.assertTrue(val_report["complete_coverage"])
        self.assertEqual(len(val_report["errors"]), 0)

    def test_repeated_runs_produce_same_manifest(self) -> None:
        """Verify determinism across multiple invocations."""
        manifest1 = create_deterministic_split(self.records, seed=42)
        manifest2 = create_deterministic_split(self.records, seed=42)

        self.assertEqual(manifest1.to_dict(), manifest2.to_dict())

    def test_overlapping_ids_are_rejected(self) -> None:
        """Verify intentionally corrupted overlapping manifest fails validation."""
        corrupt_manifest = SplitManifest(
            train_ids=["s1", "s2", "s3"],
            val_ids=["s3", "s4"],  # s3 deliberately leaks into both train & val
            test_ids=["s5", "s6"],
        )
        val_report = validate_manifest(corrupt_manifest, self.all_ids)

        self.assertFalse(val_report["valid"])
        self.assertFalse(val_report["is_disjoint"])
        self.assertTrue(any("Overlap" in e for e in val_report["errors"]))

    def test_group_isolation_rejection_on_leakage(self) -> None:
        """Verify manifest with shared groups across train and val is rejected."""
        leaky_manifest = SplitManifest(
            train_ids=["s1", "s2"],  # grp_a
            val_ids=["s3", "s4"],   # s3 is grp_a, s4 is grp_b -> grp_a leaks between train and val
            test_ids=["s5", "s6"],   # grp_b
        )
        val_report = validate_manifest(leaky_manifest, self.all_ids, record_to_group=self.group_map)
        self.assertFalse(val_report["valid"])
        self.assertTrue(any("Group leakage" in e for e in val_report["errors"]))


if __name__ == "__main__":
    unittest.main()
