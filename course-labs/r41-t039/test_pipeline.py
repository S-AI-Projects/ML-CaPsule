"""Unit tests for R41-T039: Wine origin input perturbation robustness."""

import unittest

from pipeline import (
    RobustUnsupervisedAssigner,
    load_fixture,
    test_perturbation_properties,
)


class TestWineOriginInputPerturbation(unittest.TestCase):
    """Verifies input perturbation invariants and unsupported input rejection for Task R41-T039."""

    def setUp(self) -> None:
        self.fixture = load_fixture()
        self.records = self.fixture["records"]
        self.centroids = self.fixture["metadata"]["fixed_centroids"]
        self.assigner = RobustUnsupervisedAssigner()

    def test_nearest_centroid_assignments_fixture(self) -> None:
        """Verify six points (0,0),(0,1),(1,0),(5,5),(5,6),(6,5) yield 0,0,0,1,1,1."""
        assignments = self.assigner.assign(self.records, self.centroids)
        expected = {r["record_id"]: r["expected_cluster"] for r in self.records}

        self.assertEqual(assignments, expected)
        self.assertEqual([assignments[f"p{i}"] for i in range(1, 7)], [0, 0, 0, 1, 1, 1])

    def test_record_order_perturbation_invariance_after_id_alignment(self) -> None:
        """Verify shuffling row order preserves assignments once aligned by stable record_id."""
        result = test_perturbation_properties(self.records, self.centroids)
        self.assertTrue(result.is_order_invariant)
        for rid in result.original_assignments:
            self.assertEqual(
                result.original_assignments[rid],
                result.perturbed_assignments[rid],
            )

    def test_unsupported_missing_field_is_rejected(self) -> None:
        """Verify missing feature column raises clear ValueError."""
        corrupted = [dict(self.records[0])]
        del corrupted[0]["x2"]
        with self.assertRaises(ValueError) as ctx:
            self.assigner.assign(corrupted, self.centroids)
        self.assertIn("Missing required field", str(ctx.exception))

    def test_unsupported_shape_mismatch_is_rejected(self) -> None:
        """Verify passing 3D centroids to 2D records raises ValueError."""
        centroids_3d = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
        with self.assertRaises(ValueError) as ctx:
            self.assigner.assign(self.records, centroids_3d)
        self.assertIn("Shape mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
