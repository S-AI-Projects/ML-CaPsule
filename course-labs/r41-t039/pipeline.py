"""Robustness testing and input perturbation pipeline for Wine origin (UCI) LDA (Task R41-T039)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_FEATURES = ("x1", "x2")


@dataclass
class PerturbationResult:
    original_assignments: Dict[str, int]
    perturbed_assignments: Dict[str, int]
    is_order_invariant: bool
    unsupported_missing_field_rejected: bool
    unsupported_shape_mismatch_rejected: bool


class RobustUnsupervisedAssigner:
    """Assigns records to nearest centroids with explicit input perturbation handling."""

    def assign(
        self,
        records: List[Dict[str, Any]],
        centroids: List[List[float]],
    ) -> Dict[str, int]:
        """
        Assigns each record to nearest centroid.
        Rejects missing fields and dimension mismatches.
        Returns mapping from record_id to assigned cluster.
        """
        if not centroids:
            raise ValueError("Centroids cannot be empty")

        centroid_dim = len(centroids[0])
        for c in centroids:
            if len(c) != centroid_dim:
                raise ValueError("All centroids must have identical dimensionality")

        if len(REQUIRED_FEATURES) != centroid_dim:
            raise ValueError(
                f"Shape mismatch: records have {len(REQUIRED_FEATURES)} features, but centroids have {centroid_dim} dimensions"
            )

        assignments: Dict[str, int] = {}

        for r in records:
            rid = r.get("record_id")
            if not rid:
                raise ValueError("Record missing required 'record_id'")

            # Check for missing required fields
            for feat in REQUIRED_FEATURES:
                if feat not in r or r[feat] is None:
                    raise ValueError(f"Missing required field: '{feat}' in record '{rid}'")

            point = [float(r[feat]) for feat in REQUIRED_FEATURES]

            best_cluster = -1
            min_dist = float("inf")

            for c_idx, centroid in enumerate(centroids):
                dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(point, centroid)))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = c_idx

            assignments[rid] = best_cluster

        return assignments


def perturb_record_order(records: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Generates a valid deterministic order permutation of the input records (preserving record contents)."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled


def test_perturbation_properties(
    records: List[Dict[str, Any]],
    centroids: List[List[float]],
) -> PerturbationResult:
    """Tests order invariance and rejection of unsupported inputs."""
    assigner = RobustUnsupervisedAssigner()

    # 1. Base assignment
    base_assignments = assigner.assign(records, centroids)

    # 2. Valid transformation: order shuffle with ID alignment
    shuffled_records = perturb_record_order(records, seed=42)
    shuffled_assignments = assigner.assign(shuffled_records, centroids)

    order_invariant = all(
        base_assignments[rid] == shuffled_assignments[rid]
        for rid in base_assignments
    )

    # 3. Unsupported input 1: Missing field
    missing_rejected = False
    corrupt_records = [dict(records[0])]
    del corrupt_records[0]["x2"]
    try:
        assigner.assign(corrupt_records, centroids)
    except ValueError:
        missing_rejected = True

    # 4. Unsupported input 2: Shape mismatch (3D centroids for 2D data)
    shape_rejected = False
    centroids_3d = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    try:
        assigner.assign(records, centroids_3d)
    except ValueError:
        shape_rejected = True

    return PerturbationResult(
        original_assignments=base_assignments,
        perturbed_assignments=shuffled_assignments,
        is_order_invariant=order_invariant,
        unsupported_missing_field_rejected=missing_rejected,
        unsupported_shape_mismatch_rejected=shape_rejected,
    )


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
