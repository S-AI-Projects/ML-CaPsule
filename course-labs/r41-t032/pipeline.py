"""Deterministic split manifest and k-means centroid assigner for Seeds (UCI) (Task R41-T032)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple


@dataclass(frozen=True)
class SplitManifest:
    """Deterministic, leakage-free split manifest."""

    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "train": list(self.train_ids),
            "validation": list(self.val_ids),
            "test": list(self.test_ids),
        }


def create_deterministic_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    seed: int = 42,
    group_isolation: bool = False,
) -> SplitManifest:
    """
    Creates a deterministic split manifest preserving group isolation where specified.
    Uses sorted stable IDs and a fixed seed.
    """
    sorted_recs = sorted(records, key=lambda r: r["record_id"])

    if group_isolation:
        # Group records by group_id
        groups: Dict[str, List[str]] = {}
        for r in sorted_recs:
            gid = r.get("group_id", "default")
            groups.setdefault(gid, []).append(r["record_id"])

        sorted_groups = sorted(groups.keys())
        rng = random.Random(seed)
        shuffled_groups = list(sorted_groups)
        rng.shuffle(shuffled_groups)

        n_groups = len(shuffled_groups)
        n_train = max(1, int(round(n_groups * train_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))

        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train:n_train + n_val])
        test_groups = set(shuffled_groups[n_train + n_val:])

        train_ids = [rid for gid in train_groups for rid in groups[gid]]
        val_ids = [rid for gid in val_groups for rid in groups[gid]]
        test_ids = [rid for gid in test_groups for rid in groups[gid]]
    else:
        record_ids = [r["record_id"] for r in sorted_recs]
        rng = random.Random(seed)
        shuffled = list(record_ids)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train:n_train + n_val])
        test_ids = sorted(shuffled[n_train + n_val:])

    return SplitManifest(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def validate_manifest(
    manifest: SplitManifest,
    all_record_ids: Set[str],
    record_to_group: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Validates manifest for leakage prevention:
    - Splits must be mutually disjoint.
    - No record ID may appear in multiple splits.
    - Rejects intentionally overlapping IDs or groups.
    - Ensures all records are covered.
    """
    train_set = set(manifest.train_ids)
    val_set = set(manifest.val_ids)
    test_set = set(manifest.test_ids)

    errors = []

    # Check pair-wise intersections
    tv_overlap = train_set & val_set
    if tv_overlap:
        errors.append(f"Overlap between train and validation: {sorted(tv_overlap)}")

    tt_overlap = train_set & test_set
    if tt_overlap:
        errors.append(f"Overlap between train and test: {sorted(tt_overlap)}")

    vt_overlap = val_set & test_set
    if vt_overlap:
        errors.append(f"Overlap between validation and test: {sorted(vt_overlap)}")

    # Check complete coverage
    union_set = train_set | val_set | test_set
    missing = all_record_ids - union_set
    if missing:
        errors.append(f"Records missing from manifest: {sorted(missing)}")

    unknown = union_set - all_record_ids
    if unknown:
        errors.append(f"Unknown records present in manifest: {sorted(unknown)}")

    # Check group isolation if mapping provided
    if record_to_group:
        train_groups = {record_to_group[r] for r in train_set if r in record_to_group}
        val_groups = {record_to_group[r] for r in val_set if r in record_to_group}
        test_groups = {record_to_group[r] for r in test_set if r in record_to_group}

        if train_groups & val_groups:
            errors.append(f"Group leakage between train and val: {sorted(train_groups & val_groups)}")
        if train_groups & test_groups:
            errors.append(f"Group leakage between train and test: {sorted(train_groups & test_groups)}")
        if val_groups & test_groups:
            errors.append(f"Group leakage between val and test: {sorted(val_groups & test_groups)}")

    return {
        "valid": len(errors) == 0,
        "is_disjoint": len(tv_overlap | tt_overlap | vt_overlap) == 0,
        "complete_coverage": len(missing) == 0 and len(unknown) == 0,
        "errors": errors,
    }


def assign_nearest_centroids(
    records: List[Dict[str, Any]],
    centroids: List[Tuple[float, float]],
) -> List[int]:
    """
    Assigns each record to the nearest centroid using Euclidean distance.
    Returns list of assigned cluster indices matching records order.
    """
    assignments = []
    for r in records:
        x1, x2 = float(r["x1"]), float(r["x2"])
        best_cluster = -1
        min_dist = float("inf")

        for c_idx, (cx1, cx2) in enumerate(centroids):
            dist = math.sqrt((x1 - cx1) ** 2 + (x2 - cx2) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx

        assignments.append(best_cluster)

    return assignments


def load_fixture(path: Path = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
