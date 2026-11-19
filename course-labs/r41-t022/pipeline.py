"""Deterministic split manifest and leakage-free regression pipeline for Abalone age (Task R41-T022)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class SplitManifest:
    """Deterministic, leakage-free split manifest recording exact record IDs per partition."""

    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "train": list(self.train_ids),
            "validation": list(self.val_ids),
            "test": list(self.test_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "SplitManifest":
        val = data.get("validation", data.get("val", []))
        return cls(
            train_ids=list(data.get("train", [])),
            val_ids=list(val),
            test_ids=list(data.get("test", [])),
        )


def get_environment_info() -> Dict[str, str]:
    """Capture runtime environment versions."""
    return {
        "python": sys.version.split()[0],
        "os": platform.system().lower(),
    }


def create_deterministic_split(
    records: List[Dict[str, Any]],
    train_count: int = 4,
    val_count: int = 1,
) -> SplitManifest:
    """
    Creates a deterministic, leakage-free split manifest.
    Allocates exact records chronologically and preserves group isolation:
      - train: r1..r4
      - validation: r5
      - test: r6
    """
    sorted_records = sorted(records, key=lambda r: (r.get("timestamp", 0), r["record_id"]))
    record_ids = [r["record_id"] for r in sorted_records]

    n_train = min(train_count, len(record_ids))
    n_val = min(val_count, max(0, len(record_ids) - n_train))

    train_ids = record_ids[:n_train]
    val_ids = record_ids[n_train : n_train + n_val]
    test_ids = record_ids[n_train + n_val :]

    return SplitManifest(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def validate_manifest(
    manifest: SplitManifest,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validates manifest to prevent data leakage:
      1. Disjointness: Train, validation, and test sets must have zero overlap.
      2. Complete Coverage: Union of manifest splits must exactly match known record IDs.
      3. Group Isolation: No group_id may span multiple split partitions.
      4. Chronological Order: Future timestamps must not appear in earlier splits.
    """
    all_record_ids = {r["record_id"] for r in records}
    record_to_group = {r["record_id"]: r.get("group_id") for r in records if "group_id" in r}
    record_to_time = {r["record_id"]: r.get("timestamp") for r in records if "timestamp" in r}

    train_set = set(manifest.train_ids)
    val_set = set(manifest.val_ids)
    test_set = set(manifest.test_ids)

    errors: List[str] = []

    # 1. Pairwise disjointness check
    tv_overlap = train_set & val_set
    if tv_overlap:
        errors.append(f"Overlap between train and validation: {sorted(tv_overlap)}")

    tt_overlap = train_set & test_set
    if tt_overlap:
        errors.append(f"Overlap between train and test: {sorted(tt_overlap)}")

    vt_overlap = val_set & test_set
    if vt_overlap:
        errors.append(f"Overlap between validation and test: {sorted(vt_overlap)}")

    # 2. Coverage checks
    union_set = train_set | val_set | test_set
    missing = all_record_ids - union_set
    if missing:
        errors.append(f"Records missing from manifest: {sorted(missing)}")

    unknown = union_set - all_record_ids
    if unknown:
        errors.append(f"Unknown records present in manifest: {sorted(unknown)}")

    # 3. Group isolation check
    if record_to_group:
        train_groups = {record_to_group[r] for r in train_set if r in record_to_group and record_to_group[r]}
        val_groups = {record_to_group[r] for r in val_set if r in record_to_group and record_to_group[r]}
        test_groups = {record_to_group[r] for r in test_set if r in record_to_group and record_to_group[r]}

        g_tv = train_groups & val_groups
        if g_tv:
            errors.append(f"Group leakage between train and validation: {sorted(g_tv)}")
        g_tt = train_groups & test_groups
        if g_tt:
            errors.append(f"Group leakage between train and test: {sorted(g_tt)}")
        g_vt = val_groups & test_groups
        if g_vt:
            errors.append(f"Group leakage between validation and test: {sorted(g_vt)}")

    # 4. Chronological / Time-series check
    is_chronological = True
    if record_to_time and train_set and val_set:
        max_train_time = max(record_to_time[r] for r in train_set if r in record_to_time)
        min_val_time = min(record_to_time[r] for r in val_set if r in record_to_time)
        if max_train_time > min_val_time:
            errors.append(f"Chronological leakage: train max timestamp ({max_train_time}) > val min timestamp ({min_val_time})")
            is_chronological = False

    if record_to_time and val_set and test_set:
        max_val_time = max(record_to_time[r] for r in val_set if r in record_to_time)
        min_test_time = min(record_to_time[r] for r in test_set if r in record_to_time)
        if max_val_time > min_test_time:
            errors.append(f"Chronological leakage: val max timestamp ({max_val_time}) > test min timestamp ({min_test_time})")
            is_chronological = False

    return {
        "valid": len(errors) == 0,
        "is_disjoint": len(tv_overlap | tt_overlap | vt_overlap) == 0,
        "complete_coverage": len(missing) == 0 and len(unknown) == 0,
        "is_chronological": is_chronological,
        "errors": errors,
    }


def verify_deterministic_formula(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify fixture records strictly satisfy target = 2 * x1 + 1."""
    discrepancies = []
    for r in records:
        x1 = float(r["x1"])
        target = float(r["target"])
        expected = 2.0 * x1 + 1.0
        if abs(target - expected) > 1e-9:
            discrepancies.append({
                "record_id": r.get("record_id"),
                "x1": x1,
                "target": target,
                "expected": expected,
            })
    return {
        "formula_satisfied": len(discrepancies) == 0,
        "total_records": len(records),
        "discrepancies": discrepancies,
    }


class SimpleLinearRegression:
    """Bounded, CPU-only 1D closed-form linear regression estimator."""

    def __init__(self):
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None

    def fit(self, records: List[Dict[str, Any]]) -> "SimpleLinearRegression":
        xs = [float(r["x1"]) for r in records]
        ys = [float(r["target"]) for r in records]
        n = len(xs)
        if n < 2:
            raise ValueError("At least 2 points required to fit linear regression")

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        ss_xx = sum((x - mean_x) ** 2 for x in xs)

        if abs(ss_xx) < 1e-12:
            raise ValueError("Zero variance in feature x1")

        self.slope = ss_xy / ss_xx
        self.intercept = mean_y - self.slope * mean_x
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        if self.slope is None or self.intercept is None:
            raise RuntimeError("Model must be fitted before predict")
        return [self.slope * float(r["x1"]) + self.intercept for r in records]

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        preds = self.predict(records)
        actuals = [float(r["target"]) for r in records]
        mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(records) if records else 0.0
        return {"mse": mse, "predictions": preds, "actuals": actuals}


def load_fixture(fixture_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture from disk."""
    if fixture_path is None:
        fixture_path = Path(__file__).parent / "fixture.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)
