"""Deterministic split manifest & leakage prevention for Rain in Australia (Task R41-T012)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REQUIRED_FIELDS = ("record_id", "x1", "x2", "target")


def _is_finite_number(val: Any) -> bool:
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(float(val))


def validate_regression_records(records: List[Dict[str, Any]]) -> None:
    """Verify required regression schema (record_id, x1, x2, target) with unique IDs."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    seen_ids: Set[str] = set()
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"row {idx}: expected dict")
        for field in REQUIRED_FIELDS:
            if field not in rec:
                raise ValueError(f"row {idx}: missing required field {field!r}")
        rid = rec["record_id"]
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"row {idx}: record_id must be a non-empty string")
        if rid in seen_ids:
            raise ValueError(f"duplicate record_id {rid!r}")
        seen_ids.add(rid)
        for num_field in ("x1", "x2", "target"):
            if not _is_finite_number(rec[num_field]):
                raise ValueError(f"{rid}: {num_field} must be a finite scalar number")


def build_split_manifest(
    records: List[Dict[str, Any]],
    split_ids: Optional[Dict[str, List[str]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build and verify a leak-free deterministic train/validation/test split manifest."""
    validate_regression_records(records)
    by_id = {r["record_id"]: r for r in records}

    if split_ids is None:
        ordered = sorted(records, key=lambda r: (r.get("timestamp", ""), r["record_id"]))
        if len(ordered) < 3:
            raise ValueError("at least 3 records required for train/validation/test split")
        split_ids = {
            "train": [r["record_id"] for r in ordered[:-2]],
            "validation": [ordered[-2]["record_id"]],
            "test": [ordered[-1]["record_id"]],
        }

    for split_name in ("train", "validation", "test"):
        if split_name not in split_ids or not split_ids[split_name]:
            raise ValueError(f"split {split_name!r} must be a non-empty list of record IDs")

    train_list = list(split_ids["train"])
    val_list = list(split_ids["validation"])
    test_list = list(split_ids["test"])

    for split_name, ids in (("train", train_list), ("validation", val_list), ("test", test_list)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate record_id inside split {split_name!r}")
        for rid in ids:
            if rid not in by_id:
                raise ValueError(f"unknown record_id {rid!r} in split {split_name!r}")

    train_set, val_set, test_set = set(train_list), set(val_list), set(test_list)
    id_overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if id_overlap:
        raise ValueError(f"record_id leakage across splits: {sorted(id_overlap)}")

    all_assigned = train_set | val_set | test_set
    if all_assigned != set(by_id.keys()):
        missing = sorted(set(by_id.keys()) - all_assigned)
        raise ValueError(f"unassigned record_ids omitted from manifest: {missing}")

    # Enforce group_id and user_id isolation when present
    for entity_key in ("group_id", "user_id"):
        train_entities = {by_id[i][entity_key] for i in train_list if entity_key in by_id[i]}
        val_entities = {by_id[i][entity_key] for i in val_list if entity_key in by_id[i]}
        test_entities = {by_id[i][entity_key] for i in test_list if entity_key in by_id[i]}
        entity_overlap = (
            (train_entities & val_entities)
            | (train_entities & test_entities)
            | (val_entities & test_entities)
        )
        if entity_overlap:
            raise ValueError(f"{entity_key} leakage across splits: {sorted(entity_overlap)}")

    # Enforce chronological ordering when timestamps are present
    def _timestamps(ids: List[str]) -> List[str]:
        return [by_id[i]["timestamp"] for i in ids if "timestamp" in by_id[i]]

    t_train, t_val, t_test = _timestamps(train_list), _timestamps(val_list), _timestamps(test_list)
    if t_train and t_val and not (max(t_train) < min(t_val)):
        raise ValueError("chronological leakage: train timestamps must precede validation")
    if t_val and t_test and not (max(t_val) < min(t_test)):
        raise ValueError("chronological leakage: validation timestamps must precede test")

    return {
        "task_id": "R41-T012",
        "seed": seed,
        "leakage_free": True,
        "splits": {
            "train": train_list,
            "validation": val_list,
            "test": test_list,
        },
        "counts": {
            "train": len(train_list),
            "validation": len(val_list),
            "test": len(test_list),
        },
    }


def load_fixture(fixture_path: Optional[Path] = None) -> Dict[str, Any]:
    path = fixture_path or (Path(__file__).resolve().parent / "fixture.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    payload = load_fixture()
    manifest = build_split_manifest(
        payload["records"],
        split_ids=payload["metadata"]["expected_split"],
        seed=payload["metadata"]["seed"],
    )
    print(json.dumps(manifest, indent=2))
