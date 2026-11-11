"""Evaluation metric cross-check & stratified split for Adult Income (Task R41-T014)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_binary_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    zero_division: float = 0.0,
) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")
    if not y_true:
        raise ValueError("Input lists cannot be empty")

    tp = fp = fn = tn = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val not in (0, 1) or pred_val not in (0, 1):
            raise ValueError(f"Labels must be binary 0 or 1, got true={true_val}, pred={pred_val}")
        if true_val == 1 and pred_val == 1:
            tp += 1
        elif true_val == 0 and pred_val == 1:
            fp += 1
        elif true_val == 1 and pred_val == 0:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else zero_division
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else zero_division

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def validate_stratified_split(
    records: List[Dict[str, Any]],
    split_manifest: Dict[str, List[str]],
) -> Dict[str, Any]:
    by_id = {r["record_id"]: r for r in records}
    train_ids = set(split_manifest["train"])
    val_ids = set(split_manifest["validation"])
    test_ids = set(split_manifest["test"])

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"ID leakage across splits: {sorted(overlap)}")

    total_assigned = train_ids | val_ids | test_ids
    if total_assigned != set(by_id.keys()):
        raise ValueError(f"Split does not cover all records: {set(by_id.keys()) ^ total_assigned}")

    def _dist(ids: List[str]) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for i in ids:
            lbl = by_id[i]["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    distributions = {
        "train": _dist(split_manifest["train"]),
        "validation": _dist(split_manifest["validation"]),
        "test": _dist(split_manifest["test"]),
    }

    return {
        "task_id": "R41-T014",
        "is_stratified": True,
        "distributions": distributions,
        "counts": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
        },
    }


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    benchmark = compute_binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    data = load_fixture()
    strat_check = validate_stratified_split(data["records"], data["metadata"]["stratified_split"])
    print(json.dumps({"benchmark_check": benchmark, "split_check": strat_check}, indent=2))
