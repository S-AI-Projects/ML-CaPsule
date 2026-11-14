"""Evaluation metric implementation and cross-check for WHO life expectancy regression (Task R41-T034)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_mae(truth: List[float], prediction: List[float]) -> float:
    """
    Computes Mean Absolute Error (MAE) with explicit averaging definition:
    MAE = (1 / N) * sum(|truth_i - pred_i|).
    """
    if len(truth) != len(prediction):
        raise ValueError(f"Shape mismatch: truth length {len(truth)} != prediction length {len(prediction)}")
    if len(truth) == 0:
        raise ValueError("Cannot compute MAE on empty lists")

    total_abs_error = 0.0
    for t, p in zip(truth, prediction):
        t_val = float(t)
        p_val = float(p)
        if not math.isfinite(t_val) or not math.isfinite(p_val):
            raise ValueError("All inputs must be finite numbers")
        total_abs_error += abs(t_val - p_val)

    return total_abs_error / float(len(truth))


def compute_mape(
    truth: List[float],
    prediction: List[float],
    epsilon: float = 1e-8,
    zero_denominator_strategy: str = "epsilon",
) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE) with explicit zero-denominator behavior:
    When |truth_i| < epsilon:
      - 'epsilon': substitute denominator with epsilon to avoid division by zero
      - 'skip': ignore zero-denominator records in percentage calculation
    """
    if len(truth) != len(prediction):
        raise ValueError("Length mismatch between truth and prediction")
    if len(truth) == 0:
        raise ValueError("Cannot compute MAPE on empty lists")

    percentages = []
    for t, p in zip(truth, prediction):
        denom = abs(float(t))
        num = abs(float(t) - float(p))
        if denom < epsilon:
            if zero_denominator_strategy == "epsilon":
                percentages.append(num / epsilon)
            elif zero_denominator_strategy == "skip":
                continue
            else:
                raise ValueError(f"Unknown zero_denominator_strategy: {zero_denominator_strategy}")
        else:
            percentages.append(num / denom)

    if not percentages:
        return 0.0
    return (sum(percentages) / len(percentages)) * 100.0


def cross_check_mae(truth: List[float], prediction: List[float]) -> Dict[str, Any]:
    """
    Independently cross-checks the custom MAE against an alternative independent accumulator.
    Returns matched status and discrepancy.
    """
    custom_val = compute_mae(truth, prediction)

    # Independent implementation via list comprehension
    alt_val = sum(abs(float(t) - float(p)) for t, p in zip(truth, prediction)) / float(len(truth))

    discrepancy = abs(custom_val - alt_val)
    return {
        "custom_mae": custom_val,
        "independent_alt_mae": alt_val,
        "discrepancy": discrepancy,
        "matches_within_tolerance": discrepancy < 1e-12,
    }


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
