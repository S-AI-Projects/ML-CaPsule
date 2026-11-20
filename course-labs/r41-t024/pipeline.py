"""
R41-T024 · Loan Approval – Decision Tree
Classification Evaluation: Precision, Recall, F1 with explicit averaging
and zero-denominator behaviour.

All computations use the Python standard library only (no numpy/sklearn).
Data: synthetic fixture — NOT observations from the named real Loan Approval dataset.
"""

import json
import math
import os
import platform
import sys


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class ZeroDenominatorPolicy:
    """Controls what value is returned when a denominator is zero."""
    ZERO = "zero"       # return 0.0 (safe for sums/averages)
    NAN  = "nan"        # return float('nan') (flags undefined metric)
    ONE  = "one"        # return 1.0 (used for precision when no positives predicted)

    VALID = {ZERO, NAN, ONE}

    @staticmethod
    def apply(policy: str) -> float:
        if policy == ZeroDenominatorPolicy.ZERO:
            return 0.0
        if policy == ZeroDenominatorPolicy.NAN:
            return float("nan")
        if policy == ZeroDenominatorPolicy.ONE:
            return 1.0
        raise ValueError(f"Unknown zero-denominator policy: {policy!r}. "
                         f"Valid options: {ZeroDenominatorPolicy.VALID}")


def _safe_divide(numerator: float, denominator: float,
                 zero_policy: str = ZeroDenominatorPolicy.ZERO) -> float:
    """Divide numerator by denominator; apply zero_policy when denominator == 0."""
    if denominator == 0:
        return ZeroDenominatorPolicy.apply(zero_policy)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Binary confusion matrix computed from truth/prediction lists."""

    def __init__(self, truth: list, prediction: list, positive_label: int = 1):
        if len(truth) != len(prediction):
            raise ValueError(
                f"truth and prediction must have the same length; "
                f"got {len(truth)} vs {len(prediction)}."
            )
        if len(truth) == 0:
            raise ValueError("truth and prediction must not be empty.")
        self.positive_label = positive_label
        self.tp = sum(1 for t, p in zip(truth, prediction)
                      if t == positive_label and p == positive_label)
        self.fp = sum(1 for t, p in zip(truth, prediction)
                      if t != positive_label and p == positive_label)
        self.fn = sum(1 for t, p in zip(truth, prediction)
                      if t == positive_label and p != positive_label)
        self.tn = sum(1 for t, p in zip(truth, prediction)
                      if t != positive_label and p != positive_label)

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """
    Compute precision, recall and F1 for binary classification.

    Averaging modes
    ---------------
    'binary'  – treat the positive_label class only (default).
    'macro'   – unweighted mean of per-class metrics.
    'micro'   – aggregate TP/FP/FN across classes before computing metrics.

    Zero-denominator policy
    -----------------------
    When the denominator of precision or recall is zero (no predicted positives /
    no actual positives) the result is determined by `zero_policy`:
      'zero' – return 0.0  (default; safe for sums and averages)
      'nan'  – return float('nan')  (flags the undefined case explicitly)
      'one'  – return 1.0  (used in some frameworks for precision with no predictions)
    """

    VALID_AVERAGING = {"binary", "macro", "micro"}

    def __init__(self,
                 averaging: str = "binary",
                 positive_label: int = 1,
                 zero_policy: str = ZeroDenominatorPolicy.ZERO):
        if averaging not in self.VALID_AVERAGING:
            raise ValueError(
                f"Unknown averaging mode: {averaging!r}. "
                f"Valid options: {self.VALID_AVERAGING}"
            )
        if zero_policy not in ZeroDenominatorPolicy.VALID:
            raise ValueError(
                f"Unknown zero_policy: {zero_policy!r}. "
                f"Valid options: {ZeroDenominatorPolicy.VALID}"
            )
        self.averaging = averaging
        self.positive_label = positive_label
        self.zero_policy = zero_policy

    def compute(self, truth: list, prediction: list) -> dict:
        """
        Compute precision, recall, f1 and the confusion matrix.

        Parameters
        ----------
        truth      : list of ground-truth integer labels
        prediction : list of predicted integer labels (same length as truth)

        Returns
        -------
        dict with keys: precision, recall, f1, confusion_matrix, averaging, zero_policy
        """
        if len(truth) != len(prediction):
            raise ValueError(
                f"truth and prediction must have equal length; "
                f"got {len(truth)} vs {len(prediction)}."
            )
        if len(truth) == 0:
            raise ValueError("truth and prediction must not be empty.")

        if self.averaging == "binary":
            return self._binary(truth, prediction)
        if self.averaging == "macro":
            return self._macro(truth, prediction)
        if self.averaging == "micro":
            return self._micro(truth, prediction)

    # -- binary --------------------------------------------------------------

    def _binary(self, truth: list, prediction: list) -> dict:
        cm = ConfusionMatrix(truth, prediction, self.positive_label)
        precision = _safe_divide(cm.tp, cm.tp + cm.fp, self.zero_policy)
        recall    = _safe_divide(cm.tp, cm.tp + cm.fn, self.zero_policy)
        f1        = _safe_divide(2 * precision * recall,
                                  precision + recall, self.zero_policy)
        return {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "confusion_matrix": cm.as_dict(),
            "averaging":   self.averaging,
            "zero_policy": self.zero_policy,
        }

    # -- macro ---------------------------------------------------------------

    def _macro(self, truth: list, prediction: list) -> dict:
        labels = sorted(set(truth) | set(prediction))
        per_class = {}
        for lbl in labels:
            cm = ConfusionMatrix(truth, prediction, positive_label=lbl)
            p = _safe_divide(cm.tp, cm.tp + cm.fp, self.zero_policy)
            r = _safe_divide(cm.tp, cm.tp + cm.fn, self.zero_policy)
            f = _safe_divide(2 * p * r, p + r,     self.zero_policy)
            per_class[lbl] = {"precision": p, "recall": r, "f1": f,
                               "confusion_matrix": cm.as_dict()}

        def _mean(key):
            vals = [v[key] for v in per_class.values()
                    if not (isinstance(v[key], float) and math.isnan(v[key]))]
            return sum(vals) / len(vals) if vals else float("nan")

        return {
            "precision": _mean("precision"),
            "recall":    _mean("recall"),
            "f1":        _mean("f1"),
            "per_class": per_class,
            "averaging":   self.averaging,
            "zero_policy": self.zero_policy,
        }

    # -- micro ---------------------------------------------------------------

    def _micro(self, truth: list, prediction: list) -> dict:
        labels = sorted(set(truth) | set(prediction))
        total_tp = total_fp = total_fn = 0
        for lbl in labels:
            cm = ConfusionMatrix(truth, prediction, positive_label=lbl)
            total_tp += cm.tp
            total_fp += cm.fp
            total_fn += cm.fn
        precision = _safe_divide(total_tp, total_tp + total_fp, self.zero_policy)
        recall    = _safe_divide(total_tp, total_tp + total_fn, self.zero_policy)
        f1        = _safe_divide(2 * precision * recall,
                                  precision + recall, self.zero_policy)
        return {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "totals": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
            "averaging":   self.averaging,
            "zero_policy": self.zero_policy,
        }


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

def load_fixture(path: str = None) -> list:
    """Load the synthetic classification fixture and return its records."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "fixture.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["records"]


def get_split(records: list, split: str) -> list:
    """Return records belonging to the given split name."""
    return [r for r in records if r["split"] == split]


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def get_environment_info() -> dict:
    return {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "machine":        platform.machine(),
        "task_id":        "R41-T024",
    }
