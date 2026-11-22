"""Synthetic text classification fixture and split definition for R41-T003.

DISCLAIMER:
These six synthetic records are generated strictly for pipeline verification and
baseline model testing. They do NOT represent real SMS messages from the SMS Spam
Collection dataset or any other production corpus, and must not be used to claim
real-world classification performance.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# Explicit statement of synthetic purpose
DATASET_DISCLAIMER: str = (
    "Synthetic verification fixture for R41-T003 baseline comparison. "
    "These fictional records do NOT represent real SMS messages or spam classification data."
)

# Canonical six synthetic records specified in Issue #3:
# - record_id: stable unique identifier (t1..t6)
# - text: simulated text string
# - label: binary integer label (0 or 1)
SYNTHETIC_FIXTURE: List[Dict[str, Any]] = [
    {"record_id": "t1", "text": "red apple", "label": 1},
    {"record_id": "t2", "text": "green apple", "label": 1},
    {"record_id": "t3", "text": "blue sky", "label": 0},
    {"record_id": "t4", "text": "dark sky", "label": 0},
    {"record_id": "t5", "text": "red fruit", "label": 1},
    {"record_id": "t6", "text": "blue cloud", "label": 0},
]

# Fixed 3-way split specification:
# - Train: 3 records (t1, t2, t3) -> labels: [1, 1, 0] -> majority class is 1 (2/3 frequency)
# - Validation: 1 record (t4) -> label: 0 (held-out)
# - Test: 2 records (t5, t6) -> labels: [1, 0] (held-out)
# No record ID overlaps between any of the splits.
DEFAULT_SPLIT: Dict[str, List[str]] = {
    "train": ["t1", "t2", "t3"],
    "validation": ["t4"],
    "test": ["t5", "t6"],
}


def load_fixture() -> List[Dict[str, Any]]:
    """Return a fresh deep copy of the six synthetic classification records."""
    return copy.deepcopy(SYNTHETIC_FIXTURE)


def get_split_records(
    records: List[Dict[str, Any]] | None = None,
    split_manifest: Dict[str, List[str]] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition records according to a split manifest.

    Parameters
    ----------
    records : List[Dict[str, Any]] | None
        Input records (defaults to canonical synthetic fixture).
    split_manifest : Dict[str, List[str]] | None
        Dictionary mapping split names to lists of record_ids (defaults to DEFAULT_SPLIT).

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary with keys 'train', 'validation', and 'test' containing corresponding records.
    """
    recs = load_fixture() if records is None else records
    manifest = DEFAULT_SPLIT if split_manifest is None else split_manifest
    record_map = {r["record_id"]: copy.deepcopy(r) for r in recs}

    return {
        split_name: [record_map[rid] for rid in manifest[split_name] if rid in record_map]
        for split_name in ("train", "validation", "test")
    }
