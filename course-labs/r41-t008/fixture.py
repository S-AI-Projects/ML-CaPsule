"""Synthetic fixture for Task R41-T008: Air passengers – moving-average forecast.

DISCLAIMER:
This time series dataset is an explicitly synthetic, deterministic verification fixture
created solely for offline pipeline verification and error-analysis testing. It does NOT
contain real-world Box-Jenkins airline passenger counts, international flight traffic, or
real-world aviation data.
"""

from typing import Any, Dict, List

DISCLAIMER: str = (
    "Synthetic pipeline-verification fixture: ten daily observations [10,12,11,13,15,14,16,18,17,19]. "
    "Not real-world Air Passengers airline passenger data."
)

# Ten consecutive daily observations: values [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
# Schema: (record_id, timestamp, value)
SYNTHETIC_RECORDS: List[Dict[str, Any]] = [
    {"record_id": "d1", "timestamp": "2026-01-01", "value": 10},
    {"record_id": "d2", "timestamp": "2026-01-02", "value": 12},
    {"record_id": "d3", "timestamp": "2026-01-03", "value": 11},
    {"record_id": "d4", "timestamp": "2026-01-04", "value": 13},
    {"record_id": "d5", "timestamp": "2026-01-05", "value": 15},
    {"record_id": "d6", "timestamp": "2026-01-06", "value": 14},
    {"record_id": "d7", "timestamp": "2026-01-07", "value": 16},
    {"record_id": "d8", "timestamp": "2026-01-08", "value": 18},
    {"record_id": "d9", "timestamp": "2026-01-09", "value": 17},
    {"record_id": "d10", "timestamp": "2026-01-10", "value": 19},
]

# Sequential split without shuffling:
# Train: first 6 (d1..d6)
# Validation: next 2 (d7..d8)
# Test: final 2 (d9..d10)
SPLIT_MANIFEST: Dict[str, List[str]] = {
    "train": ["d1", "d2", "d3", "d4", "d5", "d6"],
    "validation": ["d7", "d8"],
    "test": ["d9", "d10"],
}

ID_FIELD: str = "record_id"
TIMESTAMP_FIELD: str = "timestamp"
VALUE_FIELD: str = "value"


def get_split_records() -> Dict[str, List[Dict[str, Any]]]:
    """Return records partitioned chronologically without shuffling."""
    records_by_id = {r[ID_FIELD]: dict(r) for r in SYNTHETIC_RECORDS}
    return {
        split_name: [dict(records_by_id[rid]) for rid in ids]
        for split_name, ids in SPLIT_MANIFEST.items()
    }
