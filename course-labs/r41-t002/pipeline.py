"""Deterministic, group-isolated split manifest for R41-T002."""

import random

SPLIT_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records):
    """Validate the input records and their identifiers."""
    if not records:
        raise ValueError("Records must not be empty.")

    if not isinstance(records, (list, tuple)):
        raise ValueError("Records must be a list or tuple of dictionaries.")

    required = {"record_id", "group_id", "x1", "x2", "label"}
    record_ids = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a dictionary.")

        if not required.issubset(record):
            raise ValueError("A record is missing required fields.")

        record_id = record["record_id"]
        group_id = record["group_id"]

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string.")

        if record_id in record_ids:
            raise ValueError(f"Duplicate record_id: {record_id}")

        record_ids.add(record_id)

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a non-empty string.")

        if isinstance(record["label"], bool) or record["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1.")

        for field_name in ("x1", "x2"):
            val = record[field_name]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{field_name} must be a numeric scalar.")


def build_split_manifest(records, seed=SPLIT_SEED):
    """Create a reproducible group-isolated and stratified split manifest."""
    validate_records(records)

    groups = sorted({record["group_id"] for record in records})

    if len(groups) < 3:
        raise ValueError("At least three groups are required.")

    expected_labels = {record["label"] for record in records}

    for group_id in groups:
        group_labels = {
            record["label"]
            for record in records
            if record["group_id"] == group_id
        }
        if group_labels != expected_labels:
            raise ValueError(
                f"Group {group_id} does not contain every class."
            )

    shuffled_groups = groups.copy()
    random.Random(seed).shuffle(shuffled_groups)

    train_count = max(1, round(len(groups) * 0.5))
    validation_count = max(1, round(len(groups) * 0.25))

    if train_count + validation_count >= len(groups):
        train_count = len(groups) - 2
        validation_count = 1

    train_groups = set(shuffled_groups[:train_count])
    validation_groups = set(
        shuffled_groups[train_count:train_count + validation_count]
    )
    test_groups = set(
        shuffled_groups[train_count + validation_count:]
    )

    group_assignments = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }

    manifest = {
        "seed": seed,
        "splits": {
            split_name: [
                record["record_id"]
                for record in records
                if record["group_id"] in group_assignments[split_name]
            ]
            for split_name in SPLIT_NAMES
        },
    }

    validate_manifest(records, manifest)
    return manifest


def validate_manifest(records, manifest):
    """Reject missing, duplicate, unknown, or group-overlapping IDs."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a dictionary.")

    validate_records(records)

    splits = manifest.get("splits", {})

    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("Manifest must contain train, validation, and test.")

    record_to_group = {
        record["record_id"]: record["group_id"]
        for record in records
    }
    expected_ids = set(record_to_group)
    seen_ids = set()
    seen_groups = set()

    for split_name in SPLIT_NAMES:
        split_ids = splits[split_name]

        if not isinstance(split_ids, (list, tuple)):
            raise ValueError(f"Split {split_name} IDs must be a list or tuple.")

        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Duplicate record ID in {split_name}.")

        unknown_ids = set(split_ids) - expected_ids
        if unknown_ids:
            raise ValueError(f"Unknown record IDs: {sorted(unknown_ids)}")

        overlapping_ids = seen_ids.intersection(split_ids)
        if overlapping_ids:
            raise ValueError("Record IDs overlap between splits.")

        split_groups = {record_to_group[item] for item in split_ids}
        if seen_groups.intersection(split_groups):
            raise ValueError("Groups overlap between splits.")

        seen_ids.update(split_ids)
        seen_groups.update(split_groups)

    if seen_ids != expected_ids:
        raise ValueError("Manifest must assign every record exactly once.")
