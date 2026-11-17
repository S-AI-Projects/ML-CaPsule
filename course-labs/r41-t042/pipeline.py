"""Deterministic, leakage-aware splitting for a small text fixture.

This module uses only the Python standard library. It creates split manifests;
it does not train a sentiment model or make claims about an IMDB corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import random
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_GROUP_FIELDS = ("group_id", "user_id")


class SplitValidationError(ValueError):
    """Raised when input records or a split manifest violate the contract."""


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if not isinstance(value, str) or not value.strip():
        return "<missing>"
    return value.strip()


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate required text-fixture fields and unique record identifiers."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SplitValidationError("record_id=<collection>, field=records, reason=expected a sequence of records")

    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SplitValidationError(
                f"record_id=<row:{index}>, field=record, reason=expected an object"
            )

        record_id = _record_id(record)
        if record_id == "<missing>":
            raise SplitValidationError(
                f"record_id=<missing>, field=record_id, reason=required non-empty string"
            )
        if record["record_id"] != record_id:
            raise SplitValidationError(
                f"record_id={record_id}, field=record_id, reason=surrounding whitespace is not allowed"
            )
        if record_id in seen:
            raise SplitValidationError(
                f"record_id={record_id}, field=record_id, reason=duplicate record ID"
            )
        seen.add(record_id)

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SplitValidationError(
                f"record_id={record_id}, field=text, reason=required non-empty string"
            )

        label = record.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise SplitValidationError(
                f"record_id={record_id}, field=label, reason=expected integer 0 or 1"
            )


def _get_groups(
    records: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[tuple[tuple[str, Any], ...]]:
    """Return row group values, rejecting partial or invalid group metadata."""
    groups: list[tuple[tuple[str, Any], ...]] = []
    for record in records:
        row_groups: list[tuple[str, Any]] = []
        for field in group_fields:
            present = [field in row for row in records]
            if any(present) and not all(present):
                raise SplitValidationError(
                    f"record_id={_record_id(record)}, field={field}, reason=group metadata must be present on every record or none"
                )
            value = record.get(field)
            if field in record:
                if value is None or value == "":
                    raise SplitValidationError(
                        f"record_id={_record_id(record)}, field={field}, reason=group value must be non-empty"
                    )
                try:
                    hash(value)
                except TypeError as error:
                    raise SplitValidationError(
                        f"record_id={_record_id(record)}, field={field}, reason=group value must be hashable"
                    ) from error
                row_groups.append((field, value))
        groups.append(tuple(row_groups))
    return groups


def _parse_time(record: Mapping[str, Any], field: str) -> datetime:
    record_id = _record_id(record)
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise SplitValidationError(
            f"record_id={record_id}, field={field}, reason=expected ISO-8601 datetime string"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SplitValidationError(
            f"record_id={record_id}, field={field}, reason=invalid ISO-8601 datetime"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _components(
    records: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[list[int]]:
    """Build atomic connected components so group/user links cannot leak."""
    row_groups = _get_groups(records, group_fields)
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    owners: dict[tuple[str, Any], int] = {}
    for index, values in enumerate(row_groups):
        for field, value in values:
            key = (field, value)
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    by_root: dict[int, list[int]] = {}
    for index in range(len(records)):
        by_root.setdefault(find(index), []).append(index)
    return list(by_root.values())


def _choose_boundary(
    units: Sequence[list[int]], target_rows: int, *, minimum_remaining: int = 1
) -> int:
    """Pick a deterministic non-empty unit boundary nearest a target size."""
    cumulative = 0
    candidates: list[tuple[int, int]] = []
    for cut, unit in enumerate(units[:-minimum_remaining], start=1):
        cumulative += len(unit)
        if len(units) - cut >= minimum_remaining:
            candidates.append((abs(cumulative - target_rows), cut))
    if not candidates:
        raise SplitValidationError(
            "record_id=<collection>, field=splits, reason=at least three independent groups are required"
        )
    return min(candidates)[1]


def create_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 41,
    group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
    time_field: str | None = None,
) -> dict[str, Any]:
    """Create stable train/validation/test ID lists with group/time isolation.

    In random mode the independent group components are shuffled by ``seed``.
    When ``time_field`` is supplied, components are kept in chronological order
    and split boundaries are contiguous. Each split aims for approximately
    67%/17%/16% of records; groups are never split to hit an exact size.
    """
    validate_records(records)
    if len(records) < 3:
        raise SplitValidationError(
            "record_id=<collection>, field=splits, reason=at least three records are required"
        )

    units = _components(records, group_fields)
    if len(units) < 3:
        raise SplitValidationError(
            "record_id=<collection>, field=splits, reason=at least three independent groups are required"
        )

    times: list[datetime] | None = None
    if time_field is not None:
        present = [time_field in row for row in records]
        if not all(present):
            missing_id = _record_id(records[present.index(False)])
            raise SplitValidationError(
                f"record_id={missing_id}, field={time_field}, reason=time field required on every record"
            )
        times = [_parse_time(record, time_field) for record in records]
        units.sort(key=lambda unit: (min(times[i] for i in unit), unit[0]))
        for current, following in zip(units, units[1:]):
            if max(times[i] for i in current) > min(times[i] for i in following):
                raise SplitValidationError(
                    f"record_id={_record_id(records[following[0]])}, field={time_field}, reason=group and chronological isolation conflict"
                )
        for unit in units:
            unit.sort(key=lambda i: (times[i], i))
    else:
        random.Random(seed).shuffle(units)

    count = len(records)
    train_target = max(1, int(count * 0.67))
    validation_target = max(1, int(count * 0.17))
    train_cut = _choose_boundary(units, train_target, minimum_remaining=2)
    remaining = units[train_cut:]
    validation_cut = _choose_boundary(remaining, validation_target)
    split_units = (units[:train_cut], remaining[:validation_cut], remaining[validation_cut:])
    splits = {
        name: [records[index]["record_id"] for unit in split for index in unit]
        for name, split in zip(SPLIT_NAMES, split_units)
    }
    manifest = {"seed": seed, "splits": splits}
    validate_manifest(records, manifest, group_fields=group_fields, time_field=time_field)
    return manifest


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
    time_field: str | None = None,
) -> None:
    """Reject unknown, missing, duplicated, cross-group, or out-of-time IDs."""
    validate_records(records)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("splits"), Mapping):
        raise SplitValidationError(
            "record_id=<manifest>, field=splits, reason=expected a split mapping"
        )
    splits = manifest["splits"]
    if set(splits) != set(SPLIT_NAMES):
        raise SplitValidationError(
            "record_id=<manifest>, field=splits, reason=expected exactly train, validation, and test"
        )

    record_by_id = {_record_id(record): record for record in records}
    assigned: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        ids = splits[split_name]
        if not isinstance(ids, list):
            raise SplitValidationError(
                f"record_id=<manifest>, field={split_name}, reason=expected a list of IDs"
            )
        for record_id in ids:
            if not isinstance(record_id, str) or record_id not in record_by_id:
                raise SplitValidationError(
                    f"record_id={record_id}, field={split_name}, reason=unknown record ID"
                )
            if record_id in assigned:
                raise SplitValidationError(
                    f"record_id={record_id}, field={split_name}, reason=ID already assigned to {assigned[record_id]}"
                )
            assigned[record_id] = split_name

    missing = set(record_by_id) - set(assigned)
    if missing:
        record_id = sorted(missing)[0]
        raise SplitValidationError(
            f"record_id={record_id}, field=splits, reason=record missing from manifest"
        )

    group_values = _get_groups(records, group_fields)
    group_splits: dict[tuple[str, Any], str] = {}
    for record, values in zip(records, group_values):
        split_name = assigned[_record_id(record)]
        for field, value in values:
            key = (field, value)
            prior = group_splits.setdefault(key, split_name)
            if prior != split_name:
                raise SplitValidationError(
                    f"record_id={_record_id(record)}, field={field}, reason=group appears in {prior} and {split_name}"
                )

    if time_field is not None:
        for record in records:
            if time_field not in record:
                raise SplitValidationError(
                    f"record_id={_record_id(record)}, field={time_field}, reason=time field required on every record"
                )
        ordered_ids = [record_id for split_name in SPLIT_NAMES for record_id in splits[split_name]]
        ordered_times = [_parse_time(record_by_id[record_id], time_field) for record_id in ordered_ids]
        if ordered_times != sorted(ordered_times):
            raise SplitValidationError(
                f"record_id={ordered_ids[0] if ordered_ids else '<missing>'}, field={time_field}, reason=splits are not chronological"
            )