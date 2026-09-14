from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def row_value(row: Any, key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if value is None or (not isinstance(value, (list, dict)) and value != value):
        return default
    return value


def stable_id_for_row(row: Any, row_index: int) -> str:
    return str(
        row_value(row, "stable_id")
        or row_value(row, "id")
        or f"row:{row_index}"
    )


def stable_ids_sha256(stable_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for stable_id in stable_ids:
        digest.update(str(stable_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def is_ordered_subsequence(values: list[str], reference: list[str]) -> bool:
    iterator = iter(reference)
    return all(any(candidate == value for candidate in iterator) for value in values)


def write_preprocessing_audit(
    processed_dir: str | Path,
    *,
    builder: str,
    split: str,
    input_stable_ids: list[str],
    output_stable_ids: list[str],
    failures: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{builder}_{split}"
    failure_path = processed_path / f"{prefix}.failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")

    coverage_ok = (
        not failures
        and len(output_stable_ids) == len(input_stable_ids)
        and output_stable_ids == input_stable_ids
        and len(set(output_stable_ids)) == len(output_stable_ids)
    )
    report = {
        "artifact_type": "retrosynflow_native_preprocessing_audit",
        "builder": builder,
        "split": split,
        "input_rows": len(input_stable_ids),
        "output_rows": len(output_stable_ids),
        "failed_rows": len(failures),
        "coverage_ok": coverage_ok,
        "order_preserved": output_stable_ids == input_stable_ids,
        "relative_order_preserved": is_ordered_subsequence(
            output_stable_ids, input_stable_ids
        ),
        "input_stable_ids_sha256": stable_ids_sha256(input_stable_ids),
        "output_stable_ids_sha256": stable_ids_sha256(output_stable_ids),
        "failure_file": str(failure_path.resolve()),
        "statistics": statistics,
    }
    report_path = processed_path / f"{prefix}.preprocessing.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
