from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None


BASE_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = BASE_DIR / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".txt", ".jl"}
REQUIRED_SOURCE_FIELDS = {"text"}
OPTIONAL_REVIEW_FIELDS = [
    "nmId",
    "productValuation",
    "color",
    "text",
    "answer",
    "fake_label",
    "_reviewmark_id",
    "_reviewmark_source_index",
    "predicted_label",
    "predicted_proba",
    "source_file",
]

FIELD_ALIASES = {
    "nmId": "nmId",
    "product_id": "nmId",
    "productValuation": "productValuation",
    "rating": "productValuation",
    "color": "color",
    "text": "text",
    "review_text": "text",
    "answer": "answer",
    "seller_answer": "answer",
    "fake_label": "fake_label",
    "label": "fake_label",
    "_reviewmark_id": "_reviewmark_id",
    "_reviewmark_source_index": "_reviewmark_source_index",
    "predicted_label": "predicted_label",
    "predicted_proba": "predicted_proba",
    "probability_fake": "predicted_proba",
    "source_file": "source_file",
}


def parse_json_stream(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    records: list[dict[str, Any]] = []
    text = text.lstrip("\ufeff")

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break

        item, next_index = decoder.raw_decode(text, index)
        if isinstance(item, list):
            for nested in item:
                if isinstance(nested, dict):
                    records.append(nested)
        elif isinstance(item, dict):
            records.append(item)
        index = next_index

    return records


def read_text_dataset(path: Path) -> list[dict[str, Any]]:
    return parse_json_stream(path.read_text(encoding="utf-8"))


def read_zst_dataset(path: Path) -> list[dict[str, Any]]:
    if zstd is None:
        raise ValueError("Для чтения .json.zst нужен установленный пакет zstandard.")
    with path.open("rb") as file:
        text = zstd.ZstdDecompressor().stream_reader(file).read().decode("utf-8")
    return parse_json_stream(text)


def load_records(path: Path) -> list[dict[str, Any]]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".json", ".zst"]:
        records = read_zst_dataset(path)
    elif suffixes and suffixes[-1] in {".json", ".jsonl", ".txt", ".jl"}:
        records = read_text_dataset(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.name}")

    if not records:
        raise ValueError("Файл не содержит ни одной JSON-записи.")
    return records


def normalize_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        target = FIELD_ALIASES.get(key)
        if target:
            normalized[target] = value

    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in normalized]
    if missing:
        raise ValueError(
            "В записи отсутствуют обязательные поля: " + ", ".join(sorted(missing))
        )

    normalized.setdefault("_reviewmark_source_index", source_index)
    normalized.setdefault(
        "_reviewmark_id",
        f"{normalized.get('nmId', 'record')}__{normalized['_reviewmark_source_index']}",
    )
    normalized.setdefault("fake_label", record.get("fake_label"))
    return normalized


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        normalized.append(normalize_record(record, index))
    if not normalized:
        raise ValueError("После нормализации не осталось корректных записей.")
    return normalized


def save_jsonl(path: Path, records: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def build_internal_id(record: dict[str, Any], source_index: int) -> str:
    basis = "|".join(
        [
            to_text(record.get("nmId")),
            to_text(record.get("productValuation")),
            to_text(record.get("color")),
            to_text(record.get("text")),
            to_text(record.get("answer")),
            str(source_index),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"generated_{source_index}_{digest}"


def canonicalize_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    normalized = {field: None for field in OPTIONAL_REVIEW_FIELDS}
    for key, value in record.items():
        target = FIELD_ALIASES.get(key)
        if target:
            normalized[target] = value

    normalized["text"] = to_text(normalized.get("text"))
    normalized["answer"] = to_text(normalized.get("answer"))
    normalized["color"] = to_text(normalized.get("color"))
    normalized["source_file"] = to_text(normalized.get("source_file"))
    normalized["_reviewmark_source_index"] = safe_int(
        normalized.get("_reviewmark_source_index"),
        default=source_index,
    )
    normalized["_reviewmark_id"] = to_text(normalized.get("_reviewmark_id")) or build_internal_id(
        normalized,
        source_index,
    )
    return normalized


def comparable_payload(record: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(record)
    comparable.pop("_reviewmark_source_index", None)
    comparable.pop("source_file", None)
    return comparable


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            normalized.append(canonicalize_record(record, index))
    if not normalized:
        raise ValueError("После нормализации не осталось корректных записей.")
    return normalized


def deduplicate_records(
    records: list[dict[str, Any]],
    *,
    key: str = "_reviewmark_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []
    duplicates: list[str] = []
    conflicting_duplicates: list[str] = []

    for record in records:
        value = to_text(record.get(key))
        if not value:
            unique.append(record)
            continue

        existing = seen.get(value)
        if existing is not None:
            duplicates.append(value)
            if comparable_payload(existing) != comparable_payload(record):
                conflicting_duplicates.append(value)
            continue

        seen[value] = record
        unique.append(record)

    return unique, {
        "deduplicated_by": key,
        "duplicate_count": len(duplicates),
        "duplicate_ids": duplicates[:20],
        "conflicting_duplicate_count": len(conflicting_duplicates),
        "conflicting_duplicate_ids": conflicting_duplicates[:20],
        "duplicate_policy": "kept_first_record",
    }


def validate_labeling_records(
    records: list[dict[str, Any]],
    *,
    source_file: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = normalize_records(records)
    valid: list[dict[str, Any]] = []
    invalid_count = 0

    for record in normalized:
        record["source_file"] = source_file or record.get("source_file")
        if to_text(record.get("text")):
            valid.append(record)
        else:
            invalid_count += 1

    if not valid:
        raise ValueError("Для разметки нужен хотя бы один отзыв с непустым полем text.")

    unique, duplicate_info = deduplicate_records(valid)
    return unique, {
        "source_file": source_file,
        "total_input": len(records),
        "valid_records": len(unique),
        "filtered_empty_text": invalid_count,
        **duplicate_info,
    }


def validate_training_records(
    records: list[dict[str, Any]],
    *,
    source_file: str | None = None,
    deduplicate: bool = True,
    min_records: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = normalize_records(records)
    valid: list[dict[str, Any]] = []
    filtered_empty_text = 0
    filtered_invalid_label = 0

    for record in normalized:
        record["source_file"] = source_file or record.get("source_file")
        if not to_text(record.get("text")):
            filtered_empty_text += 1
            continue

        label = safe_int(record.get("fake_label"))
        if label not in {0, 1}:
            filtered_invalid_label += 1
            continue

        record["fake_label"] = label
        valid.append(record)

    if deduplicate:
        valid, duplicate_info = deduplicate_records(valid)
    else:
        duplicate_info = {
            "deduplicated_by": None,
            "duplicate_count": 0,
            "duplicate_ids": [],
            "conflicting_duplicate_count": 0,
            "conflicting_duplicate_ids": [],
            "duplicate_policy": "disabled",
        }

    if len(valid) < min_records:
        raise ValueError(
            f"После фильтрации осталось слишком мало размеченных записей: {len(valid)}. Нужно минимум {min_records}."
        )

    labels = {record["fake_label"] for record in valid}
    if labels != {0, 1}:
        raise ValueError("Для обучения нужны корректные метки обоих классов: fake_label = 0 и fake_label = 1.")

    return valid, {
        "source_file": source_file,
        "total_input": len(records),
        "valid_records": len(valid),
        "filtered_empty_text": filtered_empty_text,
        "filtered_invalid_label": filtered_invalid_label,
        **duplicate_info,
    }


def validate_prediction_records(
    records: list[dict[str, Any]],
    *,
    source_file: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = normalize_records(records)
    valid: list[dict[str, Any]] = []
    filtered_empty_text = 0

    for record in normalized:
        record["source_file"] = source_file or record.get("source_file")
        record.pop("predicted_label", None)
        record.pop("predicted_proba", None)
        if to_text(record.get("text")):
            valid.append(record)
        else:
            filtered_empty_text += 1

    if not valid:
        raise ValueError("Для прогнозирования нужен хотя бы один отзыв с непустым полем text.")

    return valid, {
        "source_file": source_file,
        "total_input": len(records),
        "valid_records": len(valid),
        "filtered_empty_text": filtered_empty_text,
    }
