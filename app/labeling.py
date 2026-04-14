from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .dataio import save_jsonl, validate_labeling_records


BASE_DIR = Path(__file__).resolve().parent.parent
LABELING_DIR = BASE_DIR / "data" / "labeling_sessions"
LABELING_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_paths(session_id: str) -> dict[str, Path]:
    session_dir = LABELING_DIR / session_id
    return {
        "dir": session_dir,
        "state": session_dir / "session.json",
        "source": session_dir / "source.jsonl",
        "output": session_dir / "labeled_output.jsonl",
        "skipped": session_dir / "skipped.jsonl",
    }


def create_session(records: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    session_id = f"label_{uuid4().hex}"
    paths = session_paths(session_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    normalized, stats = validate_labeling_records(records, source_file=source_name)
    save_jsonl(paths["source"], normalized)

    session = {
        "session_id": session_id,
        "source_name": source_name,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "total": len(normalized),
        "labeled_count": 0,
        "skipped_count": 0,
        "current_index": 0,
        "decisions": {},
        "validation": stats,
    }
    save_session(session)
    return session


def save_session(session: dict[str, Any]) -> None:
    paths = session_paths(session["session_id"])
    paths["dir"].mkdir(parents=True, exist_ok=True)
    session["updated_at"] = utc_now()
    paths["state"].write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> dict[str, Any]:
    paths = session_paths(session_id)
    return json.loads(paths["state"].read_text(encoding="utf-8"))


def list_sessions() -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for state_file in LABELING_DIR.glob("*/session.json"):
        sessions.append(json.loads(state_file.read_text(encoding="utf-8")))
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def load_source_records(session_id: str) -> list[dict[str, Any]]:
    paths = session_paths(session_id)
    records: list[dict[str, Any]] = []
    with paths["source"].open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_progress(session: dict[str, Any]) -> dict[str, int]:
    total = int(session["total"])
    labeled = int(session.get("labeled_count", 0))
    skipped = int(session.get("skipped_count", 0))
    processed = labeled + skipped
    return {
        "total": total,
        "labeled": labeled,
        "skipped": skipped,
        "remaining": max(total - processed, 0),
        "percent": int((processed / total) * 100) if total else 0,
    }


def get_next_record(session: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    decisions = session.get("decisions", {})
    for index, record in enumerate(records):
        record_id = record["_reviewmark_id"]
        if record_id not in decisions:
            session["current_index"] = index
            return record
    session["current_index"] = len(records)
    return None


def apply_decision(session_id: str, record_id: str, decision: str) -> dict[str, Any]:
    session = load_session(session_id)
    records = load_source_records(session_id)
    record_map = {record["_reviewmark_id"]: record for record in records}
    if record_id not in record_map:
        raise ValueError("Запись для разметки не найдена.")
    if decision not in {"0", "1", "skip"}:
        raise ValueError("Некорректное действие разметки.")

    current = record_map[record_id].copy()
    session["decisions"][record_id] = decision

    paths = session_paths(session_id)
    if decision == "skip":
        session["skipped_count"] = int(session.get("skipped_count", 0)) + 1
        current["status"] = "skipped"
        current["labeling_session_id"] = session_id
        current["labeled_at"] = utc_now()
        current["source_file"] = session.get("source_name")
        save_jsonl(paths["skipped"], [current], append=True)
    else:
        current["fake_label"] = int(decision)
        current["labeling_session_id"] = session_id
        current["labeled_at"] = utc_now()
        current["source_file"] = session.get("source_name")
        session["labeled_count"] = int(session.get("labeled_count", 0)) + 1
        save_jsonl(paths["output"], [current], append=True)

    save_session(session)
    return session
