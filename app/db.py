import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "reviews.db"
FALLBACK_DB_PATH = Path(tempfile.gettempdir()) / "trustberry" / "reviews.db"


def resolve_db_path() -> Path:
    configured = os.getenv("TRUSTBERRY_DB_PATH")
    if configured:
        path = Path(configured).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    FALLBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FALLBACK_DB_PATH


DB_PATH = resolve_db_path()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                rating INTEGER,
                color TEXT,
                review_text TEXT NOT NULL,
                seller_answer TEXT,
                fake_label INTEGER,
                probability_fake REAL,
                predicted_label INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_reviews(rows: List[Dict[str, Any]]) -> int:
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO reviews (
                product_id, rating, color, review_text, seller_answer,
                fake_label, probability_fake, predicted_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("product_id"),
                    row.get("rating"),
                    row.get("color"),
                    row.get("review_text"),
                    row.get("seller_answer"),
                    row.get("fake_label"),
                    row.get("probability_fake"),
                    row.get("predicted_label"),
                )
                for row in rows
            ],
        )
        conn.commit()
        return len(rows)


def fetch_recent(limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]


def count_reviews() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM reviews")
        return int(cur.fetchone()[0])
