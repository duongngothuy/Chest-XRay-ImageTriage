"""All database code lives here. Plain sqlite3, no frameworks.

One table called "cases" stores every prediction and its review.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("TRIAGE_DB", "triage.db")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    label TEXT NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    review_note TEXT,
    reviewed_at TEXT
)
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = connect()
    conn.execute(CREATE_TABLE)
    conn.commit()
    conn.close()


def add_case(image_path, label, probability, model_version):
    """Save one prediction and return the new row."""
    conn = connect()
    cur = conn.execute(
        "INSERT INTO cases (image_path, label, probability, model_version, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (image_path, label, probability, model_version, now()),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def list_cases(limit, offset):
    """Return (rows, total count), newest first."""
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM cases ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    conn.close()
    return [dict(r) for r in rows], total


def get_case(case_id):
    conn = connect()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_review(case_id, decision, note):
    """Store a human review for one case."""
    conn = connect()
    conn.execute(
        "UPDATE cases SET review_status = ?, review_note = ?, reviewed_at = ?"
        " WHERE id = ?",
        (decision, note, now(), case_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    return dict(row)


def stats():
    """Simple counts for the stats page."""
    conn = connect()
    rows = conn.execute(
        "SELECT review_status, COUNT(*) AS n FROM cases GROUP BY review_status"
    ).fetchall()
    conn.close()
    counts = {row["review_status"]: row["n"] for row in rows}
    total = sum(counts.values())
    confirmed = counts.get("confirmed", 0)
    overridden = counts.get("overridden", 0)
    reviewed = confirmed + overridden
    # How often the human agreed with the model, out of all reviewed cases.
    accuracy = confirmed / reviewed if reviewed else None
    return {
        "total": total,
        "pending": counts.get("pending", 0),
        "confirmed": confirmed,
        "overridden": overridden,
        "accuracy": accuracy,
    }
