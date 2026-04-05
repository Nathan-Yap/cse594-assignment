import sqlite3
import os
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).parent / "labelling.db")))


def get_conn() -> sqlite3.Connection:
    """
    Open a new SQLite connection.
    check_same_thread=False is safe here because:
      - All endpoints are plain `def` (FastAPI runs them in a threadpool).
      - Each request gets its own connection opened and closed in the same thread.
      - WAL mode allows concurrent reads without locking.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db():
    """FastAPI dependency — yields one connection per request."""
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS vehicles (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path  TEXT NOT NULL UNIQUE,
        image_name  TEXT NOT NULL,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS descriptions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id  INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
        text        TEXT NOT NULL,
        is_correct  INTEGER NOT NULL DEFAULT 0,
        clip_score  REAL,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id                  TEXT PRIMARY KEY,
        mturk_worker_id     TEXT,
        condition           TEXT NOT NULL DEFAULT 'ai',   -- 'ai' | 'control'
        started_at          TEXT DEFAULT (datetime('now')),
        completed_at        TEXT,
        feedback_text       TEXT,
        ai_helped           TEXT,
        ai_comments         TEXT,
        confidence_rating   INTEGER,   -- 1-7 Likert: confidence in classifications
        mental_effort       INTEGER    -- 1-7 Likert: mental effort invested
    );

    CREATE TABLE IF NOT EXISTS task_responses (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        vehicle_id          INTEGER NOT NULL REFERENCES vehicles(id),
        task_order          INTEGER NOT NULL,
        selected_desc_id    INTEGER REFERENCES descriptions(id),
        not_listed          INTEGER NOT NULL DEFAULT 0,
        is_correct          INTEGER,
        time_started_ms     INTEGER,
        time_submitted_ms   INTEGER,
        duration_ms         INTEGER GENERATED ALWAYS AS (
                                CASE WHEN time_submitted_ms IS NOT NULL AND time_started_ms IS NOT NULL
                                THEN time_submitted_ms - time_started_ms ELSE NULL END
                            ) STORED,
        created_at          TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_task_session ON task_responses(session_id);
    CREATE INDEX IF NOT EXISTS idx_desc_vehicle  ON descriptions(vehicle_id);
    """)
    conn.commit()
    # ── Migrations (safe to run on existing DBs) ──────────────────────────────
    cols = [row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "condition" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN condition TEXT NOT NULL DEFAULT 'ai'")
        conn.commit()
        print("Migrated: added sessions.condition column")
    if "confidence_rating" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN confidence_rating INTEGER")
        conn.commit()
        print("Migrated: added sessions.confidence_rating column")
    if "mental_effort" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN mental_effort INTEGER")
        conn.commit()
        print("Migrated: added sessions.mental_effort column")

    conn.close()
    print(f"DB at {DB_PATH}")