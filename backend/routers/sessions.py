import secrets
import os
import random
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database import get_db

router = APIRouter()

BASE_DIR   = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "static" / "images"

VALID_CONDITIONS = {"ai", "control"}


# ── Pydantic models ────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    mturk_worker_id: Optional[str] = None
    condition: str = "ai"          # "ai" | "control"


class FeedbackSubmit(BaseModel):
    session_id:    str
    feedback_text: Optional[str] = None
    ai_helped:     Optional[str] = None   # 'yes' | 'no' | 'neutral'
    ai_comments:   Optional[str] = None


class AnswerSubmit(BaseModel):
    session_id:       str
    task_order:       int
    selected_desc_id: Optional[int] = None
    not_listed:       bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tasks_per_session() -> int:
    return int(os.getenv("TASKS_PER_SESSION", "10"))


def _pick_vehicles(conn, n: int) -> List[dict]:
    """Pick n random vehicles that have at least 1 description."""
    rows = conn.execute("""
        SELECT v.id, v.image_path, v.image_name,
               COUNT(d.id) AS desc_count
        FROM   vehicles v
        JOIN   descriptions d ON d.vehicle_id = v.id
        GROUP  BY v.id
        HAVING desc_count >= 1
        ORDER  BY RANDOM()
        LIMIT  ?
    """, (n,)).fetchall()
    return [dict(r) for r in rows]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/create")
def create_session(body: SessionCreate, conn=Depends(get_db)):
    """
    Generate a 32-char hex survey code and pre-assign vehicles.

    condition="ai"      → descriptions served CLIP-ranked (highest similarity first)
    condition="control" → descriptions served in random order, no CLIP scores exposed
    """
    condition = body.condition.lower().strip()
    if condition not in VALID_CONDITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"condition must be one of: {', '.join(VALID_CONDITIONS)}"
        )

    n = _tasks_per_session()

    total_vehicles = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    eligible = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT v.id FROM vehicles v
            JOIN descriptions d ON d.vehicle_id = v.id
            GROUP BY v.id HAVING COUNT(d.id) >= 1
        )
    """).fetchone()[0]

    vehicles = _pick_vehicles(conn, n)

    if not vehicles:
        if total_vehicles == 0:
            detail = "No vehicles in the database. Upload images and descriptions via the admin panel first."
        elif eligible == 0:
            detail = f"{total_vehicles} vehicle(s) found but none have any descriptions attached."
        else:
            detail = (
                f"Only {eligible} eligible vehicle(s) available but {n} tasks required. "
                "Add more vehicles or reduce 'Tasks per session' in admin config."
            )
        raise HTTPException(status_code=503, detail=detail)

    survey_code = secrets.token_hex(16)

    conn.execute(
        "INSERT INTO sessions (id, mturk_worker_id, condition) VALUES (?, ?, ?)",
        (survey_code, body.mturk_worker_id, condition)
    )
    for idx, v in enumerate(vehicles, start=1):
        conn.execute(
            "INSERT INTO task_responses (session_id, vehicle_id, task_order) VALUES (?, ?, ?)",
            (survey_code, v["id"], idx)
        )
    conn.commit()

    return {"survey_code": survey_code, "total_tasks": len(vehicles), "condition": condition}


@router.get("/{session_id}/task/{task_order}")
def get_task(session_id: str, task_order: int, conn=Depends(get_db)):
    """
    Return vehicle image + descriptions for one task.

    AI condition:      descriptions sorted by CLIP score DESC (most similar first).
                       clip_score values included in response.
    Control condition: descriptions shuffled randomly each time.
                       clip_score is always None in the response (scores are hidden).
    """
    session = conn.execute(
        "SELECT id, condition FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    condition = (session["condition"] or "ai").lower()

    task = conn.execute("""
        SELECT tr.id as task_id, tr.vehicle_id, tr.time_started_ms,
               v.image_path, v.image_name
        FROM   task_responses tr
        JOIN   vehicles v ON v.id = tr.vehicle_id
        WHERE  tr.session_id = ? AND tr.task_order = ?
    """, (session_id, task_order)).fetchone()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task = dict(task)

    # Stamp start time on first visit
    if task["time_started_ms"] is None:
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        conn.execute(
            "UPDATE task_responses SET time_started_ms = ? WHERE session_id = ? AND task_order = ?",
            (now_ms, session_id, task_order)
        )
        conn.commit()
        task["time_started_ms"] = now_ms

    # Fetch raw descriptions
    descs = conn.execute(
        "SELECT id, text, is_correct, clip_score FROM descriptions WHERE vehicle_id = ?",
        (task["vehicle_id"],)
    ).fetchall()
    descs = [dict(d) for d in descs]

    # Order and expose scores depending on condition
    if condition == "ai":
        # CLIP-ranked: scored rows first (DESC), unscored rows after (RANDOM within)
        scored   = sorted([d for d in descs if d["clip_score"] is not None],
                          key=lambda d: d["clip_score"], reverse=True)
        unscored = [d for d in descs if d["clip_score"] is None]
        random.shuffle(unscored)
        descs = scored + unscored
        clip_ranked = bool(scored)
        desc_payload = [
            {"id": d["id"], "text": d["text"], "clip_score": d["clip_score"]}
            for d in descs
        ]
    else:
        # Control: fully random order, scores hidden from client
        random.shuffle(descs)
        clip_ranked = False
        desc_payload = [
            {"id": d["id"], "text": d["text"], "clip_score": None}
            for d in descs
        ]

    total = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    completed = conn.execute(
        """SELECT COUNT(*) FROM task_responses
           WHERE session_id = ? AND (selected_desc_id IS NOT NULL OR not_listed = 1)""",
        (session_id,)
    ).fetchone()[0]

    return {
        "task_id":      task["task_id"],
        "task_order":   task_order,
        "total":        total,
        "completed":    completed,
        "condition":    condition,
        "vehicle_id":   task["vehicle_id"],
        "image_url":    f"/static/images/{task['image_name']}",
        "image_name":   task["image_name"],
        "descriptions": desc_payload,
        "clip_ranked":  clip_ranked,
    }


@router.post("/submit_answer")
def submit_answer(body: AnswerSubmit, conn=Depends(get_db)):
    """Record user's answer for a single task."""
    if not body.not_listed and body.selected_desc_id is None:
        raise HTTPException(status_code=422, detail="Must select a description or mark as not listed")

    task = conn.execute(
        """SELECT tr.id, tr.vehicle_id, tr.time_started_ms
           FROM task_responses tr
           WHERE tr.session_id = ? AND tr.task_order = ?""",
        (body.session_id, body.task_order)
    ).fetchone()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    now_ms = int(datetime.utcnow().timestamp() * 1000)

    is_correct = None
    if body.not_listed:
        correct_count = conn.execute(
            "SELECT COUNT(*) FROM descriptions WHERE vehicle_id = ? AND is_correct = 1",
            (task["vehicle_id"],)
        ).fetchone()[0]
        is_correct = 1 if correct_count == 0 else 0
    elif body.selected_desc_id:
        row = conn.execute(
            "SELECT is_correct FROM descriptions WHERE id = ?", (body.selected_desc_id,)
        ).fetchone()
        if row:
            is_correct = row["is_correct"]

    conn.execute(
        """UPDATE task_responses
           SET selected_desc_id  = ?,
               not_listed        = ?,
               is_correct        = ?,
               time_submitted_ms = ?
           WHERE session_id = ? AND task_order = ?""",
        (body.selected_desc_id, 1 if body.not_listed else 0,
         is_correct, now_ms, body.session_id, body.task_order)
    )
    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE session_id = ?", (body.session_id,)
    ).fetchone()[0]
    answered = conn.execute(
        """SELECT COUNT(*) FROM task_responses
           WHERE session_id = ? AND (selected_desc_id IS NOT NULL OR not_listed = 1)""",
        (body.session_id,)
    ).fetchone()[0]

    return {
        "is_correct": is_correct,
        "all_done":   answered >= total,
        "answered":   answered,
        "total":      total,
    }


@router.post("/feedback")
def submit_feedback(body: FeedbackSubmit, conn=Depends(get_db)):
    """Submit post-study feedback and mark session complete."""
    conn.execute(
        """UPDATE sessions
           SET feedback_text = ?, ai_helped = ?, ai_comments = ?,
               completed_at  = datetime('now')
           WHERE id = ?""",
        (body.feedback_text, body.ai_helped, body.ai_comments, body.session_id)
    )
    conn.commit()
    return {"status": "ok"}


@router.get("/{session_id}/summary")
def session_summary(session_id: str, conn=Depends(get_db)):
    """Accuracy and timing summary for a completed session."""
    tasks = conn.execute(
        """SELECT task_order, is_correct, duration_ms, not_listed
           FROM task_responses WHERE session_id = ? ORDER BY task_order""",
        (session_id,)
    ).fetchall()
    if not tasks:
        raise HTTPException(status_code=404, detail="Session not found")

    rows      = [dict(t) for t in tasks]
    correct   = sum(1 for r in rows if r["is_correct"] == 1)
    total     = len(rows)
    durations = [r["duration_ms"] for r in rows if r["duration_ms"] is not None]
    avg_dur   = sum(durations) / len(durations) if durations else None

    session = conn.execute(
        "SELECT condition FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()

    return {
        "session_id":      session_id,
        "condition":       session["condition"] if session else None,
        "total_tasks":     total,
        "correct":         correct,
        "accuracy_pct":    round(correct / total * 100, 1) if total else 0,
        "avg_duration_ms": round(avg_dur) if avg_dur else None,
        "tasks":           rows,
    }