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

TASKS_PER_SESSION = int(os.getenv("TASKS_PER_SESSION", "10"))
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "static" / "images"


# ── Pydantic models ────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    mturk_worker_id: Optional[str] = None


class FeedbackSubmit(BaseModel):
    session_id: str
    feedback_text: Optional[str] = None
    ai_helped: Optional[str] = None        # 'yes' | 'no' | 'neutral'
    ai_comments: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pick_vehicles_for_session(conn, n: int) -> List[dict]:
    """Pick n random vehicles that have at least 2 descriptions."""
    rows = conn.execute("""
        SELECT v.id, v.image_path, v.image_name,
               COUNT(d.id) AS desc_count
        FROM vehicles v
        JOIN descriptions d ON d.vehicle_id = v.id
        GROUP BY v.id
        HAVING desc_count >= 2
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,)).fetchall()
    return [dict(r) for r in rows]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/create")
def create_session(body: SessionCreate, conn=Depends(get_db)):
    """
    Generate a new 32-char hex survey code and pre-assign vehicles to this session.
    Called when a user first visits the landing page.
    """
    survey_code = secrets.token_hex(16)   # 32 hex chars

    # Check we have enough vehicles
    vehicles = _pick_vehicles_for_session(conn, TASKS_PER_SESSION)
    if not vehicles:
        raise HTTPException(
            status_code=503,
            detail="Not enough vehicle data loaded yet. Please contact the study administrator."
        )

    conn.execute(
        "INSERT INTO sessions (id, mturk_worker_id) VALUES (?, ?)",
        (survey_code, body.mturk_worker_id)
    )

    # Pre-assign task order
    for idx, v in enumerate(vehicles, start=1):
        conn.execute(
            """INSERT INTO task_responses
               (session_id, vehicle_id, task_order)
               VALUES (?, ?, ?)""",
            (survey_code, v["id"], idx)
        )

    conn.commit()
    return {"survey_code": survey_code, "total_tasks": len(vehicles)}


@router.get("/{session_id}/task/{task_order}")
def get_task(session_id: str, task_order: int, conn=Depends(get_db)):
    """
    Return the vehicle image + shuffled candidate descriptions for a given task.
    Descriptions are ordered by CLIP score if available, otherwise random.
    """
    # Validate session
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the pre-assigned task row
    task = conn.execute(
        """SELECT tr.id as task_id, tr.vehicle_id, tr.time_started_ms,
                  v.image_path, v.image_name
           FROM task_responses tr
           JOIN vehicles v ON v.id = tr.vehicle_id
           WHERE tr.session_id = ? AND tr.task_order = ?""",
        (session_id, task_order)
    ).fetchone()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task = dict(task)

    # Stamp start time if first visit
    if task["time_started_ms"] is None:
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        conn.execute(
            "UPDATE task_responses SET time_started_ms = ? WHERE session_id = ? AND task_order = ?",
            (now_ms, session_id, task_order)
        )
        conn.commit()
        task["time_started_ms"] = now_ms

    # Fetch descriptions — CLIP-ranked if scores exist, random otherwise.
    # SQLite does not support NULLS LAST so we use a two-key sort:
    #   key 1: 0 when scored, 1 when not (puts scored rows first)
    #   key 2: clip_score DESC (highest similarity first among scored rows)
    descs = conn.execute(
        """SELECT id, text, is_correct, clip_score
           FROM descriptions
           WHERE vehicle_id = ?
           ORDER BY
               CASE WHEN clip_score IS NULL THEN 1 ELSE 0 END ASC,
               clip_score DESC""",
        (task["vehicle_id"],)
    ).fetchall()
    descs = [dict(d) for d in descs]

    # Determine total tasks in session
    total = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE session_id = ?",
        (session_id,)
    ).fetchone()[0]

    # Count completed tasks
    completed = conn.execute(
        """SELECT COUNT(*) FROM task_responses
           WHERE session_id = ? AND (selected_desc_id IS NOT NULL OR not_listed = 1)""",
        (session_id,)
    ).fetchone()[0]

    clip_ranked = any(d["clip_score"] is not None for d in descs)

    return {
        "task_id":    task["task_id"],
        "task_order": task_order,
        "total":      total,
        "completed":  completed,
        "vehicle_id": task["vehicle_id"],
        "image_url":  f"/static/images/{task['image_name']}",
        "image_name": task["image_name"],
        "descriptions": [
            {
                "id":         d["id"],
                "text":       d["text"],
                "clip_score": d["clip_score"],   # None when unscored
            }
            for d in descs
        ],
        "clip_ranked": clip_ranked,
    }


class AnswerSubmit(BaseModel):
    session_id: str
    task_order: int
    selected_desc_id: Optional[int] = None
    not_listed: bool = False


@router.post("/submit_answer")
def submit_answer(body: AnswerSubmit, conn=Depends(get_db)):
    """Record user's answer for a single task."""
    if not body.not_listed and body.selected_desc_id is None:
        raise HTTPException(status_code=422, detail="Must select a description or mark as not listed")

    # Verify task exists
    task = conn.execute(
        """SELECT tr.id, tr.vehicle_id, tr.time_started_ms
           FROM task_responses tr
           WHERE tr.session_id = ? AND tr.task_order = ?""",
        (body.session_id, body.task_order)
    ).fetchone()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    now_ms = int(datetime.utcnow().timestamp() * 1000)

    # Determine correctness
    is_correct = None
    if body.not_listed:
        # Correct if no description has is_correct=1 for this vehicle
        correct_count = conn.execute(
            "SELECT COUNT(*) FROM descriptions WHERE vehicle_id = ? AND is_correct = 1",
            (task["vehicle_id"],)
        ).fetchone()[0]
        is_correct = 1 if correct_count == 0 else 0
    elif body.selected_desc_id:
        row = conn.execute(
            "SELECT is_correct FROM descriptions WHERE id = ?",
            (body.selected_desc_id,)
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
        (
            body.selected_desc_id,
            1 if body.not_listed else 0,
            is_correct,
            now_ms,
            body.session_id,
            body.task_order,
        )
    )
    conn.commit()

    # Check if all tasks done
    total = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE session_id = ?",
        (body.session_id,)
    ).fetchone()[0]
    answered = conn.execute(
        """SELECT COUNT(*) FROM task_responses
           WHERE session_id = ? AND (selected_desc_id IS NOT NULL OR not_listed = 1)""",
        (body.session_id,)
    ).fetchone()[0]

    return {
        "is_correct": is_correct,
        "all_done": answered >= total,
        "answered": answered,
        "total": total,
    }


@router.post("/feedback")
def submit_feedback(body: FeedbackSubmit, conn=Depends(get_db)):
    """Submit post-study feedback and mark session complete."""
    conn.execute(
        """UPDATE sessions
           SET feedback_text  = ?,
               ai_helped      = ?,
               ai_comments    = ?,
               completed_at   = datetime('now')
           WHERE id = ?""",
        (body.feedback_text, body.ai_helped, body.ai_comments, body.session_id)
    )
    conn.commit()
    return {"status": "ok"}


@router.get("/{session_id}/summary")
def session_summary(session_id: str, conn=Depends(get_db)):
    """Return accuracy and timing summary for a completed session."""
    tasks = conn.execute(
        """SELECT task_order, is_correct, duration_ms, not_listed
           FROM task_responses
           WHERE session_id = ?
           ORDER BY task_order""",
        (session_id,)
    ).fetchall()
    if not tasks:
        raise HTTPException(status_code=404, detail="Session not found")

    rows = [dict(t) for t in tasks]
    correct = sum(1 for r in rows if r["is_correct"] == 1)
    total   = len(rows)
    durations = [r["duration_ms"] for r in rows if r["duration_ms"] is not None]
    avg_dur = sum(durations) / len(durations) if durations else None

    return {
        "session_id":       session_id,
        "total_tasks":      total,
        "correct":          correct,
        "accuracy_pct":     round(correct / total * 100, 1) if total else 0,
        "avg_duration_ms":  round(avg_dur) if avg_dur else None,
        "tasks":            rows,
    }