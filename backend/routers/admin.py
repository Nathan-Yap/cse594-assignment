import os
import csv
import io
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from database import get_db, get_conn

router = APIRouter()
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic ───────────────────────────────────────────────────────────────────

class DescriptionAdd(BaseModel):
    vehicle_id: int
    text: str
    is_correct: bool = False


class VehicleBulkAdd(BaseModel):
    image_name: str
    descriptions: List[dict]


class ConfigUpdate(BaseModel):
    tasks_per_session: Optional[int] = None


# ── Dataset management ─────────────────────────────────────────────────────────

@router.get("/vehicles")
def list_vehicles(conn=Depends(get_db)):
    rows = conn.execute("""
        SELECT v.id, v.image_name, v.image_path,
               COUNT(d.id) as desc_count,
               SUM(d.is_correct) as correct_count
        FROM vehicles v
        LEFT JOIN descriptions d ON d.vehicle_id = v.id
        GROUP BY v.id
        ORDER BY v.id DESC
    """).fetchall()
    return [dict(r) for r in rows]


@router.post("/vehicles")
def add_vehicle(body: VehicleBulkAdd, conn=Depends(get_db)):
    image_path = str(IMAGES_DIR / body.image_name)
    if not Path(image_path).exists():
        raise HTTPException(status_code=400, detail=f"Image not found at {image_path}")

    cur = conn.execute(
        "INSERT OR IGNORE INTO vehicles (image_path, image_name) VALUES (?, ?)",
        (image_path, body.image_name)
    )
    vehicle_id = cur.lastrowid or conn.execute(
        "SELECT id FROM vehicles WHERE image_name = ?", (body.image_name,)
    ).fetchone()["id"]

    for d in body.descriptions:
        conn.execute(
            "INSERT INTO descriptions (vehicle_id, text, is_correct) VALUES (?, ?, ?)",
            (vehicle_id, d["text"], 1 if d.get("is_correct") else 0)
        )

    conn.commit()
    return {"vehicle_id": vehicle_id}


# ── File uploads (async to allow await file.read(), DB via get_conn()) ─────────

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    async only for await file.read() — no SQLite used here so no thread issue.
    """
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Image must be jpg/png/webp/gif")

    contents = await file.read()          # async file read — fine in event loop
    dest = IMAGES_DIR / file.filename
    dest.write_bytes(contents)            # sync disk write — fine, no SQLite
    return {"filename": file.filename, "url": f"/static/images/{file.filename}"}


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Reads the uploaded JSON, then does all DB work with a fresh connection
    opened in THIS coroutine's execution context to avoid thread-id mismatch.

    Format:
    [
      {
        "image_name": "car_001.jpg",
        "descriptions": [
          {"text": "A red sedan", "is_correct": true},
          {"text": "A blue SUV",  "is_correct": false}
        ]
      }
    ]
    """
    contents = await file.read()
    try:
        data = json.loads(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON — could not parse file")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be a list of vehicle objects")

    # Open a dedicated connection right here so SQLite sees one consistent thread
    conn = get_conn()
    try:
        added, skipped = 0, 0
        for entry in data:
            img   = (entry.get("image_name") or "").strip()
            descs = entry.get("descriptions", [])
            if not img:
                skipped += 1
                continue
            image_path = str(IMAGES_DIR / img)
            if not Path(image_path).exists():
                skipped += 1
                continue

            cur = conn.execute(
                "INSERT OR IGNORE INTO vehicles (image_path, image_name) VALUES (?, ?)",
                (image_path, img)
            )
            if cur.lastrowid:
                vehicle_id = cur.lastrowid
            else:
                row = conn.execute(
                    "SELECT id FROM vehicles WHERE image_name = ?", (img,)
                ).fetchone()
                vehicle_id = row["id"]

            for d in descs:
                text = (d.get("text") or "").strip()
                if not text:
                    continue
                # Accept pre-computed CLIP scores from the Colab notebook output
                clip_score = d.get("clip_score")
                if clip_score is not None:
                    try:
                        clip_score = float(clip_score)
                    except (TypeError, ValueError):
                        clip_score = None
                conn.execute(
                    """INSERT INTO descriptions (vehicle_id, text, is_correct, clip_score)
                       VALUES (?, ?, ?, ?)""",
                    (vehicle_id, text, 1 if d.get("is_correct") else 0, clip_score)
                )
            added += 1

        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {exc}")
    finally:
        conn.close()

    return {"added": added, "skipped_missing_images": skipped}


@router.delete("/vehicles/{vehicle_id}")
def delete_vehicle(vehicle_id: int, conn=Depends(get_db)):
    conn.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
    conn.commit()
    return {"status": "deleted"}


# ── CLIP scoring ───────────────────────────────────────────────────────────────

@router.post("/run_clip")
def run_clip(conn=Depends(get_db)):
    try:
        from clip_service import score_all_vehicles
        updated = score_all_vehicles(conn)
        return {"status": "ok", "updated": updated}
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="CLIP dependencies not installed. Run: pip install torch transformers pillow"
        )


# ── Results & export ───────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(conn=Depends(get_db)):
    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    completed      = conn.execute("SELECT COUNT(*) FROM sessions WHERE completed_at IS NOT NULL").fetchone()[0]
    total_tasks    = conn.execute("SELECT COUNT(*) FROM task_responses").fetchone()[0]
    answered       = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE selected_desc_id IS NOT NULL OR not_listed=1"
    ).fetchone()[0]
    correct        = conn.execute(
        "SELECT COUNT(*) FROM task_responses WHERE is_correct=1"
    ).fetchone()[0]
    avg_dur        = conn.execute(
        "SELECT AVG(duration_ms) FROM task_responses WHERE duration_ms IS NOT NULL"
    ).fetchone()[0]
    vehicles       = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    descs          = conn.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0]

    return {
        "total_sessions":     total_sessions,
        "completed_sessions": completed,
        "total_tasks":        total_tasks,
        "answered_tasks":     answered,
        "correct_tasks":      correct,
        "accuracy_pct":       round(correct / answered * 100, 1) if answered else 0,
        "avg_duration_ms":    round(avg_dur) if avg_dur else None,
        "vehicles":           vehicles,
        "descriptions":       descs,
    }


@router.get("/sessions")
def list_sessions(conn=Depends(get_db)):
    rows = conn.execute("""
        SELECT s.id, s.mturk_worker_id, s.condition, s.started_at, s.completed_at,
               s.ai_helped, s.feedback_text, s.ai_comments,
               COUNT(tr.id) as total_tasks,
               SUM(CASE WHEN tr.is_correct=1 THEN 1 ELSE 0 END) as correct,
               AVG(tr.duration_ms) as avg_duration_ms
        FROM sessions s
        LEFT JOIN task_responses tr ON tr.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC
    """).fetchall()
    return [dict(r) for r in rows]


@router.get("/export/csv")
def export_csv(conn=Depends(get_db)):
    rows = conn.execute("""
        SELECT
            s.id            as session_id,
            s.mturk_worker_id,
            s.started_at    as session_started,
            s.completed_at  as session_completed,
            s.confidence_rating,
            s.mental_effort,
            tr.task_order,
            v.image_name,
            d.text          as selected_description,
            tr.not_listed,
            tr.is_correct,
            tr.duration_ms,
            tr.time_started_ms,
            tr.time_submitted_ms
        FROM task_responses tr
        JOIN sessions  s ON s.id = tr.session_id
        JOIN vehicles  v ON v.id = tr.vehicle_id
        LEFT JOIN descriptions d ON d.id = tr.selected_desc_id
        ORDER BY s.started_at, tr.task_order
    """).fetchall()

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])
        [print(dict(r)) for r in rows]

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=labelling_results.csv"}
    )


@router.get("/export/json")
def export_json(conn=Depends(get_db)):
    rows = conn.execute("""
        SELECT s.id as session_id, s.mturk_worker_id, s.ai_helped, s.ai_comments,
               tr.task_order, v.image_name, d.text as selected_desc,
               tr.not_listed, tr.is_correct, tr.duration_ms
        FROM task_responses tr
        JOIN sessions s ON s.id = tr.session_id
        JOIN vehicles v ON v.id = tr.vehicle_id
        LEFT JOIN descriptions d ON d.id = tr.selected_desc_id
        ORDER BY s.started_at, tr.task_order
    """).fetchall()
    return [dict(r) for r in rows]


# ── Config ─────────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config():
    return {"tasks_per_session": int(os.getenv("TASKS_PER_SESSION", "10"))}


@router.post("/config")
def update_config(body: ConfigUpdate):
    if body.tasks_per_session is not None:
        os.environ["TASKS_PER_SESSION"] = str(body.tasks_per_session)
    return {"tasks_per_session": int(os.getenv("TASKS_PER_SESSION", "10"))}