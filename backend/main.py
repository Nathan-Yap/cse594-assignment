import os
import uuid
import json
import time
import random
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from database import init_db, get_db
from routers import sessions, tasks, admin

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TASKS_PER_SESSION = int(os.getenv("TASKS_PER_SESSION", "10"))

app = FastAPI(title="Vehicle Labelling Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    init_db()
    print(f"✅  Database initialised")
    print(f"✅  Tasks per session: {TASKS_PER_SESSION}")

# ── Include routers ───────────────────────────────────────────────────────────
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(tasks.router,    prefix="/api/tasks",    tags=["tasks"])
app.include_router(admin.router,    prefix="/api/admin",    tags=["admin"])

# ── HTML routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """AI condition landing page (CLIP-ranked descriptions)."""
    return templates.TemplateResponse("index.html", {"request": request, "condition": "ai"})

@app.get("/control", response_class=HTMLResponse)
async def index_control(request: Request):
    """Control condition landing page (randomly ordered descriptions)."""
    return templates.TemplateResponse("index.html", {"request": request, "condition": "control"})

@app.get("/survey", response_class=HTMLResponse)
async def survey(request: Request):
    return templates.TemplateResponse("survey.html", {"request": request})

@app.get("/complete", response_class=HTMLResponse)
async def complete(request: Request):
    return templates.TemplateResponse("complete.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)