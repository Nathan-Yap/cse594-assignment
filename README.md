# Vehicle Description Labelling Service

A full-stack MTurk-compatible labelling platform for vehicle image–description matching studies.

---

## Features

| Feature | Details |
|---|---|
| **Survey codes** | 32-char hex code generated per visitor, entered into MTurk |
| **Image + descriptions** | Shows vehicle image alongside N candidate descriptions |
| **"Not listed" option** | Participant can indicate no description matches |
| **CLIP ranking** | Optional: AI re-ranks descriptions by cosine similarity to image |
| **Per-task timing** | Start/submit timestamps recorded for every task |
| **Accuracy tracking** | Ground-truth correct answers compared automatically |
| **Post-study feedback** | Form asks about AI intervention impact |
| **Admin dashboard** | Upload images/dataset, view sessions, export CSV/JSON |
| **Configurable tasks** | `TASKS_PER_SESSION` env var (default 10) |
| **Portable** | Runs locally or in Docker, SQLite database |

---

## Quick Start (Local)

```bash
# 1. Install Python deps
cd backend
pip install -r requirements.txt

# 2. Seed the database with demo data (optional)
python seed_demo.py

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit:
- **Study**: http://localhost:8000
- **Admin**: http://localhost:8000/admin

---

## Docker Deployment

```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

The database and images persist in `./data/` on the host.

---

## Adding Your Dataset

### Option A — Admin UI (recommended)

1. Go to **http://localhost:8000/admin → Dataset**
2. Upload vehicle images (jpg/png/webp)
3. Upload a JSON file in this format:

```json
[
  {
    "image_name": "car_001.jpg",
    "descriptions": [
      {"text": "A red sedan with chrome trim", "is_correct": true},
      {"text": "A blue SUV with roof rack",    "is_correct": false},
      {"text": "A white minivan with sliding doors", "is_correct": false}
    ]
  }
]
```

> **At least one `is_correct: true` description per vehicle** is needed for accuracy tracking.  
> Vehicles with no correct description will still work — users selecting "Not listed" score as correct.

### Option B — Python script

```python
import requests, json

# 1. Upload image
with open("car_001.jpg", "rb") as f:
    requests.post("http://localhost:8000/api/admin/upload_image", files={"file": f})

# 2. Add vehicle + descriptions
requests.post("http://localhost:8000/api/admin/vehicles", json={
    "image_name": "car_001.jpg",
    "descriptions": [
        {"text": "A red sedan", "is_correct": True},
        {"text": "A blue SUV",  "is_correct": False},
    ]
})
```

---

## CLIP Integration

CLIP ranks descriptions by visual–text similarity, surfacing the most relevant options first.

```bash
# Install CLIP dependencies
pip install torch torchvision transformers pillow

# Run scoring via admin UI: Admin → CLIP → "Run CLIP Scoring"
# Or via API:
curl -X POST http://localhost:8000/api/admin/run_clip
```

Once scored, the `clip_ranked` flag appears on the survey and descriptions are sorted by score.  
An "AI-ranked" badge is shown to participants.

---

## Changing Tasks Per Session

**At runtime** (no restart needed — affects new sessions only):

```bash
curl -X POST http://localhost:8000/api/admin/config \
  -H "Content-Type: application/json" \
  -d '{"tasks_per_session": 15}'
```

**Via environment variable**:

```bash
# In .env or docker-compose.yml
TASKS_PER_SESSION=15
```

---

## MTurk Integration

1. Create an MTurk HIT with an input field labelled "Survey Code"
2. Set the HIT URL to `http://YOUR_IP:8000/` (or your domain)
3. Workers visit the link, receive their code, copy it into MTurk, then complete the study
4. The same 32-char code is shown again at the end for final submission

> **Tip**: Use [ngrok](https://ngrok.com) for a public URL during local testing:  
> `ngrok http 8000` → copy the HTTPS URL into MTurk

---

## Exporting Results

```bash
# CSV (all task responses + session metadata)
curl http://localhost:8000/api/admin/export/csv -o results.csv

# JSON
curl http://localhost:8000/api/admin/export/json -o results.json
```

Or use the **Export** tab in the admin dashboard.

### CSV Columns

| Column | Description |
|---|---|
| `session_id` | 32-char survey code |
| `mturk_worker_id` | Worker ID if provided |
| `task_order` | 1-based position in session |
| `image_name` | Vehicle image filename |
| `selected_description` | Text of chosen description |
| `not_listed` | 1 if user chose "not listed" |
| `is_correct` | 1/0 correctness vs ground truth |
| `duration_ms` | Milliseconds spent on this task |
| `ai_helped` | Post-study feedback: yes/no/neutral |
| `ai_comments` | Free-text AI feedback |

---

## Project Structure

```
labelling-service/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── database.py          # SQLite schema + helpers
│   ├── clip_service.py      # CLIP scoring pipeline
│   ├── seed_demo.py         # Demo data loader
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── routers/
│   │   ├── sessions.py      # Session create, task fetch, answer submit, feedback
│   │   └── admin.py         # Dataset management, stats, export
│   ├── templates/
│   │   ├── index.html       # Landing page (survey code generation)
│   │   ├── survey.html      # Task labelling UI
│   │   ├── complete.html    # Feedback form + final code display
│   │   └── admin.html       # Admin dashboard
│   └── static/
│       └── images/          # Vehicle images go here
├── docker-compose.yml
└── README.md
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions/create` | Generate survey code, assign vehicles |
| `GET` | `/api/sessions/{id}/task/{n}` | Fetch task n for session |
| `POST` | `/api/sessions/submit_answer` | Record user's answer |
| `POST` | `/api/sessions/feedback` | Submit post-study feedback |
| `GET` | `/api/sessions/{id}/summary` | Accuracy + timing summary |
| `GET` | `/api/admin/vehicles` | List all vehicles |
| `POST` | `/api/admin/vehicles` | Add vehicle + descriptions |
| `POST` | `/api/admin/upload_image` | Upload image file |
| `POST` | `/api/admin/upload_dataset` | Bulk JSON import |
| `DELETE` | `/api/admin/vehicles/{id}` | Delete vehicle |
| `POST` | `/api/admin/run_clip` | Run CLIP scoring |
| `GET` | `/api/admin/stats` | Aggregate statistics |
| `GET` | `/api/admin/sessions` | All sessions |
| `GET` | `/api/admin/export/csv` | Download CSV |
| `GET` | `/api/admin/export/json` | Download JSON |
| `GET/POST` | `/api/admin/config` | Get/set tasks per session |
