#!/usr/bin/env python3
"""
seed_demo.py  –  Populate the database with demo vehicle data.

Usage:
    cd backend
    python seed_demo.py

This script:
  1. Creates placeholder vehicle image files (grey squares) for demo purposes.
  2. Inserts vehicles + descriptions into the database.

Replace the placeholder images with real vehicle images before running
your study. The image filenames must match what you put in the DB.
"""

import sys, json
from pathlib import Path

# Make sure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, get_db
from PIL import Image as PILImage
import io

IMAGES_DIR = Path(__file__).parent / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Demo dataset ──────────────────────────────────────────────────────────────
DEMO_VEHICLES = [
    {
        "image_name": "vehicle_001.jpg",
        "color": (180, 60, 60),
        "descriptions": [
            {"text": "A red sedan with a sleek aerodynamic profile and chrome trim.", "is_correct": True},
            {"text": "A silver pickup truck with off-road tyres and a roof rack.", "is_correct": False},
            {"text": "A white minivan with sliding rear doors and roof rails.", "is_correct": False},
            {"text": "A dark blue luxury SUV with tinted windows.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_002.jpg",
        "color": (60, 100, 180),
        "descriptions": [
            {"text": "A navy blue compact hatchback with alloy wheels.", "is_correct": True},
            {"text": "A yellow sports car with wide body kit and spoiler.", "is_correct": False},
            {"text": "A green electric vehicle with futuristic headlights.", "is_correct": False},
            {"text": "A beige estate car with roof bars and tow hitch.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_003.jpg",
        "color": (50, 170, 100),
        "descriptions": [
            {"text": "A lime green compact city car with a panoramic sunroof.", "is_correct": True},
            {"text": "A matte black muscle car with racing stripes.", "is_correct": False},
            {"text": "An orange convertible roadster with leather interior.", "is_correct": False},
            {"text": "A white panel van with a high roof.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_004.jpg",
        "color": (200, 180, 40),
        "descriptions": [
            {"text": "A golden-yellow vintage coupe with wire-spoke wheels.", "is_correct": True},
            {"text": "A pearl white crossover SUV with roof-mounted sensors.", "is_correct": False},
            {"text": "A charcoal grey saloon with a long bonnet.", "is_correct": False},
            {"text": "A bright red ambulance with roof-mounted lights.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_005.jpg",
        "color": (80, 80, 80),
        "descriptions": [
            {"text": "A dark grey city SUV with panoramic windows and 18-inch alloys.", "is_correct": True},
            {"text": "A pastel blue microcar parked on a city street.", "is_correct": False},
            {"text": "A brown delivery van with cargo doors.", "is_correct": False},
            {"text": "A metallic purple hatchback with lowered suspension.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_006.jpg",
        "color": (220, 120, 40),
        "descriptions": [
            {"text": "An amber-orange pickup truck with a roll bar and spotlights.", "is_correct": True},
            {"text": "A light silver hybrid sedan with a charging port.", "is_correct": False},
            {"text": "A forest green land cruiser with mud flaps.", "is_correct": False},
            {"text": "A midnight blue sports saloon with quad exhausts.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_007.jpg",
        "color": (240, 240, 240),
        "descriptions": [
            {"text": "A pearl white luxury saloon with polished chrome details.", "is_correct": True},
            {"text": "A bright yellow taxi cab with roof signage.", "is_correct": False},
            {"text": "A camouflage military jeep with canvas doors.", "is_correct": False},
            {"text": "A red fire engine with an extended ladder.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_008.jpg",
        "color": (100, 60, 140),
        "descriptions": [
            {"text": "A deep violet sports car with aggressive front splitter.", "is_correct": True},
            {"text": "A white ice cream van with side-serving hatch.", "is_correct": False},
            {"text": "A brown estate car with roof box attached.", "is_correct": False},
            {"text": "A sand-coloured dune buggy with roll cage.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_009.jpg",
        "color": (40, 160, 160),
        "descriptions": [
            {"text": "A teal blue electric hatchback with flush door handles.", "is_correct": True},
            {"text": "A crimson red limousine with elongated wheelbase.", "is_correct": False},
            {"text": "An olive drab military transport truck.", "is_correct": False},
            {"text": "A chrome silver concept car with butterfly doors.", "is_correct": False},
        ],
    },
    {
        "image_name": "vehicle_010.jpg",
        "color": (160, 100, 60),
        "descriptions": [
            {"text": "A tan-brown retro off-roader with round headlights.", "is_correct": True},
            {"text": "A neon green electric scooter with cargo box.", "is_correct": False},
            {"text": "A jet black executive SUV with privacy glass.", "is_correct": False},
            {"text": "A baby blue vintage campervan with roof pod.", "is_correct": False},
        ],
    },
]


def make_placeholder_image(path: Path, color: tuple, size=(640, 480)):
    """Create a solid colour JPEG as a placeholder."""
    try:
        img = PILImage.new("RGB", size, color=color)
        # Add a simple label
        img.save(str(path), "JPEG", quality=85)
    except Exception as e:
        # Fallback: write a tiny valid JPEG manually if Pillow not available
        print(f"  ⚠  Could not create image with PIL: {e} — writing blank file")
        path.write_bytes(b"")


def seed():
    init_db()
    conn_gen = get_db()
    conn = next(conn_gen)

    # Check if already seeded
    existing = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    if existing > 0:
        print(f"Database already has {existing} vehicles. Skipping seed.")
        print("To re-seed, delete labelling.db first.")
        return

    print("Seeding demo vehicles…")
    for v in DEMO_VEHICLES:
        img_path = IMAGES_DIR / v["image_name"]
        print(f"  Creating image: {img_path}")
        make_placeholder_image(img_path, tuple(v["color"]))

        cur = conn.execute(
            "INSERT OR IGNORE INTO vehicles (image_path, image_name) VALUES (?, ?)",
            (str(img_path), v["image_name"])
        )
        vehicle_id = cur.lastrowid

        for d in v["descriptions"]:
            conn.execute(
                "INSERT INTO descriptions (vehicle_id, text, is_correct) VALUES (?, ?, ?)",
                (vehicle_id, d["text"], 1 if d["is_correct"] else 0)
            )

    conn.commit()
    print(f"✅  Seeded {len(DEMO_VEHICLES)} demo vehicles with descriptions.")
    print("Note: demo images are solid colour placeholders. Replace with real images.")


if __name__ == "__main__":
    try:
        seed()
    except ImportError:
        print("⚠  Pillow not installed (pip install pillow). Images will be empty placeholders.")
        # Seed DB without images
        init_db()
        conn = next(get_db())
        for v in DEMO_VEHICLES:
            img_path = IMAGES_DIR / v["image_name"]
            img_path.write_bytes(b"")
            cur = conn.execute(
                "INSERT OR IGNORE INTO vehicles (image_path, image_name) VALUES (?, ?)",
                (str(img_path), v["image_name"])
            )
            vehicle_id = cur.lastrowid
            for d in v["descriptions"]:
                conn.execute(
                    "INSERT INTO descriptions (vehicle_id, text, is_correct) VALUES (?, ?, ?)",
                    (vehicle_id, d["text"], 1 if d["is_correct"] else 0)
                )
        conn.commit()
        print("Seeded DB (no images). Install pillow for placeholder images.")
