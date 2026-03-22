"""
CLIP scoring service.
Scores each description against its vehicle image using OpenAI CLIP.

Install deps:
    pip install torch torchvision transformers pillow

Usage:
    from clip_service import score_all_vehicles
    updated = score_all_vehicles(conn)
"""

from pathlib import Path
from typing import List


def score_all_vehicles(conn) -> int:
    """
    For every (vehicle, description) pair without a clip_score,
    compute CLIP cosine similarity and store it.
    Returns the number of descriptions updated.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install torch transformers pillow")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model ({model_name}) on {device}…")
    model     = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Fetch all vehicles with un-scored descriptions
    vehicles = conn.execute("""
        SELECT DISTINCT v.id, v.image_path
        FROM vehicles v
        JOIN descriptions d ON d.vehicle_id = v.id
        WHERE d.clip_score IS NULL
    """).fetchall()

    updated = 0
    for vehicle in vehicles:
        image_path = Path(vehicle["image_path"])
        if not image_path.exists():
            print(f"  ⚠  Image not found: {image_path} — skipping")
            continue

        descs = conn.execute(
            "SELECT id, text FROM descriptions WHERE vehicle_id = ? AND clip_score IS NULL",
            (vehicle["id"],)
        ).fetchall()
        if not descs:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠  Cannot open {image_path}: {e}")
            continue

        texts = [d["text"] for d in descs]

        with torch.no_grad():
            inputs = processor(
                text=texts, images=image,
                return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Cosine similarity between image and each text
            img_emb  = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores   = (img_emb @ txt_emb.T).squeeze(0).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for desc, score in zip(descs, scores):
            conn.execute(
                "UPDATE descriptions SET clip_score = ? WHERE id = ?",
                (float(score), desc["id"])
            )
            updated += 1

        conn.commit()
        print(f"  ✓  Vehicle {vehicle['id']}: scored {len(descs)} descriptions")

    print(f"CLIP scoring complete. {updated} descriptions updated.")
    return updated


def score_single_vehicle(vehicle_id: int, conn) -> List[dict]:
    """Score descriptions for a single vehicle. Returns updated scores."""
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
    except ImportError as e:
        raise ImportError(str(e))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    vehicle = conn.execute(
        "SELECT id, image_path FROM vehicles WHERE id = ?", (vehicle_id,)
    ).fetchone()
    if not vehicle:
        return []

    descs = conn.execute(
        "SELECT id, text FROM descriptions WHERE vehicle_id = ?", (vehicle_id,)
    ).fetchall()

    image = Image.open(vehicle["image_path"]).convert("RGB")
    texts = [d["text"] for d in descs]

    with torch.no_grad():
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores  = (img_emb @ txt_emb.T).squeeze(0).cpu().tolist()

    if isinstance(scores, float):
        scores = [scores]

    results = []
    for desc, score in zip(descs, scores):
        conn.execute("UPDATE descriptions SET clip_score = ? WHERE id = ?", (float(score), desc["id"]))
        results.append({"id": desc["id"], "text": desc["text"], "clip_score": float(score)})
    conn.commit()
    return results
