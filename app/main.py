"""The web API for the Chest X-Ray Image Triage demo.

Routes:
  POST /predict            upload an image, get a prediction
  GET  /cases              list past predictions
  POST /cases/{id}/review  a human confirms or overrides a prediction
  GET  /stats              simple counts
  GET  /health             model load status
"""

import io
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import db, model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("triage")

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Chest X-Ray Image Triage")


@app.on_event("startup")
def startup():
    db.init_db()
    model.start_loading()


class ReviewIn(BaseModel):
    decision: str  # must be "confirmed" or "overridden"
    note: str | None = None


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not model.is_ready():
        status = model.state["status"]
        raise HTTPException(
            status_code=503,
            detail=f"Model is not ready yet (status: {status}). Try again soon.",
        )

    data = file.file.read()

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not an image.")

    # Broad except on purpose: anything that goes wrong here means we
    # could not process the image, and the caller should hear about it.
    try:
        image = image.convert("RGB")
        label, probability = model.predict(image)
    except Exception as exc:
        log.exception("Could not process the image")
        raise HTTPException(
            status_code=422,
            detail=f"Could not process this image: {exc}",
        )

    suffix = Path(file.filename or "image.png").suffix or ".png"
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / name
    path.write_bytes(data)

    case = db.add_case(str(path), label, probability, model.MODEL_VERSION)
    return case


@app.get("/cases")
def cases(limit: int = 20, offset: int = 0):
    rows, total = db.list_cases(limit, offset)
    return {"cases": rows, "total": total}


@app.post("/cases/{case_id}/review")
def review(case_id: int, body: ReviewIn):
    if body.decision not in ("confirmed", "overridden"):
        raise HTTPException(
            status_code=422,
            detail="Decision must be 'confirmed' or 'overridden'.",
        )
    case = db.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return db.save_review(case_id, body.decision, body.note)


@app.get("/stats")
def stats():
    return db.stats()


@app.get("/health")
def health():
    return {
        "status": model.state["status"],
        "model": model.MODEL_NAME,
        "detail": model.state["detail"],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# So the cases screen can show the uploaded images.
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
