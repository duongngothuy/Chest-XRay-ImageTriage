# Chest X-Ray Image Triage: architecture

One Docker container runs the whole app. A browser talks to a single
uvicorn process that serves the UI, runs the model itself, and stores
everything in a Docker volume.

## Parts

- Browser: single page UI (app/static/index.html), screens Upload / Cases / Stats.
- app/main.py: the HTTP routes (/predict, /cases, /cases/{id}/review, /stats,
  /health, /, /uploads) and the input checks.
- app/model.py: loads the Hugging Face model in a background thread at
  startup and runs it on CPU.
- app/db.py: all database code (plain sqlite3, one "cases" table).
- Volume triage-data: holds triage.db and the saved upload images.
- Hugging Face Hub: the model weights (about 350 MB) are pulled once
  during docker build.

## What happens on a prediction

1. The browser POSTs an image to /predict (the button stays off until
   /health says "ready").
2. PIL checks the file really is an image (else 400).
3. model.py prepares the image and runs the model, which returns a label
   and a probability.
4. The image is saved to uploads/ as <uuid>.png and a case row is added
   with review_status "pending".
5. A human confirms or overrides the case with POST /cases/{id}/review.
6. /stats counts the cases and how often the human agreed with the model.

## When things go wrong

- Not an image: 400. The model cannot handle the image: 422 (logged).
- Model still loading: 503. Unknown case id: 404.

## If this ever grows

The two places to change are model.py (if the model moves to its own
service) and db.py (if SQLite becomes Postgres). Everything else is thin
HTTP and one HTML page.

## Diagram

- chest-xray-triage-architecture.png (and the same diagram as .svg):
  the full flow from upload and validation, through the model run and
  storage, to physician review and statistics, with the error paths.
