#!/bin/sh
# Run the app without Docker.
# The first run downloads the model (about 350 MB), later runs start fast.
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
