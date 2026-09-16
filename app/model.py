"""Loads the chest X-ray model and runs predictions.

The model comes from Hugging Face. It was trained on chest X-ray
images to tell pneumonia apart from normal. It runs on a normal CPU,
no GPU is needed.
"""

import threading

from PIL import Image

MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
MODEL_VERSION = MODEL_NAME

# Shared load state. The health check reads this.
state = {"status": "loading", "detail": ""}

_processor = None
_model = None


def load_model():
    """Download (once) and load the model. Runs in a background thread."""
    global _processor, _model
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        state["status"] = "ready"
    except Exception as exc:  # model download or load failed
        state["status"] = "error"
        state["detail"] = str(exc)


def start_loading():
    """Kick off loading so the web server can start right away."""
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()


def is_ready():
    return state["status"] == "ready"


def predict(image: Image.Image):
    """Run one image through the model.

    Returns (label, probability), for example ("PNEUMONIA", 0.93).
    """
    inputs = _processor(images=image, return_tensors="pt")
    outputs = _model(**inputs)
    probs = outputs.logits.softmax(dim=-1)[0]
    best = probs.argmax().item()
    label = _model.config.id2label[best]
    return label, float(probs[best])
