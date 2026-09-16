FROM python:3.11-slim

WORKDIR /app

# Install the CPU build of torch first (much smaller than the GPU build),
# then the rest of the requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model weights now (about 350 MB) so the container starts fast
# and works without internet access later.
RUN python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; \
AutoImageProcessor.from_pretrained('nickmuchi/vit-finetuned-chest-xray-pneumonia'); \
AutoModelForImageClassification.from_pretrained('nickmuchi/vit-finetuned-chest-xray-pneumonia')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
