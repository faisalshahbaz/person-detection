FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Official YOLOv8n ONNX (~12 MB) — no PyTorch/Ultralytics install at build time.
ARG ONNX_MODEL_URL=https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx
RUN mkdir -p /app/models \
    && curl -fL "${ONNX_MODEL_URL}" -o /app/models/yolov8n.onnx

COPY app/ ./app/

ENV DETECTOR=onnx
ENV ONNX_MODEL_PATH=/app/models/yolov8n.onnx
EXPOSE 9000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
