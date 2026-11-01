import os

ML_INTERNAL_TOKEN = os.getenv("ML_INTERNAL_TOKEN", "dev-ml-token")
# onnx = YOLOv8n ONNX + onnxruntime (default, lightweight)
# ultralytics = full PyTorch Ultralytics (slow/large Docker build)
# hog = OpenCV HOG people detector
_raw_detector = os.getenv("DETECTOR", "onnx").lower()
# "yolo" kept for older .env files — maps to lightweight ONNX runtime.
DETECTOR = "onnx" if _raw_detector == "yolo" else _raw_detector
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/app/models/yolov8n.onnx")
ONNX_INPUT_SIZE = int(os.getenv("ONNX_INPUT_SIZE", "640"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
ML_WORKERS = int(os.getenv("ML_WORKERS", "2"))
ML_QUEUE_MAX = int(os.getenv("ML_QUEUE_MAX", "32"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
PERSON_CLASS_ID = 0

HOG_HIT_THRESHOLD = float(os.getenv("HOG_HIT_THRESHOLD", "0.0"))
HOG_WIN_STRIDE = int(os.getenv("HOG_WIN_STRIDE", "8"))
HOG_PADDING = int(os.getenv("HOG_PADDING", "4"))
HOG_SCALE = float(os.getenv("HOG_SCALE", "1.05"))
HOG_MIN_WEIGHT = float(os.getenv("HOG_MIN_WEIGHT", "0.3"))
