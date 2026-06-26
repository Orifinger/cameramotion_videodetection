from pathlib import Path

"""
Central configuration for CameraBench object proposal pipeline.

All file paths + runtime hyperparameters should be defined here.
No CLI arguments are required for standard execution.
"""

# =========================
# Root paths
# =========================
VIDEO_ROOT = Path("/tmp/cambench_train/cam_train/video")
CAM_MOTION_ANNOS = Path("/tmp/cambench_train/cam_train/anno/cam_motion")
WORK_ROOT = Path("/tmp/cambench_train/cam_train/object_discovery")

# =========================
# Manifest
# =========================
MANIFEST_PATH = WORK_ROOT / "manifests" / "cambench_videos.json"

# =========================
# Qwen server
# =========================
QWEN_API_BASE = "http://127.0.0.1:8000/v1"
QWEN_MODEL_NAME = "qwen3-vl-8b-object"
QWEN_MODEL_PATH = "/home/admin/Qwen3-VL-8B-Instruct"

# =========================
# Inference settings
# =========================
MAX_CONCURRENCY = 40
MAX_RETRIES = 3
REQUEST_TIMEOUT_SEC = 900
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0

# =========================
# Video constraints (future use)
# =========================
MAX_VIDEO_LENGTH_SEC = None  # optional filter later
MAX_FRAMES = None

# =========================
# SAM stage placeholder
# =========================
SAM_MODEL_NAME = "sam3.1"
