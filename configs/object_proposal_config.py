from pathlib import Path

"""
Central configuration for CameraBench object proposal pipeline.

Standard execution never requires CLI arguments. Edit this file, then run:
    python scripts/build_video_manifest.py
    python scripts/run_qwen_object_proposals.py
"""

# ============================================================
# Filesystem paths
# ============================================================
VIDEO_ROOT = Path("/tmp/cambench_train/cam_train/video")
CAM_MOTION_ANNOS = Path("/tmp/cambench_train/cam_train/anno/cam_motion")
WORK_ROOT = Path("/tmp/cambench_train/cam_train/object_discovery")

# The CameraBench MP4 inventory. This is the only input list used by Qwen.
MANIFEST_PATH = WORK_ROOT / "manifests" / "cambench_videos.json"

# Per-video Qwen object-proposal outputs. Each successful video writes one file.
QWEN_OUTPUT_ROOT = WORK_ROOT / "qwen_object_proposals"
RESULT_DIR = QWEN_OUTPUT_ROOT / "results"
FAILURE_DIR = QWEN_OUTPUT_ROOT / "failures"
RUN_SUMMARY_PATH = QWEN_OUTPUT_ROOT / "run_summary.json"

# ============================================================
# Qwen3-VL vLLM server
# ============================================================
QWEN_API_BASE = "http://127.0.0.1:8000/v1"
QWEN_MODEL_NAME = "qwen3-vl-8b-object"
QWEN_MODEL_PATH = "/home/admin/Qwen3-VL-8B-Instruct"

# ============================================================
# Object proposal inference
# ============================================================
# One TP=16 vLLM engine receives up to this many in-flight HTTP requests.
MAX_CONCURRENCY = 40
MAX_RETRIES = 3
REQUEST_TIMEOUT_SEC = 900
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0

# vLLM structured output. Keep enabled for vLLM 0.11.0. Set False only if the
# deployed server rejects response_format/json_schema, then inspect failures.
ENABLE_JSON_SCHEMA = True

# False enables resumable execution: successful results/<video_id>.json files
# are skipped. Set True only when intentionally regenerating every video.
OVERWRITE_EXISTING = False

# Smoke-test control. Set 20 for the first run; set None for all remaining
# videos after checking the generated result files.
MAX_VIDEOS = 20
SHUFFLE_MANIFEST = False
PROGRESS_EVERY = 10

# ============================================================
# Reserved for later SAM 3.1 stage; unused in Qwen proposal stage
# ============================================================
MAX_VIDEO_LENGTH_SEC = None
MAX_FRAMES = None
SAM_MODEL_NAME = "sam3.1"
