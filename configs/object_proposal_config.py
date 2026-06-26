from pathlib import Path

"""Central configuration for CameraBench object discovery.

Normal execution:
    python scripts/build_video_manifest.py
    python scripts/run_qwen_object_proposals.py

Only original large video data stays under /tmp. All JSON metadata and indexes
are written under the project data/ or res/ directories.
"""

# ============================================================
# Filesystem paths
# ============================================================
VIDEO_ROOT = Path("/tmp/cambench_train/cam_train/video")
CAM_MOTION_ANNOS = Path("/tmp/cambench_train/cam_train/anno/cam_motion")

PROJECT_ROOT = Path("/input/workflow_58770161/workspace/test/cameramotion_det")
DATA_ROOT = PROJECT_ROOT / "data"
RES_ROOT = PROJECT_ROOT / "res"

# ============================================================
# Small data / metadata files
# ============================================================
MANIFEST_PATH = DATA_ROOT / "cambench_videos.json"
CAM_MOTION_INDEX_PATH = DATA_ROOT / "cam_motion_metadata_index.json"

# ============================================================
# Unified Qwen outputs (no per-video JSON files)
# ============================================================
QWEN_RESULT_ROOT = RES_ROOT / "qwen_region_candidates"
ALL_CANDIDATES_PATH = QWEN_RESULT_ROOT / "qwen_region_candidates_all.json"
SAM3_CANDIDATES_PATH = QWEN_RESULT_ROOT / "qwen_sam3_candidates.json"
SCENE_TEXT_GRAPHIC_PATH = QWEN_RESULT_ROOT / "qwen_deferred_scene_text_graphic.json"
SCREEN_OVERLAY_PATH = QWEN_RESULT_ROOT / "qwen_deferred_screen_overlay.json"
PERSISTENT_WATERMARK_PATH = QWEN_RESULT_ROOT / "qwen_deferred_persistent_watermark.json"
RUN_SUMMARY_PATH = QWEN_RESULT_ROOT / "qwen_run_summary.json"

# ============================================================
# Qwen3-VL vLLM server
# ============================================================
QWEN_API_BASE = "http://127.0.0.1:8000/v1"
QWEN_MODEL_NAME = "qwen3-vl-8b-object"
QWEN_MODEL_PATH = "/home/admin/Qwen3-VL-8B-Instruct"

# ============================================================
# Object proposal inference
# ============================================================
# One TP=16 vLLM engine receives at most this many in-flight HTTP requests.
MAX_CONCURRENCY = 40
MAX_RETRIES = 3
REQUEST_TIMEOUT_SEC = 900
MAX_OUTPUT_TOKENS = 1400
TEMPERATURE = 0.0
ENABLE_JSON_SCHEMA = True

# Resumption policy. Existing success/no_sam3_candidate records are skipped
# when False. Failed records are retried unless RETRY_FAILURE_RECORDS is False.
OVERWRITE_EXISTING = False
RETRY_FAILURE_RECORDS = True

# Save the unified JSON files after this many completed requests.
SAVE_EVERY = 10
PROGRESS_EVERY = 10

# Smoke-test control. Set 20 initially; set None for full processing.
MAX_VIDEOS = 20
SHUFFLE_MANIFEST = False

# ============================================================
# Reserved for later SAM 3.1 stage
# ============================================================
SAM_LARGE_OUTPUT_ROOT = Path("/tmp/cambench_train/cam_train/object_discovery_sam")
SAM_RESULT_ROOT = RES_ROOT / "sam_focus_regions"
MAX_VIDEO_LENGTH_SEC = None
MAX_FRAMES = None
SAM_MODEL_NAME = "sam3.1"
