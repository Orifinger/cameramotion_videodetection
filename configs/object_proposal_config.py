from pathlib import Path

"""Central configuration for CameraBench object discovery.

Normal execution:
    python scripts/build_video_manifest.py
    python scripts/run_qwen_object_proposals.py

This v4 Qwen stage discovers a compact, high-recall bank of segmentable visual
concepts. It does not make final edit decisions. Source MP4s remain in /tmp;
all JSON metadata is written under the project res/ directory.
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
# Unified Qwen v4 outputs (no per-video JSON files)
# ============================================================
# Kept separate from res/qwen_region_candidates/ so the audited v3 result is
# preserved as a baseline and cannot be overwritten by this new discovery run.
QWEN_RESULT_ROOT = RES_ROOT / "qwen_region_candidates_v4"
ALL_CANDIDATES_PATH = QWEN_RESULT_ROOT / "qwen_region_candidates_all.json"
SAM3_CANDIDATES_PATH = QWEN_RESULT_ROOT / "qwen_sam3_candidates.json"
SCENE_TEXT_GRAPHIC_PATH = QWEN_RESULT_ROOT / "qwen_deferred_scene_text_graphic.json"
SCREEN_OVERLAY_PATH = QWEN_RESULT_ROOT / "qwen_deferred_screen_overlay.json"
PERSISTENT_WATERMARK_PATH = QWEN_RESULT_ROOT / "qwen_deferred_persistent_watermark.json"
PARSER_REJECTIONS_PATH = QWEN_RESULT_ROOT / "qwen_parser_rejections.json"
RUN_SUMMARY_PATH = QWEN_RESULT_ROOT / "qwen_run_summary.json"

# ============================================================
# Qwen3-VL vLLM server
# ============================================================
QWEN_API_BASE = "http://127.0.0.1:8000/v1"
QWEN_MODEL_NAME = "qwen3-vl-8b-object"
QWEN_MODEL_PATH = "/home/admin/Qwen3-VL-8B-Instruct"

# ============================================================
# Candidate-concept discovery inference
# ============================================================
# One TP=16 vLLM engine receives at most this many in-flight HTTP requests.
MAX_CONCURRENCY = 40
MAX_RETRIES = 3
REQUEST_TIMEOUT_SEC = 900
MAX_OUTPUT_TOKENS = 1600
TEMPERATURE = 0.0
ENABLE_JSON_SCHEMA = True

# The goal is controlled recall, not a scene-wide inventory. Deferred candidates
# are retained for audit but do not enter the SAM3 main route.
MAX_SAM3_CANDIDATES = 6
MAX_DEFERRED_CANDIDATES = 6

# Resumption policy. Existing v4 terminal records are skipped when False.
# The v3 baseline is in a separate directory and is never read by this run.
OVERWRITE_EXISTING = False
RETRY_FAILURE_RECORDS = True

# Save the unified JSON files after this many completed requests.
SAVE_EVERY = 10
PROGRESS_EVERY = 10

# Smoke-test control. Keep 20 for v4 prompt/schema validation; set None only
# after manually auditing the v4 candidate output.
MAX_VIDEOS = 20
SHUFFLE_MANIFEST = False

# ============================================================
# Reserved for the later SAM3 track-bank stage
# ============================================================
SAM_LARGE_OUTPUT_ROOT = Path("/tmp/cambench_train/cam_train/object_discovery_sam")
SAM_RESULT_ROOT = RES_ROOT / "sam_track_bank"
MAX_VIDEO_LENGTH_SEC = None
MAX_FRAMES = None
SAM_MODEL_NAME = "sam3"
