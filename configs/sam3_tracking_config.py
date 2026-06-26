from pathlib import Path

"""Configuration for the SAM3 track-bank stage.

Run one physical GPU at a time:
    CUDA_VISIBLE_DEVICES=<physical_gpu_id> python scripts/run_sam3_tracking.py

This stage consumes only qwen_region_candidates_v4 structured candidates and
produces a reusable object-track bank. It never selects a final focus region or
decides an editing operation.
"""

PROJECT_ROOT = Path("/input/workflow_58770161/workspace/test/cameramotion_det")
RES_ROOT = PROJECT_ROOT / "res"

# Input emitted by the Qwen v4 high-recall candidate-discovery stage.
QWEN_INPUT_SCHEMA_VERSION = "qwen_region_candidates_v4"
QWEN_SAM3_CANDIDATES_PATH = (
    RES_ROOT / "qwen_region_candidates_v4" / "qwen_sam3_candidates.json"
)

# Official source and checkpoint. This project never edits SAM3 source.
SAM3_SOURCE_ROOT = Path("/input/workflow_58770161/workspace/test/sam3-main")
SAM3_CHECKPOINT_PATH = Path("/home/admin/sam3/sam3.pt")

# Aggregate metadata stays in the project. There are no per-video JSON files.
SAM3_RESULT_ROOT = RES_ROOT / "sam_track_bank"
SAM3_TRACKS_ALL_PATH = SAM3_RESULT_ROOT / "sam3_tracks_all.json"
SAM3_QUALITY_TRACKS_PATH = SAM3_RESULT_ROOT / "sam3_quality_tracks.json"
SAM3_FAILURES_PATH = SAM3_RESULT_ROOT / "sam3_failures.json"
SAM3_RUN_SUMMARY_PATH = SAM3_RESULT_ROOT / "sam3_run_summary.json"

# Dense masks are stored as compressed, sparse frame-indexed NPZ tubes. They are
# reusable editing-region assets; unified JSON stores only paths and statistics.
SAM3_LARGE_OUTPUT_ROOT = Path("/tmp/cambench_train/cam_train/object_discovery_sam")
SAM3_TRACK_MASK_ROOT = SAM3_LARGE_OUTPUT_ROOT / "track_masks_v1"
SAM3_SAVE_MASK_TUBES = True

# Runtime. Bind the physical GPU in the shell; Python sees it as local device 0.
SAM3_LOCAL_GPU_IDS = (0,)
SAM3_EXPECTED_VISIBLE_GPU_COUNT = 1
SAM3_PROMPT_FRAME_INDEX = 0
SAM3_PROPAGATION_DIRECTION = "forward"
SAM3_OUTPUT_PROB_THRESH = 0.50
SAM3_MAX_CANDIDATES_PER_VIDEO = 6

# Start with a one-video smoke run. Increase only after the real outputs and
# overlays have been manually checked.
SAM3_MAX_VIDEOS = 1
SAM3_SAVE_EVERY = 1
SAM3_PROGRESS_EVERY = 1
SAM3_CLOSE_SESSION_RUN_GC = True
SAM3_CLEAR_CACHE_THRESHOLD = 80

# Rule-level quality gate. A failed track remains in sam3_tracks_all.json; it is
# simply omitted from sam3_quality_tracks.json. This is not a final edit choice.
SAM3_MIN_VISIBLE_FRAME_RATIO = 0.20
SAM3_MIN_LONGEST_VISIBLE_RUN = 8
SAM3_MIN_MEDIAN_AREA_RATIO = 0.002
SAM3_MAX_MEDIAN_AREA_RATIO = 0.90
SAM3_MAX_BORDER_TOUCH_RATIO = 0.95

SAM3_SCHEMA_VERSION = "sam3_track_bank_v1"
