import sys
import json
import time
import uuid
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# -----------------------------------------------------------------------------
# Project config import
# -----------------------------------------------------------------------------
from configs.sam3_tracking_config import (
    PROJECT_ROOT,
    RES_ROOT,
    QWEN_SAM3_CANDIDATES_PATH,
    SAM3_SOURCE_ROOT,
    SAM3_CHECKPOINT_PATH,
    SAM3_RESULT_ROOT,
    SAM3_TRACKS_ALL_PATH,
    SAM3_FOCUS_REGIONS_PATH,
    SAM3_FAILURES_PATH,
    SAM3_RUN_SUMMARY_PATH,
    SAM3_LARGE_OUTPUT_ROOT,
    SAM3_FOCUS_MASK_ROOT,
    SAM3_WORK_MASK_ROOT,
    SAM3_PROMPT_FRAME_INDEX,
    SAM3_PROPAGATION_DIRECTION,
    SAM3_OUTPUT_PROB_THRESH,
    SAM3_MAX_CANDIDATES_PER_VIDEO,
    SAM3_MAX_VIDEOS,
    SAM3_OVERWRITE_EXISTING,
)

# -----------------------------------------------------------------------------
# Add SAM3 source to path
# -----------------------------------------------------------------------------
sys.path.insert(0, str(SAM3_SOURCE_ROOT))

from sam3.model_builder import build_sam3_video_predictor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pack_mask(mask: np.ndarray) -> np.ndarray:
    # uint8 packed along width axis (bitpack)
    return np.packbits(mask.astype(np.uint8), axis=-1)


# -----------------------------------------------------------------------------
# SAM3 session wrapper
# -----------------------------------------------------------------------------

class Sam3Runner:
    def __init__(self):
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=str(SAM3_CHECKPOINT_PATH),
            gpus_to_use=[0],
        )
        print("SAM3 loaded")

    def start(self, video_path):
        return self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_path),
            }
        )["session_id"]

    def add_prompt(self, session_id, text):
        return self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": SAM3_PROMPT_FRAME_INDEX,
                "text": text,
            }
        )

    def propagate(self, session_id):
        stream = self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": SAM3_PROPAGATION_DIRECTION,
                "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
            }
        )
        for item in stream:
            yield item

    def close(self, session_id):
        self.predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
            }
        )


# -----------------------------------------------------------------------------
# Core tracking logic (smoke version)
# -----------------------------------------------------------------------------

def process_video(runner, video):
    video_id = video["video_id"]
    video_path = video["video_path"]
    candidates = video.get("sam3_candidates", [])[:SAM3_MAX_CANDIDATES_PER_VIDEO]

    session_id = runner.start(video_path)

    video_result = {
        "video_id": video_id,
        "tracks": [],
        "focus_track": None,
        "status": "ok",
    }

    try:
        for c in candidates:
            text = c["sam_prompt"]
            candidate_id = c.get("candidate_id", str(uuid.uuid4()))

            runner.add_prompt(session_id, text)

            tracks = defaultdict(lambda: {"frames": [], "areas": []})

            for frame in runner.propagate(session_id):
                fidx = frame["frame_index"]
                outputs = frame["outputs"]

                obj_ids = outputs.get("out_obj_ids", [])
                masks = outputs.get("out_binary_masks", [])

                for i, oid in enumerate(obj_ids):
                    mask = masks[i]
                    area = float(mask.sum())

                    tracks[int(oid)]["frames"].append(fidx)
                    tracks[int(oid)]["areas"].append(area)

            # score tracks
            best_track = None
            best_score = -1

            for oid, t in tracks.items():
                frames = t["frames"]
                areas = t["areas"]

                if len(frames) == 0:
                    continue

                vis_ratio = len(frames) / max(frames[-1] + 1, 1)
                score = vis_ratio + np.mean(areas) * 1e-6

                if score > best_score:
                    best_score = score
                    best_track = oid

            video_result["tracks"].append(
                {
                    "candidate_id": candidate_id,
                    "best_object_id": best_track,
                    "score": float(best_score),
                }
            )

        video_result["focus_track"] = max(
            video_result["tracks"], key=lambda x: x["score"] if x["score"] is not None else -1,
            default=None,
        )

    except Exception as e:
        video_result["status"] = f"error:{str(e)}"

    finally:
        runner.close(session_id)

    return video_result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    data = load_json(QWEN_SAM3_CANDIDATES_PATH)

    runner = Sam3Runner()

    results = []

    for vid in data[:SAM3_MAX_VIDEOS]:
        results.append(process_video(runner, vid))

    save_json(SAM3_TRACKS_ALL_PATH, results)
    print("done")


if __name__ == "__main__":
    main()
