#!/usr/bin/env python3
"""Build CameraBench camera-motion labels for Data A VACE frame folders.

The output format intentionally matches the existing DataB camera-motion
jsonl rows:

    {"path": ".../frames/<case_id>/fake", "labels": [...], "caption": "..."}

This script does not extract frames and does not touch VACE outputs. It only
joins:

* CameraBench cam_motion captions / yes-no VQA files.
* Data A VACE preflight reports, to recover the original target video name.
* The already extracted Data A real/fake frame folders.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json


SCHEMA_VERSION = "dataA_v1_camerabench_camera_motion_json_v1"

DEFAULT_RUN_ROOTS = [
    Path("/tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1"),
    Path("/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_dataset_v2_vace13b_v1"),
    Path("/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_textedit_reserve_vace13b_v1"),
]

LABEL_ORDER = [
    "very-unsteady",
    "unsteady",
    "minimal-shaking",
    "no-shaking",
    "complex-motion",
    "minor-motion",
    "no-motion",
    "static",
    "fast-speed",
    "regular-speed",
    "slow-speed",
    "dolly-in",
    "dolly-out",
    "truck-left",
    "truck-right",
    "pedestal-up",
    "pedestal-down",
    "pan-left",
    "pan-right",
    "tilt-up",
    "tilt-down",
    "roll-CW",
    "roll-CCW",
    "zoom-in",
    "zoom-out",
    "arc-CW",
    "arc-CCW",
    "side-tracking",
    "lead-tracking",
    "tail-tracking",
    "aerial-tracking",
    "arc-tracking",
    "pan-tracking",
    "tilt-tracking",
]

STEADINESS_LABELS = {"very-unsteady", "unsteady", "minimal-shaking", "no-shaking"}
MOTION_LABELS = {"complex-motion", "minor-motion", "no-motion"}
SPEED_LABELS = {"fast-speed", "regular-speed", "slow-speed"}
GROUP_PRIORITY = {
    "very-unsteady": 4,
    "unsteady": 3,
    "minimal-shaking": 2,
    "no-shaking": 1,
    "complex-motion": 3,
    "minor-motion": 2,
    "no-motion": 1,
    "fast-speed": 3,
    "regular-speed": 2,
    "slow-speed": 1,
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataAError(f"missing_json:{path}") from exc
    except json.JSONDecodeError as exc:
        raise DataAError(f"invalid_json:{path}:{exc}") from exc


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u2019", "'").split())


def _video_basename(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def _video_key_from_record(record: Mapping[str, Any]) -> str:
    videos = record.get("videos")
    if isinstance(videos, Sequence) and not isinstance(videos, (str, bytes)) and videos:
        return _video_basename(videos[0])
    return ""


def _assistant_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            return _clean_text(message.get("content"))
    return ""


def _user_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _clean_text(message.get("content"))
    return ""


def _answer_bool(text: str) -> bool | None:
    value = _clean_text(text).strip().lower()
    if value.startswith("yes"):
        return True
    if value.startswith("no"):
        return False
    return None


def _normalized_question(text: str) -> str:
    value = _clean_text(text)
    value = value.replace("<video>", "").strip()
    value = re.sub(r"\s*please only answer yes or no\.?\s*$", "", value, flags=re.I)
    return value.strip()


def _vote(votes: Counter[str], label: str, weight: int = 1) -> None:
    if label in LABEL_ORDER:
        votes[label] += int(weight)


def _labels_from_text(text: str) -> Counter[str]:
    """Infer coarse CameraBench-style labels from an official caption string."""
    value = _clean_text(text).lower()
    votes: Counter[str] = Counter()
    if not value:
        return votes

    if any(token in value for token in ("no shaking", "without any shaking", "no vibration", "shake-free")):
        _vote(votes, "no-shaking")
    if any(token in value for token in ("minimal shaking", "minor vibration", "minor vibrations", "slightly unsteady", "some shaking")):
        _vote(votes, "minimal-shaking")
    if any(token in value for token in ("unsteady", "shaky", "noticeable shaking", "wobbl", "vibration", "shaking")):
        _vote(votes, "unsteady")
    if any(token in value for token in ("very unsteady", "intense shaking", "erratic", "rapidly pans", "noticeable shaking throughout")):
        _vote(votes, "very-unsteady")

    if any(token in value for token in ("completely fixed", "completely still", "no movement", "remains fixed", "fixed position")):
        _vote(votes, "no-motion")
    if any(token in value for token in ("minor movement", "barely moves", "subtle", "slight dolly", "slightly pans")):
        _vote(votes, "minor-motion")
    if any(token in value for token in ("complex motion", "multiple movements", "different motions", "backward and then forward", "pan and tilt", "pans and tilts")):
        _vote(votes, "complex-motion")
    if "static" in value:
        _vote(votes, "static")

    if any(token in value for token in ("fast", "quick", "rapid", "swift")):
        _vote(votes, "fast-speed")
    if any(token in value for token in ("slow", "slowly")):
        _vote(votes, "slow-speed")
    if "consistent speed" in value or "regular speed" in value:
        _vote(votes, "regular-speed")

    if any(token in value for token in ("moves forward", "moving forward", "dollies forward", "dolly forward", "dolly in", "dollies in")):
        _vote(votes, "dolly-in")
    if any(token in value for token in ("moves backward", "moving backward", "dollies backward", "dolly backward", "dolly out", "pulls back", "pull away")):
        _vote(votes, "dolly-out")
    if any(token in value for token in ("trucks left", "trucking left", "moves left", "leftward truck", "right to left")):
        _vote(votes, "truck-left")
    if any(token in value for token in ("trucks right", "trucking right", "moves right", "rightward truck", "left to right")):
        _vote(votes, "truck-right")
    if any(token in value for token in ("moves upward", "moving upward", "pedestaling upward", "crane up", "cranes up")):
        _vote(votes, "pedestal-up")
    if any(token in value for token in ("moves downward", "moving downward", "pedestaling downward", "crane down", "cranes down")):
        _vote(votes, "pedestal-down")

    if any(token in value for token in ("pans left", "panning left", "pan to the left")):
        _vote(votes, "pan-left")
    if any(token in value for token in ("pans right", "panning right", "pan to the right")):
        _vote(votes, "pan-right")
    if any(token in value for token in ("tilts up", "tilting up", "tilt upward", "tilts upward")):
        _vote(votes, "tilt-up")
    if any(token in value for token in ("tilts down", "tilting down", "tilt downward", "tilts downward")):
        _vote(votes, "tilt-down")
    if "rolls clockwise" in value or "rolling clockwise" in value or "roll clockwise" in value:
        _vote(votes, "roll-CW")
    if "rolls counterclockwise" in value or "rolling counterclockwise" in value or "roll counterclockwise" in value:
        _vote(votes, "roll-CCW")
    if "zoom in" in value or "zooms in" in value or "zooming in" in value:
        _vote(votes, "zoom-in")
    if "zoom out" in value or "zooms out" in value or "zooming out" in value:
        _vote(votes, "zoom-out")
    if "arc clockwise" in value or "arcs clockwise" in value or "clockwise arc" in value:
        _vote(votes, "arc-CW")
    if "arc counterclockwise" in value or "arcs counterclockwise" in value or "counterclockwise arc" in value:
        _vote(votes, "arc-CCW")

    if "side-track" in value or "side tracking" in value or "from the side" in value:
        _vote(votes, "side-tracking")
    if "leading the subject" in value or "from the front" in value or "front-side" in value:
        _vote(votes, "lead-tracking")
    if "following behind" in value or "from behind" in value or "rear-side" in value:
        _vote(votes, "tail-tracking")
    if "aerial" in value or "bird's-eye" in value or "drone" in value:
        _vote(votes, "aerial-tracking")
    if "tracking" in value and ("arc" in value or "around" in value):
        _vote(votes, "arc-tracking")
    if "panning" in value and "track" in value:
        _vote(votes, "pan-tracking")
    if "tilting" in value and "track" in value:
        _vote(votes, "tilt-tracking")
    return votes


def _maybe_vote(votes: Counter[str], yes: bool, label: str, *, yes_means_label: bool = True, weight: int = 2) -> bool:
    if yes == yes_means_label:
        _vote(votes, label, weight)
    return True


def _labels_from_question_answer(question: str, answer: str) -> Counter[str]:
    yes = _answer_bool(answer)
    if yes is None:
        return Counter()
    q = _normalized_question(question)
    lower = q.lower()
    votes: Counter[str] = Counter()

    match = re.search(r"description:\s*'(.+)'\??$", q, flags=re.I)
    if match and yes:
        votes.update(_labels_from_text(match.group(1)))
        return votes

    # Coarse camera motion and steadiness.
    if "completely still without any visible movement" in lower:
        _maybe_vote(votes, yes, "no-motion")
    if "completely still without any motion or shaking" in lower:
        _maybe_vote(votes, yes, "no-motion")
        _maybe_vote(votes, yes, "no-shaking")
    if "not completely still and shows visible movement" in lower:
        _maybe_vote(votes, yes, "no-motion", yes_means_label=False)
    if "noticeable motion beyond minor shake or wobble" in lower and "free from" not in lower:
        _maybe_vote(votes, yes, "complex-motion")
        _maybe_vote(votes, yes, "minor-motion", yes_means_label=False)
    if "free from noticeable motion beyond minor shake or wobble" in lower:
        _maybe_vote(votes, yes, "minor-motion")
        _maybe_vote(votes, yes, "complex-motion", yes_means_label=False)
    if "show complex motion" in lower:
        _maybe_vote(votes, yes, "complex-motion")
    if "show simple motion" in lower:
        _maybe_vote(votes, yes, "minor-motion")
    if "stationary with minor vibrations or shaking" in lower:
        _maybe_vote(votes, yes, "minimal-shaking")
        _maybe_vote(votes, yes, "no-motion")
    if "show any vibrations" in lower or "show noticable vibrations" in lower or "show noticeable vibrations" in lower:
        _maybe_vote(votes, yes, "unsteady")
        _maybe_vote(votes, yes, "no-shaking", yes_means_label=False)
    if "exceptionally smooth and highly stable" in lower or "movement smooth and stable" in lower:
        _maybe_vote(votes, yes, "no-shaking")
        _maybe_vote(votes, yes, "unsteady", yes_means_label=False)
    if "scene in the video completely static" in lower or "scene in the video mostly static" in lower:
        _maybe_vote(votes, yes, "static")
    if "scene in the video dynamic" in lower:
        _maybe_vote(votes, yes, "static", yes_means_label=False)

    # Speed.
    if "fast motion speed" in lower or "fast speed" in lower:
        _maybe_vote(votes, yes, "fast-speed")
    if "slow motion speed" in lower:
        _maybe_vote(votes, yes, "slow-speed")

    negated_motion_rules = [
        ("not just move laterally to the left", "truck-left"),
        ("not just move laterally to the right", "truck-right"),
        ("not just move left", "truck-left"),
        ("not just move right", "truck-right"),
        ("not just pan left", "pan-left"),
        ("not just pan right", "pan-right"),
        ("not just tilt up", "tilt-up"),
        ("not just tilt down", "tilt-down"),
        ("not just roll clockwise", "roll-CW"),
        ("not just roll counterclockwise", "roll-CCW"),
        ("not just zoom in", "zoom-in"),
        ("not just zoom out", "zoom-out"),
        ("not craning upward in an arc", "arc-tracking"),
        ("not craning downward in an arc", "arc-tracking"),
        ("not following behind the subject", "tail-tracking"),
        ("not leading the subject", "lead-tracking"),
        ("not moving ahead of the subject", "lead-tracking"),
        ("not panning to track", "pan-tracking"),
        ("not tilting to track", "tilt-tracking"),
        ("not a tracking shot from an aerial perspective", "aerial-tracking"),
        ("not a tracking shot with arc movement", "arc-tracking"),
        ("not a tracking shot", "side-tracking"),
    ]
    for phrase, label in negated_motion_rules:
        if phrase in lower:
            _maybe_vote(votes, yes, label, yes_means_label=False)
            return votes

    # Translational movement.
    if "free from any forward motion" in lower:
        _maybe_vote(votes, yes, "dolly-in", yes_means_label=False)
    elif "moving forward in the scene" in lower or "move forward (not zooming in)" in lower or "physically move forward" in lower:
        _maybe_vote(votes, yes, "dolly-in")
    if "move only forward" in lower or "only move forward" in lower:
        _maybe_vote(votes, yes, "dolly-in")

    if "free from any backward motion" in lower:
        _maybe_vote(votes, yes, "dolly-out", yes_means_label=False)
    elif "moving backward in the scene" in lower or "move backward (not zooming out)" in lower or "physically move backward" in lower:
        _maybe_vote(votes, yes, "dolly-out")
    if "move only backward" in lower or "only move backward" in lower:
        _maybe_vote(votes, yes, "dolly-out")

    if "move laterally to the left" in lower or "move leftward" in lower or "only move left" in lower:
        _maybe_vote(votes, yes, "truck-left")
    if "move laterally to the right" in lower or "move rightward" in lower or "only move right" in lower:
        _maybe_vote(votes, yes, "truck-right")
    if "move physically upward" in lower or "move upward (not tilting up)" in lower or "pedestal up" in lower:
        _maybe_vote(votes, yes, "pedestal-up")
    if "move physically downward" in lower or "move downward (not tilting down)" in lower or "pedestal down" in lower:
        _maybe_vote(votes, yes, "pedestal-down")

    # Rotation / zoom / arc.
    if "free from any leftward panning" in lower:
        _maybe_vote(votes, yes, "pan-left", yes_means_label=False)
    elif "pan to the left" in lower or "pan left" in lower or "pan leftward" in lower:
        _maybe_vote(votes, yes, "pan-left")
    if "free from any rightward panning" in lower:
        _maybe_vote(votes, yes, "pan-right", yes_means_label=False)
    elif "pan to the right" in lower or "pan right" in lower or "pan rightward" in lower:
        _maybe_vote(votes, yes, "pan-right")
    if "free from any upward tilting" in lower:
        _maybe_vote(votes, yes, "tilt-up", yes_means_label=False)
    elif "tilt upward" in lower or "tilt up" in lower:
        _maybe_vote(votes, yes, "tilt-up")
    if "free from any downward tilting" in lower:
        _maybe_vote(votes, yes, "tilt-down", yes_means_label=False)
    elif "tilt downward" in lower or "tilt down" in lower:
        _maybe_vote(votes, yes, "tilt-down")
    if "free from any clockwise rolling" in lower:
        _maybe_vote(votes, yes, "roll-CW", yes_means_label=False)
    elif "roll clockwise" in lower:
        _maybe_vote(votes, yes, "roll-CW")
    if "free from any counterclockwise rolling" in lower:
        _maybe_vote(votes, yes, "roll-CCW", yes_means_label=False)
    elif "roll counterclockwise" in lower:
        _maybe_vote(votes, yes, "roll-CCW")
    if "free from any zoom in" in lower:
        _maybe_vote(votes, yes, "zoom-in", yes_means_label=False)
    elif "zoom in" in lower or "zooming in" in lower:
        _maybe_vote(votes, yes, "zoom-in")
    if "free from any zoom out" in lower:
        _maybe_vote(votes, yes, "zoom-out", yes_means_label=False)
    elif "zoom out" in lower or "zooming out" in lower:
        _maybe_vote(votes, yes, "zoom-out")
    if "free from any clockwise arc" in lower:
        _maybe_vote(votes, yes, "arc-CW", yes_means_label=False)
    elif "clockwise arc" in lower:
        _maybe_vote(votes, yes, "arc-CW")
    if "free from any counterclockwise arc" in lower:
        _maybe_vote(votes, yes, "arc-CCW", yes_means_label=False)
    elif "counterclockwise arc" in lower:
        _maybe_vote(votes, yes, "arc-CCW")

    # Tracking attributes.
    if "video not a tracking shot" in lower:
        _maybe_vote(votes, yes, "side-tracking", yes_means_label=False)
    elif "track the subject as they move" in lower:
        _maybe_vote(votes, yes, "side-tracking")
    if "side-tracking shot" in lower or "moving from the side" in lower:
        _maybe_vote(votes, yes, "side-tracking")
    if "leading the subject" in lower or "moving ahead of the subject" in lower or "front-side angle" in lower:
        _maybe_vote(votes, yes, "lead-tracking")
    if "following behind the subject" in lower or "rear-side angle" in lower:
        _maybe_vote(votes, yes, "tail-tracking")
    if "aerial perspective" in lower:
        _maybe_vote(votes, yes, "aerial-tracking")
    if "follow the subject while moving in an arc" in lower or "tracking shot with arc movement" in lower:
        _maybe_vote(votes, yes, "arc-tracking")
    if "pan to track" in lower or "panning to track" in lower:
        _maybe_vote(votes, yes, "pan-tracking")
    if "tilt to track" in lower or "tilting to track" in lower:
        _maybe_vote(votes, yes, "tilt-tracking")
    if "subject appear larger during the tracking shot" in lower:
        _maybe_vote(votes, yes, "dolly-in")
    if "subject appear smaller during the tracking shot" in lower:
        _maybe_vote(votes, yes, "dolly-out")

    return votes


def _pick_group(votes: Counter[str], labels: set[str]) -> str | None:
    candidates = [label for label in labels if votes.get(label, 0) > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda label: (votes[label], GROUP_PRIORITY.get(label, 0)))


def _label_order_index(label: str) -> int:
    try:
        return LABEL_ORDER.index(label)
    except ValueError:
        return len(LABEL_ORDER)


def _ensure_core_label(votes: Counter[str], labels: set[str], group: set[str], fallback: str) -> None:
    if labels.isdisjoint(group):
        labels.add(fallback)
        votes[fallback] += 1


def _finalize_labels(votes: Counter[str]) -> list[str]:
    votes = Counter({label: count for label, count in votes.items() if count > 0 and label in LABEL_ORDER})
    if not votes:
        return []

    # Official CameraBench-style rows almost always include one speed label.
    # When the VQA set does not explicitly ask speed, regular-speed is the
    # neutral default instead of leaving the row with only 1-2 labels.
    if not any(votes.get(label, 0) > 0 for label in SPEED_LABELS):
        votes["regular-speed"] += 1

    # Static/no-motion examples in CameraBench are represented as their own
    # label plus a motion label. Keep that form even when only one wording is
    # present in the source VQA/caption.
    if votes.get("no-motion", 0) > 0:
        votes["static"] += 1

    selected: set[str] = set()
    for group in (STEADINESS_LABELS, MOTION_LABELS, SPEED_LABELS):
        label = _pick_group(votes, group)
        if label:
            selected.add(label)

    _ensure_core_label(votes, selected, STEADINESS_LABELS, "minimal-shaking")
    _ensure_core_label(votes, selected, MOTION_LABELS, "minor-motion")
    _ensure_core_label(votes, selected, SPEED_LABELS, "regular-speed")

    extras = [
        label
        for label, count in votes.items()
        if count > 0 and label not in STEADINESS_LABELS and label not in MOTION_LABELS and label not in SPEED_LABELS
    ]
    extras.sort(key=lambda label: (-votes[label], _label_order_index(label)))
    selected.update(extras)

    if len(selected) > 8:
        core = [
            label
            for group in (STEADINESS_LABELS, MOTION_LABELS, SPEED_LABELS)
            for label in [_pick_group(votes, group)]
            if label
        ]
        keep = set(core)
        if "no-motion" in keep and votes.get("static", 0) > 0:
            keep.add("static")
        for label in extras:
            if len(keep) >= 8:
                break
            keep.add(label)
        selected = keep

    return [label for label in LABEL_ORDER if label in selected]


def _load_captions(captionset_path: Path) -> dict[str, str]:
    rows = _read_json(captionset_path)
    if not isinstance(rows, list):
        raise DataAError(f"captionset_not_list:{captionset_path}")
    captions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = _video_key_from_record(row)
        text = _assistant_text(row)
        if key and text:
            captions[key][text] += 1
    return {
        key: max(counter, key=lambda text: (counter[text], len(text)))
        for key, counter in captions.items()
    }


def _load_label_votes(paths: Sequence[Path], captions: Mapping[str, str]) -> dict[str, Counter[str]]:
    votes_by_video: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise DataAError(f"vqa_not_list:{path}")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            key = _video_key_from_record(row)
            if not key:
                continue
            votes_by_video[key].update(_labels_from_question_answer(_user_text(row), _assistant_text(row)))

    for key, caption in captions.items():
        votes_by_video[key].update(_labels_from_text(caption))
    return votes_by_video


def _walk_values(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            next_path = path + (str(key),)
            yield next_path, value
            yield from _walk_values(value, next_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _walk_values(value, path + (str(index),))


def _extract_target_video_path(report: Mapping[str, Any]) -> str:
    candidates: list[tuple[int, str, str]] = []
    for path, value in _walk_values(report):
        key = path[-1] if path else ""
        if key not in {"video_path", "target_video_path", "original_video_path", "canonical_video_path"}:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        path_text = "/".join(path).lower()
        value_text = value.replace("\\", "/")
        score = 0
        if "target" in path_text:
            score += 20
        if "donor" in path_text or "reference" in path_text:
            score -= 50
        if "/video/" in value_text or value_text.endswith(".mp4"):
            score += 5
        if any(token in path_text for token in ("source_clip", "full_fake", "full_real", "generated")):
            score -= 25
        candidates.append((score, "/".join(path), value))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][2]


def _case_video_map(run_roots: Sequence[Path]) -> tuple[dict[str, dict[str, str]], Counter[str], list[dict[str, str]]]:
    case_map: dict[str, dict[str, str]] = {}
    skipped: Counter[str] = Counter()
    duplicates: list[dict[str, str]] = []
    for run_root in run_roots:
        if not run_root.is_dir():
            skipped["missing_run_root"] += 1
            continue
        for report_path in sorted(run_root.rglob("preflight_report.json")):
            case_id = report_path.parent.name
            try:
                report = _read_json(report_path)
            except DataAError:
                skipped["invalid_preflight_report"] += 1
                continue
            if not isinstance(report, Mapping):
                skipped["preflight_not_object"] += 1
                continue
            target_video_path = _extract_target_video_path(report)
            video_name = _video_basename(target_video_path)
            if not video_name:
                skipped["missing_target_video_path"] += 1
                continue
            item = {
                "case_id": case_id,
                "target_video_path": target_video_path,
                "video_name": video_name,
                "run_root": str(run_root),
                "preflight_report": str(report_path),
            }
            if case_id in case_map and case_map[case_id].get("video_name") != video_name:
                duplicates.append(
                    {
                        "case_id": case_id,
                        "first_video_name": case_map[case_id].get("video_name", ""),
                        "duplicate_video_name": video_name,
                        "duplicate_preflight_report": str(report_path),
                    }
                )
                skipped["duplicate_case_conflict"] += 1
                continue
            case_map[case_id] = item
    return case_map, skipped, duplicates


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + "\n")


def build_camera_motion_json(
    *,
    camera_motion_root: Path,
    run_roots: Sequence[Path],
    frame_root: Path,
    out_dir: Path,
    out_name: str,
    audit_name: str,
    allow_empty_labels: bool,
    roles: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    captions = _load_captions(camera_motion_root / "captionset.json")
    label_votes = _load_label_votes(
        [camera_motion_root / "balanced_vqa.json", camera_motion_root / "imb_raw.json"],
        captions,
    )
    labels_by_video = {key: _finalize_labels(votes) for key, votes in label_votes.items()}
    case_map, preflight_skipped, duplicates = _case_video_map(run_roots)

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter(preflight_skipped)
    label_counts: Counter[str] = Counter()
    run_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    missing_examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for case_id, case in sorted(case_map.items()):
        video_name = case["video_name"]
        caption = captions.get(video_name, "")
        labels = labels_by_video.get(video_name, [])
        if not caption:
            skipped["missing_caption"] += 1
            if len(missing_examples["missing_caption"]) < 20:
                missing_examples["missing_caption"].append(case)
            continue
        if not labels and not allow_empty_labels:
            skipped["missing_labels"] += 1
            if len(missing_examples["missing_labels"]) < 20:
                missing_examples["missing_labels"].append(case)
            continue
        case_frame_root = frame_root / case_id
        if not case_frame_root.is_dir():
            skipped["missing_case_frame_dir"] += 1
            if len(missing_examples["missing_case_frame_dir"]) < 20:
                missing_examples["missing_case_frame_dir"].append(case)
            continue
        wrote_case = False
        for role in roles:
            role_dir = case_frame_root / role
            if not role_dir.is_dir():
                skipped[f"missing_{role}_frame_dir"] += 1
                continue
            rows.append(
                {
                    "path": str(role_dir),
                    "labels": labels,
                    "caption": caption,
                }
            )
            wrote_case = True
            role_counts[role] += 1
            label_counts.update(labels)
        if wrote_case:
            run_counts[Path(case["run_root"]).name] += 1

    out_jsonl = out_dir / out_name
    out_audit = out_dir / audit_name
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "dry_run": bool(dry_run),
        "camera_motion_root": str(camera_motion_root),
        "frame_root": str(frame_root),
        "run_roots": [str(path) for path in run_roots],
        "out_jsonl": str(out_jsonl),
        "out_audit": str(out_audit),
        "case_count": len(case_map),
        "row_count": len(rows),
        "role_counts": dict(role_counts),
        "run_counts": dict(run_counts),
        "caption_video_count": len(captions),
        "label_video_count": sum(1 for labels in labels_by_video.values() if labels),
        "label_counts": dict(label_counts),
        "skipped_counts": dict(skipped),
        "duplicate_case_conflicts": duplicates[:50],
        "missing_examples": dict(missing_examples),
    }
    if not dry_run:
        _write_jsonl(out_jsonl, rows)
        write_json(out_audit, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-motion-root",
        type=Path,
        default=Path("/input/workflow_58770161/workspace/test/cameramotion_det/camera/cam_motion"),
        help="CameraBench cam_motion directory containing captionset.json, balanced_vqa.json, imb_raw.json.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=None,
        help="Data A VACE run root. Repeat to override the default three production runs.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path("/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames"),
        help="Root with <case_id>/real and <case_id>/fake frame folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson"),
    )
    parser.add_argument("--out-name", default="dataa_cameramotion_labels_v2.jsonl")
    parser.add_argument("--audit-name", default="dataa_cameramotion_labels_v2_audit.json")
    parser.add_argument("--roles", default="real,fake", help="Comma-separated frame roles to emit.")
    parser.add_argument("--allow-empty-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_roots = args.run_root if args.run_root else DEFAULT_RUN_ROOTS
    roles = [role.strip() for role in str(args.roles).split(",") if role.strip()]
    if not roles:
        raise DataAError("empty_roles")
    summary = build_camera_motion_json(
        camera_motion_root=args.camera_motion_root,
        run_roots=run_roots,
        frame_root=args.frame_root,
        out_dir=args.out_dir,
        out_name=args.out_name,
        audit_name=args.audit_name,
        allow_empty_labels=bool(args.allow_empty_labels),
        roles=roles,
        dry_run=bool(args.dry_run),
    )
    print(
        "camera_motion_json "
        f"dry_run={summary['dry_run']} cases={summary['case_count']} rows={summary['row_count']} "
        f"roles={summary['role_counts']} skipped={summary['skipped_counts']} "
        f"out={summary['out_jsonl']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



