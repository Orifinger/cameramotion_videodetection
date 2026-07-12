#!/usr/bin/env python3
"""Build the first CASPR validation data without touching the held-out identities.

The output keeps each DataA Real/Fake case as one record.  Both videos are
scored independently by the trainer; the pair relation exists only in the
loss.  DataB replay remains an independent, label-balanced sample set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)
VERDICT_INSTRUCTION = (
    "For the auxiliary authenticity score, the assistant response begins with "
    "<verdict> Real</verdict> or <verdict> Fake</verdict> before any analysis. "
    "Camera motion is context and must not be treated as direct authenticity evidence."
)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    return normalize_path(images[0]) if isinstance(images, list) and images else ""


def assistant_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def case_identity(record: Mapping[str, Any]) -> tuple[str, str] | None:
    match = CASE_RE.search(first_image(record))
    return (match.group(1), match.group(2)) if match else None


def source_family(case_id: str) -> str:
    if "textedit_reserve" in case_id:
        return "vace13b_textedit_40step_v3"
    if "dataset_v2" in case_id:
        return "vace13b_dataset_40step_v3"
    return "vace14b_reused"


def detection_label(record: Mapping[str, Any]) -> str:
    identity = case_identity(record)
    if identity:
        return "Real" if identity[1] == "real" else "Fake"
    match = ANSWER_RE.search(assistant_text(record))
    if match:
        return match.group(1).title()
    path = first_image(record).casefold()
    if "/real/" in path:
        return "Real"
    if "/fake/" in path:
        return "Fake"
    return "UNKNOWN"


def prompt_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages")
    output: list[dict[str, str]] = []
    system_seen = False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        if role == "assistant":
            continue
        content = str(message.get("content", ""))
        if role == "system":
            content = content.rstrip() + "\n\n" + VERDICT_INSTRUCTION
            system_seen = True
        output.append({"role": role, "content": content})
    if not system_seen:
        output.insert(0, {"role": "system", "content": VERDICT_INSTRUCTION})
    return output


def evenly_select(values: Sequence[str], limit: int) -> list[str]:
    items = list(values)
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def scoring_sample(record: Mapping[str, Any], frames_per_video: int) -> dict[str, Any]:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("record has no images")
    label = detection_label(record)
    if label not in {"Real", "Fake"}:
        raise ValueError("record has no Real/Fake label")
    selected_images = evenly_select([normalize_path(path) for path in images], frames_per_video)
    messages = prompt_messages(record)
    original_tokens = sum(message["content"].count("<image>") for message in messages)
    if original_tokens != len(images):
        raise ValueError(f"image token/path mismatch before frame selection: {original_tokens} vs {len(images)}")
    if len(selected_images) != len(images):
        selected = set(selected_images)
        kept = 0
        for message in messages:
            parts = message["content"].split("<image>")
            if len(parts) == 1:
                continue
            rebuilt = parts[0]
            for index, suffix in enumerate(parts[1:]):
                image_path = normalize_path(images[kept])
                kept += 1
                if image_path in selected:
                    rebuilt += "<image>" + suffix
                else:
                    rebuilt += suffix
            message["content"] = rebuilt
    image_tokens = sum(message["content"].count("<image>") for message in messages)
    if image_tokens != len(selected_images):
        raise ValueError(f"image token/path mismatch after frame selection: {image_tokens} vs {len(selected_images)}")
    return {
        "messages": messages,
        "images": selected_images,
        "label": label,
        "assistant_prefix": "<verdict>",
    }


def load_detection(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def case_ids_from_split(path: str | Path) -> set[str]:
    output: set[str] = set()
    for row in load_detection(path):
        identity = case_identity(row)
        if identity:
            output.add(identity[0])
    if not output:
        raise ValueError(f"no DataA case ids found in split: {path}")
    return output


def load_camera(path: str | Path) -> dict[tuple[str, str], list[str]]:
    output: dict[tuple[str, str], list[str]] = {}
    for row in read_jsonl(path):
        match = CASE_RE.search(normalize_path(row.get("path")))
        if not match:
            continue
        labels = sorted({str(label).strip() for label in row.get("labels", []) if str(label).strip()})
        output[(match.group(1), match.group(2))] = labels
    return output


def motion_bucket(labels: Sequence[str]) -> str:
    present = {str(label).strip().casefold().replace("_", "-") for label in labels}
    for candidate in ("complex-motion", "minor-motion", "no-motion"):
        if candidate in present:
            return candidate
    return "unknown"


def round_robin_stratified(
    candidates: Sequence[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(str(row["source_family"]), str(row["motion_bucket"]))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    positions = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < min(limit, len(candidates)):
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][position])
                positions[key] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def balanced_datab_replay(
    records: Sequence[dict[str, Any]], count: int, frames_per_video: int, seed: int
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {"Real": [], "Fake": []}
    for row in records:
        label = detection_label(row)
        if label in by_label:
            by_label[label].append(row)
    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    real_count = count // 2
    fake_count = count - real_count
    if len(by_label["Real"]) < real_count or len(by_label["Fake"]) < fake_count:
        raise ValueError(
            f"not enough balanced DataB replay: Real={len(by_label['Real'])}, Fake={len(by_label['Fake'])}"
        )
    selected = by_label["Real"][:real_count] + by_label["Fake"][:fake_count]
    rng.shuffle(selected)
    output = []
    for index, row in enumerate(selected):
        sample = scoring_sample(row, frames_per_video)
        sample.update({"sample_id": f"datab_replay_{index:05d}", "source": "DataB"})
        output.append(sample)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--dataa-train-json")
    parser.add_argument("--dataa-dev-json", required=True)
    parser.add_argument("--datab-detection-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-train-pairs", type=int, default=256)
    parser.add_argument("--num-datab-replay", type=int, default=512)
    parser.add_argument("--frames-per-video", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataa = load_detection(args.dataa_detection_json)
    datab = load_detection(args.datab_detection_json)
    dev_ids = case_ids_from_split(args.dataa_dev_json)

    camera = load_camera(args.dataa_camera_jsonl)
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in dataa:
        identity = case_identity(row)
        if identity:
            pairs[identity[0]][identity[1]] = row

    complete = {case_id: pair for case_id, pair in pairs.items() if set(pair) == {"real", "fake"}}
    if args.dataa_train_json:
        train_ids = case_ids_from_split(args.dataa_train_json)
        train_partition_source = "explicit_train_json"
    else:
        train_ids = set(complete) - dev_ids
        train_partition_source = "all_complete_pairs_minus_fixed_dev"
    if train_ids & dev_ids:
        raise ValueError(f"DataA train/dev leakage: {sorted(train_ids & dev_ids)[:20]}")
    candidates: list[dict[str, Any]] = []
    camera_mismatches: list[str] = []
    for case_id in sorted(train_ids & complete.keys()):
        real_labels = camera.get((case_id, "real"), [])
        fake_labels = camera.get((case_id, "fake"), [])
        if real_labels and fake_labels and real_labels != fake_labels:
            camera_mismatches.append(case_id)
            continue
        labels = real_labels or fake_labels
        candidates.append(
            {
                "case_id": case_id,
                "source_family": source_family(case_id),
                "motion_bucket": motion_bucket(labels),
                "camera_labels": labels,
                "real": complete[case_id]["real"],
                "fake": complete[case_id]["fake"],
            }
        )
    if len(candidates) < args.num_train_pairs:
        raise ValueError(f"only {len(candidates)} complete, camera-consistent train pairs are available")

    selected = round_robin_stratified(candidates, args.num_train_pairs, args.seed)
    train_pairs = []
    for row in selected:
        train_pairs.append(
            {
                "pair_id": row["case_id"],
                "case_id": row["case_id"],
                "source_family": row["source_family"],
                "motion_bucket": row["motion_bucket"],
                "camera_labels": row["camera_labels"],
                "real": scoring_sample(row["real"], args.frames_per_video),
                "fake": scoring_sample(row["fake"], args.frames_per_video),
            }
        )

    dev_pairs = []
    missing_dev: list[str] = []
    for case_id in sorted(dev_ids):
        pair = complete.get(case_id)
        if not pair:
            missing_dev.append(case_id)
            continue
        real_labels = camera.get((case_id, "real"), [])
        fake_labels = camera.get((case_id, "fake"), [])
        labels = real_labels or fake_labels
        dev_pairs.append(
            {
                "pair_id": case_id,
                "case_id": case_id,
                "source_family": source_family(case_id),
                "motion_bucket": motion_bucket(labels),
                "camera_labels": labels,
                "camera_pair_consistent": not (real_labels and fake_labels) or real_labels == fake_labels,
                "real": scoring_sample(pair["real"], args.frames_per_video),
                "fake": scoring_sample(pair["fake"], args.frames_per_video),
            }
        )
    if missing_dev:
        raise ValueError(f"missing complete detection pairs for dev cases: {missing_dev[:20]}")

    replay = balanced_datab_replay(datab, args.num_datab_replay, args.frames_per_video, args.seed + 1)
    if args.check_images:
        all_paths = {
            path
            for pair in train_pairs + dev_pairs
            for split in ("real", "fake")
            for path in pair[split]["images"]
        } | {path for row in replay for path in row["images"]}
        missing = [path for path in sorted(all_paths) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} image files; first={missing[0]}")

    out_dir = Path(args.out_dir)
    paths = {
        "train_pairs": out_dir / "dataa_train_pairs_256.jsonl",
        "datab_replay": out_dir / "datab_replay_512.jsonl",
        "dev_pairs": out_dir / "dataa_dev_pairs.jsonl",
    }
    write_jsonl(paths["train_pairs"], train_pairs)
    write_jsonl(paths["datab_replay"], replay)
    write_jsonl(paths["dev_pairs"], dev_pairs)
    selected_counts = Counter((row["source_family"], row["motion_bucket"]) for row in train_pairs)
    summary = {
        "schema_version": "caspr_gate1_v1",
        "question": "Does camera-stratified exact-pair verdict ranking improve independent DataA detection?",
        "seed": args.seed,
        "data": {
            "dataa_detection_json": args.dataa_detection_json,
            "dataa_camera_jsonl": args.dataa_camera_jsonl,
            "dataa_train_json": args.dataa_train_json or "derived_as_complete_pairs_minus_dev",
            "dataa_dev_json": args.dataa_dev_json,
            "datab_detection_json": args.datab_detection_json,
        },
        "counts": {
            "complete_dataa_pairs": len(complete),
            "eligible_train_pairs": len(candidates),
            "selected_train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "datab_replay_records": len(replay),
            "camera_mismatch_train_pairs_excluded": len(camera_mismatches),
        },
        "train_partition_source": train_partition_source,
        "complete_pairs_not_assigned_to_train_or_dev": sorted(set(complete) - train_ids - dev_ids),
        "selected_train_source_motion": {
            f"{source}|{bucket}": count for (source, bucket), count in sorted(selected_counts.items())
        },
        "datab_replay_labels": dict(Counter(row["label"] for row in replay)),
        "leakage_audit": {
            "train_dev_overlap": sorted(train_ids & dev_ids),
            "selected_train_dev_overlap": sorted({row["case_id"] for row in train_pairs} & dev_ids),
            "dev_usage": "development gate only; repeatedly evaluated and not a pristine final paper test",
        },
        "scoring": {
            "assistant_prefix": "<verdict>",
            "candidate_texts": [" Real", " Fake"],
            "camera_text_in_detection_prompt": False,
            "independent_real_fake_forward": True,
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
    }
    summary_path = out_dir / "caspr_gate1_data_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
