#!/usr/bin/env python3
"""Create an exact A/B-order swap for DataA pair-selection JSON.

Input records are produced by tools/build_dataa_pair_region_pretext.py --task pair.
The swapped output keeps the same case order and target region, but exchanges:
  - Video A/B user-prompt frame sections
  - image list halves
  - top-level edited_video label
  - assistant <edited_video> ground-truth tag

This is a control for position bias: a content-aware model should flip its A/B
prediction when the two videos are swapped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


EDITED_RE = re.compile(r"<edited_video>\s*([AB])\s*</edited_video>", re.IGNORECASE)
IMAGE_TAG_RE = re.compile(r"<image>", re.IGNORECASE)
USER_PAIR_RE = re.compile(
    r"\AVideo A frames:\n(?P<a>.*?)\n\nVideo B frames:\n(?P<b>.*?)\n\n(?P<rest>.*)\Z",
    re.DOTALL,
)
VIDEO_LABEL_RE = re.compile(r"\[Video [AB] ")


def invert_choice(value: str) -> str:
    value = str(value).strip().upper()
    if value == "A":
        return "B"
    if value == "B":
        return "A"
    raise ValueError(f"expected A or B, got {value!r}")


def get_message(record: Mapping[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def relabel_frame_section(section: str, label: str) -> str:
    return VIDEO_LABEL_RE.sub(f"[Video {label} ", section)


def replace_message(record: dict[str, Any], role: str, content: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            message["content"] = content
            return
    raise ValueError(f"record has no {role!r} message")


def swap_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    out = dict(record)
    out["messages"] = [dict(message) for message in record.get("messages", [])]

    user = get_message(record, "user")
    match = USER_PAIR_RE.match(user)
    if not match:
        raise ValueError(f"record {index}: user prompt is not a recognized pair-selection prompt")

    a_section = match.group("a")
    b_section = match.group("b")
    rest = match.group("rest")
    num_a = len(IMAGE_TAG_RE.findall(a_section))
    num_b = len(IMAGE_TAG_RE.findall(b_section))

    images = list(record.get("images") or [])
    if len(images) != num_a + num_b:
        raise ValueError(
            f"record {index}: image count mismatch, images={len(images)} prompt={num_a}+{num_b}"
        )
    images_a = images[:num_a]
    images_b = images[num_a:]

    new_user = "\n\n".join(
        [
            "Video A frames:\n" + relabel_frame_section(b_section, "A"),
            "Video B frames:\n" + relabel_frame_section(a_section, "B"),
            rest,
        ]
    )
    replace_message(out, "user", new_user)
    out["images"] = images_b + images_a

    edited = str(record.get("edited_video") or "").strip().upper()
    if not edited:
        assistant = get_message(record, "assistant")
        edited_match = EDITED_RE.search(assistant)
        edited = edited_match.group(1).upper() if edited_match else ""
    new_edited = invert_choice(edited)
    out["edited_video"] = new_edited
    out["original_edited_video"] = edited
    out["pair_order_control"] = "ab_swapped"

    assistant = get_message(record, "assistant")
    if assistant:
        new_assistant, count = EDITED_RE.subn(f"<edited_video>{new_edited}</edited_video>", assistant, count=1)
        if count != 1:
            raise ValueError(f"record {index}: assistant has no <edited_video> tag")
        replace_message(out, "assistant", new_assistant)

    if "video_a_source_split" in out or "video_b_source_split" in out:
        out["video_a_source_split"], out["video_b_source_split"] = (
            out.get("video_b_source_split", ""),
            out.get("video_a_source_split", ""),
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original pair-selection JSON.")
    parser.add_argument("--out", required=True, help="Output swapped pair-selection JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {args.input}")
    swapped = [swap_record(record, index) for index, record in enumerate(data)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(swapped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "input": args.input,
                "out": args.out,
                "records": len(swapped),
                "control": "ab_swapped",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
