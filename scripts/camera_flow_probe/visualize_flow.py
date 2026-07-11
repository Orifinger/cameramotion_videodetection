#!/usr/bin/env python3
"""Render camera-flow diagnostics for a few Data A cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl
from scripts.camera_flow_probe.geometry import (
    dense_transform_flow,
    fit_global_camera_transform,
    forward_backward_error,
    resize_and_pad,
)
from scripts.camera_flow_probe.masks import load_mask_tube
from scripts.camera_flow_probe.models import TorchvisionRaft
from scripts.camera_flow_probe.video import paired_dense_frames


def _flow_color(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    scale = max(float(np.nanpercentile(magnitude, 95)), 1e-6)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 90.0 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _heatmap(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.nanpercentile(values, 95)), 1e-6)
    normalized = np.clip(np.nan_to_num(values) / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 330), 28), (0, 0, 0), -1)
    cv2.putText(output, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _pair_index(row: Mapping[str, Any], timestamps: np.ndarray) -> int:
    value = row.get("edit_time_range_source_sec") or []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        center = (float(value[0]) + float(value[1])) / 2.0
    else:
        center = float(timestamps[len(timestamps) // 2])
    return max(0, min(timestamps.size - 2, int(np.argmin(np.abs(timestamps - center)))))


def _render_role(
    *,
    frames: np.ndarray,
    timestamp: float,
    pair_index: int,
    mask: np.ndarray,
    raft: TorchvisionRaft,
) -> tuple[np.ndarray, dict[str, Any]]:
    pair = frames[pair_index : pair_index + 2]
    forward, backward, geometry = raft.infer_pairs(pair, backward=True)
    flow = forward[0]
    fb = forward_backward_error(flow, backward[0]) if backward is not None else None
    transform, stats = fit_global_camera_transform(flow, fb_error=fb)
    global_flow = dense_transform_flow(transform, flow.shape[0], flow.shape[1])
    residual = np.linalg.norm(flow - global_flow, axis=2)
    frame1 = resize_and_pad(pair[0], geometry)
    frame2 = resize_and_pad(pair[1], geometry)
    warped2 = cv2.warpPerspective(
        frame2,
        np.linalg.inv(transform),
        (geometry.canvas_width, geometry.canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    difference = np.abs(frame1.astype(np.float32) - warped2.astype(np.float32)).mean(axis=2)
    mask_canvas = resize_and_pad((mask > 0).astype(np.uint8) * 255, geometry)
    overlay = frame1.copy()
    overlay[mask_canvas > 0] = (0.55 * overlay[mask_canvas > 0] + 0.45 * np.array([255, 32, 32])).astype(np.uint8)
    tiles = [
        _label(frame1, f"frame t={timestamp:.3f}s"),
        _label(frame2, "next frame"),
        _label(_flow_color(flow), "RAFT total flow"),
        _label(_flow_color(global_flow), f"global {stats['model']}"),
        _label(warped2, "next warped to current"),
        _label(_heatmap(difference), "photometric residual"),
        _label(_heatmap(residual), "flow minus global"),
        _label(overlay, "GT mask (audit only)"),
    ]
    panel = np.concatenate(
        [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)],
        axis=0,
    )
    stats["timestamp_sec"] = float(timestamp)
    stats["median_residual_flow_px"] = float(np.median(residual))
    stats["median_photometric_residual"] = float(np.median(difference))
    return panel, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"))
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    raft = TorchvisionRaft(args.raft_checkpoint, device=device, long_side=512, batch_size=1)
    rows = read_jsonl(args.manifest_jsonl)[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for row in rows:
        real, fake, timestamps, meta = paired_dense_frames(
            Path(str(row["real_video"])),
            Path(str(row["fake_video"])),
            target_fps=args.target_fps,
        )
        index = _pair_index(row, timestamps)
        tube = load_mask_tube(Path(str(row["mask_npz"])), Path(str(row["case_manifest"])))
        mask = tube.sample(float(timestamps[index]), height=meta.height, width=meta.width)
        case_summary: dict[str, Any] = {"case_id": row["case_id"], "motion_bucket": row.get("motion_bucket")}
        for role, frames in (("real", real), ("fake", fake)):
            panel, stats = _render_role(
                frames=frames,
                timestamp=float(timestamps[index]),
                pair_index=index,
                mask=mask,
                raft=raft,
            )
            destination = args.output_dir / f"{row['case_id']}_{role}.jpg"
            cv2.imwrite(str(destination), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            case_summary[role] = stats
        summaries.append(case_summary)
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "cases": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
