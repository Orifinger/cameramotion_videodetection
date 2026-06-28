#!/usr/bin/env python3
"""Sample a concrete Data A v1 generation plan from a catalog and donor pair pool.

Use this only after route smoke tests. Example for the first VACE-only stage:
  --operations object_swap person_appearance_swap surface_content_edit \
               object_attribute_edit surface_attribute_edit

The script caps each target source video to one formal Fake and caps donor reuse.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def weighted_choice(rng: random.Random, rows: Sequence[Dict[str, Any]], key: str = "weight") -> Dict[str, Any]:
    total = sum(max(0.0, float(x.get(key, 0.0))) for x in rows)
    if total <= 0:
        return rng.choice(list(rows))
    x = rng.random() * total
    acc = 0.0
    for row in rows:
        acc += max(0.0, float(row.get(key, 0.0)))
        if acc >= x:
            return row
    return rows[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--pair-pool", required=True, type=Path)
    p.add_argument("--operation-registry", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-cases", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260628)
    p.add_argument("--max-donor-reuse", type=int, default=3)
    p.add_argument("--max-target-video-use", type=int, default=1)
    p.add_argument("--operations", nargs="+", required=True)
    args = p.parse_args()

    catalog = read_json(args.catalog)
    pair_pool = read_json(args.pair_pool)
    op_registry = read_json(args.operation_registry)["operations"]
    allowed_ops = set(args.operations)
    unknown = allowed_ops - set(op_registry)
    if unknown:
        raise ValueError(f"Unknown operations: {sorted(unknown)}")

    rng = random.Random(args.seed)
    tracks = catalog["tracks"]
    pairs = pair_pool["pairs"]
    by_op: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for pair in pairs:
        op = pair["operation"]
        if op not in allowed_ops:
            continue
        for route in pair.get("route_candidates", []):
            if float(route.get("weight", 0.0)) <= 0:
                continue
            by_op[op].append({
                "kind": "donor_pair",
                "operation": op,
                "weight": float(pair["compatibility"].get("pair_score", 0.0)) * float(route["weight"]),
                "route": route["route_id"],
                "pair": pair,
                "target_video_id": pair["target"]["video_id"],
                "donor_track_id": pair["donor"]["track_id"],
            })

    for track in tracks:
        if not track.get("eligible"):
            continue
        for op_item in track.get("editable_operations", []):
            op = op_item["operation"]
            if op not in allowed_ops or op_item.get("status") != "eligible":
                continue
            for route in op_item.get("route_candidates", []):
                route_id = route["route_id"]
                if "reference" in route_id:
                    continue
                by_op[op].append({
                    "kind": "direct_track",
                    "operation": op,
                    "weight": float(op_item.get("operation_weight", 0.0)) * float(route.get("weight", 0.0)),
                    "route": route_id,
                    "track": track,
                    "target_video_id": track["video_id"],
                    "donor_track_id": None,
                })

    available_ops = [op for op in allowed_ops if by_op.get(op)]
    if not available_ops:
        raise RuntimeError("No planning candidates for the requested operations")
    op_weights = [{"operation": op, "weight": float(op_registry[op]["global_weight"])} for op in available_ops]

    selected: List[Dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    donor_counts: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(1000, args.num_cases * 200)

    while len(selected) < args.num_cases and attempts < max_attempts:
        attempts += 1
        op = weighted_choice(rng, op_weights)["operation"]
        candidate = weighted_choice(rng, by_op[op])
        target_vid = candidate["target_video_id"]
        donor_tid = candidate["donor_track_id"]
        if target_counts[target_vid] >= args.max_target_video_use:
            continue
        if donor_tid and donor_counts[donor_tid] >= args.max_donor_reuse:
            continue

        case_id = f"dataA_v1_{len(selected) + 1:05d}"
        if candidate["kind"] == "donor_pair":
            pair = candidate["pair"]
            case = {
                "case_id": case_id,
                "status": "planned",
                "operation": candidate["operation"],
                "generator_route": candidate["route"],
                "target": pair["target"],
                "donor": pair["donor"],
                "compatibility": pair["compatibility"],
                "reference_materialization": {
                    "status": "pending",
                    "strategy": pair["donor"]["reference_frame_strategy"],
                },
                "sampling_meta": {
                    "candidate_kind": "donor_pair",
                    "sample_weight": candidate["weight"],
                    "seed": args.seed,
                },
            }
        else:
            t = candidate["track"]
            case = {
                "case_id": case_id,
                "status": "planned",
                "operation": candidate["operation"],
                "generator_route": candidate["route"],
                "target": {
                    "video_id": t["video_id"],
                    "video_path": t.get("video_path"),
                    "track_id": t["track_id"],
                    "candidate_class": t["candidate_class"],
                    "canonical_concept": t["canonical_concept"],
                    "mask_tube_path": t.get("mask_tube_path"),
                    "bbox_tube_xywh": t.get("bbox_tube_xywh"),
                },
                "donor": None,
                "reference_materialization": None,
                "sampling_meta": {
                    "candidate_kind": "direct_track",
                    "sample_weight": candidate["weight"],
                    "seed": args.seed,
                },
            }

        selected.append(case)
        target_counts[target_vid] += 1
        if donor_tid:
            donor_counts[donor_tid] += 1

    if len(selected) < args.num_cases:
        print(f"WARNING: sampled {len(selected)}/{args.num_cases}; constraints may be too strict.")

    payload = {
        "schema_version": "dataA_v1_generation_plan",
        "seed": args.seed,
        "requested_operations": sorted(allowed_ops),
        "requested_case_count": args.num_cases,
        "actual_case_count": len(selected),
        "constraints": {
            "max_donor_reuse": args.max_donor_reuse,
            "max_target_video_use": args.max_target_video_use,
        },
        "operation_counts": dict(Counter(c["operation"] for c in selected)),
        "route_counts": dict(Counter(c["generator_route"] for c in selected)),
        "cases": selected,
    }
    write_json(args.out, payload)
    print(json.dumps({
        "saved": str(args.out),
        "actual_case_count": len(selected),
        "operation_counts": payload["operation_counts"],
        "route_counts": payload["route_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
