#!/usr/bin/env python3
"""Sample a Data A v1 plan with exact per-operation quotas.

Use this for smoke sets, where coverage matters more than globally random proportions.
Example:
  python scripts/sample_dataa_v1_quota_plan.py \
    --catalog res/dataA_v1/registries/track_editability_catalog_v1.json \
    --pair-pool res/dataA_v1/registries/donor_pair_pool_v1.json \
    --out res/dataA_v1/plans/vace14b_stage1_quota_plan.json \
    --quotas object_swap=5 person_appearance_swap=3 surface_content_edit=3 \
              object_attribute_edit=2 surface_attribute_edit=2 \
    --max-target-video-use 1 --max-donor-reuse 3 --seed 20260629
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


def weighted_choice(rng: random.Random, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(max(0.0, float(r.get("weight", 0.0))) for r in rows)
    if total <= 0:
        return rng.choice(list(rows))
    needle = rng.random() * total
    acc = 0.0
    for row in rows:
        acc += max(0.0, float(row.get("weight", 0.0)))
        if acc >= needle:
            return row
    return rows[-1]


def parse_quotas(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Quota must use OPERATION=COUNT: {item}")
        op, count = item.split("=", 1)
        count_i = int(count)
        if count_i < 0:
            raise ValueError(f"Quota must be non-negative: {item}")
        out[op.strip()] = count_i
    if not out or sum(out.values()) <= 0:
        raise ValueError("At least one positive quota is required")
    return out


def build_candidates(catalog: Dict[str, Any], pair_pool: Dict[str, Any], wanted_ops: set[str]) -> Dict[str, List[Dict[str, Any]]]:
    by_op: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # reference-driven candidates: source-to-source donor pairs
    for pair in pair_pool.get("pairs", []):
        op = pair.get("operation")
        if op not in wanted_ops:
            continue
        for route in pair.get("route_candidates", []):
            route_weight = float(route.get("weight", 0.0))
            if route_weight <= 0:
                continue
            by_op[op].append({
                "kind": "donor_pair",
                "operation": op,
                "weight": float(pair.get("compatibility", {}).get("pair_score", 0.0)) * route_weight,
                "route": route["route_id"],
                "pair": pair,
                "target_video_id": pair["target"]["video_id"],
                "target_track_id": pair["target"]["track_id"],
                "donor_track_id": pair["donor"]["track_id"],
            })

    # no-donor / text-driven candidates
    for track in catalog.get("tracks", []):
        if not track.get("eligible"):
            continue
        for op_item in track.get("editable_operations", []):
            op = op_item.get("operation")
            if op not in wanted_ops or op_item.get("status") != "eligible":
                continue
            for route in op_item.get("route_candidates", []):
                route_id = route.get("route_id", "")
                if "reference" in route_id:
                    continue
                route_weight = float(route.get("weight", 0.0))
                if route_weight <= 0:
                    continue
                by_op[op].append({
                    "kind": "direct_track",
                    "operation": op,
                    "weight": float(op_item.get("operation_weight", 0.0)) * route_weight,
                    "route": route_id,
                    "track": track,
                    "target_video_id": track["video_id"],
                    "target_track_id": track["track_id"],
                    "donor_track_id": None,
                })
    return by_op


def make_case(case_id: str, candidate: Dict[str, Any], seed: int) -> Dict[str, Any]:
    if candidate["kind"] == "donor_pair":
        pair = candidate["pair"]
        return {
            "case_id": case_id,
            "status": "planned_needs_visual_review",
            "operation": candidate["operation"],
            "generator_route": candidate["route"],
            "target": pair["target"],
            "donor": pair["donor"],
            "compatibility": pair["compatibility"],
            "reference_materialization": {
                "status": "pending",
                "strategy": pair["donor"].get("reference_frame_strategy", "deferred_pick_largest_visible_mask_frame"),
            },
            "edit_spec": {
                "source_description": pair["target"].get("canonical_concept"),
                "target_description": None,
                "prompt": None,
                "review_decision": "pending",
            },
            "sampling_meta": {
                "candidate_kind": "donor_pair",
                "sample_weight": candidate["weight"],
                "seed": seed,
            },
        }
    track = candidate["track"]
    return {
        "case_id": case_id,
        "status": "planned_needs_visual_review",
        "operation": candidate["operation"],
        "generator_route": candidate["route"],
        "target": {
            "video_id": track["video_id"],
            "video_path": track.get("video_path"),
            "track_id": track["track_id"],
            "candidate_class": track["candidate_class"],
            "canonical_concept": track["canonical_concept"],
            "mask_tube_path": track.get("mask_tube_path"),
            "bbox_tube_xywh": track.get("bbox_tube_xywh"),
        },
        "donor": None,
        "compatibility": None,
        "reference_materialization": None,
        "edit_spec": {
            "source_description": track.get("canonical_concept"),
            "target_description": None,
            "prompt": None,
            "review_decision": "pending",
        },
        "sampling_meta": {
            "candidate_kind": "direct_track",
            "sample_weight": candidate["weight"],
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--pair-pool", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quotas", required=True, nargs="+", help="e.g. object_swap=5 person_appearance_swap=3")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--max-donor-reuse", type=int, default=3)
    parser.add_argument("--max-target-video-use", type=int, default=1)
    parser.add_argument("--max-attempts-per-case", type=int, default=5000)
    args = parser.parse_args()

    quotas = parse_quotas(args.quotas)
    rng = random.Random(args.seed)
    catalog = read_json(args.catalog)
    pair_pool = read_json(args.pair_pool)
    by_op = build_candidates(catalog, pair_pool, set(quotas))

    unavailable = [op for op, count in quotas.items() if count > 0 and not by_op.get(op)]
    if unavailable:
        raise RuntimeError(f"No candidates for operations: {unavailable}")

    target_counts: Counter[str] = Counter()
    donor_counts: Counter[str] = Counter()
    selected: List[Dict[str, Any]] = []
    unsatisfied: Dict[str, int] = {}

    # least-populated operations first; this prevents a large candidate pool from consuming target videos.
    op_order = sorted(quotas, key=lambda op: (len(by_op[op]), op))
    for op in op_order:
        needed = quotas[op]
        made = 0
        attempts = 0
        while made < needed and attempts < args.max_attempts_per_case * max(1, needed):
            attempts += 1
            candidate = weighted_choice(rng, by_op[op])
            if target_counts[candidate["target_video_id"]] >= args.max_target_video_use:
                continue
            donor_track_id = candidate.get("donor_track_id")
            if donor_track_id and donor_counts[donor_track_id] >= args.max_donor_reuse:
                continue
            case = make_case(f"dataA_v1_{len(selected) + 1:05d}", candidate, args.seed)
            selected.append(case)
            target_counts[candidate["target_video_id"]] += 1
            if donor_track_id:
                donor_counts[donor_track_id] += 1
            made += 1
        if made < needed:
            unsatisfied[op] = needed - made

    # Stable order by case ID after selection; quota order does not imply execution order.
    selected.sort(key=lambda x: x["case_id"])
    payload = {
        "schema_version": "dataA_v1_quota_generation_plan",
        "seed": args.seed,
        "requested_quotas": quotas,
        "actual_operation_counts": dict(Counter(c["operation"] for c in selected)),
        "unsatisfied_quotas": unsatisfied,
        "constraints": {
            "max_target_video_use": args.max_target_video_use,
            "max_donor_reuse": args.max_donor_reuse,
        },
        "status": "draft_needs_visual_review",
        "cases": selected,
    }
    write_json(args.out, payload)
    print(json.dumps({
        "saved": str(args.out),
        "actual_case_count": len(selected),
        "actual_operation_counts": payload["actual_operation_counts"],
        "unsatisfied_quotas": unsatisfied,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
