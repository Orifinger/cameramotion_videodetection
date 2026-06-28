#!/usr/bin/env python3
"""Create the video-level domain-label skeleton required by donor pairing."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--tracks', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    payload = json.loads(args.tracks.read_text(encoding='utf-8'))
    tracks = payload.get('tracks', payload if isinstance(payload, list) else [])
    if not isinstance(tracks, list):
        raise ValueError('Expected a JSON object with tracks or a JSON list')
    by_video = {}
    for x in tracks:
        if not isinstance(x, dict) or not x.get('video_id'):
            continue
        vid = str(x['video_id'])
        by_video.setdefault(vid, {
            'video_id': vid,
            'video_path': x.get('video_path'),
            'content_domain': 'unknown',
            'style_domain': 'unknown',
            'domain_confidence': 0.0,
            'status': 'needs_qwen3_label',
        })
    out = {
        'schema_version': 'dataA_v1_video_domain_index',
        'label_space': [
            'real_live_action', 'animation_cartoon', 'game_scene',
            'cg_rendered', 'mixed', 'unknown'
        ],
        'videos': [by_video[k] for k in sorted(by_video)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {args.out} ({len(out["videos"])} videos)')

if __name__ == '__main__':
    main()
