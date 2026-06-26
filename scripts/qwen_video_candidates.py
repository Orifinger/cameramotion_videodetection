import json
import requests
from pathlib import Path


PROMPT = """
You are proposing semantic object candidates for video segmentation.
Return up to 6 salient, trackable objects.
Return JSON only.
"""


def call_qwen(api_base, video_path, model):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    r = requests.post(f"{api_base}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    videos = [json.loads(l) for l in open(args.manifest, "r", encoding="utf-8")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for v in videos:
            try:
                res = call_qwen(args.api_base, v["video_path"], args.model)
                f.write(json.dumps({"video_id": v["video_id"], "result": res}, ensure_ascii=False) + "\n")
            except Exception as e:
                f.write(json.dumps({"video_id": v["video_id"], "error": str(e)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
