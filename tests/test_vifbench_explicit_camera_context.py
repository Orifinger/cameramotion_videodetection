from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_vifbench_camera_context import audit_prompts, prepare


class VifbenchCameraContextTest(unittest.TestCase):
    def test_prepare_matches_different_roots_by_test_normalized_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_dir = root / "index"
            index_dir.mkdir()
            (index_dir / "test_index.rank0.json").write_text(
                json.dumps(
                    {
                        "real": ["/frames/test_normalized/real/sample-1"],
                        "fakegen": ["/frames/test_normalized/fakegen/sample-1"],
                    }
                ),
                encoding="utf-8",
            )
            (index_dir / "test_index.rank1.json").write_text("{}", encoding="utf-8")
            camera_jsonl = root / "camera.jsonl"
            camera_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "path": "/another/test_normalized/real/sample-1",
                                "labels": ["no-motion"],
                                "caption": "The camera is static.",
                            }
                        ),
                        json.dumps(
                            {
                                "path": "/another/test_normalized/fakegen/sample-1",
                                "labels": ["pan-left"],
                                "caption": "The camera pans left.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "canonical.jsonl"
            summary = root / "summary.json"
            prepare(
                argparse.Namespace(
                    index_dir=index_dir,
                    camera_json=camera_jsonl,
                    output_jsonl=output,
                    summary_json=summary,
                    expected_ranks=2,
                    min_coverage=1.0,
                )
            )
            result = json.loads(summary.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["coverage"], 1.0)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["video_id"], "real/sample-1")

    def test_prepare_resolves_basename_collision_by_one_to_one_elimination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_dir = root / "index"
            index_dir.mkdir()
            (index_dir / "test_index.rank0.json").write_text(
                json.dumps(
                    {
                        "real": ["/frames/test_normalized/real/shared-id"],
                        "fakegen": ["/frames/test_normalized/fakegen/shared-id"],
                    }
                ),
                encoding="utf-8",
            )
            camera_jsonl = root / "camera.jsonl"
            camera_jsonl.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "path": "/frames/test_normalized/real/shared-id",
                                "labels": ["no-motion"],
                                "caption": "The camera is static.",
                            }
                        ),
                        json.dumps(
                            {
                                "path": "/legacy/unmapped-generator/shared-id",
                                "labels": ["pan-left"],
                                "caption": "The camera pans left.",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "canonical.jsonl"
            summary = root / "summary.json"
            prepare(
                argparse.Namespace(
                    index_dir=index_dir,
                    camera_json=camera_jsonl,
                    output_jsonl=output,
                    summary_json=summary,
                    expected_ranks=1,
                    min_coverage=1.0,
                )
            )
            result = json.loads(summary.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["one_to_one_elimination_count"], 1)
            self.assertEqual(result["ambiguous_count"], 0)
            self.assertEqual(rows[1]["match_method"], "basename_one_to_one_elimination")
            self.assertEqual(rows[1]["labels"], ["pan-left"])

    def test_prompt_audit_enforces_exact_training_append(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detection = root / "detection.json"
            detection.write_text(
                json.dumps(
                    [
                        {
                            "messages": [
                                {"role": "system", "content": "system\nprompt"},
                                {
                                    "role": "user",
                                    "content": "frames\n\nPlease analyze the video.",
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            system = root / "system.txt"
            no_camera = root / "no_camera.txt"
            with_camera = root / "with_camera.txt"
            system.write_text("system\\nprompt\n", encoding="utf-8")
            no_camera.write_text("\\n\\nPlease analyze the video.\n", encoding="utf-8")
            with_camera.write_text(
                "\\n\\nPlease analyze the video.\\n\\n"
                "<camera_motion>\\n<labels>{camera_labels}</labels>\\n"
                "<caption>{camera_caption}</caption>\\n</camera_motion>\n",
                encoding="utf-8",
            )
            output = root / "audit.json"
            audit_prompts(
                argparse.Namespace(
                    detection_json=detection,
                    system_prompt_file=system,
                    no_camera_suffix_file=no_camera,
                    with_camera_suffix_file=with_camera,
                    output_json=output,
                )
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "passed")
            self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
