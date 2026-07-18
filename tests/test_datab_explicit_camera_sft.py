from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_datab_explicit_camera_sft as builder
from tools import install_datab_explicit_camera_sft as installer


def detection_record(frame_dir: str, answer: str, suffix: str = "") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Detect generated video artifacts."},
            {"role": "user", "content": "Frame 1: <image>\nFrame 2: <image>" + suffix},
            {
                "role": "assistant",
                "content": f"<think>unchanged explanation</think>\n<answer>{answer}</answer>",
            },
        ],
        "images": [f"{frame_dir}/1.png", f"{frame_dir}/2.png"],
        "extra": {"must": "stay unchanged"},
    }


class DataBExplicitCameraBuilderTests(unittest.TestCase):
    def test_builds_equal_rows_with_only_camera_user_suffix(self) -> None:
        records = [
            detection_record("/frames/a", "Real"),
            detection_record("/frames/b", "Fake"),
            detection_record("/frames/a", "Real", " duplicate explanation"),
            detection_record("/frames/missing", "Fake"),
        ]
        camera = {
            "/frames/a": {
                "path": "/frames/a",
                "labels": ["no-motion", "no-shaking"],
                "caption": "The camera remains stationary.",
            },
            "/frames/b": {
                "path": "/frames/b",
                "labels": ["complex-motion", "pan-right"],
                "caption": "The camera pans right rapidly.",
            },
        }
        plain, conditioned, manifest, summary = builder.build_paired_datasets(records, camera)
        self.assertEqual(len(plain), 3)
        self.assertEqual(len(conditioned), 3)
        self.assertEqual([row["source_index"] for row in manifest], [0, 1, 2])
        self.assertEqual(summary["missing_records"], 1)
        self.assertEqual(summary["matched_unique_camera_paths"], 2)
        self.assertEqual(summary["answer_counts"], {"Real": 2, "Fake": 1})
        self.assertTrue(summary["paired_integrity"])
        self.assertFalse(summary["no_camera_prompts_contain_camera_block"])
        for baseline, camera_record in zip(plain, conditioned):
            self.assertTrue(builder.records_equal_except_camera_user(baseline, camera_record))
            self.assertEqual(baseline["messages"][0], camera_record["messages"][0])
            self.assertEqual(baseline["messages"][2], camera_record["messages"][2])
            self.assertEqual(baseline["images"], camera_record["images"])
            self.assertEqual(baseline["extra"], camera_record["extra"])
            camera_user = camera_record["messages"][1]["content"]
            self.assertIn("<labels>", camera_user)
            self.assertIn("<caption>", camera_user)

    def test_rejects_existing_camera_prompt(self) -> None:
        record = detection_record("/frames/a", "Real", "\n<camera_motion>old</camera_motion>")
        with self.assertRaisesRegex(ValueError, "already contains"):
            builder.append_camera_context(
                record,
                {"labels": ["no-motion"], "caption": "Stationary camera."},
            )


class DataBExplicitCameraInstallerTests(unittest.TestCase):
    def test_installs_paired_datasets_with_shared_smoke_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            data_dir = root / "llamafactory" / "data"
            source.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            (data_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
            records = [
                detection_record(f"/frames/{index}", "Real" if index % 2 == 0 else "Fake")
                for index in range(8)
            ]
            camera = {
                f"/frames/{index}": {
                    "path": f"/frames/{index}",
                    "labels": ["no-motion" if index % 2 == 0 else "complex-motion"],
                    "caption": f"Camera caption {index}.",
                }
                for index in range(8)
            }
            plain, conditioned, _manifest, _summary = builder.build_paired_datasets(records, camera)
            (source / "datab_sft_no_camera_5739.json").write_text(
                json.dumps(plain), encoding="utf-8"
            )
            (source / "datab_sft_with_camera_labels_caption_5739.json").write_text(
                json.dumps(conditioned), encoding="utf-8"
            )
            argv = [
                "install_datab_explicit_camera_sft.py",
                "--source-dir",
                str(source),
                "--llamafactory-data-dir",
                str(data_dir),
                "--expected-records",
                "8",
                "--smoke-samples",
                "4",
            ]
            with patch.object(sys, "argv", argv):
                installer.main()
            summary = json.loads(
                (source / "llamafactory_install_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["paired_integrity"])
            self.assertEqual(summary["paired_record_count"], 8)
            self.assertEqual(len(summary["shared_smoke_indices"]), 4)
            dataset_info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertIn("datab_explicit_camera_no_camera", dataset_info)
            self.assertIn("datab_explicit_camera_labels_caption", dataset_info)


if __name__ == "__main__":
    unittest.main()
