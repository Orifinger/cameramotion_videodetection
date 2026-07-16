import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.camera_binary_route_gate.diagnose_experts import diagnose, exact_binomial_two_sided


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BinaryExpertDiagnosticTest(unittest.TestCase):
    def test_large_exact_binomial_does_not_overflow(self) -> None:
        self.assertEqual(exact_binomial_two_sided(1500, 1500), 1.0)
        value = exact_binomial_two_sided(500, 2500)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
    def test_detects_intended_semantic_crossover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "indices"
            definitions = [
                ("real", "case-no-real", "no-motion"),
                ("model-a", "case-no-fake", "no-motion"),
                ("real", "case-motion-real", "motion"),
                ("model-a", "case-motion-fake", "motion"),
            ]
            index_payload = {"real": [], "model-a": []}
            route_rows = []
            for source, case_id, route in definitions:
                frame_dir = root / "test_normalized" / source / case_id
                frame_dir.mkdir(parents=True)
                index_payload[source].append(str(frame_dir))
                route_rows.append(
                    {
                        "video_id": f"{source}/{case_id}",
                        "binary_route_bucket": route,
                    }
                )
            write_json(index_dir / "test_index.rank0.json", index_payload)
            route_manifest = root / "route.jsonl"
            write_jsonl(route_manifest, route_rows)

            no_predictions = []
            motion_predictions = []
            for source, case_id, route in definitions:
                video_id = f"{source}/{case_id}"
                gold = "Real" if source == "real" else "Fake"
                wrong = "Fake" if gold == "Real" else "Real"
                no_predictions.append(
                    {
                        "video_id": video_id,
                        "aigc_model_name": source,
                        "answer": gold if route == "no-motion" else wrong,
                    }
                )
                motion_predictions.append(
                    {
                        "video_id": video_id,
                        "aigc_model_name": source,
                        "answer": gold if route == "motion" else wrong,
                    }
                )
            no_dir = root / "no"
            motion_dir = root / "motion"
            write_json(no_dir / "rank_0" / "predictions.json", no_predictions)
            write_json(motion_dir / "rank_0" / "predictions.json", motion_predictions)

            output = root / "full.json"
            compact = root / "compact.json"
            diagnose(
                argparse.Namespace(
                    index_dir=str(index_dir),
                    route_manifest=str(route_manifest),
                    no_motion_prediction_dir=str(no_dir),
                    motion_prediction_dir=str(motion_dir),
                    output_json=str(output),
                    output_compact_json=str(compact),
                    expected_ranks=1,
                )
            )
            summary = json.loads(compact.read_text(encoding="utf-8"))
            self.assertEqual(summary["pattern"], "semantic_crossover")
            self.assertEqual(
                summary["balanced_accuracy"]["route_no_motion"]["no-motion"], 1.0
            )
            self.assertEqual(summary["balanced_accuracy"]["route_motion"]["motion"], 1.0)


if __name__ == "__main__":
    unittest.main()
