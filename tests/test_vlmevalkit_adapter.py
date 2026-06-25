from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


def _require_runtime_deps() -> None:
    missing = []
    for module in ("PIL", "yaml"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise unittest.SkipTest(f"Missing runtime dependencies: {', '.join(missing)}")


class VLMEvalKitAdapterTest(unittest.TestCase):
    def test_manifest_rows_prompt_and_scoring(self) -> None:
        _require_runtime_deps()
        from PIL import Image
        from tlg.vlmevalkit_adapter import TLGVLMEvalDataset, score_vlmeval_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "samples": [
                            {
                                "id": "sample-1",
                                "dataset": "labliquidvision",
                                "split": "test",
                                "rgb_image": str(image_path),
                                "annotations": [
                                    {
                                        "image_id": "sample-1",
                                        "dataset": "labliquidvision",
                                        "task_type": "count",
                                        "question": "How many containers?",
                                        "answer": 2,
                                        "metadata": {"task": "container_count"},
                                    },
                                    {
                                        "image_id": "sample-1",
                                        "dataset": "labliquidvision",
                                        "task_type": "classification",
                                        "question": "Which type?",
                                        "choices": ["Tube", "Flask"],
                                        "answer": "Tube",
                                        "metadata": {"task": "container_type_simple"},
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = TLGVLMEvalDataset(
                manifest_path,
                split="test",
                tasks=["container_count"],
                image_cache_dir=root / "cache",
            )
            self.assertEqual(len(dataset), 1)
            row = dataset[0]
            self.assertEqual(row["task"], "container_count")

            messages = dataset.build_prompt(row)
            self.assertEqual(messages[0]["type"], "image")
            self.assertEqual(messages[0]["value"], str(image_path))
            self.assertEqual(messages[-1]["type"], "text")
            self.assertIn("How many containers?", messages[-1]["value"])

            scored = score_vlmeval_predictions([{**row, "model": "dummy", "prediction": "2"}])
            self.assertEqual(scored["predictions"][0]["correct"], True)
            self.assertEqual(scored["metrics"]["overall"]["container_count"]["n"], 1)

    def test_ordered_images_builds_cached_strip(self) -> None:
        _require_runtime_deps()
        from PIL import Image
        from tlg.vlmevalkit_adapter import TLGVLMEvalDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            for i in range(3):
                path = root / f"frame_{i}.png"
                Image.new("RGB", (12, 12), color=(255, 255 - i * 20, 255)).save(path)
                paths.append(str(path))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "samples": [
                            {
                                "id": "sample-order",
                                "dataset": "pouring",
                                "split": "test",
                                "rgb_image": paths[0],
                                "ordered_rgb_images": paths,
                                "ordered_image_labels": ["A", "B", "C"],
                                "annotations": [
                                    {
                                        "image_id": "sample-order",
                                        "dataset": "pouring",
                                        "task_type": "ordering",
                                        "question": "Order frames.",
                                        "answer": "A, B, C",
                                        "metadata": {"task": "pouring_frame_ordering"},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = TLGVLMEvalDataset(manifest_path, image_cache_dir=root / "cache")
            messages = dataset.build_prompt(0)

            self.assertEqual([msg["type"] for msg in messages], ["image", "text"])
            self.assertTrue(Path(messages[0]["value"]).exists())
            self.assertIn("horizontal strip of frames", messages[1]["value"])

    def test_runner_finds_nested_vlmeval_model_registry(self) -> None:
        import scripts.run_tlg_vlmeval as runner

        class Config:
            supported_VLM = {"direct": lambda: "direct-model"}
            grouped = {"group": {"nested": lambda: "nested-model"}}

        supported = runner._vlmeval_supported_models(Config)
        self.assertEqual(supported["direct"](), "direct-model")
        self.assertEqual(supported["nested"](), "nested-model")


if __name__ == "__main__":
    unittest.main()
