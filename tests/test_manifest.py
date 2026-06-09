from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_runtime_deps() -> None:
    missing = []
    for module in ("numpy", "PIL", "yaml"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise unittest.SkipTest(f"Missing runtime dependencies: {', '.join(missing)}")


class ManifestTest(unittest.TestCase):
    def test_serialize_and_load_manifest_sample(self) -> None:
        sys.path.append(str(REPO_ROOT / "src"))
        from tlg.annotations import TLGAnnotation
        from tlg.manifest import TLGManifestDataset, serialize_manifest_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "image.png"
            sample = {
                "id": "sample-1",
                "split": "source-train",
                "rgb_image": image_path,
                "bbox_visib": [1, 2, 3, 4],
                "scene_id": "scene-a",
                "nested": {"path": image_path, "values": [1, True, None]},
                "liquid_label": {"0": [384, 411]},
                "annotations": [
                    TLGAnnotation(
                        image_id="sample-1",
                        dataset="dtld",
                        task_type="classification",
                        question="question",
                        choices=["A", "B"],
                        answer="A",
                        metadata={"task": "container_type"},
                    )
                ],
            }

            serialized = serialize_manifest_sample(sample, dataset_name="dtld", split="test")
            self.assertEqual(serialized["rgb_image"], str(image_path))
            self.assertEqual(serialized["scene_id"], "scene-a")
            self.assertEqual(serialized["source_split"], "source-train")
            self.assertEqual(serialized["annotations"][0]["answer"], "A")
            self.assertNotIn("nested", serialized)
            self.assertNotIn("liquid_label", serialized)

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "samples": [
                            serialized,
                            {**serialized, "id": "sample-2", "split": "train"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            ds_test = TLGManifestDataset(manifest_path, split="test")
            ds_all = TLGManifestDataset(manifest_path, split="all")
            self.assertEqual(len(ds_test), 1)
            self.assertEqual(len(ds_all), 2)
            self.assertIsInstance(ds_test[0]["annotations"][0], TLGAnnotation)

    def test_export_and_dry_run_manifest(self) -> None:
        _require_runtime_deps()
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "LabLiquidVision"
            manifest_path = root / "manifest.json"
            for name, vol_liquid in (("sample_001", "10"), ("sample_002", "30")):
                sample_dir = data_root / "processed" / name
                sample_dir.mkdir(parents=True)
                (sample_dir / "Input_vessel.txt").write_text("Tube_50mL\n", encoding="utf-8")
                (sample_dir / "Input_vol_liquid.txt").write_text(f"{vol_liquid}\n", encoding="utf-8")
                (sample_dir / "Input_vol_vessel.txt").write_text("50\n", encoding="utf-8")
                (sample_dir / "Input_color_liquid.txt").write_text("red\n", encoding="utf-8")
                Image.new("RGB", (8, 8), color=(255, 255, 255)).save(sample_dir / "Input_RGBImage.png")

            export_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_tlg_manifest.py"),
                "--datasets",
                "labliquidvision",
                "--data-root",
                f"labliquidvision={data_root}",
                "--out",
                str(manifest_path),
                "--max-samples-per-dataset",
                "2",
                "--shuffle",
                "--seed",
                "7",
                "--train-ratio",
                "0.5",
            ]
            first = subprocess.run(export_cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            first_payload = manifest_path.read_text(encoding="utf-8")

            second = subprocess.run(export_cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(first_payload, manifest_path.read_text(encoding="utf-8"))

            dry_run = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_vlm_tasks.py"),
                    "--dry-run",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-split",
                    "test",
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            output = dry_run.stdout + dry_run.stderr
            self.assertEqual(dry_run.returncode, 0, output)
            self.assertIn("Loading manifest from:", output)
            self.assertIn("Dry run OK: annotations are well-formed.", output)


if __name__ == "__main__":
    unittest.main()
