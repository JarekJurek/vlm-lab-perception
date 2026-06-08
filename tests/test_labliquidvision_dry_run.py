from __future__ import annotations

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


class LabLiquidVisionDryRunTest(unittest.TestCase):
    def test_dry_run_prints_container_count_annotation(self) -> None:
        _require_runtime_deps()
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "processed" / "sample_001"
            sample_dir.mkdir(parents=True)
            (sample_dir / "Input_vessel.txt").write_text("Tube_50mL\n", encoding="utf-8")
            (sample_dir / "Input_vol_liquid.txt").write_text("10\n", encoding="utf-8")
            (sample_dir / "Input_vol_vessel.txt").write_text("50\n", encoding="utf-8")
            (sample_dir / "Input_color_liquid.txt").write_text("red\n", encoding="utf-8")
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(sample_dir / "Input_RGBImage.png")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_vlm_tasks.py"),
                    "--dry-run",
                    "--datasets",
                    "labliquidvision",
                    "--max-samples",
                    "1",
                    "--data-root",
                    f"labliquidvision={root}",
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("Dry run OK: annotations are well-formed.", output)
        self.assertIn("- container_count: 1", output)


if __name__ == "__main__":
    unittest.main()
