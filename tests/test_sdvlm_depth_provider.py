from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.sdvlm_runner import DepthAnything3Provider, _extract_da3_depth


class FakeDepthAnything3Provider(DepthAnything3Provider):
    def __init__(self, *, cache_dir: Path) -> None:
        super().__init__(model_path="fake-da3", cache_dir=cache_dir, device="cpu")
        self.calls = 0

    def _infer_depth(self, image):
        self.calls += 1
        width, height = image.size
        return np.arange(width * height, dtype=np.float32).reshape(height, width)


class DepthAnything3ProviderTest(unittest.TestCase):
    def test_cache_miss_then_hit_for_same_input_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeDepthAnything3Provider(cache_dir=Path(tmpdir))
            image = Image.new("RGB", (4, 3), color=(10, 20, 30))

            depth_image_1, meta_1 = provider.get_depth_image(
                image=image,
                sample={},
                sample_id="sample",
                task_name="task",
            )
            depth_image_2, meta_2 = provider.get_depth_image(
                image=image,
                sample={},
                sample_id="sample",
                task_name="task",
            )

            self.assertEqual(provider.calls, 1)
            self.assertEqual(meta_1["sdvlm_depth_cache_status"], "miss")
            self.assertEqual(meta_2["sdvlm_depth_cache_status"], "hit")
            self.assertEqual(depth_image_1.mode, "L")
            self.assertEqual(depth_image_1.size, image.size)
            self.assertEqual(depth_image_2.size, image.size)

    def test_extract_da3_depth_accepts_object_and_batch_dimension(self) -> None:
        pred = SimpleNamespace(depth=np.ones((1, 2, 3), dtype=np.float32))

        out = _extract_da3_depth(pred)

        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(float(out[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
