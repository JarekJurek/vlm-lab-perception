from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.target_overlay import make_target_box_overlay, should_mark_target_box
from tlg.annotations import TLGAnnotation
from tlg import model_runner


class DTLDImageRoutingTest(unittest.TestCase):
    def test_container_count_uses_unmarked_image_and_object_tasks_use_bbox(self) -> None:
        self.assertFalse(should_mark_target_box("dtld", "container_count"))
        self.assertTrue(should_mark_target_box("dtld", "container_type"))
        self.assertTrue(should_mark_target_box("dtld", "container_type_simple"))
        self.assertTrue(should_mark_target_box("dtld", "fill_height_mm"))
        self.assertTrue(should_mark_target_box("dtld", "liquid_presence"))

    def test_target_box_overlay_can_use_bbox_without_mask(self) -> None:
        image = Image.new("RGB", (20, 20), color=(255, 255, 255))

        marked = make_target_box_overlay(image, image, bbox=[5, 5, 8, 8], width=2, padding=0)

        self.assertEqual(marked.getpixel((5, 5)), (255, 0, 0))
        self.assertEqual(marked.getpixel((0, 0)), (255, 255, 255))

    def test_runner_passes_original_image_for_count_and_boxed_image_for_object_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = Path(tmpdir) / "rgb.png"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(rgb_path)
            sample = {
                "rgb_image": rgb_path,
                "bbox_visib": [5, 5, 8, 8],
            }
            annotations = [
                TLGAnnotation(
                    image_id="dtld/000000/000000/000000",
                    dataset="dtld",
                    task_type="count",
                    question="count?",
                    answer=1,
                    metadata={"task": "container_count"},
                ),
                TLGAnnotation(
                    image_id="dtld/000000/000000/000000",
                    dataset="dtld",
                    task_type="type",
                    question="type?",
                    answer="T25 Flask",
                    metadata={"task": "container_type"},
                ),
            ]
            captured_pixels = {}
            original_run_task = model_runner.run_task

            def fake_run_task(_model, _processor, image, _prompt, **kwargs):
                captured_pixels[kwargs["task_name"]] = image.getpixel((5, 5))
                return "ok"

            try:
                model_runner.run_task = fake_run_task
                model_runner.test_sample(
                    model=None,
                    processor=None,
                    sample=sample,
                    annotations=annotations,
                    dataset_name="dtld",
                    sample_id="sample",
                    viz_attn=False,
                    viz_out_dir=None,
                )
            finally:
                model_runner.run_task = original_run_task

            self.assertEqual(captured_pixels["container_count"], (255, 255, 255))
            self.assertEqual(captured_pixels["container_type"], (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
