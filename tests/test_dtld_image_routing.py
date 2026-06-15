from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg.target_overlay import make_target_box_overlay, should_mark_target_box
from tlg.annotations import TLGAnnotation
from tlg import PromptBank, model_runner


class DTLDImageRoutingTest(unittest.TestCase):
    def test_container_count_uses_unmarked_image_and_object_tasks_use_bbox(self) -> None:
        self.assertFalse(should_mark_target_box("dtld", "container_count"))
        self.assertTrue(should_mark_target_box("dtld", "container_type"))
        self.assertTrue(should_mark_target_box("dtld", "container_type_simple"))
        self.assertTrue(should_mark_target_box("dtld", "fill_height_mm"))

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
            captured_prompts = {}
            original_run_task = model_runner.run_task

            def fake_run_task(_model, _processor, image, prompt, **kwargs):
                captured_pixels[kwargs["task_name"]] = image.getpixel((5, 5))
                captured_prompts[kwargs["task_name"]] = prompt
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
            self.assertFalse(captured_prompts["container_count"].startswith("The target region"))
            self.assertTrue(
                captured_prompts["container_type"].startswith(PromptBank.default().target_region_bbox_prompt())
            )

    def test_runner_uses_custom_prompt_bank_for_target_box_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rgb_path = root / "rgb.png"
            prompt_defs_path = root / "prompt_bank.yaml"
            dataset_prompts_path = root / "dataset_prompts.yaml"
            Image.new("RGB", (20, 20), color=(255, 255, 255)).save(rgb_path)
            prompt_defs_path.write_text(
                """
target_region_bbox_prompt:
  question: "CUSTOM TARGET PREFIX."
container_type:
  question: "What is the marked container?"
  task_type: categorical
  choices:
    - Flask
""",
                encoding="utf-8",
            )
            dataset_prompts_path.write_text(
                """
dtld:
  include:
    - container_type
  task_prompt_prefixes:
    container_type:
      - target_region_bbox_prompt
""",
                encoding="utf-8",
            )
            sample = {"rgb_image": rgb_path, "bbox_visib": [5, 5, 8, 8]}
            annotations = [
                TLGAnnotation(
                    image_id="dtld/000000/000000/000000",
                    dataset="dtld",
                    task_type="type",
                    question="type?",
                    answer="Flask",
                    metadata={"task": "container_type"},
                ),
            ]
            prompt_bank = PromptBank.from_files(prompt_defs_path, dataset_prompts_path)
            captured_prompts = {}
            original_run_task = model_runner.run_task

            def fake_run_task(_model, _processor, _image, prompt, **kwargs):
                captured_prompts[kwargs["task_name"]] = prompt
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
                    prompt_bank=prompt_bank,
                )
            finally:
                model_runner.run_task = original_run_task

            self.assertTrue(captured_prompts["container_type"].startswith("CUSTOM TARGET PREFIX."))


if __name__ == "__main__":
    unittest.main()
