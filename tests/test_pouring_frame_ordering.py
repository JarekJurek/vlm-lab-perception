from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.pouring import PouringFrameOrderingDataset
from tlg import model_runner
from tlg.prompts import PromptBank


class PouringFrameOrderingTest(unittest.TestCase):
    def test_builds_four_frame_ordering_sample_from_one_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 10, 20, 30])

            dataset = PouringFrameOrderingDataset(root, prompt_bank=PromptBank.default(), seed=7)
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertEqual(len(ordering_samples), 1)
            sample = ordering_samples[0]
            ann = sample["annotations"][0]

            self.assertEqual(sample["video_id"], "video_0")
            self.assertEqual(sample["frame_count"], 4)
            self.assertEqual(len(sample["ordered_rgb_images"]), 4)
            self.assertEqual(sample["ordered_image_labels"], ["A", "B", "C", "D"])
            self.assertEqual(sample["chronological_frame_indices"], [0, 10, 20, 30])
            self.assertNotEqual(sample["display_frame_indices"], sample["chronological_frame_indices"])
            self.assertEqual(ann.metadata["task"], "pouring_frame_ordering")
            self.assertEqual(ann.answer, sample["label"])
            self.assertEqual(set(ann.answer.replace(",", "").split()), {"A", "B", "C", "D"})

    def test_qwen_runner_passes_ordered_frames_as_multiple_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 10, 20])
            dataset = PouringFrameOrderingDataset(root, prompt_bank=PromptBank.default(), seed=3)
            sample = next(sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering")
            captured = {}
            original_run_task = model_runner.run_task

            def fake_run_task(_model, _processor, image, prompt, **kwargs):
                captured["primary_pixel"] = image.getpixel((0, 0))
                captured["extra_count"] = len(kwargs.get("extra_images") or [])
                captured["prompt"] = prompt
                return "ok"

            try:
                model_runner.run_task = fake_run_task
                model_runner.test_sample(
                    model=None,
                    processor=None,
                    sample=sample,
                    annotations=sample["annotations"],
                    dataset_name="pouring",
                    sample_id=sample["id"],
                    viz_attn=False,
                    viz_out_dir=None,
                )
            finally:
                model_runner.run_task = original_run_task

            first_display_index = sample["display_frame_indices"][0]
            self.assertEqual(captured["primary_pixel"], _color_for_frame(first_display_index))
            self.assertEqual(captured["extra_count"], 2)
            self.assertIn("A, B, C", captured["prompt"])

    def _make_pouring_root(self, base: Path, frame_indices: list[int]) -> Path:
        video_dir = base / "pouring_testset_videos" / "video_0"
        video_dir.mkdir(parents=True)
        for frame_index in frame_indices:
            Image.new("RGB", (8, 8), color=_color_for_frame(frame_index)).save(video_dir / f"rgb_{frame_index}.jpg")
        return base


def _color_for_frame(frame_index: int) -> tuple[int, int, int]:
    return (frame_index % 251, (frame_index * 2) % 251, (frame_index * 3) % 251)


if __name__ == "__main__":
    unittest.main()
