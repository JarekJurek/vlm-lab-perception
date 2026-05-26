from __future__ import annotations

from collections import Counter
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
    def test_builds_multiple_four_frame_ordering_samples_from_one_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 10, 20, 30, 40])

            dataset = PouringFrameOrderingDataset(root, prompt_bank=PromptBank.default(), seed=7)
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertEqual(len(ordering_samples), 5)
            sample = ordering_samples[0]
            ann = sample["annotations"][0]

            self.assertEqual(sample["video_id"], "video_0")
            self.assertEqual(sample["frame_count"], 4)
            self.assertEqual(len(sample["ordered_rgb_images"]), 4)
            self.assertEqual(sample["ordered_image_labels"], ["A", "B", "C", "D"])
            self.assertEqual(sample["chronological_frame_indices"], [0, 10, 20, 30])
            self.assertEqual(sample["ordering_frame_positions"], [0, 1, 2, 3])
            self.assertEqual(sorted(sample["display_frame_indices"]), sample["chronological_frame_indices"])
            self.assertEqual(ann.metadata["task"], "pouring_frame_ordering")
            self.assertEqual(ann.answer, sample["label"])
            self.assertEqual(set(ann.answer.replace(",", "").split()), {"A", "B", "C", "D"})

    def test_caps_ordering_combinations_per_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=list(range(8)))

            dataset = PouringFrameOrderingDataset(
                root,
                prompt_bank=PromptBank.default(),
                seed=7,
                max_ordering_samples_per_video=6,
                min_ordering_frame_gap=0,
            )
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertEqual(len(ordering_samples), 6)
            self.assertEqual(len({sample["id"] for sample in ordering_samples}), 6)

    def test_balances_display_permutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=list(range(11)))

            dataset = PouringFrameOrderingDataset(
                root,
                prompt_bank=PromptBank.default(),
                seed=7,
                max_ordering_samples_per_video=48,
                min_ordering_frame_gap=0,
            )
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]
            answers = Counter("".join(sample["answer_order_labels"]) for sample in ordering_samples)

            self.assertEqual(len(ordering_samples), 48)
            self.assertEqual(len(answers), 24)
            self.assertEqual(set(answers.values()), {2})

    def test_min_ordering_frame_gap_filters_close_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 2, 4, 10, 20, 30])

            dataset = PouringFrameOrderingDataset(
                root,
                prompt_bank=PromptBank.default(),
                seed=7,
                min_ordering_frame_gap=10,
            )
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertEqual(len(ordering_samples), 1)
            sample = ordering_samples[0]
            self.assertEqual(sample["chronological_frame_indices"], [0, 10, 20, 30])
            self.assertEqual(sample["chronological_frame_gaps"], [10, 10, 10])
            self.assertEqual(sample["min_ordering_frame_gap"], 10)

    def test_default_min_ordering_frame_gap_skips_adjacent_saved_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 2, 4, 6, 8, 10, 12])

            dataset = PouringFrameOrderingDataset(root, prompt_bank=PromptBank.default(), seed=7)
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertTrue(ordering_samples)
            for sample in ordering_samples:
                self.assertGreaterEqual(min(sample["chronological_frame_gaps"]), 4)
                self.assertEqual(sample["min_ordering_frame_gap"], 4)

    def test_min_ordering_frame_gap_can_return_no_ordering_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_pouring_root(Path(tmpdir), frame_indices=[0, 2, 4, 6])

            dataset = PouringFrameOrderingDataset(
                root,
                prompt_bank=PromptBank.default(),
                seed=7,
                min_ordering_frame_gap=10,
            )
            ordering_samples = [sample for sample in dataset if sample["primary_task"] == "pouring_frame_ordering"]

            self.assertEqual(ordering_samples, [])

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
            self.assertTrue(_colors_close(captured["primary_pixel"], _color_for_frame(first_display_index)))
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


def _colors_close(a: tuple[int, int, int], b: tuple[int, int, int], tolerance: int = 3) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


if __name__ == "__main__":
    unittest.main()
