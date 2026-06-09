from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.pouring import PouringFrameOrderingDataset
from tlg.prompts import PromptBank


class PouringLiquidColorTest(unittest.TestCase):
    def test_liquid_color_transparent_only_for_visible_liquid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_pouring_root(Path(tmpdir), frame_indices=[0, 10])
            video_dir = root / "pouring_testset_videos" / "video_0"
            _write_labelme_json(video_dir / "rgb_0.json", has_liquid=True)
            _write_labelme_json(video_dir / "rgb_10.json", has_liquid=False)

            dataset = PouringFrameOrderingDataset(root, prompt_bank=PromptBank.default(), seed=7)
            single_frame_samples = {
                sample["frame_a_index"]: sample
                for sample in dataset
                if sample["primary_task"] == "liquid_grounding"
            }

            visible_tasks = _annotation_tasks(single_frame_samples[0])
            empty_tasks = _annotation_tasks(single_frame_samples[10])

            self.assertEqual(visible_tasks["liquid_color"], "transparent")
            self.assertNotIn("liquid_color", empty_tasks)


def _make_pouring_root(base: Path, frame_indices: list[int]) -> Path:
    video_dir = base / "pouring_testset_videos" / "video_0"
    video_dir.mkdir(parents=True)
    for frame_index in frame_indices:
        Image.new("RGB", (8, 8), color=(frame_index, frame_index, frame_index)).save(
            video_dir / f"rgb_{frame_index}.jpg"
        )
    return base


def _write_labelme_json(path: Path, *, has_liquid: bool) -> None:
    shapes = []
    if has_liquid:
        shapes.append(
            {
                "label": "liquid_mask",
                "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                "shape_type": "polygon",
            }
        )
    path.write_text(
        json.dumps(
            {
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )


def _annotation_tasks(sample: dict) -> dict[str, object]:
    return {ann.metadata["task"]: ann.answer for ann in sample["annotations"]}


if __name__ == "__main__":
    unittest.main()
