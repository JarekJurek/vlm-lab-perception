from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from datasets.dtld import DTLDDataset


def _write_scene(root: Path, scene_id: str, frame_count: int, object_count: int = 1) -> None:
    scene_dir = root / scene_id
    scene_dir.mkdir()
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir()
    liquid = {
        str(frame_id): [
            {"obj_id": 15 + min(obj_idx, 2), "liquid_h": float(frame_id + obj_idx)}
            for obj_idx in range(object_count)
        ]
        for frame_id in range(frame_count)
    }
    for frame_id in range(frame_count):
        (rgb_dir / f"{frame_id:06d}.png").touch()
    (scene_dir / "scene_gt_liquid.json").write_text(
        json.dumps(liquid),
        encoding="utf-8",
    )


def _write_scene_with_info(
    root: Path,
    scene_id: str,
    liquid: dict[str, list[dict[str, object]]],
    info: dict[str, list[dict[str, object]]],
) -> None:
    scene_dir = root / scene_id
    scene_dir.mkdir()
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir()
    for frame_key in liquid:
        (rgb_dir / f"{int(frame_key):06d}.png").touch()
    (scene_dir / "scene_gt_liquid.json").write_text(json.dumps(liquid), encoding="utf-8")
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(info), encoding="utf-8")


def _task_names(sample: dict[str, object]) -> set[str]:
    return {ann.metadata["task"] for ann in sample["annotations"]}


class DTLDBalancingTest(unittest.TestCase):
    def test_shuffle_interleaves_scene_folders_before_repeating(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene(root, "000000", 8)
            _write_scene(root, "000001", 1)
            _write_scene(root, "000002", 2)
            _write_scene(root, "000003", 3)

            dataset = DTLDDataset(root, shuffle=True, seed=7)

            first_round_scene_ids = [entry[0] for entry in dataset._entries[:4]]
            self.assertEqual(
                set(first_round_scene_ids),
                {"000000", "000001", "000002", "000003"},
            )

    def test_balanced_indices_cover_scene_folders_under_max_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene(root, "000000", 20)
            _write_scene(root, "000001", 1)
            _write_scene(root, "000002", 1)
            _write_scene(root, "000003", 1)

            dataset = DTLDDataset(root, shuffle=False)
            indices = dataset.balanced_indices_by_scene(
                max_samples=4,
                shuffle=True,
                seed=11,
            )

            selected_scene_ids = {dataset._entries[i][0] for i in indices}
            self.assertEqual(selected_scene_ids, {"000000", "000001", "000002", "000003"})

    def test_shuffle_max_samples_can_cover_all_fourteen_scene_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for scene_idx in range(14):
                frame_count = 1 + (scene_idx % 3)
                _write_scene(root, f"{scene_idx:06d}", frame_count, object_count=3)

            dataset = DTLDDataset(root, shuffle=True, seed=13)
            indices = dataset.balanced_indices_by_scene(
                max_samples=14,
                shuffle=True,
                seed=13,
            )

            selected_entries = [dataset._entries[i] for i in indices]
            selected_scene_ids = {scene_id for scene_id, _frame_id, _obj_idx in selected_entries}
            selected_frame_ids = {frame_id for _scene_id, frame_id, _obj_idx in selected_entries}
            selected_obj_indices = {obj_idx for _scene_id, _frame_id, obj_idx in selected_entries}
            selected_samples = [dataset[i] for i in indices]

            self.assertEqual(len(selected_scene_ids), 14)
            self.assertEqual(selected_frame_ids, {0})
            self.assertEqual(selected_obj_indices, {0})
            for sample in selected_samples:
                tasks = {ann.metadata["task"] for ann in sample["annotations"]}
                self.assertIn("container_count", tasks)

    def test_first_selected_scene_entry_uses_lowest_frame_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "000000"
            scene_dir.mkdir()
            rgb_dir = scene_dir / "rgb"
            rgb_dir.mkdir()
            for frame_id in range(3):
                (rgb_dir / f"{frame_id:06d}.png").touch()
            (scene_dir / "scene_gt_liquid.json").write_text(
                json.dumps(
                    {
                        "2": [{"obj_id": 15, "liquid_h": 2.0}],
                        "0": [{"obj_id": 15, "liquid_h": 0.0}],
                        "1": [{"obj_id": 15, "liquid_h": 1.0}],
                    }
                ),
                encoding="utf-8",
            )

            dataset = DTLDDataset(root, shuffle=True, seed=5)

            self.assertEqual(dataset._entries[0], ("000000", 0, 0))

    def test_non_overlapping_low_visibility_target_keeps_object_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene_with_info(
                root,
                "000000",
                {
                    "0": [
                        {"obj_id": 15, "liquid_h": 5.0},
                        {"obj_id": 16, "liquid_h": 7.0},
                    ]
                },
                {
                    "0": [
                        {"bbox_obj": [0, 0, 20, 20], "bbox_visib": [0, 0, 20, 20], "visib_fract": 0.5},
                        {"bbox_obj": [30, 0, 20, 20], "bbox_visib": [30, 0, 20, 20], "visib_fract": 1.0},
                    ]
                },
            )

            dataset = DTLDDataset(root)
            low_visibility_sample = dataset[0]
            visible_sample = dataset[1]

            self.assertFalse(low_visibility_sample["target_occluded"])
            self.assertFalse(low_visibility_sample["container_count_ambiguous"])
            self.assertIn("container_count", _task_names(low_visibility_sample))
            self.assertIn("container_type", _task_names(low_visibility_sample))
            self.assertIn("container_type_simple", _task_names(low_visibility_sample))
            self.assertIn("fill_height_mm", _task_names(low_visibility_sample))
            self.assertIn("liquid_presence", _task_names(low_visibility_sample))
            self.assertFalse(visible_sample["target_occluded"])
            self.assertIn("container_type", _task_names(visible_sample))
            self.assertIn("container_type_simple", _task_names(visible_sample))
            self.assertIn("fill_height_mm", _task_names(visible_sample))
            self.assertIn("liquid_presence", _task_names(visible_sample))

    def test_overlapping_dtld_boxes_suppress_object_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene_with_info(
                root,
                "000000",
                {
                    "0": [
                        {"obj_id": 15, "liquid_h": 5.0},
                        {"obj_id": 16, "liquid_h": 7.0},
                    ]
                },
                {
                    "0": [
                        {"bbox_obj": [0, 0, 20, 20], "bbox_visib": [0, 0, 20, 20], "visib_fract": 1.0},
                        {"bbox_obj": [4, 4, 20, 20], "bbox_visib": [4, 4, 20, 20], "visib_fract": 1.0},
                    ]
                },
            )

            dataset = DTLDDataset(root)

            self.assertTrue(dataset[0]["target_occluded"])
            self.assertTrue(dataset[1]["target_occluded"])
            self.assertTrue(dataset[0]["container_count_ambiguous"])
            self.assertTrue(dataset[1]["container_count_ambiguous"])
            self.assertEqual(_task_names(dataset[0]), set())
            self.assertEqual(_task_names(dataset[1]), set())

    def test_occluded_target_filter_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene_with_info(
                root,
                "000000",
                {
                    "0": [
                        {"obj_id": 15, "liquid_h": 5.0},
                        {"obj_id": 16, "liquid_h": 7.0},
                    ]
                },
                {
                    "0": [
                        {"bbox_obj": [0, 0, 20, 20], "bbox_visib": [0, 0, 20, 20], "visib_fract": 1.0},
                        {"bbox_obj": [4, 4, 20, 20], "bbox_visib": [4, 4, 20, 20], "visib_fract": 1.0},
                    ]
                },
            )

            dataset = DTLDDataset(root, skip_occluded_target_questions=False)

            self.assertTrue(dataset[0]["target_occluded"])
            self.assertIn("container_type", _task_names(dataset[0]))
            self.assertIn("container_type_simple", _task_names(dataset[0]))

    def test_container_type_simple_uses_no_volume_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_scene(root, "000000", 1)

            dataset = DTLDDataset(root)
            annotations = {
                ann.metadata["task"]: ann
                for ann in dataset[0]["annotations"]
            }

            self.assertEqual(annotations["container_type"].answer, "T25 Flask")
            self.assertEqual(annotations["container_type_simple"].answer, "Flask")


if __name__ == "__main__":
    unittest.main()
