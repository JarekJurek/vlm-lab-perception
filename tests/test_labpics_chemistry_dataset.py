from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Tuple


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from datasets.labpics_chemistry import LabPicsChemistryDataset
from tlg import PromptBank


def _write_mask(path: Path, size: Tuple[int, int] = (10, 10), box: Tuple[int, int, int, int] = (2, 2, 4, 4)) -> None:
    img = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            img.putpixel((x, y), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_sample(
    split_root: Path,
    name: str,
    *,
    vessel_label: str = "Beaker",
    material_labels: Tuple[str, ...] = ("Liquid",),
    material_box: Tuple[int, int, int, int] = (2, 2, 4, 4),
    vessel_box: Tuple[int, int, int, int] = (1, 1, 8, 8),
    material_id: int = 1,
    vessel_id: int = 1,
) -> Path:
    sample_dir = split_root / name
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(sample_dir / "Image.jpg")
    _write_mask(sample_dir / "Ignore.png", box=(0, 0, 0, 0))
    _write_mask(sample_dir / "Vessels" / f"{vessel_id}.png", box=vessel_box)
    _write_mask(sample_dir / "MaterialsAndParts" / f"{material_id}.png", box=material_box)
    for label in material_labels:
        _write_mask(sample_dir / "SemanticMaps" / "FullImage" / f"{label}.png", box=material_box)
    _write_mask(sample_dir / "SemanticMaps" / "FullImage" / "Vessel.png", box=vessel_box)
    (sample_dir / "Source.txt").write_text(f"Source:{name}\n", encoding="utf-8")

    data: Dict[str, object] = {
        "Vessels": {
            str(vessel_id): {
                "Indx": vessel_id,
                "MaskFilePath": f"/Vessels/{vessel_id}.png",
                "All_ClassNames": ["Vessel", vessel_label, "Transparent"],
                "VesselType_ClassNames": ["Vessel", vessel_label],
                "Proprties_ClassNames": ["Transparent"],
                "VesselContentAll_Indx": [material_id],
                "VesselContentMaterial_Indx": [material_id],
                "VesselContentParts_Indx": [],
            }
        },
        "MaterialsAndParts": {
            str(material_id): {
                "Indx": material_id,
                "MaskFilePath": f"/MaterialsAndParts/{material_id}.png",
                "IsPart": False,
                "MaterialType_ClassNames": list(material_labels) + ["Filled"],
                "All_ClassNames": list(material_labels) + ["Filled"],
                "InsideOfVessel_Indx": [vessel_id],
                "ContainMaterials_Indx": [],
                "InsideOfMaterials_Indx": [],
            }
        },
    }
    (sample_dir / "Data.json").write_text(json.dumps(data), encoding="utf-8")
    return sample_dir


class LabPicsChemistryDatasetTest(unittest.TestCase):
    def test_discovers_train_test_and_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sample(root / "Train", "1001Train")
            _write_sample(root / "Test", "1000Eval")

            self.assertEqual(len(LabPicsChemistryDataset(root, split="train")), 1)
            self.assertEqual(len(LabPicsChemistryDataset(root, split="test")), 1)
            self.assertEqual(len(LabPicsChemistryDataset(root, split="all")), 2)
            self.assertEqual(
                LabPicsChemistryDataset(root, ids=["1000Eval"], split="all")[0]["id"],
                "labpics_chemistry/test/1000Eval",
            )

    def test_builds_core_annotations_from_labpics_chemistry_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sample(
                root / "Train",
                "1001Train",
                vessel_label="RoundFlask",
                material_labels=("Liquid", "Suspension", "Solid"),
            )

            sample = LabPicsChemistryDataset(root, split="train", prompt_bank=PromptBank.default())[0]
            annotations = {ann.metadata["task"]: ann for ann in sample["annotations"]}

            self.assertEqual(sample["id"], "labpics_chemistry/train/1001Train")
            self.assertEqual(sample["split"], "train")
            self.assertIn("Source:1001Train", sample["source_text"])
            self.assertEqual(annotations["container_count"].answer, 1)
            self.assertEqual(annotations["container_type"].answer, "RoundFlask")
            self.assertEqual(annotations["container_type"].metadata["bbox_xyxy"], [1, 1, 8, 8])
            self.assertEqual(annotations["material_types_present"].answer, "LIQUID, SOLID")
            self.assertEqual(annotations["multiphase_material_presence"].answer, "YES")
            self.assertEqual(
                annotations["liquid_grounding"].answer,
                [{"bbox_2d": [222, 222, 444, 444], "label": "Liquid"}],
            )

    def test_highlighted_region_type_can_emit_gel_and_granular(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sample(root / "Train", "gelTrain", material_labels=("Gel",), material_box=(0, 0, 9, 9), vessel_box=(0, 0, 9, 9))
            _write_sample(root / "Train", "granularTrain", material_labels=("Granular",), material_box=(0, 0, 9, 9), vessel_box=(0, 0, 9, 9))
            for sample_id in ("gelTrain", "granularTrain"):
                data_path = root / "Train" / sample_id / "Data.json"
                data = json.loads(data_path.read_text(encoding="utf-8"))
                data["Vessels"] = {}
                data_path.write_text(json.dumps(data), encoding="utf-8")

            labels = {
                LabPicsChemistryDataset(root, ids=[sample_id], split="train")[0]["annotations"][0].answer
                for sample_id in ("gelTrain", "granularTrain")
            }

            self.assertIn("GEL", labels)
            self.assertIn("GRANULAR", labels)

    def test_prompts_are_registered_for_labpics_chemistry(self) -> None:
        bank = PromptBank.default()

        for task in (
            "highlighted_region_type",
            "material_types_present",
            "multiphase_material_presence",
            "liquid_grounding",
            "container_count",
            "container_type",
        ):
            self.assertTrue(bank.supports("labpics_chemistry", task))

        container_prompt = bank.get("labpics_chemistry", "container_type")
        self.assertIn("RoundFlask", container_prompt["choices"])
        self.assertNotIn("Tube_50mL", container_prompt["choices"])


if __name__ == "__main__":
    unittest.main()
