"""Test Qwen-VL on liquid container vision tasks.

This script evaluates Qwen-VL on visual understanding tasks for multiple
liquid-container datasets:
    - LabLiquidVision
    - Glasses in the Wild
    - DTLD
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.datasets.dtld import DTLDDataset
from src.datasets.glassesinthewild import GlassesInTheWildDataset
from src.datasets.labliquidvision import LabLiquidVisionDataset

from src.tlg import TLGAnnotation


MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
}

DTLD_OBJECT_NAMES = {
    15: "T25 Flask",
    16: "T75 Flask",
    17: "T175 Flask",
    19: "G-rex",
}


def load_model():
    """Load Qwen3-VL model and processor."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    print("Model loaded successfully.")
    
    return model, processor


def run_task(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    mask: Image.Image = None
) -> str:
    """Run a single VLM task.
    
    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        image: RGB image of the container.
        prompt: Task prompt text.
        mask: Optional vessel mask image.
    
    Returns:
        Model's text response.
    """
    # Build content list (image placeholders) and images list
    content = [{"type": "image"}]
    images = [image]

    if mask is not None:
        content.append({"type": "image"})
        images.append(mask)
        prompt = f"The second image shows the container mask.\n\n{prompt}"

    content.append({"type": "text", "text": prompt})

    # Build messages
    messages = [{"role": "user", "content": content}]

    # Prepare inputs using chat template + images
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
    
    # Decode
    response = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]
    
    # Extract just the answer (remove the prompt echo if present)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response


def test_sample(
    model,
    processor,
    sample: Dict[str, Any],
    annotations: List[TLGAnnotation],
) -> Dict[str, str]:
    """Test all tasks on a single sample.
    
    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample: Dataset sample containing 'rgb_image'.
    
    Returns:
        Dictionary mapping task names to model responses.
    """
    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        rgb_image = Image.open(rgb_path).convert("RGB")
    else:
        rgb_array = sample.get("rgb")
        if rgb_array is None:
            raise FileNotFoundError(
                "Input_RGBImage.png or Input_RGBImage.npy not found for sample."
            )
        if rgb_array.shape[0] in (1, 3, 4):
            rgb_array = rgb_array.transpose(1, 2, 0)
        rgb_image = Image.fromarray(rgb_array.astype("uint8"))

    mask_image = None
    mask_path = sample.get("mask_image")
    if mask_path is not None:
        mask_image = Image.open(mask_path).convert("RGB")

    results: Dict[str, str] = {}

    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  Running task: {task_name}...")
        response = run_task(model, processor, rgb_image, ann.question, mask=mask_image)
        results[task_name] = response
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-VL tasks on liquid datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["labliquidvision", "glasses", "dtld"],
        default=["labliquidvision"],
        help="One or more datasets to evaluate.",
    )
    return parser.parse_args()


def build_dataset(dataset_name: str, data_root: Path):
    if dataset_name == "labliquidvision":
        return LabLiquidVisionDataset(data_root, rgb_only=True)
    if dataset_name == "glasses":
        return GlassesInTheWildDataset(data_root)
    if dataset_name == "dtld":
        return DTLDDataset(data_root, rgb_only=True)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def print_ground_truth(dataset_name: str, sample: Dict[str, Any]) -> None:
    print("\nGround Truth:")

    annotations = sample.get("annotations")
    if isinstance(annotations, list) and annotations:
        print("  Standardized annotations:")
        for ann in annotations:
            if not isinstance(ann, TLGAnnotation):
                continue
            task_key = str(ann.metadata.get("task", ann.task_type))
            print(f"    - {task_key}: {ann.answer}")

    if dataset_name == "labliquidvision":
        print(f"  Vessel type: {sample.get('vessel', 'N/A')}")
        print(f"  Liquid color: {sample.get('color_liquid', 'N/A')}")
        vol_liquid = sample.get('vol_liquid', 0)
        vol_vessel = sample.get('vol_vessel', 0)
        print(f"  Liquid volume (mL): {vol_liquid}")
        print(f"  Vessel volume (mL): {vol_vessel}")
        if vol_vessel and vol_liquid:
            percentage = f"{(float(vol_liquid) / float(vol_vessel)) * 100:.1f}%"
        else:
            percentage = "N/A"
        print(f"  Fill percentage: {percentage}")
        return

    if dataset_name == "glasses":
        print(f"  Image name: {sample.get('image_name', 'N/A')}")
        print(f"  Num keypoints: {sample.get('num_keypoints', 'N/A')}")
        fill_percentage = sample.get("fill_percentage")
        if fill_percentage is not None:
            print(f"  Fill percentage (from keypoints): {float(fill_percentage):.1f}%")
        else:
            print("  Fill percentage (from keypoints): N/A")
        keypoints = sample.get("keypoints", {})
        fluid_level = keypoints.get("fluid_level") if isinstance(keypoints, dict) else None
        if fluid_level is not None:
            print(f"  Fluid-level keypoint: ({fluid_level['x']:.1f}, {fluid_level['y']:.1f})")
        return

    if dataset_name == "dtld":
        obj_id = int(sample.get("obj_id", -1))
        print(f"  Scene/frame/object: {sample.get('scene_id')}/{sample.get('frame_id'):06d}/{sample.get('obj_idx'):06d}")
        print(f"  Object ID: {obj_id}")
        print(f"  Object type: {DTLD_OBJECT_NAMES.get(obj_id, 'Unknown')}")
        print(f"  Liquid height (mm): {sample.get('liquid_h', 'N/A')}")
        print(f"  Visible fraction: {sample.get('visib_fract', 'N/A')}")
        liquid_label = sample.get("liquid_label")
        if liquid_label is not None:
            print(f"  Liquid label points: {liquid_label}")
        return


def main():
    """Run all VLM tests."""
    args = parse_args()
    dataset_names = args.datasets

    model, processor = load_model()

    for dataset_name in dataset_names:
        data_root = DATA_ROOTS[dataset_name]

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"Loading dataset from: {data_root}")
        dataset = build_dataset(dataset_name, data_root)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            continue

        for idx, sample in enumerate(dataset, start=1):
            print("\n" + "-" * 60)
            print(f"Sample {idx}/{len(dataset)}")
            print(f"Sample ID: {sample.get('id', 'unknown')}")

            print_ground_truth(dataset_name, sample)

            annotations = sample.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                print("  No standardized annotations available for this sample; skipping.")
                continue

            results = test_sample(model, processor, sample, annotations)

            print("\nPredictions:")
            for task_name, response in results.items():
                print(f"  {task_name}: {response}")


if __name__ == "__main__":
    main()
