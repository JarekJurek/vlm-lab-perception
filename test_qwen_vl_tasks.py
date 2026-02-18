"""Test Qwen-VL on liquid container vision tasks.

This script evaluates Qwen-VL on various visual understanding tasks for
laboratory liquid containers without requiring depth information.

Tasks tested:
  - Binary liquid presence detection
  - Ordinal fill level classification (empty, low, mid, high, full)
  - Bubble presence detection (none, some, lots)
  - Relative fill level comparison
  - Fill level estimation (percentage)
"""

from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.labliquidvision import LabLiquidVisionDataset


MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data") / "LabLiquidVision"
MAX_NEW_TOKENS = 256


TASK_PROMPTS = {
    "binary_liquid": (
        "Look at this container. Does it contain any liquid? "
        "Answer with only 'YES' or 'NO'."
    ),
    "ordinal_level": (
        "What is the fill level of liquid in this container? "
        "Choose exactly one: EMPTY, LOW, MID, HIGH, or FULL."
    ),
    "bubble_presence": (
        "Are there bubbles visible in the liquid? "
        "Choose exactly one: NONE, SOME, or LOTS."
    ),
    "fill_percentage": (
        "Estimate the fill level as a percentage of the container's capacity. "
        "Answer with only a number between 0 and 100."
    ),
    "container_type": (
        "What type of laboratory container is this? "
        "Answer briefly (e.g., beaker, flask, graduated cylinder)."
    ),
    "liquid_color": (
        "What color is the liquid in this container? "
        "Answer with a single color name."
    ),
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
    sample: Dict[str, Any]
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

    results = {}
    
    for task_name, prompt in TASK_PROMPTS.items():
        print(f"  Running task: {task_name}...")
        response = run_task(model, processor, rgb_image, prompt)
        results[task_name] = response
    
    return results


def main():
    """Run all VLM tests."""
    model, processor = load_model()

    print(f"\nLoading dataset from: {DATA_ROOT}")
    dataset = LabLiquidVisionDataset(DATA_ROOT, rgb_only=True)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    for idx, sample in enumerate(dataset, start=1):
        print("\n" + "-" * 60)
        print(f"Sample {idx}/{len(dataset)}")
        print(f"Sample ID: {sample.get('id', 'unknown')}")

        print("\nGround Truth:")
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
        print(f"  Fill percentage: {vol_liquid}/{vol_vessel} ({percentage})")

        results = test_sample(model, processor, sample)

        print("\nPredictions:")
        for task_name, response in results.items():
            print(f"  {task_name}: {response}")


if __name__ == "__main__":
    main()
