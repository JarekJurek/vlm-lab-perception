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
DATA_ROOT = Path("data") / "LabLiquidVision" / "processed"
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
    if rgb_path is None:
        raise FileNotFoundError("Input_RGBImage.png not found for sample.")

    rgb_image = Image.open(rgb_path).convert("RGB")

    results = {}
    
    for task_name, prompt in TASK_PROMPTS.items():
        print(f"  Running task: {task_name}...")
        response = run_task(model, processor, rgb_image, prompt)
        results[task_name] = response
    
    return results


def compare_containers(
    model,
    processor,
    sample1: Dict[str, Any],
    sample2: Dict[str, Any]
) -> str:
    """Compare fill levels of two containers.
    
    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample1: First container sample.
        sample2: Second container sample.
    
    Returns:
        Model's comparison response.
    """
    rgb_path1 = sample1.get("rgb_image")
    rgb_path2 = sample2.get("rgb_image")
    if rgb_path1 is None or rgb_path2 is None:
        raise FileNotFoundError("Input_RGBImage.png not found for comparison samples.")

    image1 = Image.open(rgb_path1).convert("RGB")
    image2 = Image.open(rgb_path2).convert("RGB")
    
    prompt = (
        "You see two laboratory containers. "
        "Which one contains more liquid? "
        "Answer with 'FIRST', 'SECOND', or 'EQUAL'."
    )
    
    # Build messages with two images (placeholders) and prepare inputs
    content = [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image1, image2],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )
    
    response = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]
    
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response


def main():
    """Run all VLM tests."""
    # Load model
    model, processor = load_model()

    # Load dataset
    print(f"\nLoading dataset from: {DATA_ROOT}")
    dataset = LabLiquidVisionDataset(DATA_ROOT, rgb_only=True)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    # Test on first sample
    print("\n" + "=" * 60)
    print("SINGLE SAMPLE TASKS")
    print("=" * 60)

    sample = dataset[0]
    print(f"\nSample ID: {sample.get('id', 'unknown')}")
    print(f"Ground truth volume: {sample.get('vol_liquid', 'N/A')}")
    print(f"Vessel type: {sample.get('vessel', 'N/A')}")
    print(f"Liquid color: {sample.get('color_liquid', 'N/A')}")

    results = test_sample(model, processor, sample)

    print("\nResults:")
    for task_name, response in results.items():
        print(f"  {task_name}: {response}")

    # Test comparison task if we have multiple samples
    if len(dataset) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON TASK")
        print("=" * 60)

        sample1 = dataset[0]
        sample2 = dataset[1]

        print(f"\nSample 1: {sample1.get('vol_liquid', 'N/A')}")
        print(f"Sample 2: {sample2.get('vol_liquid', 'N/A')}")

        comparison = compare_containers(model, processor, sample1, sample2)
        print(f"\nComparison result: {comparison}")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
