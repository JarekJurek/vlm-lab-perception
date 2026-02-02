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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from labliquidvision import LabLiquidVisionDataset


# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-VL-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data") / "LabLiquidVision" / "processed"
MAX_NEW_TOKENS = 256


# ============================================================================
# Task Prompts
# ============================================================================
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


# ============================================================================
# Model Setup
# ============================================================================
def load_model():
    """Load Qwen3-VL model and processor."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    return model, processor


# ============================================================================
# Inference
# ============================================================================
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
    # Build content list
    content = [{"type": "image", "image": image}]
    
    if mask is not None:
        content.append({"type": "image", "image": mask})
        prompt = f"The second image shows the container mask.\n\n{prompt}"
    
    content.append({"type": "text", "text": prompt})
    
    # Build messages
    messages = [{"role": "user", "content": content}]
    
    # Prepare inputs
    inputs = processor(messages, return_tensors="pt").to(model.device)
    
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
    use_mask: bool = True
) -> Dict[str, str]:
    """Test all tasks on a single sample.
    
    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample: Dataset sample containing 'rgb' and optionally 'mask'.
        use_mask: Whether to include vessel mask in prompts.
    
    Returns:
        Dictionary mapping task names to model responses.
    """
    # Convert numpy arrays to PIL Images
    rgb_array = sample["rgb"]
    if rgb_array.shape[0] in (1, 3, 4):  # CHW format
        rgb_array = rgb_array.transpose(1, 2, 0)
    
    rgb_image = Image.fromarray(rgb_array.astype('uint8'))
    
    mask_image = None
    if use_mask and "mask" in sample:
        mask_array = sample["mask"]
        if mask_array.ndim == 3:
            if mask_array.shape[0] == 1:  # CHW format
                mask_array = mask_array.transpose(1, 2, 0).squeeze(-1)
            elif mask_array.shape[-1] == 1:
                mask_array = mask_array.squeeze(-1)
        # Convert to RGB for visualization
        mask_rgb = (mask_array * 255).astype('uint8')
        mask_image = Image.fromarray(mask_rgb).convert("RGB")
    
    results = {}
    
    for task_name, prompt in TASK_PROMPTS.items():
        print(f"  Running task: {task_name}...")
        response = run_task(model, processor, rgb_image, prompt, mask_image)
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
    # Convert to PIL Images
    rgb1 = sample1["rgb"]
    if rgb1.shape[0] in (1, 3, 4):
        rgb1 = rgb1.transpose(1, 2, 0)
    image1 = Image.fromarray(rgb1.astype('uint8'))
    
    rgb2 = sample2["rgb"]
    if rgb2.shape[0] in (1, 3, 4):
        rgb2 = rgb2.transpose(1, 2, 0)
    image2 = Image.fromarray(rgb2.astype('uint8'))
    
    prompt = (
        "You see two laboratory containers. "
        "Which one contains more liquid? "
        "Answer with 'FIRST', 'SECOND', or 'EQUAL'."
    )
    
    # Build messages with two images
    content = [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "text", "text": prompt}
    ]
    
    messages = [{"role": "user", "content": content}]
    
    inputs = processor(messages, return_tensors="pt").to(model.device)
    
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
    dataset = LabLiquidVisionDataset(DATA_ROOT)
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

    results = test_sample(model, processor, sample, use_mask=True)

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
