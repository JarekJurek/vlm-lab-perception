"""Model loading and task execution helpers for Qwen-VL evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tlg.attention_viz import write_attention_overlay
from tlg.annotations import TLGAnnotation


# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 256


def load_model(*, viz_attn: bool = False) -> Tuple[Any, Any]:
    """Load Qwen3-VL model and processor.

    Args:
        viz_attn: Whether to enable attention outputs for visualization.

    Returns:
        Tuple of (model, processor).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model_kwargs: Dict[str, object] = {
        "dtype": torch.float16,
        "device_map": "auto",
    }
    if viz_attn:
        # Some attention backends do not materialize attention weights.
        # When visualizing, prefer an implementation that can return weights.
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **model_kwargs).eval()

    print("Model loaded successfully.")

    return model, processor


def run_task(
    model,
    processor,
    image,
    prompt: str,
    mask=None,
    extra_images: Optional[Sequence[Any]] = None,
    *,
    dataset_name: str = "",
    sample_id: str = "",
    task_name: str = "",
    viz_attn: bool = False,
    viz_out_dir: Optional[Path] = None,
    viz_attn_mode: str = "all",
) -> str:
    """Run a single VLM task.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        image: RGB image of the container.
        prompt: Task prompt text.
        mask: Optional vessel mask image.
        extra_images: Optional additional RGB images to pass after ``image``.
        dataset_name: Dataset name for visualization output.
        sample_id: Sample identifier for visualization output.
        task_name: Task name for visualization output.
        viz_attn: Whether to emit attention visualizations.
        viz_out_dir: Output directory for attention visualizations.
        viz_attn_mode: Which generated tokens to use for attention visualization.

    Returns:
        Model's text response.
    """
    import torch

    # Build content list (image placeholders) and images list.
    content = [{"type": "image"}]
    images = [image]

    if extra_images:
        for extra_image in extra_images:
            content.append({"type": "image"})
            images.append(extra_image)

    if mask is not None:
        content.append({"type": "image"})
        images.append(mask)
        if extra_images:
            prompt = f"The final image shows the container mask.\n\n{prompt}"
        else:
            prompt = f"The second image shows the container mask.\n\n{prompt}"

    content.append({"type": "text", "text": prompt})

    # Build messages.
    messages = [{"role": "user", "content": content}]

    # Prepare inputs using chat template + images.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    # Generate.
    gen_out = None
    with torch.no_grad():
        if viz_attn:
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            output_ids = gen_out.sequences
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

    # Decode.
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract just the answer (remove the prompt echo if present).
    if prompt in response:
        response = response.split(prompt)[-1].strip()

    # Attention visualization (opt-in).
    if viz_attn and viz_out_dir is not None:
        write_attention_overlay(
            model=model,
            processor=processor,
            inputs=inputs,
            gen_out=gen_out,
            output_ids=output_ids,
            image=image,
            images=images,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_out_dir=viz_out_dir,
            viz_attn_mode=viz_attn_mode,
        )

    return response


def test_sample(
    model,
    processor,
    sample: Dict[str, Any],
    annotations: List[TLGAnnotation],
    *,
    dataset_name: str,
    sample_id: str,
    viz_attn: bool,
    viz_out_dir: Optional[Path],
    viz_attn_mode: str = "all",
) -> Dict[str, str]:
    """Test all tasks on a single sample.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample: Dataset sample containing an RGB image.
        annotations: Standardized task annotations.
        dataset_name: Dataset name for logging and visualization output.
        sample_id: Sample identifier for logging and visualization output.
        viz_attn: Whether to emit attention visualizations.
        viz_out_dir: Output directory for attention visualizations.
        viz_attn_mode: Which generated tokens to use for attention visualization.

    Returns:
        Dictionary mapping task names to model responses.
    """
    from PIL import Image, UnidentifiedImageError

    def _open_rgb_image(path: object, *, label: str):
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise OSError(f"Failed to open {label}: {path} ({exc})") from exc

    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        rgb_image = _open_rgb_image(rgb_path, label="RGB image")
    else:
        rgb_array = sample.get("rgb")
        if rgb_array is None:
            raise FileNotFoundError("Input_RGBImage.png or Input_RGBImage.npy not found for sample.")
        if rgb_array.shape[0] in (1, 3, 4):
            rgb_array = rgb_array.transpose(1, 2, 0)
        rgb_image = Image.fromarray(rgb_array.astype("uint8"))

    comparison_image = None
    comparison_path = sample.get("comparison_rgb_image")
    if comparison_path is not None:
        comparison_image = _open_rgb_image(comparison_path, label="comparison RGB image")

    mask_image = None
    mask_path = sample.get("mask_image")
    if mask_path is not None:
        try:
            mask_image = _open_rgb_image(mask_path, label="mask image")
        except OSError:
            mask_image = None

    results: Dict[str, str] = {}
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  Running task: {task_name}...")
        prompt = ann.question
        extra_images = None
        if comparison_image is not None:
            prompt = (
                "The first image is the earlier frame. "
                "The second image is the later frame.\n\n"
                f"{prompt}"
            )
            extra_images = [comparison_image]

        response = run_task(
            model,
            processor,
            rgb_image,
            prompt,
            mask=mask_image,
            extra_images=extra_images,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_attn=viz_attn,
            viz_out_dir=viz_out_dir,
            viz_attn_mode=viz_attn_mode,
        )
        results[task_name] = response

    return results
