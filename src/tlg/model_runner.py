"""Model loading and task execution helpers for Qwen-VL evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tlg.attention_viz import write_attention_overlay
from tlg.annotations import TLGAnnotation
from tlg.debug_viz import debug_dtld_box_state, save_dtld_box_debug_images
from tlg.prompts import PromptBank
from tlg.target_overlay import is_valid_bbox_xywh, make_target_box_overlay, should_mark_target_box


# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
DEFAULT_QWEN_MAX_PIXELS = 512 * 28 * 28


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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

    processor_kwargs: Dict[str, object] = {}
    qwen_max_pixels = _env_int("QWEN_MAX_PIXELS", DEFAULT_QWEN_MAX_PIXELS)
    qwen_min_pixels = _env_int("QWEN_MIN_PIXELS")
    if qwen_max_pixels is not None:
        processor_kwargs["max_pixels"] = qwen_max_pixels
    if qwen_min_pixels is not None:
        processor_kwargs["min_pixels"] = qwen_min_pixels

    processor = AutoProcessor.from_pretrained(MODEL_NAME, **processor_kwargs)
    if processor_kwargs:
        print(f"Qwen processor image limits: {processor_kwargs}")

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
    extra_images: Optional[Sequence[Any]] = None,
    *,
    dataset_name: str = "",
    sample_id: str = "",
    task_name: str = "",
    viz_attn: bool = False,
    viz_out_dir: Optional[Path] = None,
    viz_attn_mode: str = "all",
    viz_image_id: str = "",
    viz_gt: object = None,
    viz_save_input_image: bool = False,
) -> str:
    """Run a single VLM task.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        image: RGB image of the container.
        prompt: Task prompt text.
        extra_images: Optional additional RGB images to pass after ``image``.
        dataset_name: Dataset name for visualization output.
        sample_id: Sample identifier for visualization output.
        task_name: Task name for visualization output.
        viz_attn: Whether to emit attention visualizations.
        viz_out_dir: Output directory for attention visualizations.
        viz_attn_mode: Which generated tokens to use for attention visualization.
        viz_image_id: Image/sample identifier shown in visualization titles.
        viz_gt: Ground-truth answer shown in visualization titles.
        viz_save_input_image: Whether to save the exact input image separately.

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

    content.append({"type": "text", "text": prompt})

    # Build messages.
    messages = [{"role": "user", "content": content}]

    # Prepare inputs using chat template + images.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    input_ids = inputs.get("input_ids")
    image_grid = inputs.get("image_grid_thw")
    if input_ids is not None:
        msg = f"    Qwen input tokens: {int(input_ids.shape[-1])}"
        if image_grid is not None:
            grid_values = image_grid.detach().cpu().tolist()
            visual_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid_values)
            msg += f", image_grid_thw={grid_values}, visual_tokens={visual_tokens}"
        print(msg)

    # Generate.
    gen_out = None
    with torch.no_grad():
        if viz_attn:
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
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
            viz_image_id=viz_image_id,
            viz_gt=viz_gt,
            viz_prediction=response,
            viz_save_input_image=viz_save_input_image,
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
    debug_dtld_boxes: bool = False,
    debug_image_out_dir: Optional[Path] = None,
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

    ordered_images = None
    ordered_paths = sample.get("ordered_rgb_images")
    if isinstance(ordered_paths, (list, tuple)) and ordered_paths:
        ordered_images = [
            _open_rgb_image(path, label=f"ordered RGB image {i + 1}")
            for i, path in enumerate(ordered_paths)
        ]

    target_box_image = None
    bbox = sample.get("bbox_visib") or sample.get("bbox_obj")
    if is_valid_bbox_xywh(bbox):
        target_box_image = make_target_box_overlay(rgb_image, rgb_image, bbox=bbox)

    results: Dict[str, str] = {}
    target_region_prompt = PromptBank.default().target_region_bbox_prompt()
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  Running task: {task_name}...")
        region_box_image = _make_annotation_box_image(rgb_image, ann)
        use_target_box = (
            target_box_image is not None
            and should_mark_target_box(dataset_name, task_name)
        )
        use_region_box = region_box_image is not None
        task_image = region_box_image if use_region_box else target_box_image if use_target_box else rgb_image
        save_task_input_image = bool(use_region_box or use_target_box)
        prompt = ann.question
        extra_images = None
        if debug_dtld_boxes and dataset_name == "dtld":
            debug_dtld_box_state(
                prefix="Qwen",
                sample=sample,
                task_name=task_name,
                use_target_box=use_target_box,
                use_region_box=use_region_box,
                target_box_image=target_box_image,
            )
            if debug_image_out_dir is not None:
                save_dtld_box_debug_images(
                    out_dir=debug_image_out_dir,
                    model_key="qwen",
                    sample_id=sample_id,
                    task_name=task_name,
                    rgb_image=rgb_image,
                    target_box_image=target_box_image,
                    task_image=task_image,
                )
        if use_region_box:
            prompt = target_region_prompt + prompt
        elif use_target_box:
            prompt = target_region_prompt + prompt
        if comparison_image is not None:
            prompt = (
                "The first image is the first frame. "
                "The second image is the second frame.\n\n"
                f"{prompt}"
            )
            extra_images = [comparison_image]
        if ordered_images is not None and len(ordered_images) >= 2:
            labels = sample.get("ordered_image_labels")
            if not isinstance(labels, (list, tuple)) or len(labels) != len(ordered_images):
                labels = [chr(ord("A") + i) for i in range(len(ordered_images))]
            prompt = (
                "The images are provided in this label order: "
                f"{', '.join(str(label) for label in labels)}. "
                "Use these labels in your answer.\n\n"
                f"{prompt}"
            )
            task_image = ordered_images[0]
            extra_images = ordered_images[1:]

        response = run_task(
            model,
            processor,
            task_image,
            prompt,
            extra_images=extra_images,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_attn=viz_attn,
            viz_out_dir=viz_out_dir,
            viz_attn_mode=viz_attn_mode,
            viz_image_id=ann.image_id,
            viz_gt=ann.answer,
            viz_save_input_image=save_task_input_image,
        )
        results[task_name] = response

    return results


def _make_annotation_box_image(image: Any, ann: TLGAnnotation) -> Optional[Any]:
    bbox = ann.metadata.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return make_target_box_overlay(image, image, bbox=[x0, y0, x1 - x0, y1 - y0])
