"""Model loading and task execution helpers for SD-VLM evaluation.

SD-VLM (Spatial Measuring and Understanding with Depth-Encoded VLM) extends
LLaVA-1.5 with a depth-aware encoder.  The model is published at
https://github.com/cpystan/SD-VLM and must be installed separately::

    git clone https://github.com/cpystan/SD-VLM.git
    cd SD-VLM
    pip install -e .

The HuggingFace checkpoint ``cpystan/SD-VLM-7B`` is loaded automatically when
``load_sdvlm_model`` is called.

Depth handling
--------------
SD-VLM can consume depth information in two ways controlled by the flags that
are set on the model object after loading:

* ``model.use_depth = True`` (default) — depth is used.
* ``model.gt_depth = False`` (default) — depth is *estimated* internally from
  the RGB image via Depth Anything V2; ``ori_imgs`` should be a copy of the RGB
  image.
* ``model.gt_depth = True`` — a ground-truth depth image is passed directly via
  ``ori_imgs``; ``ori_imgs`` should contain the depth as a PIL Image.

LabLiquidVision provides ``Input_DepthMap.npy`` for every sample, so GT depth
is usable via ``--sdvlm-gt-depth``.  All other datasets fall back to the
internally estimated depth.

Attention visualization
-----------------------
Pass ``viz_attn=True`` and a ``viz_out_dir`` path to ``run_task_sdvlm`` to
obtain per-task attention heatmap overlays.  The model is loaded with the
standard (eager) attention implementation so that attention weights are
materialised.  The visualization logic lives in
:func:`tlg.attention_viz.write_attention_overlay_sdvlm`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tlg.annotations import TLGAnnotation
from tlg.attention_viz import write_attention_overlay_sdvlm
from tlg.debug_viz import debug_dtld_box_state, save_dtld_box_debug_images
from tlg.prompts import PromptBank
from tlg.target_overlay import is_valid_bbox_xywh, make_target_box_overlay, should_mark_target_box


SDVLM_MODEL_NAME = "cpystan/SD-VLM-7B"
SDVLM_VISION_TOWER = "openai/clip-vit-large-patch14-336"
MAX_NEW_TOKENS = 256

# Channel counts for (C, H, W) shaped arrays (grayscale, RGB, RGBA).
_LEADING_CHANNEL_COUNTS = (1, 3, 4)


def _require_llava() -> None:
    """Raise a helpful error if the SD-VLM / llava package is not installed."""
    try:
        import llava  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'llava' package (from SD-VLM) is required to run SD-VLM.\n"
            "Install it with:\n"
            "    git clone https://github.com/cpystan/SD-VLM.git\n"
            "    cd SD-VLM && pip install -e ."
        ) from exc


def load_sdvlm_model(
    model_path: str = SDVLM_MODEL_NAME,
    vision_tower: str = SDVLM_VISION_TOWER,
    *,
    use_depth: bool = True,
    gt_depth: bool = False,
    viz_attn: bool = False,
) -> Tuple[Any, Any, Any]:
    """Load SD-VLM model, tokenizer, and image processor.

    Args:
        model_path: HuggingFace model ID or local path to SD-VLM checkpoint.
        vision_tower: Vision encoder identifier (CLIP ViT).
        use_depth: Whether depth encoding is active.
        gt_depth: When *True* the caller must supply ground-truth depth images.
            When *False* depth is estimated internally from the RGB image.
        viz_attn: When *True* load the model with ``attn_implementation="eager"``
            so that attention weights are materialised during generation.

    Returns:
        Tuple of (model, tokenizer, image_processor).
    """
    _require_llava()

    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    disable_torch_init()

    print(f"Loading SD-VLM model: {model_path}")
    print(f"  use_depth={use_depth}  gt_depth={gt_depth}  viz_attn={viz_attn}")

    model_name = get_model_name_from_path(model_path)

    load_kwargs: Dict[str, Any] = {}
    if viz_attn:
        load_kwargs["attn_implementation"] = "eager"

    tokenizer, model, image_processor, _ctx_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        mm_vision_tower=vision_tower,
        **load_kwargs,
    )
    
    # SD-VLM model names may not trigger the LLaVA branch that sets
    # image_processor inside load_pretrained_model(), so recover it here.
    if image_processor is None:
        vision_tower_model = model.get_vision_tower()
        if vision_tower_model is None:
            raise RuntimeError("SD-VLM loaded without a vision tower.")
        if not getattr(vision_tower_model, "is_loaded", False):
            vision_tower_model.load_model()
        image_processor = vision_tower_model.image_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model.to(device=device, dtype=dtype)
    _set_sdvlm_runtime_flags(model, use_depth=use_depth, gt_depth=gt_depth, vit_depth=False)

    print("SD-VLM model loaded successfully.")
    return model, tokenizer, image_processor


def _set_sdvlm_runtime_flags(
    model: Any,
    *,
    use_depth: Optional[bool] = None,
    gt_depth: Optional[bool] = None,
    vit_depth: Optional[bool] = None,
) -> None:
    """Keep SD-VLM runtime flags in sync with the HF config object.

    The published checkpoint has depth flags in ``config.json``.  Some SD-VLM
    code paths read attributes from the model while others read from
    ``model.config``, so update both to avoid stale checkpoint defaults.
    """

    config = getattr(model, "config", None)
    for name, value in (
        ("use_depth", use_depth),
        ("gt_depth", gt_depth),
        ("vit_depth", vit_depth),
    ):
        if value is None:
            continue
        value = bool(value)
        setattr(model, name, value)
        if config is not None:
            setattr(config, name, value)


def depth_array_to_pil(depth_array: np.ndarray) -> Any:
    """Convert a depth numpy array to a PIL Image for SD-VLM.

    The depth values are normalised to the [0, 255] uint8 range and returned
    as an RGB PIL Image, which is the format expected by the depth encoder used
    inside SD-VLM.

    Args:
        depth_array: Depth map of arbitrary shape and dtype.

    Returns:
        PIL.Image.Image (mode "RGB").
    """
    from PIL import Image

    arr = np.asarray(depth_array, dtype=np.float32)

    # Flatten channel dim if present (C, H, W) -> (H, W).
    if arr.ndim == 3 and arr.shape[0] in _LEADING_CHANNEL_COUNTS:
        arr = arr[0]

    d_min, d_max = float(arr.min()), float(arr.max())
    if d_max > d_min:
        arr = (arr - d_min) / (d_max - d_min)
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr_uint8, mode="L")


def make_side_by_side_image(left_image: Any, right_image: Any) -> Any:
    """Combine two RGB PIL images into one left/right comparison image."""

    from PIL import Image

    left = left_image.convert("RGB")
    right = right_image.convert("RGB")
    target_height = max(left.height, right.height)

    def _resize_to_height(img: Any) -> Any:
        if img.height == target_height:
            return img
        new_width = max(1, round(img.width * target_height / img.height))
        return img.resize((new_width, target_height))

    left = _resize_to_height(left)
    right = _resize_to_height(right)
    out = Image.new("RGB", (left.width + right.width, target_height), color=(255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def make_labeled_image_strip(images: Sequence[Any], labels: Sequence[object]) -> Any:
    """Combine ordered-task frames into one labeled horizontal strip."""

    from PIL import Image, ImageDraw

    rgb_images = [image.convert("RGB") for image in images]
    target_height = max(image.height for image in rgb_images)
    label_band = 28

    def _resize_to_height(img: Any) -> Any:
        if img.height == target_height:
            return img
        new_width = max(1, round(img.width * target_height / img.height))
        return img.resize((new_width, target_height))

    resized = [_resize_to_height(image) for image in rgb_images]
    out = Image.new("RGB", (sum(image.width for image in resized), target_height + label_band), color=(255, 255, 255))
    draw = ImageDraw.Draw(out)
    x = 0
    for image, label in zip(resized, labels):
        out.paste(image, (x, label_band))
        draw.text((x + 8, 7), str(label), fill=(0, 0, 0))
        x += image.width
    return out


def run_task_sdvlm(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    image: Any,
    prompt: str,
    depth_image: Optional[Any] = None,
    *,
    dataset_name: str = "",
    sample_id: str = "",
    task_name: str = "",
    viz_attn: bool = False,
    viz_out_dir: Optional[Path] = None,
    viz_attn_mode: str = "all",
) -> str:
    """Run a single VLM task with SD-VLM.

    Args:
        model: SD-VLM (LLaVA-based) model.
        tokenizer: LLaVA tokenizer.
        image_processor: LLaVA image processor.
        image: RGB PIL Image of the container.
        prompt: Task prompt text (without image tokens).
        depth_image: Optional depth PIL Image (required when ``model.gt_depth``
            is *True*; ignored when *False*).
        dataset_name: Dataset name for logging.
        sample_id: Sample identifier for logging.
        task_name: Task name for logging.
        viz_attn: Whether to extract attention weights and write heatmap overlays.
        viz_out_dir: Output directory for attention visualizations.
        viz_attn_mode: Which generated tokens to use for visualization –
            ``"all"`` or ``"first_answer_token"``.

    Returns:
        Model's text response.
    """
    import torch
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    # Build the full prompt with the image token prepended.
    full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    # LLaVA-1.5 conversation template.
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    formatted_prompt = conv.get_prompt()

    # Tokenise.
    input_ids = tokenizer_image_token(
        formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

    # Process image.
    image_tensor = process_images([image], image_processor, model.config)[0]

    # ori_imgs: depth image when gt_depth is set, copy of RGB otherwise.
    # If a dataset has no GT depth, force SD-VLM back to its internal depth
    # estimator for this call instead of leaving the checkpoint's gt_depth=True
    # config active.
    requested_gt_depth = bool(getattr(model, "gt_depth", False))
    effective_gt_depth = requested_gt_depth and depth_image is not None
    if effective_gt_depth:
        ori_img = depth_image
    else:
        ori_img = copy.deepcopy(image)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    image_tensor = image_tensor.unsqueeze(0).half().to(device)

    old_model_gt_depth = getattr(model, "gt_depth", None)
    old_config_gt_depth = getattr(getattr(model, "config", None), "gt_depth", None)

    gen_out = None
    try:
        _set_sdvlm_runtime_flags(model, gt_depth=effective_gt_depth)
        with torch.inference_mode():
            if viz_attn:
                gen_out = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    ori_imgs=[ori_img],
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                output_ids = gen_out.sequences
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    ori_imgs=[ori_img],
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                )
    finally:
        if old_model_gt_depth is not None:
            setattr(model, "gt_depth", old_model_gt_depth)
        config = getattr(model, "config", None)
        if config is not None and old_config_gt_depth is not None:
            setattr(config, "gt_depth", old_config_gt_depth)

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Attention visualization (opt-in).
    if viz_attn and viz_out_dir is not None:
        write_attention_overlay_sdvlm(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            gen_out=gen_out,
            output_ids=output_ids,
            image=image,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_out_dir=viz_out_dir,
            viz_attn_mode=viz_attn_mode,
        )

    return response


def test_sample_sdvlm(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    sample: Dict[str, Any],
    annotations: List[TLGAnnotation],
    *,
    dataset_name: str,
    sample_id: str,
    viz_attn: bool = False,
    viz_out_dir: Optional[Path] = None,
    viz_attn_mode: str = "all",
    debug_dtld_boxes: bool = False,
    debug_image_out_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Test all tasks on a single sample with SD-VLM.

    Args:
        model: SD-VLM model.
        tokenizer: LLaVA tokenizer.
        image_processor: LLaVA image processor.
        sample: Dataset sample dict.  Must contain either ``rgb_image`` (path)
            or ``rgb`` (numpy array).  Optionally contains ``depth_map`` (numpy
            array) used when ``model.gt_depth`` is *True*.
        annotations: Standardised task annotations.
        dataset_name: Dataset name for logging and visualization output.
        sample_id: Sample identifier for logging and visualization output.
        viz_attn: Whether to extract attention weights and write heatmap overlays.
        viz_out_dir: Output directory for attention visualizations.
        viz_attn_mode: Which generated tokens to use for visualization –
            ``"all"`` or ``"first_answer_token"``.

    Returns:
        Dictionary mapping task names to model responses.
    """
    from PIL import Image, UnidentifiedImageError

    # --- Load RGB image ---
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
            raise FileNotFoundError(
                "RGB image data not found in sample (missing both 'rgb_image' path and 'rgb' array)."
            )
        if rgb_array.shape[0] in _LEADING_CHANNEL_COUNTS:
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

    # --- Load depth image (optional, only used when model.gt_depth is True) ---
    depth_image: Optional[Any] = None
    if model.gt_depth:
        depth_array = sample.get("depth_map")
        if depth_array is not None:
            depth_image = depth_array_to_pil(depth_array)
        else:
            print(
                f"  [SD-VLM] gt_depth=True but no 'depth_map' found in sample "
                f"'{sample_id}'; falling back to RGB-based depth estimation."
            )

    # --- Run per-task inference ---
    results: Dict[str, str] = {}
    target_region_prompt = PromptBank.default().target_region_bbox_prompt()
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  [SD-VLM] Running task: {task_name}...")
        region_box_image = _make_annotation_box_image(rgb_image, ann)
        use_target_box = (
            target_box_image is not None
            and should_mark_target_box(dataset_name, task_name)
        )
        use_region_box = region_box_image is not None
        task_image = region_box_image if use_region_box else target_box_image if use_target_box else rgb_image
        prompt = ann.question
        if debug_dtld_boxes and dataset_name == "dtld":
            debug_dtld_box_state(
                prefix="SD-VLM",
                sample=sample,
                task_name=task_name,
                use_target_box=use_target_box,
                use_region_box=use_region_box,
                target_box_image=target_box_image,
            )
            if debug_image_out_dir is not None:
                save_dtld_box_debug_images(
                    out_dir=debug_image_out_dir,
                    model_key="sdvlm",
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
            task_image = make_side_by_side_image(task_image, comparison_image)
            prompt = (
                "The input image is a side-by-side comparison. "
                "The left panel is the first frame. "
                "The right panel is the second frame.\n\n"
                f"{prompt}"
            )
        if ordered_images is not None and len(ordered_images) >= 2:
            labels = sample.get("ordered_image_labels")
            if not isinstance(labels, (list, tuple)) or len(labels) != len(ordered_images):
                labels = [chr(ord("A") + i) for i in range(len(ordered_images))]
            task_image = make_labeled_image_strip(ordered_images, labels)
            prompt = (
                "The input image is a horizontal strip of frames. "
                "The frames are provided from left to right with labels: "
                f"{', '.join(str(label) for label in labels)}. "
                "Use these labels in your answer.\n\n"
                f"{prompt}"
            )

        response = run_task_sdvlm(
            model,
            tokenizer,
            image_processor,
            task_image,
            prompt,
            depth_image,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_attn=viz_attn,
            viz_out_dir=viz_out_dir,
            viz_attn_mode=viz_attn_mode,
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
