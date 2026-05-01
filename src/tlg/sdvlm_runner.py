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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tlg.annotations import TLGAnnotation
from tlg.attention_viz import write_attention_overlay_sdvlm


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
    model.use_depth = use_depth
    model.gt_depth = gt_depth
    model.vit_depth = False

    print("SD-VLM model loaded successfully.")
    return model, tokenizer, image_processor


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
            ``"all"``, ``"first_answer_token"``, or ``"mcq_answer"``.

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
    if model.gt_depth and depth_image is not None:
        ori_img = depth_image
    else:
        ori_img = copy.deepcopy(image)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    image_tensor = image_tensor.unsqueeze(0).half().to(device)

    gen_out = None
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
            ``"all"``, ``"first_answer_token"``, or ``"mcq_answer"``.

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
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  [SD-VLM] Running task: {task_name}...")
        task_image = rgb_image
        prompt = ann.question
        if comparison_image is not None:
            task_image = make_side_by_side_image(rgb_image, comparison_image)
            prompt = (
                "The input image is a side-by-side comparison. "
                "The left panel is the first, earlier frame. "
                "The right panel is the second, later frame.\n\n"
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
