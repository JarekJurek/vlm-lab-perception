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
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tlg.annotations import TLGAnnotation


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
) -> Tuple[Any, Any, Any]:
    """Load SD-VLM model, tokenizer, and image processor.

    Args:
        model_path: HuggingFace model ID or local path to SD-VLM checkpoint.
        vision_tower: Vision encoder identifier (CLIP ViT).
        use_depth: Whether depth encoding is active.
        gt_depth: When *True* the caller must supply ground-truth depth images.
            When *False* depth is estimated internally from the RGB image.

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
    print(f"  use_depth={use_depth}  gt_depth={gt_depth}")

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ctx_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        mm_vision_tower=vision_tower,
    )

    model.to(torch.float16)
    model.use_depth = use_depth
    model.gt_depth = gt_depth

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
    # Convert grayscale to RGB by repeating channels.
    arr_rgb = np.stack([arr_uint8] * 3, axis=-1)
    return Image.fromarray(arr_rgb, mode="RGB")


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

    with torch.inference_mode():
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
        dataset_name: Dataset name for logging.
        sample_id: Sample identifier for logging.

    Returns:
        Dictionary mapping task names to model responses.
    """
    from PIL import Image, UnidentifiedImageError

    # --- Load RGB image ---
    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise OSError(f"Failed to open RGB image: {rgb_path} ({exc})") from exc
    else:
        rgb_array = sample.get("rgb")
        if rgb_array is None:
            raise FileNotFoundError(
                "RGB image data not found in sample (missing both 'rgb_image' path and 'rgb' array)."
            )
        if rgb_array.shape[0] in _LEADING_CHANNEL_COUNTS:
            rgb_array = rgb_array.transpose(1, 2, 0)
        rgb_image = Image.fromarray(rgb_array.astype("uint8"))

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
        response = run_task_sdvlm(
            model,
            tokenizer,
            image_processor,
            rgb_image,
            ann.question,
            depth_image,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
        )
        results[task_name] = response

    return results
