"""Attention visualization helpers for VLM outputs.

Supports two backends:
- Qwen-VL: ``write_attention_overlay`` – uses the processor's ``image_grid_thw``
  tensor to locate image tokens in the input sequence.
- SD-VLM (LLaVA-1.5): ``write_attention_overlay_sdvlm`` – locates image tokens
  via LLaVA's ``IMAGE_TOKEN_INDEX`` constant and the known patch count (576 for
  a 336 × 336 image with a 14 × 14 patch stride).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Sequence


def write_attention_overlay(
    *,
    model,
    processor,
    inputs,
    gen_out,
    output_ids=None,
    image,
    images: Sequence,
    dataset_name: str,
    sample_id: str,
    task_name: str,
    viz_out_dir: Optional[Path],
    viz_attn_mode: str = "all",
) -> None:
    """Write Qwen answer-to-image attention overlays for a single task.

    Args:
        model: Qwen3-VL model used for forward passes.
        processor: Qwen3-VL processor for tokenizer metadata.
        inputs: Model inputs returned by the processor.
        gen_out: Generation outputs from model.generate, or None.
        output_ids: Generated token ids (batch, seq_len). Defaults to gen_out.sequences.
        image: PIL image for the first input image.
        images: Sequence of input images (RGB and optional mask).
        dataset_name: Dataset name used in output path.
        sample_id: Sample identifier used in output path.
        task_name: Task name used in output filename.
        viz_out_dir: Output directory root for visualization files.
        viz_attn_mode: Accepted for CLI compatibility; Qwen uses answer tokens.
    """
    if viz_out_dir is None:
        return

    import torch

    def _warn(msg: str) -> None:
        """Print a visualization warning with context."""
        prefix = "[viz-attn]"
        ctx = ""
        if dataset_name or sample_id or task_name:
            ctx = f" dataset={dataset_name!r} sample_id={sample_id!r} task={task_name!r}"
        print(f"{prefix}{ctx} {msg}")

    try:
        prompt_ids = inputs.get("input_ids")
        if prompt_ids is None or getattr(prompt_ids, "ndim", None) != 2 or int(prompt_ids.shape[0]) != 1:
            _warn(f"Unexpected input_ids shape={getattr(prompt_ids, 'shape', None)}; skipping viz.")
            return
        input_len = int(prompt_ids.shape[1])

        if images is not None and len(images) > 1:
            _warn("Multiple images were provided; visualizing attention for the first image only.")

        if output_ids is None and gen_out is not None:
            output_ids = getattr(gen_out, "sequences", None)
        if output_ids is None or getattr(output_ids, "ndim", None) != 2:
            _warn("Missing generated sequences; skipping viz.")
            return

        image_start, image_end = _qwen_image_range(prompt_ids[0], model)
        num_image_tokens = int(image_end - image_start)
        grid_h, grid_w = _qwen_grid_size(inputs, model, num_image_tokens)

        generated_ids = output_ids[:, input_len:]
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            _warn("Tokenizer unavailable; skipping viz.")
            return
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0].detach().cpu().tolist())
        answer_offsets = _qwen_answer_token_offsets(generated_tokens)
        if not answer_offsets:
            _warn("No non-special generated answer tokens found; skipping viz.")
            return
        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        full_inputs = _qwen_extend_inputs(inputs, output_ids, input_len)
        with torch.no_grad():
            outputs = model(
                **full_inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        base_path = (
            viz_out_dir
            / _safe_name(dataset_name)
            / _safe_name(sample_id)
            / _safe_name(task_name)
        )
        _save_qwen_layer_grid(
            image=image,
            outputs=outputs,
            answer=answer,
            answer_offsets=answer_offsets,
            input_len=input_len,
            image_start=image_start,
            image_end=image_end,
            grid_h=grid_h,
            grid_w=grid_w,
            out_path=Path(f"{base_path}.png"),
        )
    except torch.cuda.OutOfMemoryError as exc:
        try:
            del outputs
        except UnboundLocalError:
            pass
        try:
            del full_inputs
        except UnboundLocalError:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _warn(f"CUDA OOM while extracting attentions ({exc}); continuing without viz.")
    except ValueError as exc:
        _warn(f"{exc}; skipping viz.")
    except Exception as exc:
        _warn(f"Failed to write visualization ({type(exc).__name__}: {exc}); continuing without viz.")


_QWEN_LAYERS = [-29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -16, -12, -8, -4]
_QWEN_SIGMA = 30


def _qwen_image_range(input_ids, model) -> tuple[int, int]:
    """Return the first contiguous Qwen image-token span."""
    import torch

    image_token_id = int(model.config.image_token_id)
    positions = torch.where(input_ids == image_token_id)[0]
    if int(positions.numel()) == 0:
        raise ValueError("No image token positions found in prompt")

    start = int(positions[0].item())
    end = start + 1
    for pos in positions[1:].detach().cpu().tolist():
        if int(pos) != end:
            break
        end += 1
    return start, end


def _qwen_grid_size(inputs, model, num_image_tokens: int) -> tuple[int, int]:
    """Return the supervisor-style merged image-token grid size."""
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("Missing image_grid_thw")
    t, h, w = [int(v) for v in grid[0].detach().cpu().tolist()]
    if t != 1:
        raise ValueError(f"Expected one image frame, got image_grid_thw[0][0]={t}")
    merge = int(model.config.vision_config.spatial_merge_size)
    grid_h = h // merge
    grid_w = w // merge
    if grid_h * grid_w != int(num_image_tokens):
        raise ValueError(
            f"Grid size {grid_h}x{grid_w} does not match {num_image_tokens} image tokens"
        )
    return grid_h, grid_w


def _qwen_extend_inputs(inputs, sequences, input_len: int):
    """Append generated ids and masks for a full answer-attention forward pass."""
    import torch

    full = dict(inputs)
    full["input_ids"] = sequences

    generated_len = int(sequences.shape[1]) - int(input_len)
    if generated_len < 0:
        raise ValueError("Generated sequence is shorter than the prompt")

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Missing attention_mask")
    full["attention_mask"] = torch.cat(
        [
            attention_mask,
            torch.ones(
                attention_mask.shape[0],
                generated_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            ),
        ],
        dim=1,
    )

    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in inputs:
            extra = inputs[key][:, -1:].expand(-1, generated_len)
            full[key] = torch.cat([inputs[key], extra], dim=1)

    for key in ("position_ids", "rope_deltas", "cache_position"):
        full.pop(key, None)

    return full


def _qwen_answer_token_offsets(tokens: Sequence[str]) -> List[int]:
    """Return generated token offsets that look like answer content."""
    offsets: List[int] = []
    special_tokens = {"<s>", "</s>", "<|im_end|>", "<|endoftext|>"}
    for i, tok in enumerate(tokens):
        if tok.startswith("<|") and tok.endswith("|>"):
            continue
        if tok in special_tokens:
            continue
        offsets.append(i)
    return offsets


def _qwen_attention_for_layer(
    *,
    outputs,
    layer: int,
    answer_offsets: Sequence[int],
    input_len: int,
    image_start: int,
    image_end: int,
):
    """Average one layer's answer-token attention to Qwen image tokens."""
    import torch

    attn = outputs.attentions[layer][0]
    maps = []
    for offset in answer_offsets:
        query = int(input_len) + int(offset)
        heads_to_image = attn[:, query, image_start:image_end]
        maps.append(heads_to_image.mean(dim=0))

    vec = torch.stack(maps).mean(dim=0)
    return _normalize_array(vec.float().detach().cpu().numpy())


def _save_qwen_layer_grid(
    *,
    image,
    outputs,
    answer: str,
    answer_offsets: Sequence[int],
    input_len: int,
    image_start: int,
    image_end: int,
    grid_h: int,
    grid_w: int,
    out_path: Path,
) -> None:
    """Save the supervisor-style grid of Qwen layer overlays."""
    import math
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = int(math.ceil(len(_QWEN_LAYERS) / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    for i, layer in enumerate(_QWEN_LAYERS):
        vec = _qwen_attention_for_layer(
            outputs=outputs,
            layer=layer,
            answer_offsets=answer_offsets,
            input_len=input_len,
            image_start=image_start,
            image_end=image_end,
        )
        grid = vec.reshape(int(grid_h), int(grid_w))
        overlay = _qwen_make_overlay(image, grid)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer}")
        ax.axis("off")

    fig.suptitle(f"Mean answer-to-image attention\nAnswer: {answer}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _qwen_make_overlay(image, grid):
    """Blend a Qwen attention grid onto the source image."""
    import cv2
    import numpy as np

    image_np = np.asarray(image.convert("RGB"))
    h, w = image_np.shape[:2]
    heatmap = cv2.resize(grid, (w, h))
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), _QWEN_SIGMA)
    heatmap = _normalize_array(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.65 * image_np + 0.35 * heatmap
    return np.uint8(np.clip(overlay, 0, 255))


def _normalize_array(x):
    """Normalize an array to 0-1."""
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    arr = arr - float(arr.min())
    return arr / (float(arr.max()) + 1e-8)


def write_attention_overlay_sdvlm(
    *,
    model,
    tokenizer,
    input_ids,
    gen_out,
    output_ids=None,
    image,
    dataset_name: str,
    sample_id: str,
    task_name: str,
    viz_out_dir: Optional[Path],
    viz_attn_mode: str = "all",
) -> None:
    """Write an attention heatmap overlay for a single SD-VLM (LLaVA-1.5) task.

    SD-VLM is based on LLaVA-1.5 which encodes each image as a fixed grid of
    N_PATCHES = (image_size / patch_size)² tokens that are inserted into the
    sequence at the position of the ``IMAGE_TOKEN_INDEX`` placeholder.  For the
    default 336 × 336 CLIP ViT-L/14 vision tower this gives 576 patch tokens.

    The function derives the number of image tokens from the model's vision
    tower configuration at runtime, so it adapts automatically to other CLIP
    configurations without hard-coding 576.

    Args:
        model: SD-VLM (LLaVA-based) model.
        tokenizer: LLaVA tokenizer, used for decoding generated tokens.
        input_ids: Token ids fed into the model (before image expansion),
            shape (1, seq_len).
        gen_out: GenerateOutput returned by ``model.generate`` with
            ``return_dict_in_generate=True, output_attentions=True``.
        output_ids: Full generated token id sequence (batch, seq_len).
            Defaults to ``gen_out.sequences`` when *None*.
        image: PIL Image of the RGB input (used as overlay base).
        dataset_name: Dataset name used in the output path.
        sample_id: Sample identifier used in the output path.
        task_name: Task name used in the output filename.
        viz_out_dir: Output directory root.  If *None* the function is a no-op.
        viz_attn_mode: Token selection strategy – ``"all"`` or
            ``"first_answer_token"``.
    """
    if viz_out_dir is None:
        return

    import torch

    def _warn(msg: str) -> None:
        prefix = "[viz-attn/sdvlm]"
        ctx = ""
        if dataset_name or sample_id or task_name:
            ctx = f" dataset={dataset_name!r} sample_id={sample_id!r} task={task_name!r}"
        print(f"{prefix}{ctx} {msg}")

    try:
        # --- Resolve the number of image patch tokens ---
        n_image_tokens: Optional[int] = None
        try:
            # LLaVA-1.5: model.model.vision_tower exposes mm_hidden_size and
            # the underlying CLIP model which stores image_size and patch_size.
            vt = getattr(model, "model", model)
            vt = getattr(vt, "vision_tower", None)
            if vt is not None:
                clip_model = getattr(vt, "vision_tower", vt)
                cfg = getattr(clip_model, "config", None)
                if cfg is not None:
                    img_size = int(getattr(cfg, "image_size", 336))
                    patch_size = int(getattr(cfg, "patch_size", 14))
                    n_image_tokens = (img_size // patch_size) ** 2
        except Exception:
            pass

        if n_image_tokens is None:
            # Fallback: standard CLIP ViT-L/14-336 produces 576 tokens.
            n_image_tokens = 576
            _warn(f"Could not read vision tower config; assuming {n_image_tokens} image tokens.")

        # --- Find where IMAGE_TOKEN_INDEX sits in the flat input ---
        try:
            from llava.constants import IMAGE_TOKEN_INDEX
        except ImportError:
            IMAGE_TOKEN_INDEX = -200

        if input_ids is None or getattr(input_ids, "ndim", None) != 2:
            _warn("Unexpected input_ids shape; skipping viz.")
            return

        img_placeholder_positions = (input_ids[0] == int(IMAGE_TOKEN_INDEX)).nonzero(as_tuple=False).flatten()
        if img_placeholder_positions.numel() == 0:
            _warn("IMAGE_TOKEN_INDEX not found in input_ids; skipping viz.")
            return

        # After the model expands the image placeholder, each original token
        # before the placeholder stays in place; the placeholder itself is
        # replaced by n_image_tokens patch embeddings; tokens after shift right.
        img_start = int(img_placeholder_positions[0].item())
        img_positions = list(range(img_start, img_start + n_image_tokens))

        # The expanded prompt length accounts for the token expansion.
        n_prefix_tokens = int(input_ids.shape[1])
        # Prefix tokens before image: img_start
        # Image tokens: n_image_tokens  (replaces 1 placeholder)
        # Remaining prefix tokens: n_prefix_tokens - img_start - 1
        expanded_prompt_len = img_start + n_image_tokens + (n_prefix_tokens - img_start - 1)

        if output_ids is None and gen_out is not None:
            output_ids = getattr(gen_out, "sequences", None)

        gen_attns = None
        if gen_out is not None:
            gen_attns = getattr(gen_out, "attentions", None)

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        attn_accum = _collect_generated_token_attention(
            gen_attns=gen_attns,
            output_ids=output_ids,
            prompt_len=expanded_prompt_len,
            img_positions=img_positions,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            viz_attn_mode=viz_attn_mode,
            warn=_warn,
        )

        if attn_accum is None:
            _warn("Could not reduce attentions into a vector; skipping viz.")
            return

        attn_vec = attn_accum.detach().float().cpu()
        if int(attn_vec.numel()) != n_image_tokens:
            _warn(
                f"Attention vector length {attn_vec.numel()} != expected {n_image_tokens}; skipping viz."
            )
            return

        # Reshape into a square patch grid (h, w).
        hw = _infer_grid_hw(n_image_tokens)
        if hw is None:
            _warn(f"Cannot reshape {n_image_tokens} image tokens into a square grid; skipping viz.")
            return
        h_i, w_i = hw
        heat_hw = attn_vec.view(h_i, w_i)

        base_path = (
            viz_out_dir
            / _safe_name(dataset_name)
            / _safe_name(sample_id)
            / _safe_name(task_name)
        )
        _save_overlay_png(rgb_pil=image, heatmap_hw=heat_hw.numpy(), out_path=Path(f"{base_path}.png"))
    except torch.cuda.OutOfMemoryError as exc:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        _warn(f"CUDA OOM while extracting attentions ({exc}); continuing without viz.")
    except Exception as exc:
        _warn(f"Failed to write visualization ({type(exc).__name__}: {exc}); continuing without viz.")


def _collect_generated_token_attention(
    *,
    gen_attns,
    output_ids,
    prompt_len: int,
    img_positions: List[int],
    tokenizer,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    viz_attn_mode: str,
    warn,
) -> Optional["torch.Tensor"]:
    """Aggregate attention over selected generated tokens."""
    import torch

    if not isinstance(gen_attns, (tuple, list)) or not gen_attns:
        return None
    if output_ids is None or not hasattr(output_ids, "shape"):
        return None
    if tokenizer is None:
        warn("Tokenizer unavailable; cannot decode generated tokens for attention selection.")
        return None

    def _collect_for_mode(mode: str) -> Optional[torch.Tensor]:
        vectors: List[torch.Tensor] = []
        for step_idx, step_attns in enumerate(gen_attns):
            token_pos = int(prompt_len + step_idx)
            if int(output_ids.shape[1]) <= token_pos:
                continue
            token_id = int(output_ids[0, token_pos].item())
            if eos_token_id is not None and token_id == int(eos_token_id):
                continue
            if pad_token_id is not None and token_id == int(pad_token_id):
                continue
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if _is_trivial_generated_token(token_text):
                continue

            if not isinstance(step_attns, (tuple, list)):
                continue

            layer_vecs: List[torch.Tensor] = []
            for layer_attn in step_attns:
                if layer_attn is None:
                    continue
                if getattr(layer_attn, "ndim", None) != 4:
                    continue
                if int(layer_attn.shape[-1]) <= max(img_positions):
                    continue
                if int(layer_attn.shape[-2]) < 1:
                    continue
                q_idx = int(layer_attn.shape[-2]) - 1
                vec = layer_attn[0, :, q_idx, img_positions]
                vec = vec.mean(dim=0)
                layer_vecs.append(vec)

            if not layer_vecs:
                continue
            vectors.append(torch.stack(layer_vecs, dim=0).mean(dim=0))
            if mode == "first_answer_token":
                break

        if not vectors:
            return None
        return torch.stack(vectors, dim=0).mean(dim=0)

    return _collect_for_mode(viz_attn_mode)


def _is_trivial_generated_token(token_text: str) -> bool:
    """Return True for tokens that should be ignored in visualization selection."""
    trivial = {
        "",
        " ",
        "\n",
        "\t",
        ".",
        ",",
        ":",
        ";",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        '"',
        "'",
        "`",
    }
    return token_text in trivial or token_text.strip() == ""


def _infer_grid_hw(n: int) -> Optional[tuple[int, int]]:
    """Infer a square grid (h, w) for a token count, if possible."""
    if n <= 0:
        return None
    side = int(round(n ** 0.5))
    if side * side == n:
        return side, side
    return None


def _save_overlay_png(
    *,
    rgb_pil,
    heatmap_hw,
    out_path: Path,
    alpha: float = 0.65,
    cmap: str = "inferno",
) -> None:
    """Save a heatmap overlay PNG on top of a reference image."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    heat = _normalize_heatmap(heatmap_hw)
    heat_img = Image.fromarray((heat * 255.0).astype("uint8"), mode="L")
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR"))
    heat_img = heat_img.resize(rgb_pil.size, resample=resample)
    heat_up = np.asarray(heat_img, dtype=np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(rgb_pil.size[0] / 100.0, rgb_pil.size[1] / 100.0), dpi=100)
    ax.imshow(rgb_pil)
    ax.imshow(heat_up, cmap=cmap, alpha=float(alpha), vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _normalize_heatmap(heatmap_hw):
    """Normalize a 2D heatmap to 0-1 for visualization."""
    import numpy as np

    heat = np.asarray(heatmap_hw, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape={heat.shape}")

    hmin = float(heat.min())
    hmax = float(heat.max())
    if hmax > hmin:
        heat = (heat - hmin) / (hmax - hmin)
    else:
        heat = np.zeros_like(heat, dtype=np.float32)
    return heat


def _safe_name(value: str) -> str:
    """Normalize a string for safe filesystem usage."""
    import re

    safe_name_re = re.compile(r"[^a-zA-Z0-9_.-]+")
    s2 = safe_name_re.sub("_", str(value).strip())
    s2 = s2.strip("_.")
    return s2 or "unknown"
