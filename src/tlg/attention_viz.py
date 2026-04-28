"""Attention visualization helpers for Qwen-VL outputs."""

from __future__ import annotations

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
    """Write an attention heatmap overlay for a single task.

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
        viz_attn_mode: Which generated tokens to use for attention visualization.
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

    if viz_attn_mode not in {"all", "first_answer_token", "mcq_answer"}:
        raise ValueError(f"Unsupported viz_attn_mode: {viz_attn_mode}")

    try:
        # NOTE: This is a qualitative raw self-attention diagnostic, not a faithful
        # reproduction of LABSHIELD or attention rollout.

        img_token_id = _get_image_token_id(processor)
        if img_token_id is None:
            _warn("Could not resolve image token id; skipping viz for this task.")
            return

        prompt_ids = inputs.get("input_ids")
        if prompt_ids is None or getattr(prompt_ids, "ndim", None) != 2 or int(prompt_ids.shape[0]) != 1:
            _warn(f"Unexpected input_ids shape={getattr(prompt_ids, 'shape', None)}; skipping viz.")
            return
        prompt_len = int(prompt_ids.shape[1])

        # Find image-token positions in the prompt only.
        img_positions_all = (prompt_ids[0] == int(img_token_id)).nonzero(as_tuple=False).flatten().tolist()
        if not img_positions_all:
            _warn("No image token positions found in prompt; skipping viz.")
            return

        if len(images) > 1:
            _warn("Multiple images were provided; visualizing attention for the first image only.")

        grid = inputs.get("image_grid_thw")
        t_raw = h_raw = w_raw = None
        t_i = h_i = w_i = 1
        if grid is not None:
            try:
                grid0 = grid[0].tolist() if hasattr(grid, "__getitem__") else None
                if isinstance(grid0, (list, tuple)) and len(grid0) >= 3:
                    t_raw, h_raw, w_raw = int(grid0[0]), int(grid0[1]), int(grid0[2])
            except Exception:
                t_raw = h_raw = w_raw = None

        n_img1 = None
        if (
            isinstance(t_raw, int)
            and isinstance(h_raw, int)
            and isinstance(w_raw, int)
            and t_raw > 0
            and h_raw > 0
            and w_raw > 0
        ):
            n_img1 = int(t_raw * h_raw * w_raw)
        if n_img1 is None:
            # If multiple images were provided (e.g. mask), try splitting evenly.
            n_imgs = max(1, int(len(images)))
            if n_imgs > 1 and len(img_positions_all) % n_imgs == 0:
                per = int(len(img_positions_all) // n_imgs)
                aspect = None
                hw = _infer_grid_hw_from_aspect(per, aspect=aspect)
                if hw is not None:
                    t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])
                    n_img1 = int(h_i * w_i)
            if n_img1 is None:
                # Last resort: factor the token count.
                hw = _infer_grid_hw_from_aspect(len(img_positions_all), aspect=None)
                if hw is None:
                    _warn("Could not infer image grid size (missing image_grid_thw); skipping viz.")
                    return
                t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])
                n_img1 = int(h_i * w_i)
        else:
            assert t_raw is not None and h_raw is not None and w_raw is not None
            t_i, h_i, w_i = int(t_raw), int(h_raw), int(w_raw)

            # Reconcile expected token count from `image_grid_thw` with actual image-token count.
            # Empirically, Qwen-VL often uses a downsampled token grid (e.g. 30x40 -> 15x20).
            # Use the total image-token count if multiple images are present.
            expected_total = None
            try:
                if grid is not None and hasattr(grid, "shape") and int(grid.shape[0]) >= 1:
                    exp = 0
                    for i in range(int(grid.shape[0])):
                        gi = grid[i].tolist()
                        if isinstance(gi, (list, tuple)) and len(gi) >= 3:
                            exp += int(gi[0]) * int(gi[1]) * int(gi[2])
                    expected_total = int(exp) if exp > 0 else None
            except Exception:
                expected_total = None

            actual_total = int(len(img_positions_all))
            expected_for_ratio = (
                int(expected_total) if isinstance(expected_total, int) and expected_total > 0 else int(n_img1)
            )
            if actual_total > 0 and expected_for_ratio > 0 and expected_for_ratio != actual_total:
                if expected_for_ratio % actual_total == 0:
                    ratio = int(expected_for_ratio // actual_total)
                    s = _is_perfect_square(ratio)
                    if s is not None and s > 1:
                        # Adjust grid dims by s if divisible.
                        if h_i % s == 0 and w_i % s == 0:
                            h_i //= s
                            w_i //= s
                            n_img1 = int(t_i * h_i * w_i)

        img_positions = img_positions_all[:n_img1]
        if len(img_positions) != n_img1:
            exp_before = int(n_img1)
            found = int(len(img_positions_all))
            _warn(
                f"Image token count mismatch: expected {exp_before}, found {found}; "
                "using available tokens for first image."
            )
            n_img1 = len(img_positions)
            aspect = None
            if isinstance(h_raw, int) and isinstance(w_raw, int) and h_raw > 0 and w_raw > 0:
                aspect = float(h_raw) / float(w_raw)
            hw = _infer_grid_hw_from_aspect(n_img1, aspect=aspect)
            if hw is None:
                _warn("Could not reshape image tokens into a grid; skipping viz.")
                return
            t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])

        # Prefer attentions emitted by generate() to avoid an expensive (O(L^2)) full forward pass.
        gen_attns = None
        if gen_out is not None:
            gen_attns = getattr(gen_out, "attentions", None)
            if gen_attns is None:
                gen_attns = getattr(gen_out, "decoder_attentions", None)

        if output_ids is None and gen_out is not None:
            output_ids = getattr(gen_out, "sequences", None)

        tokenizer = getattr(processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

        attn_accum = _collect_generated_token_attention(
            gen_attns=gen_attns,
            output_ids=output_ids,
            prompt_len=prompt_len,
            img_positions=img_positions,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            viz_attn_mode=viz_attn_mode,
            warn=_warn,
        )

        # Fallback: do a prompt-only forward pass if generation did not provide attentions.
        if attn_accum is None:
            _warn("Falling back to prompt-only attention; this is not answer-token attention.")
            fwd_inputs = dict(inputs)
            fwd_inputs["attention_mask"] = torch.ones_like(prompt_ids, device=prompt_ids.device)
            with torch.no_grad():
                out = model(
                    **fwd_inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

            attns = getattr(out, "attentions", None)
            if not isinstance(attns, (tuple, list)) or not attns:
                _warn("Model did not return attentions in fallback; skipping viz.")
                return

            # Use the last prompt token as query.
            q_idx = int(prompt_len) - 1
            n_used = 0
            for layer_attn in attns:
                if layer_attn is None:
                    continue
                # Expected: (bs, heads, seq, seq)
                if getattr(layer_attn, "ndim", None) != 4:
                    continue
                a = layer_attn[0]  # (heads, seq, seq)
                if int(a.shape[-1]) <= max(img_positions) or int(a.shape[-2]) <= q_idx:
                    continue
                vec = a[:, q_idx, img_positions]  # (heads, n_img1)
                vec = vec.mean(dim=0)  # (n_img1,)
                attn_accum = vec if attn_accum is None else (attn_accum + vec)
                n_used += 1

            if n_used > 0 and attn_accum is not None:
                attn_accum = attn_accum / float(n_used)

        if attn_accum is None:
            _warn("Could not reduce attentions into a vector; skipping viz.")
            return

        attn_vec = attn_accum.detach().float().cpu()
        if int(attn_vec.numel()) != int(t_i * h_i * w_i):
            # Fallback to square reshape.
            hw = _infer_grid_hw(int(attn_vec.numel()))
            if hw is None:
                _warn("Attention vector length does not match grid; skipping viz.")
                return
            t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])
        attn_thw = attn_vec.view(int(t_i), int(h_i), int(w_i))
        heat_hw = attn_thw.mean(dim=0)  # (h, w)

        base_path = (
            viz_out_dir
            / _safe_name(dataset_name)
            / _safe_name(sample_id)
            / _safe_name(task_name)
        )
        _save_original_png(rgb_pil=image, out_path=Path(f"{base_path}.original.png"))
        _save_heatmap_png(heatmap_hw=heat_hw.numpy(), out_path=Path(f"{base_path}.heatmap.png"), size=image.size)
        _save_overlay_png(rgb_pil=image, heatmap_hw=heat_hw.numpy(), out_path=Path(f"{base_path}.overlay.png"))
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

            stripped = token_text.strip()
            if mode == "mcq_answer" and stripped not in {"A", "B", "C", "D"}:
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
                vec = layer_attn[0, :, 0, img_positions]
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

    if viz_attn_mode == "mcq_answer":
        vec = _collect_for_mode("mcq_answer")
        if vec is None:
            warn("No MCQ answer token found; falling back to all generated tokens.")
            return _collect_for_mode("all")
        return vec

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


def _get_image_token_id(processor) -> Optional[int]:
    """Infer the image token id from a processor tokenizer."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    # Many VLM tokenizers expose a direct id.
    for attr in ("image_token_id", "image_pad_token_id"):
        val = getattr(tok, attr, None)
        if isinstance(val, int):
            return int(val)
    # Try to infer from additional special tokens.
    candidates: List[str] = []
    for t in getattr(tok, "additional_special_tokens", []) or []:
        ts = str(t)
        if "image" in ts.lower():
            candidates.append(ts)
    if candidates:
        preferred = None
        for key in ("image_pad", "image", "img"):
            for c in candidates:
                if key in c.lower():
                    preferred = c
                    break
            if preferred is not None:
                break
        token = preferred or candidates[0]
        try:
            token_id = tok.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)
        except Exception:
            return None
    return None


def _infer_grid_hw(n: int) -> Optional[tuple[int, int]]:
    """Infer a square grid (h, w) for a token count, if possible."""
    if n <= 0:
        return None
    side = int(round(n ** 0.5))
    if side * side == n:
        return side, side
    return None


def _is_perfect_square(n: int) -> Optional[int]:
    """Return the integer square root if n is a perfect square."""
    if n <= 0:
        return None
    r = int(round(n ** 0.5))
    return r if r * r == n else None


def _infer_grid_hw_from_aspect(n: int, *, aspect: Optional[float]) -> Optional[tuple[int, int]]:
    """Find (h, w) such that h*w == n and h/w ~= aspect.

    Falls back to any factor pair if aspect is missing.
    """
    if n <= 0:
        return None
    if aspect is not None and aspect <= 0:
        aspect = None

    best = None
    best_err = None
    r = int(n ** 0.5)
    for h in range(1, r + 1):
        if n % h != 0:
            continue
        w = n // h
        for hh, ww in ((h, w), (w, h)):
            if aspect is None:
                return int(hh), int(ww)
            err = abs((float(hh) / float(ww)) - float(aspect))
            if best is None or best_err is None or err < best_err:
                best = (int(hh), int(ww))
                best_err = err
    return best


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


def _save_original_png(*, rgb_pil, out_path: Path) -> None:
    """Save the original image as a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_pil.save(out_path, format="PNG")


def _save_heatmap_png(*, heatmap_hw, out_path: Path, size, cmap: str = "inferno") -> None:
    """Save a heatmap-only PNG resized to the original image size."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)

    heat = _normalize_heatmap(heatmap_hw)
    heat_img = Image.fromarray((heat * 255.0).astype("uint8"), mode="L")
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR"))
    heat_img = heat_img.resize(size, resample=resample)
    heat_up = np.asarray(heat_img, dtype=np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    ax.imshow(heat_up, cmap=cmap, vmin=0.0, vmax=1.0)
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
