"""Test Qwen-VL on liquid container vision tasks.

This script evaluates Qwen-VL on visual understanding tasks for multiple
liquid-container datasets:
    - LabLiquidVision
    - Glasses in the Wild
    - DTLD
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Ensure `src/` is on PYTHONPATH so imports match README usage.
sys.path.append(str(Path(__file__).parent / "src"))

from datasets.dtld import DTLDDataset
from datasets.glassesinthewild import GlassesInTheWildDataset
from datasets.labliquidvision import LabLiquidVisionDataset
from datasets.ot2 import OT2Dataset
from datasets.pouring import PouringDataset

from tlg import PromptBank, TLGAnnotation
from tlg.evaluation import TaskStats, score_prediction


# MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 256


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


DATA_ROOTS = {
    "labliquidvision": Path("data") / "LabLiquidVision",
    "glasses": Path("data") / "glasses_in_the_wild" / "glasses_in_the_wild_1000",
    "dtld": Path("data") / "DTLD_dataset",
    # These defaults may vary depending on how you lay out `data/`.
    "pouring": Path("data") / "pouring",
    "ot2": Path("data") / "OT2-Computer-Vision",
}


def load_model(*, viz_attn: bool = False):
    """Load Qwen3-VL model and processor."""
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
        # Some attention backends don't materialize attention weights.
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
    *,
    dataset_name: str = "",
    sample_id: str = "",
    task_name: str = "",
    viz_attn: bool = False,
    viz_out_dir: Optional[Path] = None,
) -> str:
    """Run a single VLM task.
    from PIL import Image, UnidentifiedImageError
    
    Args:
        processor: Qwen3-VL processor.
        image: RGB image of the container.
        prompt: Task prompt text.
        mask: Optional vessel mask image.
    
    Returns:
        Model's text response.
    """
    import torch

    def _warn(msg: str) -> None:
        prefix = "[viz-attn]"
        ctx = ""
        if dataset_name or sample_id or task_name:
            ctx = f" dataset={dataset_name!r} sample_id={sample_id!r} task={task_name!r}"
        print(f"{prefix}{ctx} {msg}")

    def _get_image_token_id() -> Optional[int]:
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

    def _select_query_index(
        seq_1d: "torch.Tensor",
        prompt_len: int,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> int:
        # Choose the last non-(eos/pad) generated token if possible.
        last = int(seq_1d.numel()) - 1
        i = last
        while i >= prompt_len:
            tid = int(seq_1d[i].item())
            if pad_token_id is not None and tid == int(pad_token_id):
                i -= 1
                continue
            if eos_token_id is not None and tid == int(eos_token_id):
                i -= 1
                continue
            return i
        return last

    def _infer_grid_hw(n: int) -> Optional[tuple[int, int]]:
        if n <= 0:
            return None
        side = int(round(n ** 0.5))
        if side * side == n:
            return side, side
        return None

    def _is_perfect_square(n: int) -> Optional[int]:
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
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        out_path.parent.mkdir(parents=True, exist_ok=True)

        heat = np.asarray(heatmap_hw, dtype=np.float32)
        if heat.ndim != 2:
            raise ValueError(f"Expected 2D heatmap, got shape={heat.shape}")

        hmin = float(heat.min())
        hmax = float(heat.max())
        if hmax > hmin:
            heat = (heat - hmin) / (hmax - hmin)
        else:
            heat = np.zeros_like(heat, dtype=np.float32)

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
                do_sample=False
            )
    
    # Decode
    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]
    
    # Extract just the answer (remove the prompt echo if present)
    if prompt in response:
        response = response.split(prompt)[-1].strip()

    # Attention visualization (opt-in)
    if viz_attn and viz_out_dir is not None:
        try:
            img_token_id = _get_image_token_id()
            if img_token_id is None:
                _warn("Could not resolve image token id; skipping viz for this task.")
                return response

            prompt_ids = inputs.get("input_ids")
            if prompt_ids is None or getattr(prompt_ids, "ndim", None) != 2 or int(prompt_ids.shape[0]) != 1:
                _warn(f"Unexpected input_ids shape={getattr(prompt_ids, 'shape', None)}; skipping viz.")
                return response
            prompt_len = int(prompt_ids.shape[1])

            # Find image-token positions in the prompt only.
            img_positions_all = (prompt_ids[0] == int(img_token_id)).nonzero(as_tuple=False).flatten().tolist()
            if not img_positions_all:
                _warn("No image token positions found in prompt; skipping viz.")
                return response

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
                        return response
                    t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])
                    n_img1 = int(h_i * w_i)
            else:
                assert t_raw is not None and h_raw is not None and w_raw is not None
                t_i, h_i, w_i = int(t_raw), int(h_raw), int(w_raw)

                # Reconcile expected token count from `image_grid_thw` with actual image-token count.
                # Empirically, Qwen-VL often uses a downsampled token grid (e.g. 30x40 -> 15x20).
                # Use the *total* image-token count if multiple images are present.
                expected_total = None
                try:
                    if grid is not None and hasattr(grid, "shape") and int(grid.shape[0]) >= 1:
                        # Sum t*h*w over all images if available.
                        exp = 0
                        for i in range(int(grid.shape[0])):
                            gi = grid[i].tolist()
                            if isinstance(gi, (list, tuple)) and len(gi) >= 3:
                                exp += int(gi[0]) * int(gi[1]) * int(gi[2])
                        expected_total = int(exp) if exp > 0 else None
                except Exception:
                    expected_total = None

                actual_total = int(len(img_positions_all))
                expected_for_ratio = int(expected_total) if isinstance(expected_total, int) and expected_total > 0 else int(n_img1)
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
                    return response
                t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])

            # Prefer attentions emitted by `generate()` to avoid an expensive (O(L^2)) full forward pass.
            gen_attns = None
            if gen_out is not None:
                gen_attns = getattr(gen_out, "attentions", None)
                if gen_attns is None:
                    gen_attns = getattr(gen_out, "decoder_attentions", None)

            attn_accum = None
            n_used = 0

            if isinstance(gen_attns, (tuple, list)) and gen_attns:
                # `gen_attns` is typically a tuple over generated steps.
                last_step = gen_attns[-1]
                if isinstance(last_step, (tuple, list)) and last_step:
                    for layer_attn in last_step:
                        if layer_attn is None:
                            continue
                        # Expected: (bs, heads, 1, seq_len)
                        if getattr(layer_attn, "ndim", None) != 4:
                            continue
                        a = layer_attn[0]  # (heads, 1, seq_len)
                        if int(a.shape[-1]) <= max(img_positions):
                            continue
                        vec = a[:, 0, img_positions]  # (heads, n_img1)
                        vec = vec.mean(dim=0)  # (n_img1,)
                        attn_accum = vec if attn_accum is None else (attn_accum + vec)
                        n_used += 1

            # Fallback: do a prompt-only forward pass if generation didn't provide attentions.
            if attn_accum is None or n_used == 0:
                _warn("No usable attentions from generate(); falling back to prompt-only forward pass.")
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
                    return response

                # Use the last prompt token as query.
                q_idx = int(prompt_len) - 1
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

            if attn_accum is None or n_used == 0:
                _warn("Could not reduce attentions into a vector; skipping viz.")
                return response

            attn_vec = (attn_accum / float(n_used)).detach().float().cpu()
            if int(attn_vec.numel()) != int(t_i * h_i * w_i):
                # Fallback to square reshape.
                hw = _infer_grid_hw(int(attn_vec.numel()))
                if hw is None:
                    _warn("Attention vector length does not match grid; skipping viz.")
                    return response
                t_i, h_i, w_i = 1, int(hw[0]), int(hw[1])
            attn_thw = attn_vec.view(int(t_i), int(h_i), int(w_i))
            heat_hw = attn_thw.mean(dim=0)  # (h, w)

            out_path = (
                viz_out_dir
                / _safe_name(dataset_name)
                / _safe_name(sample_id)
                / f"{_safe_name(task_name)}.png"
            )
            _save_overlay_png(rgb_pil=image, heatmap_hw=heat_hw.numpy(), out_path=out_path)
        except torch.cuda.OutOfMemoryError as e:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _warn(f"CUDA OOM while extracting attentions ({e}); continuing without viz.")
        except Exception as e:
            _warn(f"Failed to write visualization ({type(e).__name__}: {e}); continuing without viz.")
    
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
) -> Dict[str, str]:
    """Test all tasks on a single sample.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        sample: Dataset sample containing 'rgb_image'.

    Returns:
        Dictionary mapping task names to model responses.
    """

    from PIL import Image, UnidentifiedImageError

    rgb_path = sample.get("rgb_image")
    if rgb_path is not None:
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise OSError(f"Failed to open RGB image: {rgb_path} ({e})") from e
    else:
        rgb_array = sample.get("rgb")
        if rgb_array is None:
            raise FileNotFoundError("Input_RGBImage.png or Input_RGBImage.npy not found for sample.")
        if rgb_array.shape[0] in (1, 3, 4):
            rgb_array = rgb_array.transpose(1, 2, 0)
        rgb_image = Image.fromarray(rgb_array.astype("uint8"))

    mask_image = None
    mask_path = sample.get("mask_image")
    if mask_path is not None:
        try:
            mask_image = Image.open(mask_path).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            mask_image = None

    results: Dict[str, str] = {}
    for ann in annotations:
        task_name = str(ann.metadata.get("task", ann.task_type))
        print(f"  Running task: {task_name}...")
        response = run_task(
            model,
            processor,
            rgb_image,
            ann.question,
            mask=mask_image,
            dataset_name=dataset_name,
            sample_id=sample_id,
            task_name=task_name,
            viz_attn=viz_attn,
            viz_out_dir=viz_out_dir,
        )
        results[task_name] = response

    return results


def _safe_name(s: str) -> str:
    s2 = _SAFE_NAME_RE.sub("_", str(s).strip())
    s2 = s2.strip("_.")
    return s2 or "unknown"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-VL tasks on liquid datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["labliquidvision", "glasses", "dtld", "pouring", "ot2"],
        default=["labliquidvision"],
        help="One or more datasets to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset loading and standardized annotations without loading a model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Max samples per dataset to process (default: 1).",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        help=(
            "Override a dataset root, e.g. --data-root labliquidvision=/path/to/root. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help=(
            "If a dataset sample provides a mask image, pass it as a second image. "
            "Default: off (benchmark uses RGB only)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before selecting the first --max-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used with --shuffle (default: 0).",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="",
        help="Optional path to write per-task JSONL predictions with ground truth.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to write aggregated metrics as JSON.",
    )
    parser.add_argument(
        "--viz-attn",
        action="store_true",
        help=(
            "If set, extract attention weights and write blended heatmap overlays (PNG). "
            "This is a qualitative debug aid and may be slower / more memory intensive."
        ),
    )
    parser.add_argument(
        "--viz-out",
        type=str,
        default="",
        help=(
            "Output directory for attention visualizations (used with --viz-attn). "
            "Files are written under {viz_out}/{dataset}/{sample_id}/{task}.png"
        ),
    )
    parser.add_argument(
        "--tol",
        action="append",
        default=[],
        help=(
            "Override numeric tolerances for within-tolerance scoring, e.g. "
            "--tol fill_percentage=2.5 --tol fill_height_mm=10."
        ),
    )
    return parser.parse_args()


def _parse_tol_overrides(items: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in items:
        s = str(raw).strip()
        if not s or "=" not in s:
            raise ValueError(f"--tol must be task=value. Got: {raw!r}")
        task, val = s.split("=", 1)
        task = task.strip()
        val = val.strip()
        if not task or not val:
            raise ValueError(f"Invalid --tol value: {raw!r}")
        try:
            out[task] = float(val)
        except ValueError as e:
            raise ValueError(f"Invalid tolerance for task={task!r}: {val!r}") from e
    return out


def _parse_data_root_overrides(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for raw in items:
        s = str(raw).strip()
        if not s or "=" not in s:
            raise ValueError(
                "--data-root must be of the form dataset=/path. "
                f"Got: {raw!r}"
            )
        ds, path = s.split("=", 1)
        ds = ds.strip()
        path = path.strip()
        if not ds or not path:
            raise ValueError(f"Invalid --data-root value: {raw!r}")
        out[ds] = Path(path)
    return out


def build_dataset(dataset_name: str, data_root: Path, *, prompt_bank: PromptBank):
    if dataset_name == "labliquidvision":
        return LabLiquidVisionDataset(data_root, rgb_only=True, prompt_bank=prompt_bank)
    if dataset_name == "glasses":
        return GlassesInTheWildDataset(data_root, prompt_bank=prompt_bank)
    if dataset_name == "dtld":
        return DTLDDataset(data_root, rgb_only=True, prompt_bank=prompt_bank)
    if dataset_name == "pouring":
        return PouringDataset(data_root, splits=("test",), prompt_bank=prompt_bank)
    if dataset_name == "ot2":
        return OT2Dataset(data_root, prompt_bank=prompt_bank)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def print_ground_truth(sample: Dict[str, Any]) -> None:
    print("\nGround Truth (standardized annotations):")

    annotations = sample.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        print("  (no standardized annotations)")
        return

    for ann in annotations:
        if not isinstance(ann, TLGAnnotation):
            continue
        task_key = str(ann.metadata.get("task", ann.task_type))
        print(f"  - {task_key}: {ann.answer}")


def _validate_annotations(sample: Dict[str, Any]) -> Optional[str]:
    annotations = sample.get("annotations")
    if not isinstance(annotations, list):
        return "sample['annotations'] is not a list"
    for i, ann in enumerate(annotations):
        if not isinstance(ann, TLGAnnotation):
            return f"annotations[{i}] is not a TLGAnnotation"
        if not isinstance(ann.question, str) or not ann.question.strip():
            return f"annotations[{i}] has empty question"
        if not isinstance(ann.task_type, str) or not ann.task_type.strip():
            return f"annotations[{i}] has empty task_type"
        task = ann.metadata.get("task")
        if not isinstance(task, str) or not task.strip():
            return f"annotations[{i}] missing metadata['task']"
    return None


def main():
    """Run all VLM tests."""
    args = parse_args()
    dataset_names = args.datasets

    tol_overrides = _parse_tol_overrides(args.tol)

    data_root_overrides = _parse_data_root_overrides(args.data_root)
    prompt_bank = PromptBank.default()

    if args.dry_run:
        print("Dry run: validating dataset loading + standardized annotations (no model).")

    viz_attn = bool(args.viz_attn) and (not args.dry_run)
    viz_out_dir = Path(args.viz_out) if args.viz_out else None
    if viz_attn and viz_out_dir is None:
        raise ValueError("--viz-attn requires --viz-out")

    model = processor = None
    if not args.dry_run:
        model, processor = load_model(viz_attn=viz_attn)

    pred_out_path = Path(args.predictions_out) if args.predictions_out else None
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else None
    pred_fh = None
    if pred_out_path is not None:
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = pred_out_path.open("w", encoding="utf-8")

    overall_stats: Dict[str, TaskStats] = {}
    per_dataset_stats: Dict[str, Dict[str, TaskStats]] = {}

    for dataset_name in dataset_names:
        data_root = data_root_overrides.get(dataset_name, DATA_ROOTS[dataset_name])

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"Loading dataset from: {data_root}")

        if not data_root.exists():
            print("Dataset root does not exist; skipping.")
            continue

        dataset = build_dataset(dataset_name, data_root, prompt_bank=prompt_bank)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("Error: Dataset is empty!")
            continue

        max_samples = max(1, int(args.max_samples))
        indices = list(range(len(dataset)))
        if args.shuffle:
            rng = random.Random(int(args.seed))
            rng.shuffle(indices)
        indices = indices[: min(len(indices), max_samples)]
        for j, idx0 in enumerate(indices):
            idx = j + 1
            try:
                sample = dataset[idx0]
            except (FileNotFoundError, OSError, SyntaxError) as e:
                print(f"Failed to load sample index={idx0}: {e}")
                # In dry-run we don't want a missing file to crash the whole script.
                if args.dry_run:
                    break
                raise

            print("\n" + "-" * 60)
            print(f"Sample {idx}/{len(dataset)}")
            print(f"Sample ID: {sample.get('id', 'unknown')}")

            err = _validate_annotations(sample)
            if err is not None:
                print(f"Annotation validation failed: {err}")
                continue

            print_ground_truth(sample)

            annotations = sample.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                print("  No standardized annotations available for this sample; skipping.")
                continue

            if args.dry_run:
                print("\nDry run OK: annotations are well-formed.")
                continue

            # Optionally pass a mask image as a second input.
            if not args.use_mask:
                sample_no_mask = dict(sample)
                sample_no_mask.pop("mask_image", None)
                sample = sample_no_mask

            try:
                results = test_sample(
                    model,
                    processor,
                    sample,
                    annotations,
                    dataset_name=dataset_name,
                    sample_id=str(sample.get("id", "unknown")),
                    viz_attn=viz_attn,
                    viz_out_dir=viz_out_dir,
                )
            except (FileNotFoundError, OSError, SyntaxError) as e:
                print(f"Failed to load images for sample index={idx0}: {e}")
                continue
            ds_stats = per_dataset_stats.setdefault(dataset_name, {})
            for ann in annotations:
                task = str(ann.metadata.get("task", ann.task_type))
                resp = results.get(task)
                if resp is None:
                    continue

                rec = score_prediction(ann, resp, tol_overrides=tol_overrides)

                # Update stats.
                for stats_map in (overall_stats, ds_stats):
                    st = stats_map.setdefault(task, TaskStats())
                    st.update_from_record(rec)

                # Write JSONL record.
                if pred_fh is not None:
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sample.get("id", "unknown"),
                        "image_id": ann.image_id,
                        "task": rec.get("task"),
                        "task_type": rec.get("task_type"),
                        "question": rec.get("question"),
                        "gt": rec.get("gt"),
                        "pred": rec.get("pred"),
                        "pred_raw": rec.get("pred_raw"),
                        "valid": rec.get("valid"),
                        "correct": rec.get("correct"),
                    }
                    if "abs_error" in rec:
                        row["abs_error"] = rec["abs_error"]
                    if "within_tol" in rec:
                        row["within_tol"] = rec["within_tol"]
                    pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            print("\nPredictions:")
            for task_name, response in results.items():
                print(f"  {task_name}: {response}")

    # Close prediction writer.
    if pred_fh is not None:
        pred_fh.close()

    # Print aggregated metrics.
    def _print_stats(title: str, stats: Dict[str, TaskStats]) -> None:
        if not stats:
            return
        print("\n" + "=" * 60)
        print(title)
        for task in sorted(stats.keys()):
            d = stats[task].to_dict()
            n_valid = d.get("n_valid", 0)
            n = d.get("n", 0)
            acc = d.get("accuracy")
            mae = d.get("mae")
            within = d.get("within_tol")
            parts = [f"n={n}", f"valid={n_valid}"]
            if isinstance(acc, float):
                parts.append(f"acc={acc:.3f}")
            if isinstance(mae, float):
                parts.append(f"mae={mae:.3f}")
            if isinstance(within, float):
                parts.append(f"within_tol={within:.3f}")
            print(f"  - {task}: " + ", ".join(parts))

    _print_stats("Overall Metrics", overall_stats)
    for ds in sorted(per_dataset_stats.keys()):
        _print_stats(f"Metrics for dataset={ds}", per_dataset_stats[ds])

    if metrics_out_path is not None:
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": MODEL_NAME,
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "datasets": list(dataset_names),
            "max_samples": int(args.max_samples),
            "use_mask": bool(args.use_mask),
            "overall": {k: v.to_dict() for k, v in overall_stats.items()},
            "per_dataset": {
                ds: {k: v.to_dict() for k, v in stats.items()} for ds, stats in per_dataset_stats.items()
            },
        }
        metrics_out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
