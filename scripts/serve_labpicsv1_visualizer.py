#!/usr/bin/env python3
"""Serve a live LabPicsV1 task visualizer.

Images are loaded on demand through ``LabPicsV1Dataset`` and rendered with the
same task-input path used by inference for highlighted-region tasks.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.labpicsv1 import LabPicsV1Dataset  # noqa: E402
from tlg.prompts import PromptBank  # noqa: E402
from tlg.sdvlm_runner import build_sdvlm_task_inputs  # noqa: E402


TASK_GROUNDING = "liquid_grounding"
TASK_HIGHLIGHTED = "highlighted_region_type"
MAX_IMAGE_SIDE = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live LabPicsV1 task visualizer.")
    parser.add_argument("--data-root", type=Path, default=Path("data/LabPicsV1"), help="LabPicsV1 root.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="LabPicsV1 Simple split.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser.parse_args()


class ViewerState:
    def __init__(self, data_root: Path, split: str) -> None:
        self.prompt_bank = PromptBank.default()
        self.dataset = LabPicsV1Dataset(data_root, split=split, prompt_bank=self.prompt_bank)
        self.records = self._build_records()

    def _build_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        entries = getattr(self.dataset, "_entries", None)
        if not isinstance(entries, list):
            entries = [None] * len(self.dataset)

        for idx, entry in enumerate(entries):
            split = ""
            stem = ""
            if isinstance(entry, tuple) and len(entry) >= 2:
                split = str(entry[0])
                stem = str(entry[1])
            records.append(
                {
                    "index": idx,
                    "sample_id": f"simple/{split}/{stem}" if split and stem else str(idx),
                    "image_name": stem,
                }
            )
        return records

    @lru_cache(maxsize=256)
    def image_bytes(self, idx: int, mode: str) -> bytes:
        sample = self.dataset[idx]
        anns = _annotations_by_task(sample)

        if mode == "highlighted":
            ann = anns.get(TASK_HIGHLIGHTED)
            if ann is None:
                image = _open_sample_image(sample)
            else:
                task_inputs = build_sdvlm_task_inputs(
                    sample,
                    [ann],
                    dataset_name="labpicsv1",
                    sample_id=str(sample["id"]),
                )
                image = task_inputs[0].image if task_inputs else _open_sample_image(sample)
        elif mode == "grounding":
            ann = anns.get(TASK_GROUNDING)
            image = _draw_grounding_overlay(_open_sample_image(sample), ann.answer if ann is not None else [])
        else:
            image = _open_sample_image(sample)

        return _jpeg_bytes(_resize_for_viewer(image.convert("RGB")))

    def prompt_payload(self, idx: int, mode: str) -> Dict[str, Any]:
        sample = self.dataset[idx]
        anns = _annotations_by_task(sample)
        task = TASK_HIGHLIGHTED if mode == "highlighted" else TASK_GROUNDING
        ann = anns.get(task)
        return {
            "sample_id": str(sample["id"]),
            "task": task,
            "answer": ann.answer if ann is not None else None,
            "metadata": dict(ann.metadata) if ann is not None else {},
            "question": ann.question if ann is not None else "",
            "grounding_region_count": _region_count(anns.get(TASK_GROUNDING).answer)
            if anns.get(TASK_GROUNDING) is not None
            else None,
            "highlighted_label": anns.get(TASK_HIGHLIGHTED).answer
            if anns.get(TASK_HIGHLIGHTED) is not None
            else None,
            "material_types": anns.get("material_types_present").answer
            if anns.get("material_types_present") is not None
            else None,
        }


class Handler(BaseHTTPRequestHandler):
    state: ViewerState

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            elif parsed.path == "/api/samples":
                self._send_json(self.state.records)
            elif parsed.path == "/api/prompt":
                qs = parse_qs(parsed.query)
                idx = _parse_idx(qs, len(self.state.records))
                mode = qs.get("mode", ["grounding"])[0]
                self._send_json(self.state.prompt_payload(idx, mode))
            elif parsed.path == "/image":
                qs = parse_qs(parsed.query)
                idx = _parse_idx(qs, len(self.state.records))
                mode = qs.get("mode", ["grounding"])[0]
                self._send_bytes(self.state.image_bytes(idx, mode), "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, explain=str(exc))

    def _send_json(self, payload: object) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_text(self, payload: str, content_type: str) -> None:
        self._send_bytes(payload.encode("utf-8"), content_type)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    args = parse_args()
    state = ViewerState(args.data_root, args.split)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {len(state.records)} LabPicsV1 samples at {url}")
    print("Images are rendered live through the dataloader.")
    server.serve_forever()


def _annotations_by_task(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {str(ann.metadata.get("task", ann.task_type)): ann for ann in sample.get("annotations", [])}


def _parse_idx(qs: Dict[str, List[str]], length: int) -> int:
    raw = qs.get("idx", ["0"])[0]
    idx = int(raw)
    if idx < 0 or idx >= length:
        raise IndexError(f"idx out of range: {idx}")
    return idx


def _open_sample_image(sample: Dict[str, Any]) -> Image.Image:
    path = sample.get("rgb_image")
    if path is None:
        raise FileNotFoundError(f"Sample has no rgb_image path: {sample.get('id')}")
    return Image.open(path).convert("RGB")


def _draw_grounding_overlay(image: Image.Image, gt: object) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if not isinstance(gt, list):
        return out

    width, height = out.size
    box_width = max(3, min(width, height) // 180)
    for i, item in enumerate(gt, start=1):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        xyxy = [
            round(x1 * (width - 1) / 1000.0),
            round(y1 * (height - 1) / 1000.0),
            round(x2 * (width - 1) / 1000.0),
            round(y2 * (height - 1) / 1000.0),
        ]
        draw.rectangle(xyxy, outline=(0, 220, 80), width=box_width)
        draw.text((xyxy[0] + 4, xyxy[1] + 4), f"Liquid {i}", fill=(0, 220, 80))
    return out


def _resize_for_viewer(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = min(1.0, MAX_IMAGE_SIDE / float(max(width, height)))
    if scale >= 1.0:
        return image
    return image.resize((int(round(width * scale)), int(round(height * scale))), Image.Resampling.LANCZOS)


def _jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _region_count(value: object) -> Optional[int]:
    return len(value) if isinstance(value, list) else None


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LabPicsV1 Live Visualizer</title>
  <style>
    :root { --border: #d8dde6; --text: #111827; --muted: #687083; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 15px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--text); background: #eef0f4; }
    header { height: 58px; padding: 10px 14px; border-bottom: 1px solid var(--border); background: white; display: flex; align-items: center; gap: 10px; }
    h1 { font-size: 16px; margin: 0 8px 0 0; font-weight: 650; }
    button, input { height: 36px; border: 1px solid var(--border); background: #fff; color: var(--text); border-radius: 6px; padding: 0 10px; font: inherit; }
    button { cursor: pointer; min-width: 38px; }
    button.active { background: #172033; color: white; border-color: #172033; }
    input { width: 92px; }
    .spacer { flex: 1; min-width: 20px; }
    .meta { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    main { height: calc(100vh - 58px); display: flex; align-items: center; justify-content: center; padding: 12px; }
    img { display: block; max-width: 100%; max-height: 100%; object-fit: contain; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.18); }
  </style>
</head>
<body>
  <header>
    <h1>LabPicsV1</h1>
    <button id="prev" title="Previous sample">&#8592;</button>
    <button id="next" title="Next sample">&#8594;</button>
    <input id="idx" type="number" min="0" value="0">
    <button id="go">Go</button>
    <button id="mode-raw" class="active">Raw</button>
    <button id="mode-grounding">Grounding GT</button>
    <button id="mode-highlighted">Highlighted Input</button>
    <div class="spacer"></div>
    <div id="meta" class="meta"></div>
  </header>
  <main>
    <img id="image" src="/image?idx=0&mode=raw" alt="LabPicsV1 sample">
  </main>
  <script>
    let records = [];
    let idx = 0;
    let mode = "raw";
    const img = document.getElementById("image");
    const idxInput = document.getElementById("idx");
    const meta = document.getElementById("meta");
    const buttons = {
      raw: document.getElementById("mode-raw"),
      grounding: document.getElementById("mode-grounding"),
      highlighted: document.getElementById("mode-highlighted"),
    };

    fetch("/api/samples").then(r => r.json()).then(data => {
      records = data;
      render();
    });

    function render() {
      idx = Math.max(0, Math.min(idx, Math.max(0, records.length - 1)));
      idxInput.value = idx;
      img.src = `/image?idx=${idx}&mode=${mode}&t=${Date.now()}`;
      const record = records[idx] || {};
      meta.textContent = `#${idx} / ${Math.max(0, records.length - 1)} · ${record.sample_id || ""}`;
      for (const [name, button] of Object.entries(buttons)) {
        button.classList.toggle("active", name === mode);
      }
      fetch(`/api/prompt?idx=${idx}&mode=${mode}`).then(r => r.json()).then(payload => {
        if (payload.highlighted_label || payload.grounding_region_count !== null) {
          meta.textContent += ` · boxes ${payload.grounding_region_count ?? "none"} · highlighted ${payload.highlighted_label ?? "none"}`;
        }
      });
    }

    document.getElementById("prev").onclick = () => { idx -= 1; render(); };
    document.getElementById("next").onclick = () => { idx += 1; render(); };
    document.getElementById("go").onclick = () => { idx = Number(idxInput.value || 0); render(); };
    buttons.raw.onclick = () => { mode = "raw"; render(); };
    buttons.grounding.onclick = () => { mode = "grounding"; render(); };
    buttons.highlighted.onclick = () => { mode = "highlighted"; render(); };
    window.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowLeft") { idx -= 1; render(); }
      if (e.key === "ArrowRight") { idx += 1; render(); }
      if (e.key === "r") { mode = "raw"; render(); }
      if (e.key === "g") { mode = "grounding"; render(); }
      if (e.key === "h") { mode = "highlighted"; render(); }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
