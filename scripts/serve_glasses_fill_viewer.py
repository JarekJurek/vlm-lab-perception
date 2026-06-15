#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw


KEYPOINT_COLORS = {
    "top_front": (40, 170, 255),
    "bottom_front": (255, 170, 40),
    "fluid_level": (40, 220, 100),
}
MAX_IMAGE_SIDE = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local viewer for low-fill glasses samples.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/glasses_in_the_wild/glasses_in_the_wild_1000"),
        help="Glasses in the Wild dataset root.",
    )
    parser.add_argument("--max-fill", type=float, default=21.0, help="Show samples with fill_percentage < this value.")
    parser.add_argument("--min-fill", type=float, default=0.0, help="Optional lower bound, inclusive.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8771, help="Port to bind.")
    return parser.parse_args()


class ViewerState:
    def __init__(self, root: Path, min_fill: float, max_fill: float) -> None:
        self.root = root
        self.images_dir = root / "images"
        self.min_fill = min_fill
        self.max_fill = max_fill
        self.all_records = load_records(root)
        self.records = [
            rec
            for rec in self.all_records
            if min_fill <= float(rec["fill_percentage"]) < max_fill
        ]

    @lru_cache(maxsize=512)
    def image_bytes(self, idx: int, overlay: bool) -> bytes:
        rec = self.records[idx]
        with Image.open(self.images_dir / str(rec["file_name"])) as image:
            out = image.convert("RGB")
        if overlay:
            out = draw_overlay(out, rec["keypoints"])
        out = resize_for_viewer(out)
        return jpeg_bytes(out)


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
                self._send_json(
                    {
                        "min_fill": self.state.min_fill,
                        "max_fill": self.state.max_fill,
                        "total": len(self.state.all_records),
                        "count": len(self.state.records),
                        "records": strip_keypoints(self.state.records),
                    }
                )
            elif parsed.path == "/image":
                qs = parse_qs(parsed.query)
                idx = parse_idx(qs, len(self.state.records))
                overlay = qs.get("overlay", ["1"])[0] != "0"
                self._send_bytes(self.state.image_bytes(idx, overlay), "image/jpeg")
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
    state = ViewerState(args.root, args.min_fill, args.max_fill)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(
        f"Serving {len(state.records)} / {len(state.all_records)} glasses samples "
        f"with {args.min_fill:g}% <= fill < {args.max_fill:g}% at {url}"
    )
    print("Use left/right arrow keys to move through samples. Ctrl-C stops the server.")
    server.serve_forever()


def load_records(root: Path) -> List[Dict[str, Any]]:
    annotations_path = root / "images" / "annotations.json"
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    keypoint_names = data["categories"][0]["keypoints"]
    images = {int(img["id"]): img for img in data["images"]}
    records: List[Dict[str, Any]] = []

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        image = images.get(image_id)
        if image is None:
            continue
        keypoints = named_keypoints(ann.get("keypoints", []), keypoint_names)
        fill = estimate_fill_percentage(keypoints)
        if fill is None:
            continue
        records.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "fill_percentage": fill,
                "ordinal_21_threshold": ordinal(fill, empty_threshold=21.0),
                "ordinal_1_threshold": ordinal(fill, empty_threshold=1.0),
                "keypoints": keypoints,
            }
        )

    records.sort(key=lambda rec: (rec["fill_percentage"], rec["image_id"]))
    return records


def named_keypoints(raw: List[float], names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        base = i * 3
        if base + 2 >= len(raw):
            break
        out[name] = {
            "x": float(raw[base]),
            "y": float(raw[base + 1]),
            "v": float(raw[base + 2]),
        }
    return out


def estimate_fill_percentage(keypoints: Dict[str, Dict[str, float]]) -> Optional[float]:
    bottom = keypoints.get("bottom_front")
    top = keypoints.get("top_front")
    fluid = keypoints.get("fluid_level")
    if bottom is None or top is None or fluid is None:
        return None
    if bottom["v"] <= 0 or top["v"] <= 0 or fluid["v"] <= 0:
        return None
    height = bottom["y"] - top["y"]
    if abs(height) < 1e-6:
        return None
    return max(0.0, min(1.0, (bottom["y"] - fluid["y"]) / height)) * 100.0


def ordinal(fill: float, *, empty_threshold: float) -> str:
    p = max(0.0, min(100.0, float(fill)))
    if p <= empty_threshold:
        return "EMPTY"
    if p < 50.0:
        return "LOW"
    if p < 75.0:
        return "MID"
    return "HIGH"


def strip_keypoints(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "index": idx,
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "fill_percentage": rec["fill_percentage"],
            "ordinal_21_threshold": rec["ordinal_21_threshold"],
            "ordinal_1_threshold": rec["ordinal_1_threshold"],
        }
        for idx, rec in enumerate(records)
    ]


def parse_idx(qs: Dict[str, List[str]], length: int) -> int:
    if length <= 0:
        raise IndexError("No matching records")
    idx = int(qs.get("idx", ["0"])[0])
    if idx < 0 or idx >= length:
        raise IndexError(f"idx out of range: {idx}")
    return idx


def draw_overlay(image: Image.Image, keypoints: Dict[str, Dict[str, float]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    line_width = max(2, round(min(width, height) / 150))
    radius = max(4, round(min(width, height) / 60))

    for name in ("top_front", "bottom_front", "fluid_level"):
        kp = keypoints.get(name)
        if not kp or kp["v"] <= 0:
            continue
        x, y = kp["x"], kp["y"]
        color = KEYPOINT_COLORS[name]
        draw.line([(0, y), (width, y)], fill=(*color, 140), width=line_width)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 230),
            outline=(0, 0, 0, 230),
            width=max(1, line_width // 2),
        )

    top = keypoints.get("top_front")
    bottom = keypoints.get("bottom_front")
    fluid = keypoints.get("fluid_level")
    if top and bottom and fluid:
        x = max(8.0, min(width - 8.0, (top["x"] + bottom["x"] + fluid["x"]) / 3.0))
        draw.line([(x, top["y"]), (x, bottom["y"])], fill=(255, 255, 255, 180), width=line_width)
        draw.line([(x - 12, fluid["y"]), (x + 12, fluid["y"])], fill=(40, 220, 100, 255), width=line_width + 1)

    return out


def resize_for_viewer(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((round(width * scale), round(height * scale)))


def jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Glasses Fill Viewer</title>
  <style>
    :root { color-scheme: light; --border: #d6d6d6; --text: #181818; --muted: #666; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: #f4f4f1; }
    main { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
    aside { border-right: 1px solid var(--border); background: #fff; overflow: hidden; display: flex; flex-direction: column; }
    header { padding: 14px 16px; border-bottom: 1px solid var(--border); }
    h1 { font-size: 18px; line-height: 1.2; margin: 0 0 10px; }
    .meta { font-size: 13px; color: var(--muted); line-height: 1.4; }
    .legend { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; font-size: 12px; }
    .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 4px; border-radius: 50%; vertical-align: -1px; }
    .top::before { background: rgb(40, 170, 255); }
    .bottom::before { background: rgb(255, 170, 40); }
    .fluid::before { background: rgb(40, 220, 100); }
    .list { overflow: auto; padding: 8px; }
    button.row { width: 100%; text-align: left; border: 1px solid transparent; background: transparent; padding: 8px; border-radius: 6px; cursor: pointer; }
    button.row:hover { background: #f2f2ef; }
    button.row.active { border-color: #222; background: #ececea; }
    .row strong { display: block; font-size: 13px; }
    .row span { display: block; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    section { min-width: 0; display: grid; grid-template-rows: auto 1fr; }
    .toolbar { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--border); background: #fff; }
    .toolbar button { border: 1px solid var(--border); background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    .toolbar button:hover { background: #f2f2ef; }
    .toolbar label { font-size: 13px; color: var(--muted); display: inline-flex; align-items: center; gap: 6px; }
    .spacer { flex: 1; }
    .stage { overflow: auto; padding: 18px; }
    .sample { max-width: 1280px; margin: 0 auto; }
    .sample-title { margin-bottom: 10px; display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: end; }
    .sample-title h2 { margin: 0; font-size: 20px; }
    .sample-title .stats { text-align: right; color: var(--muted); font-size: 13px; }
    img { max-width: 100%; height: auto; display: block; background: #ddd; }
    .empty { padding: 24px; color: var(--muted); }
  </style>
</head>
<body>
<main>
  <aside>
    <header>
      <h1>Glasses Fill Viewer</h1>
      <div class="meta" id="summary">Loading...</div>
      <div class="legend">
        <span class="top">top_front</span>
        <span class="bottom">bottom_front</span>
        <span class="fluid">fluid_level</span>
      </div>
    </header>
    <div class="list" id="list"></div>
  </aside>
  <section>
    <div class="toolbar">
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <label><input type="checkbox" id="overlay" checked> overlay</label>
      <div class="spacer"></div>
      <span class="meta">Keys: left/right</span>
    </div>
    <div class="stage">
      <div class="sample" id="sample"></div>
    </div>
  </section>
</main>
<script>
let records = [];
let idx = 0;
let overlay = true;

async function init() {
  const payload = await fetch('/api/samples').then(r => r.json());
  records = payload.records;
  document.getElementById('summary').textContent =
    `${payload.count} / ${payload.total} samples with ${payload.min_fill}% <= fill < ${payload.max_fill}%`;
  renderList();
  renderSample();
}

function renderList() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  records.forEach((rec, i) => {
    const b = document.createElement('button');
    b.className = 'row' + (i === idx ? ' active' : '');
    b.innerHTML = `<strong>#${rec.image_id} - ${rec.fill_percentage.toFixed(2)}%</strong>
      <span>21%: ${rec.ordinal_21_threshold} | 1%: ${rec.ordinal_1_threshold}</span>
      <span>${rec.file_name}</span>`;
    b.onclick = () => { idx = i; renderSample(); renderList(); };
    list.appendChild(b);
  });
}

function renderSample() {
  const sample = document.getElementById('sample');
  if (!records.length) {
    sample.innerHTML = '<div class="empty">No samples match this fill range.</div>';
    return;
  }
  const rec = records[idx];
  sample.innerHTML = `<div class="sample-title">
    <h2>#${rec.image_id} - ${rec.fill_percentage.toFixed(3)}%</h2>
    <div class="stats">${idx + 1} / ${records.length}<br>21%: ${rec.ordinal_21_threshold} | 1%: ${rec.ordinal_1_threshold}</div>
  </div>
  <img src="/image?idx=${idx}&overlay=${overlay ? 1 : 0}&t=${Date.now()}" alt="">
  <p class="meta">${rec.file_name}</p>`;
  const active = document.querySelector('.row.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function move(delta) {
  if (!records.length) return;
  idx = Math.max(0, Math.min(records.length - 1, idx + delta));
  renderList();
  renderSample();
}

document.getElementById('prev').onclick = () => move(-1);
document.getElementById('next').onclick = () => move(1);
document.getElementById('overlay').onchange = (ev) => { overlay = ev.target.checked; renderSample(); };
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft') move(-1);
  if (ev.key === 'ArrowRight') move(1);
});
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
