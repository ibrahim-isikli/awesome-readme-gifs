#!/usr/bin/env python3
"""
Turn a static pixel-art PNG into a small looping "cinemagraph" GIF:
breathing zoom + pulsing bloom on bright pixels + twinkling sparkles.

No external video-gen service needed — pure PIL/numpy + ffmpeg for encoding.

Usage:
    python3 scripts/pixel_to_gif.py INPUT.png OUTPUT.gif [--width 480] [--frames 30] [--fps 15]
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def seed_for(path: Path) -> int:
    return int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)


def make_bloom(frame_rgb: np.ndarray, blur_radius: float, thresh: float = 195.0) -> np.ndarray:
    lum = frame_rgb[..., 0] * 0.299 + frame_rgb[..., 1] * 0.587 + frame_rgb[..., 2] * 0.114
    mask = np.clip((lum - thresh) / (255.0 - thresh), 0.0, 1.0) ** 2
    bright = frame_rgb * mask[..., None]
    bloom_img = Image.fromarray(bright.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_radius))
    return np.asarray(bloom_img, dtype=np.float32)


def sparkle_positions(base_rgb: np.ndarray, count: int, rng: np.random.Generator):
    h, w, _ = base_rgb.shape
    lum = base_rgb[..., 0] * 0.299 + base_rgb[..., 1] * 0.587 + base_rgb[..., 2] * 0.114
    weights = np.clip(255.0 - lum, 1.0, 255.0) ** 2
    weights = weights.flatten()
    weights /= weights.sum()
    idx = rng.choice(weights.size, size=count, replace=False, p=weights)
    ys, xs = np.unravel_index(idx, (h, w))
    sparkles = []
    for y, x in zip(ys, xs):
        sparkles.append({
            "x": int(x),
            "y": int(y),
            "cycles": int(rng.integers(1, 3)),
            "phase": float(rng.random()),
            "size": float(rng.uniform(1.6, 3.2)),
        })
    return sparkles


def stamp_sparkle(frame: np.ndarray, x: int, y: int, radius: float, brightness: float):
    if brightness <= 0.02:
        return
    r = max(2, int(round(radius * 2)))
    h, w, _ = frame.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (yy - y) ** 2 + (xx - x) ** 2
    glow = np.exp(-dist2 / (2 * radius * radius)) * brightness * 255.0
    patch = frame[y0:y1, x0:x1, :]
    for c in range(3):
        patch[..., c] = np.clip(patch[..., c] + glow, 0, 255)


def generate_frames(src_path: Path, work_w: int, n_frames: int, zoom_max: float,
                     sparkle_count: int, bloom_strength: float, out_dir: Path):
    rng = np.random.default_rng(seed_for(src_path))
    im = Image.open(src_path).convert("RGB")
    work_h = round(im.height * work_w / im.width)

    supersample_w = round(work_w * zoom_max)
    supersample_h = round(work_h * zoom_max)
    super_im = im.resize((supersample_w, supersample_h), Image.LANCZOS)

    base_im = im.resize((work_w, work_h), Image.LANCZOS)
    base_rgb = np.asarray(base_im, dtype=np.float32)
    sparkles = sparkle_positions(base_rgb, sparkle_count, rng)

    blur_radius = max(2.0, work_w * 0.015)
    paths = []
    for t in range(n_frames):
        phase = t / n_frames
        s = 1.0 + (zoom_max - 1.0) * 0.5 * (1.0 - np.cos(2 * np.pi * phase))
        cur_w, cur_h = round(work_w * s), round(work_h * s)
        cur_w = min(cur_w, supersample_w)
        cur_h = min(cur_h, supersample_h)
        frame_im = super_im.resize((cur_w, cur_h), Image.LANCZOS)
        left = (cur_w - work_w) // 2
        top = (cur_h - work_h) // 2
        frame_im = frame_im.crop((left, top, left + work_w, top + work_h))
        frame = np.asarray(frame_im, dtype=np.float32).copy()

        bloom = make_bloom(frame, blur_radius)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * phase + np.pi / 2)
        frame = np.clip(frame + bloom * pulse * bloom_strength, 0, 255)

        for sp in sparkles:
            b = max(0.0, np.sin(2 * np.pi * (phase * sp["cycles"] + sp["phase"]))) ** 3
            stamp_sparkle(frame, sp["x"], sp["y"], sp["size"], b)

        out_path = out_dir / f"f{t:03d}.png"
        Image.fromarray(frame.astype(np.uint8)).save(out_path)
        paths.append(out_path)
    return paths


def encode_gif(frame_dir: Path, out_path: Path, fps: int):
    palette = frame_dir / "palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "f%03d.png"),
         "-vf", "palettegen=stats_mode=diff", str(palette)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "f%03d.png"),
         "-i", str(palette),
         "-lavfi", "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
         "-loop", "0", str(out_path)],
        check=True, capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--zoom", type=float, default=1.035)
    ap.add_argument("--sparkles", type=int, default=14)
    ap.add_argument("--bloom", type=float, default=0.55)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        generate_frames(args.input, args.width, args.frames, args.zoom,
                         args.sparkles, args.bloom, tmp_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encode_gif(tmp_dir, args.output, args.fps)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
