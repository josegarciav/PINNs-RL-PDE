#!/usr/bin/env python3
"""
panel_tighten.py

Utility for removing excessive horizontal whitespace from a 3-panel figure.
The script detects the three panels, crops them, enlarges each by a user-defined
scale factor, and re-assembles them with a uniform gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------.
# Core image utilities
# -----------------------------------------------------------------------------.
def load_image(path: Path) -> np.ndarray:
    """Load an image file into an (H, W, 3) uint8 NumPy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_image(array: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) uint8 array as PNG."""
    Image.fromarray(array).save(path, format="PNG")


# -----------------------------------------------------------------------------.
# Panel detection & cropping
# -----------------------------------------------------------------------------.
def _column_activity(mask: np.ndarray) -> np.ndarray:
    """
    For every column, count how many non-background pixels appear.
    mask is True where the pixel is *not* background.
    """
    return mask.sum(axis=0)  # shape (W,)


def _find_gaps(activity: np.ndarray, min_gap: int = 30) -> List[Tuple[int, int]]:
    """
    Identify contiguous stretches of zero activity (pure background) that are at
    least min_gap pixels wide.  Returns a list of (start, end) indices.
    """
    inactive = activity == 0
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, is_empty in enumerate(inactive):
        if is_empty and not in_gap:
            in_gap = True
            start = i
        elif not is_empty and in_gap:
            in_gap = False
            end = i
            if end - start >= min_gap:
                gaps.append((start, end))
    # trailing gap
    if in_gap and len(activity) - start >= min_gap:
        gaps.append((start, len(activity)))
    return gaps


def detect_panels(
    img: np.ndarray, white_thr: int = 240, min_gap: int = 30
) -> List[Tuple[int, int, int, int]]:
    """
    Detect three horizontal panels in img.

    Returns
    -------
    List[(x0, y0, x1, y1)]  bounding boxes for panels, left→right order.
    """
    # Background heuristic: nearly white pixels
    bg_mask = (img > white_thr).all(axis=2)
    fg_mask = ~bg_mask

    col_act = _column_activity(fg_mask)
    gaps = _find_gaps(col_act, min_gap=min_gap)

    # Build panel x-ranges = [prev_gap_end, next_gap_start)
    x_starts = [0] + [g[1] for g in gaps]
    x_ends = [g[0] for g in gaps] + [img.shape[1]]
    boxes = []
    for x0, x1 in zip(x_starts, x_ends):
        if x1 - x0 < 10:  # ignore degenerate slices
            continue
        # vertical extent: take rows containing any fg pixel in slice
        slice_mask = fg_mask[:, x0:x1]
        rows = np.where(slice_mask.any(axis=1))[0]
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        boxes.append((int(x0), y0, int(x1), y1))

    # Sort left→right
    boxes.sort(key=lambda b: b[0])
    if len(boxes) != 3:
        raise RuntimeError(
            f"Expected 3 panels, found {len(boxes)}. "
            "Try relaxing 'white_thr' or 'min_gap'."
        )
    return boxes


# -----------------------------------------------------------------------------.
# Rescaling & composition
# -----------------------------------------------------------------------------.
def rescale_panel(panel: np.ndarray, scale: float) -> np.ndarray:
    """Return a resized copy of panel (bilinear interpolation)."""
    h, w = panel.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return np.asarray(Image.fromarray(panel).resize(new_size, Image.BILINEAR))


def compose_panels(panels: List[np.ndarray], gap: int = 20) -> np.ndarray:
    """
    Horizontally concatenate panels with a uniform white gap.
    Assumes all panels share the same height.
    """
    # Ensure common height
    heights = [p.shape[0] for p in panels]
    target_h = max(heights)
    aligned = []
    for p in panels:
        if p.shape[0] != target_h:
            # add white padding top & bottom
            padding = target_h - p.shape[0]
            pad_top = padding // 2
            pad_bot = padding - pad_top
            aligned.append(
                np.pad(p, ((pad_top, pad_bot), (0, 0), (0, 0)), constant_values=255)
            )
        else:
            aligned.append(p)
    gap_arr = 255 * np.ones((target_h, gap, 3), dtype=np.uint8)
    out = aligned[0]
    for nxt in aligned[1:]:
        out = np.concatenate([out, gap_arr, nxt], axis=1)
    return out


# -----------------------------------------------------------------------------.
# CLI glue
# -----------------------------------------------------------------------------.
def process(
    input_path: Path,
    output_path: Path,
    scale: float = 1.0,
    gap: int = 20,
    white_thr: int = 240,
    min_gap: int = 30,
) -> None:
    """High-level orchestration."""
    img = load_image(input_path)
    boxes = detect_panels(img, white_thr=white_thr, min_gap=min_gap)
    panels = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    panels = [rescale_panel(p, scale) for p in panels]
    out = compose_panels(panels, gap=gap)
    save_image(out, output_path)
    print(f"Saved cleaned figure → {output_path}")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tighten 3-panel figure spacing.")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to original image (PNG/JPG).",
    )
    p.add_argument(
        "--output", "-o", type=Path, required=True, help="Output image path (PNG)."
    )
    p.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Multiplicative scale factor for each panel.",
    )
    p.add_argument(
        "--gap",
        "-g",
        type=int,
        default=20,
        help="Pixel gap between panels in final image.",
    )
    p.add_argument(
        "--white_thr",
        type=int,
        default=240,
        help="Pixel value (>thr) treated as background.",
    )
    p.add_argument(
        "--min_gap",
        type=int,
        default=30,
        help="Minimum width (px) to treat a vertical band as a gap.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    process(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        gap=args.gap,
        white_thr=args.white_thr,
        min_gap=args.min_gap,
    )


if __name__ == "__main__":
    main()
