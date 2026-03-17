"""Visualization helpers for raster and sample debugging.

The module treats plotting as optional. If matplotlib is unavailable or broken
in the current environment, it falls back to a lightweight Pillow-based writer
so sample generation and dataset checks can continue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


_VISUALIZATION_WARNING_SHOWN = False
# Cache: None = not yet checked, False = unavailable, module = ok
_PYPLOT_CACHE: object = None


def _warn_visualization_issue(message: str) -> None:
    """Print a one-time warning."""
    global _VISUALIZATION_WARNING_SHOWN
    if _VISUALIZATION_WARNING_SHOWN:
        return
    print(f"[vis_utils] {message}", flush=True)
    _VISUALIZATION_WARNING_SHOWN = True


def _get_pyplot():
    """Lazily import matplotlib.pyplot (cached)."""
    global _PYPLOT_CACHE
    if _PYPLOT_CACHE is not None:
        return _PYPLOT_CACHE if _PYPLOT_CACHE else None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PYPLOT_CACHE = plt
        return plt
    except Exception as exc:
        _warn_visualization_issue(
            "matplotlib is unavailable; falling back "
            "to Pillow preview. "
            f"Reason: {exc}",
        )
        _PYPLOT_CACHE = False
        return None


def _normalize_for_display(array: np.ndarray) -> np.ndarray:
    """Normalize a raster to [0, 255] uint8 for fallback preview writing."""
    raster = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(raster)
    if not finite.any():
        return np.zeros(raster.shape, dtype=np.uint8)

    values = raster[finite]
    lo = float(np.nanpercentile(values, 2.0))
    hi = float(np.nanpercentile(values, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((raster - lo) / (hi - lo), 0.0, 1.0)
    normalized[~finite] = 0.0
    return (normalized * 255.0).astype(np.uint8)


def _save_with_pillow(
    rasters: Mapping[str, np.ndarray],
    output_path: Path,
    max_cols: int,
) -> Path:
    """Write a simple grid preview using Pillow when matplotlib is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - optional fallback dependency.
        _warn_visualization_issue(
            "Pillow is also unavailable; skipping preview generation. "
            f"Reason: {exc}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    items = list(rasters.items())
    if not items:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    tile_images: list[Image.Image] = []
    label_height = 22
    padding = 8
    tile_width = 256
    tile_height = 256

    for name, raster in items:
        array = _normalize_for_display(raster)
        rgb = np.stack([array, array, array], axis=-1)
        tile = Image.fromarray(rgb, mode="RGB").resize((tile_width, tile_height))
        canvas = Image.new("RGB", (tile_width, tile_height + label_height), color=(18, 18, 18))
        canvas.paste(tile, (0, label_height))
        drawer = ImageDraw.Draw(canvas)
        drawer.text((6, 4), str(name), fill=(235, 235, 235))
        tile_images.append(canvas)

    num_items = len(tile_images)
    num_cols = min(max_cols, max(1, num_items))
    num_rows = int(np.ceil(num_items / num_cols))
    grid_width = (num_cols * tile_width) + ((num_cols - 1) * padding)
    grid_height = (num_rows * (tile_height + label_height)) + ((num_rows - 1) * padding)
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(10, 10, 10))

    for index, tile in enumerate(tile_images):
        row = index // num_cols
        col = index % num_cols
        left = col * (tile_width + padding)
        top = row * (tile_height + label_height + padding)
        grid_image.paste(tile, (left, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image.save(output_path)
    return output_path


def save_debug_figure(output_path: str | Path, title: str = "debug", image: np.ndarray | None = None) -> Path:
    """Save a single-image debug figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_pyplot()
    if plt is None:
        preview_rasters = {"debug": np.zeros((32, 32), dtype=np.float32) if image is None else np.asarray(image)}
        return _save_with_pillow(preview_rasters, path, max_cols=1)
    figure, axis = plt.subplots(figsize=(6, 5))
    if image is not None:
        axis.imshow(image, cmap="terrain")
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_raster_grid(
    rasters: Mapping[str, np.ndarray],
    output_path: str | Path,
    cmap: str = "terrain",
    max_cols: int = 4,
) -> Path:
    """Save a grid of raster visualizations."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_pyplot()
    if plt is None:
        return _save_with_pillow(rasters, path, max_cols=max_cols)
    items = list(rasters.items())
    num_items = len(items)
    num_cols = min(max_cols, max(1, num_items))
    num_rows = int(np.ceil(num_items / num_cols))
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3.5 * num_rows))
    axes_array = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for axis in axes_array.flat[num_items:]:
        axis.axis("off")

    for axis, (name, raster) in zip(axes_array.flat, items, strict=False):
        raster_array = np.asarray(raster)
        display = np.where(np.isfinite(raster_array), raster_array, np.nan)
        image = axis.imshow(display, cmap="gray" if "mask" in name else cmap)
        axis.set_title(name)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
