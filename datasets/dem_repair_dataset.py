"""Dataset loader for simulated-void DEM repair samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


DEM_LIKE_CHANNELS = {
    "gt_dem",
    "partial_dem",
    "coarse_dem",
    "dsm",
}
RATIO_CHANNELS = {
    "mask",
    "last_return_ratio",
    "ground_penetration_ratio",
    "confidence_map",
    "boundary_band",
}
DENSITY_CHANNELS = {
    "all_density",
    "ground_density",
    "vegetation_density",
    "building_density",
}
DERIVED_CHANNELS = {
    "slope",
    "curvature",
    "openness_s1",
    "openness_s2",
    "openness_s3",
    "openness_s4",
}


class DEMRepairDataset(Dataset):
    """Load NPZ samples and assemble configurable input tensors."""

    def __init__(
        self,
        sample_dir: str | Path,
        required_channels: list[str],
        split: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        normalization_scales: dict[str, float] | None = None,
    ) -> None:
        root = Path(sample_dir)
        self.sample_dir = root / split if split is not None and (root / split).exists() else root
        self.required_channels = required_channels
        self.transform = transform
        self.sample_paths = sorted(self.sample_dir.glob("*.npz"))
        # Fixed normalization scales (from config or dataset-wide stats).
        # Keys: channel names → fixed divisor.  If a channel is absent,
        # the old per-sample percentile fallback is used.
        self._norm_scales = normalization_scales or {}

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _load_sample_arrays(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=True) as data:
            sample = {key: data[key] for key in data.files}
        meta_value = sample.get("meta")
        if meta_value is None:
            meta = {"sample_id": path.stem, "path": str(path)}
        else:
            meta = json.loads(str(meta_value.item()))
        sample["meta"] = meta
        return sample

    def _normalize_dem(self, array: np.ndarray, mean: float, std: float, fill_value: float) -> np.ndarray:
        prepared = np.where(np.isfinite(array), array, fill_value).astype(np.float32)
        return ((prepared - mean) / std).astype(np.float32)

    def _normalize_density(
        self, array: np.ndarray, channel_name: str = "",
    ) -> np.ndarray:
        prepared = np.log1p(
            np.maximum(np.nan_to_num(array, nan=0.0), 0.0),
        ).astype(np.float32)
        fixed = self._norm_scales.get(channel_name, 0.0)
        if fixed > 0:
            scale = fixed
        else:
            scale = float(np.nanpercentile(prepared, 99)) \
                if np.isfinite(prepared).any() else 1.0
        return prepared / max(scale, 1.0)

    def _normalize_feature(
        self, array: np.ndarray, channel_name: str = "",
    ) -> np.ndarray:
        arr = array.astype(np.float32)
        finite = np.isfinite(arr)
        if finite.any():
            fill = float(np.mean(arr[finite]))
        else:
            fill = 0.0
        prepared = np.where(finite, arr, fill)
        fixed = self._norm_scales.get(channel_name, 0.0)
        if fixed > 0:
            scale = fixed
        else:
            scale = float(
                np.nanpercentile(np.abs(arr[finite]), 95),
            ) if finite.any() else 1.0
        return (prepared / max(scale, 1.0)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.sample_paths[index]
        sample_arrays = self._load_sample_arrays(path)

        gt_dem = sample_arrays["gt_dem"].astype(np.float32)
        mask = sample_arrays["mask"].astype(np.float32)
        coarse_dem = sample_arrays["coarse_dem"].astype(np.float32)
        partial_dem = sample_arrays["partial_dem"].astype(np.float32)

        # Use gt_dem for both mean and std so that the
        # normalization centre is not biased by coarse_dem
        # interpolation errors in void regions.
        dem_mean = float(np.nanmean(gt_dem)) \
            if np.isfinite(gt_dem).any() else 0.0
        dem_std = float(np.nanstd(gt_dem)) \
            if np.isfinite(gt_dem).any() else 1.0
        dem_std = dem_std if dem_std > 1e-6 else 1.0

        input_channels: list[np.ndarray] = []
        for channel_name in self.required_channels:
            channel = sample_arrays[channel_name].astype(np.float32)
            if channel_name == "partial_dem":
                channel = np.where(np.isfinite(channel), channel, coarse_dem)

            if channel_name in DEM_LIKE_CHANNELS:
                normalized = self._normalize_dem(
                    channel, dem_mean, dem_std,
                    fill_value=dem_mean,
                )
            elif channel_name in DENSITY_CHANNELS:
                normalized = self._normalize_density(
                    channel, channel_name,
                )
            elif channel_name in RATIO_CHANNELS:
                normalized = np.clip(
                    np.nan_to_num(channel, nan=0.0),
                    0.0, 1.0,
                ).astype(np.float32)
            elif (
                channel_name in DERIVED_CHANNELS
                or channel_name == "mean_intensity"
                or channel_name == "mean_return_num"
            ):
                normalized = self._normalize_feature(
                    channel, channel_name,
                )
            else:
                normalized = np.nan_to_num(
                    channel, nan=0.0,
                ).astype(np.float32)
            input_channels.append(normalized)

        batch = {
            "input_tensor": torch.from_numpy(np.stack(input_channels, axis=0)).float(),
            "gt_dem": torch.from_numpy(self._normalize_dem(gt_dem, dem_mean, dem_std, dem_mean)[None, ...]).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "coarse_dem": torch.from_numpy(self._normalize_dem(coarse_dem, dem_mean, dem_std, dem_mean)[None, ...]).float(),
            "partial_dem": torch.from_numpy(self._normalize_dem(np.where(np.isfinite(partial_dem), partial_dem, dem_mean), dem_mean, dem_std, dem_mean)[None, ...]).float(),
            "dem_mean": torch.tensor(dem_mean, dtype=torch.float32),
            "dem_std": torch.tensor(dem_std, dtype=torch.float32),
            "meta": sample_arrays["meta"],
        }
        if self.transform is not None:
            batch = self.transform(batch)
        return batch
