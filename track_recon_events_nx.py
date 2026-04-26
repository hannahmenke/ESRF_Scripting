#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import logging
import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_SAMPLE_COUNT = 1_000_000
DEFAULT_THRESHOLD_SIGMA = 5.0
DEFAULT_MAX_EVENTS = 100
DEFAULT_MIN_EVENT_SIZE = 1000
DEFAULT_MERGE_GAP = 10
DEFAULT_GIF_FPS = 2


@dataclass
class SliceComponent:
    z_index: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    voxel_count: int
    peak_abs_diff: float
    peak_signed_diff: float
    sum_abs_diff: float
    sum_signed_diff: float
    z_weighted_sum: float
    y_weighted_sum: float
    x_weighted_sum: float


@dataclass
class Event3D:
    event_id: int
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    voxel_count: int
    peak_abs_diff: float
    peak_signed_diff: float
    sum_abs_diff: float
    sum_signed_diff: float
    z_weighted_sum: float
    y_weighted_sum: float
    x_weighted_sum: float
    last_z: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track large reconstruction differences across an NX-style reconstructed time series."
    )
    parser.add_argument(
        "--reference-path",
        required=True,
        help="Dataset directory or reconstruction HDF5 file used to identify the series and anchor the first stepwise comparison.",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        required=True,
        help="First sequence number to include in stepwise processing. Both images in a comparison pair must be within the requested range. Unsuffixed dataset counts as 0.",
    )
    parser.add_argument(
        "--stop-number",
        type=int,
        required=True,
        help="Last sequence number to include in stepwise processing. Both images in a comparison pair must be within the requested range. Unsuffixed dataset counts as 0.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the reconstruction volume.",
    )
    parser.add_argument(
        "--crop-z",
        default=None,
        help="Crop range along Z as start:stop before processing.",
    )
    parser.add_argument(
        "--crop-y",
        default=None,
        help="Crop range along Y as start:stop before processing.",
    )
    parser.add_argument(
        "--crop-x",
        default=None,
        help="Crop range along X as start:stop before processing.",
    )
    parser.add_argument(
        "--output-db",
        default="recon_event_tracker.sqlite",
        help="SQLite database path for detected events. Default: recon_event_tracker.sqlite",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=DEFAULT_THRESHOLD_SIGMA,
        help="Detection threshold as baseline_noise_sigma * value. Default: 5.0",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=None,
        help="Absolute difference threshold for event detection. Overrides --threshold-sigma if provided.",
    )
    parser.add_argument(
        "--min-event-size",
        type=int,
        default=DEFAULT_MIN_EVENT_SIZE,
        help="Minimum voxel count for an event to be recorded. Default: 1000",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=DEFAULT_MERGE_GAP,
        help="Merge events whose bounding boxes are separated by at most this many voxels in X/Y. Default: 10",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help="Maximum number of events to store per comparison image. Default: 100",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for per-comparison event detection. Default: 1.",
    )
    parser.add_argument(
        "--noise-target-samples",
        type=int,
        default=DEFAULT_TARGET_SAMPLE_COUNT,
        help="Approximate number of voxels to sample when estimating baseline noise. Default: 1000000",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a threshold-tuning preview for one stepwise comparison and exit without writing outputs.",
    )
    parser.add_argument(
        "--preview-sequence",
        type=int,
        default=None,
        help="Sequence number to preview. Default uses the first valid step in range.",
    )
    parser.add_argument(
        "--preview-z",
        type=int,
        default=None,
        help="Z slice index to preview. Default chooses the slice with the strongest absolute change.",
    )
    parser.add_argument(
        "--preview-colormap",
        default="gray",
        help="Matplotlib colormap for previous/current preview slices. Default: gray.",
    )
    parser.add_argument(
        "--preview-diff-colormap",
        default="coolwarm",
        help="Matplotlib colormap for preview and GIF difference slices. Default: coolwarm.",
    )
    parser.add_argument(
        "--preview-diff-mode",
        choices=("raw", "suppressed"),
        default="suppressed",
        help="How to render preview and GIF difference images. Default: suppressed.",
    )
    parser.add_argument(
        "--preview-diff-noise-floor",
        type=float,
        default=None,
        help="Absolute deadband around zero for preview/GIF diff suppression. Overrides the threshold fraction if provided.",
    )
    parser.add_argument(
        "--preview-diff-floor-fraction",
        type=float,
        default=0.5,
        help="Preview/GIF diff suppression floor as a fraction of the detection threshold. Default: 0.5",
    )
    parser.add_argument(
        "--diff-display-min",
        type=float,
        default=-600.0,
        help="Minimum display value for difference images in previews and GIFs. Default: -600.",
    )
    parser.add_argument(
        "--diff-display-max",
        type=float,
        default=600.0,
        help="Maximum display value for difference images in previews and GIFs. Default: 600.",
    )
    parser.add_argument(
        "--save-gifs",
        action="store_true",
        help="Save orthogonal timeseries GIFs alongside the SQLite/CSV outputs.",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Export GIFs and exit without detecting events or writing the SQLite/CSV outputs.",
    )
    parser.add_argument(
        "--gif-labels",
        action="store_true",
        help="Overlay current and previous sequence numbers in the corner of GIF frames.",
    )
    parser.add_argument(
        "--gif-planes",
        default="xy,xz,yz",
        help="Comma-separated planes to export for GIFs. Choices: xy,xz,yz. Default: xy,xz,yz",
    )
    parser.add_argument(
        "--gif-mode",
        choices=("raw", "diff", "both"),
        default="diff",
        help="Whether GIFs should show raw slices, difference slices, or both. Default: diff",
    )
    parser.add_argument(
        "--orthogonal-center",
        default=None,
        help="Comma-separated z,y,x center indices for GIF slices. Defaults to the volume center.",
    )
    parser.add_argument(
        "--gif-downsample",
        type=int,
        default=1,
        help="Downsample factor for GIF frames. Default: 1.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help="Frames per second for GIF export. Default: 2.",
    )
    parser.add_argument(
        "--gif-colormap",
        default="gray",
        help="Matplotlib colormap for raw GIF frames. Default: gray.",
    )
    parser.add_argument(
        "--gif-diff-colormap",
        default="coolwarm",
        help="Matplotlib colormap for difference GIF frames. Default: coolwarm.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def log_exception_summary(message: str, exc: Exception) -> None:
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.exception("%s", message)
    else:
        LOGGER.error("%s: %s", message, exc)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_dataset(h5_file: h5py.File, dataset_path: str) -> h5py.Dataset:
    dataset = h5_file[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        raise RuntimeError(f"{dataset_path} is not a dataset")
    return dataset


def is_valid_volume_dataset(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 3 and dataset.dtype.kind in "uif" and all(int(size) > 1 for size in dataset.shape)


def find_candidate_datasets(h5_file: h5py.File) -> list[str]:
    candidates: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if not is_valid_volume_dataset(obj):
            return

        interpretation = decode_scalar(obj.attrs.get("interpretation"))
        if interpretation in {"image", "volume"} or name.endswith("/data"):
            candidates.append(name)
            return

        candidates.append(name)

    h5_file.visititems(visitor)
    candidates.sort()
    return candidates


def resolve_volume_dataset(input_path: Path, dataset_path: str | None) -> str:
    with h5py.File(input_path, "r") as h5_file:
        if dataset_path is not None:
            dataset = read_dataset(h5_file, dataset_path)
            if not is_valid_volume_dataset(dataset):
                raise RuntimeError(
                    f"{dataset_path} is not a valid 3D reconstruction volume dataset; "
                    f"got shape {tuple(int(v) for v in dataset.shape)}"
                )
            return dataset_path

        candidates = find_candidate_datasets(h5_file)
        if not candidates:
            raise RuntimeError(f"No numeric 3D datasets found in {input_path}")
        return candidates[0]


def parse_crop_range(raw_range: str | None, size: int, axis_name: str) -> tuple[int, int]:
    if raw_range is None:
        return 0, size

    parts = raw_range.split(":", 1)
    if len(parts) != 2:
        raise RuntimeError(f"{axis_name} crop must be formatted as start:stop")

    start_raw, stop_raw = parts
    start = int(start_raw) if start_raw.strip() else 0
    stop = int(stop_raw) if stop_raw.strip() else size

    if start < 0:
        start += size
    if stop < 0:
        stop += size

    start = max(0, min(start, size))
    stop = max(0, min(stop, size))
    if start >= stop:
        raise RuntimeError(f"{axis_name} crop must satisfy start < stop within 0:{size}")
    return start, stop


def crop_ranges_for_shape(
    shape: tuple[int, int, int],
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (
        parse_crop_range(crop_z, shape[0], "Z"),
        parse_crop_range(crop_y, shape[1], "Y"),
        parse_crop_range(crop_x, shape[2], "X"),
    )


def is_dataset_directory(path: Path) -> bool:
    return path.is_dir() and (
        (path / "projections").is_dir() or (path / "reconstructed_volumes").is_dir()
    )


def resolve_dataset_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        return resolved
    for parent in resolved.parents:
        if is_dataset_directory(parent):
            return parent
    return resolved.parent


def dataset_series_name(dataset_root: Path) -> str:
    return re.sub(r"_\d{4}$", "", dataset_root.name)


def dataset_sequence_number(dataset_root: Path) -> int:
    match = re.search(r"_(\d{4})$", dataset_root.name)
    if match:
        return int(match.group(1))
    return 0


def candidate_reconstruction_files(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base_name in ("reconstructed_volumes",):
        base_dir = dataset_root / base_name
        if not base_dir.is_dir():
            continue
        candidates.extend(path for path in base_dir.rglob("*.hdf5") if path.is_file())
    return sorted(candidates, key=lambda path: path.stat().st_mtime)


def is_reconstruction_file(path: Path, dataset_path: str | None = None) -> bool:
    lowered = path.name.lower()
    if "histogram" in lowered:
        return False

    try:
        resolve_volume_dataset(path, dataset_path)
        return True
    except Exception:
        return False


def find_latest_reconstruction_file(dataset_root: Path, dataset_path: str | None = None) -> Path:
    candidates = candidate_reconstruction_files(dataset_root)
    if not candidates:
        raise RuntimeError(f"No reconstruction HDF5 files found in {dataset_root}")

    valid_candidates = [path for path in candidates if is_reconstruction_file(path, dataset_path)]
    if not valid_candidates:
        raise RuntimeError(f"No valid reconstruction volume HDF5 files found in {dataset_root}")
    return valid_candidates[-1]


def resolve_reconstruction_target(raw_path: Path, dataset_path: str | None = None) -> tuple[Path, Path]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path not found: {path}")

    if path.is_file() and path.suffix in {".hdf5", ".h5"}:
        return resolve_dataset_root(path), path

    dataset_root = resolve_dataset_root(path)
    if not is_dataset_directory(dataset_root):
        raise RuntimeError(f"{path} does not resolve to a tomography dataset directory")
    return dataset_root, find_latest_reconstruction_file(dataset_root, dataset_path)


def list_series_datasets(
    reference_dataset_root: Path,
    dataset_path: str | None,
) -> list[tuple[int, Path, Path]]:
    collection_dir = reference_dataset_root.parent
    series_name = dataset_series_name(reference_dataset_root)
    results: list[tuple[int, Path, Path]] = []

    for dataset_dir in sorted(path for path in collection_dir.iterdir() if is_dataset_directory(path)):
        if dataset_series_name(dataset_dir) != series_name:
            continue
        sequence_number = dataset_sequence_number(dataset_dir)
        try:
            recon_file = find_latest_reconstruction_file(dataset_dir, dataset_path)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", dataset_dir, exc)
            continue
        results.append((sequence_number, dataset_dir, recon_file))

    results.sort(key=lambda item: item[0])

    duplicate_sequences: dict[int, list[Path]] = {}
    for sequence_number, dataset_dir, _recon_file in results:
        duplicate_sequences.setdefault(sequence_number, []).append(dataset_dir)
    ambiguous = {
        sequence_number: dataset_dirs
        for sequence_number, dataset_dirs in duplicate_sequences.items()
        if len(dataset_dirs) > 1
    }
    if ambiguous:
        details = "; ".join(
            f"#{sequence_number:04d}: {', '.join(str(path) for path in dataset_dirs)}"
            for sequence_number, dataset_dirs in sorted(ambiguous.items())
        )
        raise RuntimeError(
            "Ambiguous dataset numbering in series. Each comparison step must have a unique "
            f"4-digit sequence number. Conflicts: {details}"
        )

    return results


def list_series_dataset_roots(reference_dataset_root: Path) -> list[tuple[int, Path]]:
    collection_dir = reference_dataset_root.parent
    series_name = dataset_series_name(reference_dataset_root)
    results: list[tuple[int, Path]] = []

    for dataset_dir in sorted(path for path in collection_dir.iterdir() if is_dataset_directory(path)):
        if dataset_series_name(dataset_dir) != series_name:
            continue
        results.append((dataset_sequence_number(dataset_dir), dataset_dir))

    duplicate_sequences: dict[int, list[Path]] = {}
    for sequence_number, dataset_dir in results:
        duplicate_sequences.setdefault(sequence_number, []).append(dataset_dir)
    ambiguous = {
        sequence_number: dataset_dirs
        for sequence_number, dataset_dirs in duplicate_sequences.items()
        if len(dataset_dirs) > 1
    }
    if ambiguous:
        details = "; ".join(
            f"#{sequence_number:04d}: {', '.join(str(path) for path in dataset_dirs)}"
            for sequence_number, dataset_dirs in sorted(ambiguous.items())
        )
        raise RuntimeError(
            "Ambiguous dataset numbering in series. Each comparison step must have a unique "
            f"4-digit sequence number. Conflicts: {details}"
        )

    results.sort(key=lambda item: item[0])
    return results


def build_stepwise_comparisons(
    reference_dataset_root: Path,
    start_number: int,
    stop_number: int,
    dataset_path: str | None,
) -> list[tuple[int, Path, Path, int, Path, Path]]:
    all_datasets = list_series_datasets(reference_dataset_root, dataset_path)
    if not all_datasets:
        return []

    all_datasets = sorted(all_datasets, key=lambda item: item[0])
    for sequence_number, dataset_root, recon_file in all_datasets:
        LOGGER.info(
            "Admitted series member #%04d: dataset=%s | recon=%s",
            sequence_number,
            dataset_root,
            recon_file,
        )

    low = min(start_number, stop_number)
    high = max(start_number, stop_number)
    comparisons: list[tuple[int, Path, Path, int, Path, Path]] = []

    for index, (sequence_number, dataset_root, recon_file) in enumerate(all_datasets):
        if sequence_number < low or sequence_number > high:
            continue
        if index == 0:
            continue

        previous_sequence, previous_dataset_root, previous_recon_file = all_datasets[index - 1]
        if previous_sequence < low or previous_sequence > high:
            continue
        comparisons.append(
            (
                sequence_number,
                dataset_root,
                recon_file,
                previous_sequence,
                previous_dataset_root,
                previous_recon_file,
            )
        )

    return comparisons


def resolve_preview_comparison(
    reference_dataset_root: Path,
    start_number: int,
    stop_number: int,
    dataset_path: str | None,
    preview_sequence: int | None,
) -> tuple[int, Path, Path, int, Path, Path]:
    low = min(start_number, stop_number)
    high = max(start_number, stop_number)
    series_roots = [
        (sequence_number, dataset_root)
        for sequence_number, dataset_root in list_series_dataset_roots(reference_dataset_root)
        if low <= sequence_number <= high
    ]
    if not series_roots:
        raise RuntimeError("No series members found in the requested range")

    if preview_sequence is not None:
        selected_index = next(
            (index for index, (sequence_number, _dataset_root) in enumerate(series_roots) if sequence_number == preview_sequence),
            None,
        )
        if selected_index is None:
            raise RuntimeError(f"No stepwise comparison found for preview sequence #{preview_sequence:04d}")
        if selected_index == 0:
            raise RuntimeError(
                f"Preview sequence #{preview_sequence:04d} has no previous in-range series member for a stepwise comparison"
            )
        candidate_indices = [selected_index]
    else:
        candidate_indices = list(range(1, len(series_roots)))

    last_error: Exception | None = None
    for selected_index in candidate_indices:
        sequence_number, dataset_root = series_roots[selected_index]
        previous_sequence, previous_dataset_root = series_roots[selected_index - 1]
        try:
            recon_file = find_latest_reconstruction_file(dataset_root, dataset_path)
            previous_recon_file = find_latest_reconstruction_file(previous_dataset_root, dataset_path)
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Skipping preview candidate #%04d minus #%04d: %s",
                sequence_number,
                previous_sequence,
                exc,
            )
            continue
        return (
            sequence_number,
            dataset_root,
            recon_file,
            previous_sequence,
            previous_dataset_root,
            previous_recon_file,
        )

    if preview_sequence is not None and last_error is not None:
        raise RuntimeError(
            f"Preview sequence #{preview_sequence:04d} could not be resolved to valid reconstruction files: {last_error}"
        )
    if last_error is not None:
        raise RuntimeError(f"No valid preview comparison reconstructions found in the requested range: {last_error}")
    raise RuntimeError("No stepwise comparison reconstructions found in the requested range")


def volume_shape(
    recon_file: Path,
    dataset_path: str,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
) -> tuple[int, int, int]:
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, dataset_path)
        shape = tuple(int(v) for v in volume.shape)
    z_range, y_range, x_range = crop_ranges_for_shape(shape, crop_z, crop_y, crop_x)
    return (
        z_range[1] - z_range[0],
        y_range[1] - y_range[0],
        x_range[1] - x_range[0],
    )


def crop_offsets_for_volume(
    recon_file: Path,
    dataset_path: str,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
) -> tuple[int, int, int]:
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, dataset_path)
        shape = tuple(int(v) for v in volume.shape)
    z_range, y_range, x_range = crop_ranges_for_shape(shape, crop_z, crop_y, crop_x)
    return z_range[0], y_range[0], x_range[0]


def parse_orthogonal_center(raw_center: str | None, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if raw_center:
        values = [int(part.strip()) for part in raw_center.split(",") if part.strip()]
        if len(values) != 3:
            raise RuntimeError("--orthogonal-center must provide exactly 3 indices as z,y,x")
        center = tuple(values)
    else:
        center = tuple(size // 2 for size in shape)

    for axis, (index, axis_size) in enumerate(zip(center, shape)):
        if index < 0 or index >= axis_size:
            raise RuntimeError(f"Orthogonal center index {index} is out of range for axis {axis} with size {axis_size}")
    return center


def choose_sampling_step(shape: tuple[int, int, int], target_sample_count: int) -> int:
    total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
    if target_sample_count <= 0:
        return 1
    step = int(math.ceil((total_voxels / target_sample_count) ** (1.0 / 3.0)))
    return max(step, 1)


def iter_diff_slices(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    z_indices: list[int] | None = None,
    y_step: int = 1,
    x_step: int = 1,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
):
    with h5py.File(reference_file, "r") as ref_h5, h5py.File(comparison_file, "r") as cmp_h5:
        ref_volume = read_dataset(ref_h5, dataset_path)
        cmp_volume = read_dataset(cmp_h5, dataset_path)
        if ref_volume.shape != cmp_volume.shape:
            raise RuntimeError(
                f"Volume shape mismatch: {reference_file} {tuple(ref_volume.shape)} vs "
                f"{comparison_file} {tuple(cmp_volume.shape)}"
            )

        full_shape = tuple(int(v) for v in ref_volume.shape)
        z_range, y_range, x_range = crop_ranges_for_shape(full_shape, crop_z, crop_y, crop_x)

        if z_indices is None:
            z_iterable = range(z_range[0], z_range[1])
        else:
            z_iterable = [z_range[0] + z_index for z_index in z_indices]

        for full_z_index in z_iterable:
            ref_slice = np.asarray(
                ref_volume[full_z_index, y_range[0]:y_range[1]:y_step, x_range[0]:x_range[1]:x_step],
                dtype=np.float32,
            )
            cmp_slice = np.asarray(
                cmp_volume[full_z_index, y_range[0]:y_range[1]:y_step, x_range[0]:x_range[1]:x_step],
                dtype=np.float32,
            )
            yield full_z_index - z_range[0], cmp_slice - ref_slice


def load_slice_pair(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    z_index: int,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(reference_file, "r") as ref_h5, h5py.File(comparison_file, "r") as cmp_h5:
        ref_volume = read_dataset(ref_h5, dataset_path)
        cmp_volume = read_dataset(cmp_h5, dataset_path)
        if ref_volume.shape != cmp_volume.shape:
            raise RuntimeError(
                f"Volume shape mismatch: {reference_file} {tuple(ref_volume.shape)} vs "
                f"{comparison_file} {tuple(cmp_volume.shape)}"
            )
        full_shape = tuple(int(v) for v in ref_volume.shape)
        z_range, y_range, x_range = crop_ranges_for_shape(full_shape, crop_z, crop_y, crop_x)
        cropped_depth = z_range[1] - z_range[0]
        if z_index < 0 or z_index >= cropped_depth:
            raise RuntimeError(f"Preview z index {z_index} is out of range for cropped depth {cropped_depth}")
        full_z_index = z_range[0] + z_index
        ref_slice = np.asarray(ref_volume[full_z_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], dtype=np.float32)
        cmp_slice = np.asarray(cmp_volume[full_z_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], dtype=np.float32)
    return ref_slice, cmp_slice


def choose_preview_z(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    preview_z: int | None,
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> int:
    shape = volume_shape(reference_file, dataset_path, crop_z, crop_y, crop_x)
    if preview_z is not None:
        if preview_z < 0 or preview_z >= shape[0]:
            raise RuntimeError(f"--preview-z {preview_z} is out of range for depth {shape[0]}")
        return preview_z

    z_step = max(shape[0] // 64, 1)
    yx_step = max(min(shape[1], shape[2]) // 512, 1)
    z_indices = list(range(0, shape[0], z_step))
    if not z_indices or z_indices[-1] != shape[0] - 1:
        z_indices.append(shape[0] - 1)

    best_z = 0
    best_value = float("-inf")
    for z_index, diff_slice in iter_diff_slices(
        reference_file,
        comparison_file,
        dataset_path,
        z_indices=z_indices,
        y_step=yx_step,
        x_step=yx_step,
        crop_z=crop_z,
        crop_y=crop_y,
        crop_x=crop_x,
    ):
        candidate = float(np.max(np.abs(diff_slice)))
        if candidate > best_value:
            best_value = candidate
            best_z = z_index
    return best_z


def suppress_low_differences_for_preview(image: np.ndarray, noise_floor: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    floor = max(float(noise_floor), 0.0)
    if floor == 0.0:
        return image
    return np.sign(image) * np.maximum(np.abs(image) - floor, 0.0)


def show_detection_preview(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    threshold_value: float,
    min_event_size: int,
    merge_gap: int,
    preview_z: int,
    sequence_number: int,
    previous_sequence_number: int,
    args: argparse.Namespace,
) -> None:
    preview_window_radius = 2
    reference_slice, comparison_slice = load_slice_pair(
        reference_file,
        comparison_file,
        dataset_path,
        preview_z,
        args.crop_z,
        args.crop_y,
        args.crop_x,
    )
    diff_slice = comparison_slice - reference_slice
    mask = np.abs(diff_slice) >= threshold_value
    shape = volume_shape(reference_file, dataset_path, args.crop_z, args.crop_y, args.crop_x)
    z_start = max(0, preview_z - preview_window_radius)
    z_stop = min(shape[0], preview_z + preview_window_radius + 1)
    preview_slice_results = [
        process_diff_slice_components(z_index, diff_window_slice, threshold_value)
        for z_index, diff_window_slice in iter_diff_slices(
            reference_file,
            comparison_file,
            dataset_path,
            z_indices=list(range(z_start, z_stop)),
            crop_z=args.crop_z,
            crop_y=args.crop_y,
            crop_x=args.crop_x,
        )
    ]
    merged_events, _preview_max_abs_diff = assemble_events_from_slice_results(
        preview_slice_results,
        min_event_size,
        merge_gap,
    )
    display_events = [
        event for event in merged_events if event.z_min <= preview_z <= event.z_max
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.atleast_1d(axes).ravel()

    previous_artist = axes[0].imshow(reference_slice, cmap=args.preview_colormap)
    axes[0].set_title(f"Previous #{previous_sequence_number:04d}")
    fig.colorbar(previous_artist, ax=axes[0], fraction=0.046, pad=0.04)

    current_artist = axes[1].imshow(comparison_slice, cmap=args.preview_colormap)
    axes[1].set_title(f"Current #{sequence_number:04d}")
    fig.colorbar(current_artist, ax=axes[1], fraction=0.046, pad=0.04)

    diff_display_image = diff_slice
    diff_title = f"Difference @ z={preview_z}"
    if args.preview_diff_mode == "suppressed":
        noise_floor = (
            float(args.preview_diff_noise_floor)
            if args.preview_diff_noise_floor is not None
            else float(args.preview_diff_floor_fraction) * float(threshold_value)
        )
        diff_display_image = suppress_low_differences_for_preview(diff_slice, noise_floor)
        diff_title = f"Suppressed difference @ z={preview_z} (floor={noise_floor:.6g})"

    diff_artist = axes[2].imshow(
        diff_display_image,
        cmap=args.preview_diff_colormap,
        vmin=args.diff_display_min,
        vmax=args.diff_display_max,
    )
    axes[2].set_title(diff_title)
    fig.colorbar(diff_artist, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(reference_slice, cmap=args.preview_colormap, alpha=0.35)
    mask_overlay = np.ma.masked_where(~mask, mask.astype(np.float32))
    mask_artist = axes[3].imshow(mask_overlay, cmap="autumn", alpha=0.7, vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"Threshold mask over reference ({len(display_events)} events on shown slice)"
    )
    fig.colorbar(mask_artist, ax=axes[3], fraction=0.046, pad=0.04)
    for event in display_events:
        width = event.x_max - event.x_min + 1
        height = event.y_max - event.y_min + 1
        rectangle = plt.Rectangle(
            (event.x_min, event.y_min),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        axes[3].add_patch(rectangle)

    fig.suptitle(
        f"Stepwise event preview: #{sequence_number:04d} minus #{previous_sequence_number:04d}\n"
        f"threshold={threshold_value:.6g}, min_event_size={min_event_size}, merge_gap={merge_gap}, "
        f"showing z={preview_z} with local stack {z_start}:{z_stop}"
    )
    plt.tight_layout()
    plt.show()


def load_orthogonal_views(
    recon_file: Path,
    dataset_path: str,
    center: tuple[int, int, int],
    downsample: int,
    planes: list[str],
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> dict[str, np.ndarray]:
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, dataset_path)
        full_shape = tuple(int(v) for v in volume.shape)
        z_range, y_range, x_range = crop_ranges_for_shape(full_shape, crop_z, crop_y, crop_x)
        z_index, y_index, x_index = center
        full_z_index = z_range[0] + z_index
        full_y_index = y_range[0] + y_index
        full_x_index = x_range[0] + x_index
        views: dict[str, np.ndarray] = {}
        if "xy" in planes:
            views["xy"] = np.asarray(
                volume[full_z_index, y_range[0]:y_range[1]:downsample, x_range[0]:x_range[1]:downsample],
                dtype=np.float32,
            )
        if "xz" in planes:
            views["xz"] = np.asarray(
                volume[z_range[0]:z_range[1]:downsample, full_y_index, x_range[0]:x_range[1]:downsample],
                dtype=np.float32,
            )
        if "yz" in planes:
            views["yz"] = np.asarray(
                volume[z_range[0]:z_range[1]:downsample, y_range[0]:y_range[1]:downsample, full_x_index],
                dtype=np.float32,
            )
    return views


def normalize_frame(
    image: np.ndarray,
    colormap: str = "gray",
    display_min: float | None = None,
    display_max: float | None = None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros((*image.shape, 3), dtype=np.uint8)
    image = np.where(finite, image, 0.0)
    if display_min is not None and display_max is not None:
        low = float(display_min)
        high = float(display_max)
    else:
        low, high = np.percentile(image, (1.0, 99.0))
        if float(low) == float(high):
            low = float(np.min(image))
            high = float(np.max(image))
    if float(low) == float(high):
        return np.zeros((*image.shape, 3), dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    rgb = np.asarray(plt.get_cmap(colormap)(scaled)[..., :3] * 255.0, dtype=np.uint8)
    return rgb


def annotate_frame(image: np.ndarray, text: str) -> np.ndarray:
    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L").convert("RGB")
    else:
        pil_image = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    padding = 4
    box = (
        8,
        8,
        8 + (right - left) + 2 * padding,
        8 + (bottom - top) + 2 * padding,
    )
    draw.rectangle(box, fill=(0, 0, 0, 160))
    draw.text((8 + padding, 8 + padding), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(pil_image)


def stack_frames_horizontally(left: np.ndarray, right: np.ndarray, separator_width: int = 4) -> np.ndarray:
    if left.ndim == 2:
        left = np.repeat(left[..., None], 3, axis=2)
    if right.ndim == 2:
        right = np.repeat(right[..., None], 3, axis=2)
    height = max(left.shape[0], right.shape[0])
    left_width = left.shape[1]
    right_width = right.shape[1]
    output = np.zeros((height, left_width + separator_width + right_width, 3), dtype=np.uint8)
    output[: left.shape[0], :left_width] = left
    output[: right.shape[0], left_width + separator_width :] = right
    output[:, left_width:left_width + separator_width] = 255
    return output


def build_gif_frames_for_plane(
    comparison: tuple[int, Path, Path, int, Path, Path],
    dataset_path: str,
    center: tuple[int, int, int],
    plane: str,
    mode: str,
    downsample: int,
    labels: bool,
    raw_colormap: str,
    diff_colormap: str,
    diff_display_min: float,
    diff_display_max: float,
    preview_diff_mode: str,
    preview_diff_noise_floor: float | None,
    preview_diff_floor_fraction: float,
    threshold_value: float | None,
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> tuple[int, dict[str, np.ndarray]]:
    sequence_number, _dataset_root, recon_file, previous_sequence, _previous_dataset_root, previous_recon_file = comparison
    current_views = load_orthogonal_views(recon_file, dataset_path, center, downsample, [plane], crop_z, crop_y, crop_x)
    previous_views = load_orthogonal_views(previous_recon_file, dataset_path, center, downsample, [plane], crop_z, crop_y, crop_x)
    label = f"#{sequence_number:04d} prev #{previous_sequence:04d}"
    frames: dict[str, np.ndarray] = {}

    if mode in {"raw", "both"}:
        key = f"{plane}_raw"
        previous_frame = normalize_frame(previous_views[plane], raw_colormap)
        current_frame = normalize_frame(current_views[plane], raw_colormap)
        frame = stack_frames_horizontally(previous_frame, current_frame)
        if labels:
            frame = annotate_frame(frame, label)
        frames[key] = frame
    if mode in {"diff", "both"}:
        key = f"{plane}_diff"
        diff_view = current_views[plane] - previous_views[plane]
        if preview_diff_mode == "suppressed":
            if threshold_value is None and preview_diff_noise_floor is None:
                raise RuntimeError("Threshold value is required for suppressed GIF diff rendering")
            noise_floor = (
                float(preview_diff_noise_floor)
                if preview_diff_noise_floor is not None
                else float(preview_diff_floor_fraction) * float(threshold_value)
            )
            diff_view = suppress_low_differences_for_preview(diff_view, noise_floor)
        frame = normalize_frame(
            diff_view,
            diff_colormap,
            diff_display_min,
            diff_display_max,
        )
        if labels:
            frame = annotate_frame(frame, label)
        frames[key] = frame

    return sequence_number, frames


def write_gif_file(output_path: Path, frames: list[np.ndarray], fps: int) -> Path:
    imageio.mimsave(output_path, frames, duration=1.0 / max(fps, 1))
    return output_path


def save_timeseries_gifs(
    comparisons: list[tuple[int, Path, Path, int, Path, Path]],
    dataset_path: str,
    output_db: Path,
    center: tuple[int, int, int],
    planes: list[str],
    mode: str,
    downsample: int,
    fps: int,
    labels: bool,
    raw_colormap: str,
    diff_colormap: str,
    diff_display_min: float,
    diff_display_max: float,
    preview_diff_mode: str,
    preview_diff_noise_floor: float | None,
    preview_diff_floor_fraction: float,
    threshold_value: float | None,
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
    jobs: int,
) -> list[Path]:
    frame_sets: dict[str, list[np.ndarray]] = {}
    output_paths: list[Path] = []
    plane_tasks = [(comparison, plane) for comparison in comparisons for plane in planes]
    if jobs > 1 and len(plane_tasks) > 1:
        max_workers = min(jobs, len(plane_tasks), os.cpu_count() or jobs)
        LOGGER.info("Running GIF frame generation in parallel with %d workers", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            task_results = list(
                executor.map(
                    build_gif_frames_for_plane,
                    (comparison for comparison, _plane in plane_tasks),
                    [dataset_path] * len(plane_tasks),
                    [center] * len(plane_tasks),
                    (plane for _comparison, plane in plane_tasks),
                    [mode] * len(plane_tasks),
                    [downsample] * len(plane_tasks),
                    [labels] * len(plane_tasks),
                    [raw_colormap] * len(plane_tasks),
                    [diff_colormap] * len(plane_tasks),
                    [diff_display_min] * len(plane_tasks),
                    [diff_display_max] * len(plane_tasks),
                    [preview_diff_mode] * len(plane_tasks),
                    [preview_diff_noise_floor] * len(plane_tasks),
                    [preview_diff_floor_fraction] * len(plane_tasks),
                    [threshold_value] * len(plane_tasks),
                    [crop_z] * len(plane_tasks),
                    [crop_y] * len(plane_tasks),
                    [crop_x] * len(plane_tasks),
                )
            )
    else:
        task_results = [
            build_gif_frames_for_plane(
                comparison,
                dataset_path,
                center,
                plane,
                mode,
                downsample,
                labels,
                raw_colormap,
                diff_colormap,
                diff_display_min,
                diff_display_max,
                preview_diff_mode,
                preview_diff_noise_floor,
                preview_diff_floor_fraction,
                threshold_value,
                crop_z,
                crop_y,
                crop_x,
            )
            for comparison, plane in plane_tasks
        ]

    task_results.sort(key=lambda item: item[0])
    for _sequence_number, frames in task_results:
        for key, frame in frames.items():
            frame_sets.setdefault(key, []).append(frame)

    gif_tasks = [
        (output_db.with_name(f"{output_db.stem}_{key}.gif"), frames)
        for key, frames in frame_sets.items()
    ]
    if jobs > 1 and len(gif_tasks) > 1:
        max_workers = min(jobs, len(gif_tasks), os.cpu_count() or jobs)
        LOGGER.info("Writing GIF files in parallel with %d workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            output_paths = list(
                executor.map(
                    write_gif_file,
                    (output_path for output_path, _frames in gif_tasks),
                    (frames for _output_path, frames in gif_tasks),
                    [fps] * len(gif_tasks),
                )
            )
    else:
        for output_path, frames in gif_tasks:
            output_paths.append(write_gif_file(output_path, frames, fps))
    return output_paths


def estimate_baseline_sigma(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    target_sample_count: int,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
) -> float:
    shape = volume_shape(reference_file, dataset_path, crop_z, crop_y, crop_x)
    step = choose_sampling_step(shape, target_sample_count)
    samples: list[np.ndarray] = []
    LOGGER.info("Estimating baseline noise with sampling step %s", step)

    z_indices = list(range(0, shape[0], step))
    if not z_indices:
        raise RuntimeError("No baseline samples were collected")

    for _z_index, diff_slice in iter_diff_slices(
        reference_file,
        comparison_file,
        dataset_path,
        z_indices=z_indices,
        y_step=step,
        x_step=step,
        crop_z=crop_z,
        crop_y=crop_y,
        crop_x=crop_x,
    ):
        samples.append(diff_slice.ravel())

    if not samples:
        raise RuntimeError("No baseline samples were collected")

    sample = np.concatenate(samples).astype(np.float32, copy=False)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    sigma = mad / 0.6744897501960817 if mad > 0 else float(np.std(sample))
    if not math.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("Baseline noise estimate is not positive")
    return sigma


def find_slice_components(mask: np.ndarray, diff_slice: np.ndarray, z_index: int) -> list[SliceComponent]:
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[SliceComponent] = []

    active_positions = np.argwhere(mask)
    for start_y, start_x in active_positions:
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        y_min = y_max = int(start_y)
        x_min = x_max = int(start_x)
        voxel_count = 0
        peak_abs_diff = 0.0
        peak_signed_diff = 0.0
        sum_abs_diff = 0.0
        sum_signed_diff = 0.0
        z_weighted_sum = 0.0
        y_weighted_sum = 0.0
        x_weighted_sum = 0.0

        while stack:
            y, x = stack.pop()
            signed_value = float(diff_slice[y, x])
            value = abs(signed_value)
            voxel_count += 1
            if value > peak_abs_diff:
                peak_abs_diff = value
                peak_signed_diff = signed_value
            sum_abs_diff += value
            sum_signed_diff += signed_value
            z_weighted_sum += float(z_index)
            y_weighted_sum += float(y)
            x_weighted_sum += float(x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            x_min = min(x_min, x)
            x_max = max(x_max, x)

            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if not mask[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        components.append(
            SliceComponent(
                z_index=z_index,
                y_min=y_min,
                y_max=y_max,
                x_min=x_min,
                x_max=x_max,
                voxel_count=voxel_count,
                peak_abs_diff=peak_abs_diff,
                peak_signed_diff=peak_signed_diff,
                sum_abs_diff=sum_abs_diff,
                sum_signed_diff=sum_signed_diff,
                z_weighted_sum=z_weighted_sum,
                y_weighted_sum=y_weighted_sum,
                x_weighted_sum=x_weighted_sum,
            )
        )

    return components


def process_diff_slice_components(
    z_index: int,
    diff_slice: np.ndarray,
    threshold_value: float,
) -> tuple[int, list[SliceComponent], float]:
    abs_slice = np.abs(diff_slice)
    max_abs_diff = float(np.max(abs_slice))
    mask = abs_slice >= threshold_value
    components = find_slice_components(mask, diff_slice, z_index)
    return z_index, components, max_abs_diff


def split_indices(indices: list[int], jobs: int) -> list[list[int]]:
    if not indices:
        return []
    worker_count = max(1, min(jobs, len(indices)))
    chunk_size = int(math.ceil(len(indices) / worker_count))
    return [indices[offset:offset + chunk_size] for offset in range(0, len(indices), chunk_size)]


def process_diff_chunk_components(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    z_indices: list[int],
    threshold_value: float,
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> list[tuple[int, list[SliceComponent], float]]:
    return [
        process_diff_slice_components(z_index, diff_slice, threshold_value)
        for z_index, diff_slice in iter_diff_slices(
            reference_file,
            comparison_file,
            dataset_path,
            z_indices=z_indices,
            crop_z=crop_z,
            crop_y=crop_y,
            crop_x=crop_x,
        )
    ]


def assemble_events_from_slice_results(
    slice_results: list[tuple[int, list[SliceComponent], float]],
    min_event_size: int,
    merge_gap: int,
) -> tuple[list[Event3D], float]:
    all_events: list[Event3D] = []
    active_events: dict[int, Event3D] = {}
    next_event_id = 1
    max_abs_diff = 0.0

    for z_index, components, slice_max_abs_diff in sorted(slice_results, key=lambda item: item[0]):
        max_abs_diff = max(max_abs_diff, slice_max_abs_diff)
        current_active_ids: set[int] = set()

        for component in components:
            matching_ids = [
                event_id
                for event_id, event in active_events.items()
                if event.last_z == z_index - 1 and bboxes_touch(event, component, merge_gap)
            ]

            if not matching_ids:
                event = Event3D(
                    event_id=next_event_id,
                    z_min=component.z_index,
                    z_max=component.z_index,
                    y_min=component.y_min,
                    y_max=component.y_max,
                    x_min=component.x_min,
                    x_max=component.x_max,
                    voxel_count=0,
                    peak_abs_diff=0.0,
                    peak_signed_diff=0.0,
                    sum_abs_diff=0.0,
                    sum_signed_diff=0.0,
                    z_weighted_sum=0.0,
                    y_weighted_sum=0.0,
                    x_weighted_sum=0.0,
                    last_z=component.z_index,
                )
                absorb_component(event, component)
                active_events[event.event_id] = event
                current_active_ids.add(event.event_id)
                next_event_id += 1
                continue

            primary_id = matching_ids[0]
            primary_event = active_events[primary_id]
            for merge_id in matching_ids[1:]:
                if merge_id == primary_id or merge_id not in active_events:
                    continue
                merge_events(primary_event, active_events[merge_id])
                del active_events[merge_id]
                current_active_ids.discard(merge_id)
            absorb_component(primary_event, component)
            current_active_ids.add(primary_id)

        finished_ids = [event_id for event_id in active_events if event_id not in current_active_ids]
        for event_id in finished_ids:
            all_events.append(active_events.pop(event_id))

    all_events.extend(active_events.values())
    filtered_events = [event for event in all_events if event.voxel_count >= min_event_size]
    filtered_events.sort(key=lambda event: event.peak_abs_diff, reverse=True)
    return filtered_events, max_abs_diff


def bboxes_touch(a: Event3D, b: SliceComponent, merge_gap: int = 1) -> bool:
    return not (
        a.y_max + merge_gap < b.y_min
        or b.y_max + merge_gap < a.y_min
        or a.x_max + merge_gap < b.x_min
        or b.x_max + merge_gap < a.x_min
    )


def merge_events(target: Event3D, source: Event3D) -> None:
    target.z_min = min(target.z_min, source.z_min)
    target.z_max = max(target.z_max, source.z_max)
    target.y_min = min(target.y_min, source.y_min)
    target.y_max = max(target.y_max, source.y_max)
    target.x_min = min(target.x_min, source.x_min)
    target.x_max = max(target.x_max, source.x_max)
    target.voxel_count += source.voxel_count
    if source.peak_abs_diff > target.peak_abs_diff:
        target.peak_abs_diff = source.peak_abs_diff
        target.peak_signed_diff = source.peak_signed_diff
    target.sum_abs_diff += source.sum_abs_diff
    target.sum_signed_diff += source.sum_signed_diff
    target.z_weighted_sum += source.z_weighted_sum
    target.y_weighted_sum += source.y_weighted_sum
    target.x_weighted_sum += source.x_weighted_sum
    target.last_z = max(target.last_z, source.last_z)


def absorb_component(event: Event3D, component: SliceComponent) -> None:
    event.z_min = min(event.z_min, component.z_index)
    event.z_max = max(event.z_max, component.z_index)
    event.y_min = min(event.y_min, component.y_min)
    event.y_max = max(event.y_max, component.y_max)
    event.x_min = min(event.x_min, component.x_min)
    event.x_max = max(event.x_max, component.x_max)
    event.voxel_count += component.voxel_count
    if component.peak_abs_diff > event.peak_abs_diff:
        event.peak_abs_diff = component.peak_abs_diff
        event.peak_signed_diff = component.peak_signed_diff
    event.sum_abs_diff += component.sum_abs_diff
    event.sum_signed_diff += component.sum_signed_diff
    event.z_weighted_sum += component.z_weighted_sum
    event.y_weighted_sum += component.y_weighted_sum
    event.x_weighted_sum += component.x_weighted_sum
    event.last_z = component.z_index


def event_centroid(event: Event3D) -> tuple[float, float, float]:
    denominator = max(event.voxel_count, 1)
    return (
        event.z_weighted_sum / denominator,
        event.y_weighted_sum / denominator,
        event.x_weighted_sum / denominator,
    )


def detect_events_for_comparison(
    reference_file: Path,
    comparison_file: Path,
    dataset_path: str,
    threshold_value: float,
    min_event_size: int,
    merge_gap: int,
    jobs: int = 1,
    crop_z: str | None = None,
    crop_y: str | None = None,
    crop_x: str | None = None,
) -> tuple[list[Event3D], float]:
    slice_results: list[tuple[int, list[SliceComponent], float]]

    if jobs <= 1:
        slice_results = [
            process_diff_slice_components(z_index, diff_slice, threshold_value)
            for z_index, diff_slice in iter_diff_slices(
                reference_file,
                comparison_file,
                dataset_path,
                crop_z=crop_z,
                crop_y=crop_y,
                crop_x=crop_x,
            )
        ]
    else:
        cropped_depth = volume_shape(reference_file, dataset_path, crop_z, crop_y, crop_x)[0]
        z_chunks = split_indices(list(range(cropped_depth)), jobs)
        max_workers = min(jobs, len(z_chunks), os.cpu_count() or jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(
                executor.map(
                    process_diff_chunk_components,
                    [reference_file] * len(z_chunks),
                    [comparison_file] * len(z_chunks),
                    [dataset_path] * len(z_chunks),
                    z_chunks,
                    [threshold_value] * len(z_chunks),
                    [crop_z] * len(z_chunks),
                    [crop_y] * len(z_chunks),
                    [crop_x] * len(z_chunks),
                )
            )
        slice_results = [item for chunk in chunk_results for item in chunk]
    return assemble_events_from_slice_results(slice_results, min_event_size, merge_gap)


def process_comparison_task(
    comparison: tuple[int, Path, Path, int, Path, Path],
    dataset_path: str,
    threshold_value: float,
    min_event_size: int,
    merge_gap: int,
    jobs: int,
    crop_z: str | None,
    crop_y: str | None,
    crop_x: str | None,
) -> tuple[tuple[int, Path, Path, int, Path, Path], list[Event3D], float]:
    sequence_number, dataset_root, recon_file, previous_sequence, previous_dataset_root, previous_recon_file = comparison
    events, max_abs_diff = detect_events_for_comparison(
        previous_recon_file,
        recon_file,
        dataset_path,
        threshold_value,
        min_event_size,
        merge_gap,
        jobs,
        crop_z,
        crop_y,
        crop_x,
    )
    return (
        (
            sequence_number,
            dataset_root,
            recon_file,
            previous_sequence,
            previous_dataset_root,
            previous_recon_file,
        ),
        events,
        max_abs_diff,
    )


def initialize_database(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            reference_path TEXT NOT NULL,
            reference_dataset_root TEXT NOT NULL,
            reference_reconstruction_file TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            start_number INTEGER NOT NULL,
            stop_number INTEGER NOT NULL,
            baseline_sigma REAL NOT NULL,
            threshold_sigma REAL NOT NULL,
            threshold_value REAL NOT NULL,
            min_event_size INTEGER NOT NULL,
            merge_gap INTEGER NOT NULL,
            crop_z_start INTEGER NOT NULL DEFAULT 0,
            crop_y_start INTEGER NOT NULL DEFAULT 0,
            crop_x_start INTEGER NOT NULL DEFAULT 0,
            max_events INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            sequence_number INTEGER NOT NULL,
            previous_sequence_number INTEGER NOT NULL,
            previous_dataset_root TEXT NOT NULL,
            previous_reconstruction_file TEXT NOT NULL,
            previous_reconstruction_mtime TEXT,
            dataset_root TEXT NOT NULL,
            reconstruction_file TEXT NOT NULL,
            reconstruction_mtime TEXT,
            detected_event_count INTEGER NOT NULL,
            stored_event_count INTEGER NOT NULL,
            max_abs_diff REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id INTEGER NOT NULL,
            event_rank INTEGER NOT NULL,
            voxel_count INTEGER NOT NULL,
            peak_abs_diff REAL NOT NULL,
            peak_signed_diff REAL NOT NULL,
            mean_abs_diff REAL NOT NULL,
            mean_signed_diff REAL NOT NULL,
            z_centroid REAL NOT NULL,
            y_centroid REAL NOT NULL,
            x_centroid REAL NOT NULL,
            z_min INTEGER NOT NULL,
            z_max INTEGER NOT NULL,
            y_min INTEGER NOT NULL,
            y_max INTEGER NOT NULL,
            x_min INTEGER NOT NULL,
            x_max INTEGER NOT NULL,
            full_z_centroid REAL,
            full_y_centroid REAL,
            full_x_centroid REAL,
            full_z_min INTEGER,
            full_z_max INTEGER,
            full_y_min INTEGER,
            full_y_max INTEGER,
            full_x_min INTEGER,
            full_x_max INTEGER,
            FOREIGN KEY (comparison_id) REFERENCES comparisons(id)
        )
        """
    )
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(comparisons)")
    }
    run_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(runs)")
    }
    event_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(events)")
    }
    if "merge_gap" not in run_columns:
        connection.execute(f"ALTER TABLE runs ADD COLUMN merge_gap INTEGER NOT NULL DEFAULT {DEFAULT_MERGE_GAP}")
    if "crop_z_start" not in run_columns:
        connection.execute("ALTER TABLE runs ADD COLUMN crop_z_start INTEGER NOT NULL DEFAULT 0")
    if "crop_y_start" not in run_columns:
        connection.execute("ALTER TABLE runs ADD COLUMN crop_y_start INTEGER NOT NULL DEFAULT 0")
    if "crop_x_start" not in run_columns:
        connection.execute("ALTER TABLE runs ADD COLUMN crop_x_start INTEGER NOT NULL DEFAULT 0")
    if "previous_reconstruction_mtime" not in existing_columns:
        connection.execute("ALTER TABLE comparisons ADD COLUMN previous_reconstruction_mtime TEXT")
    if "reconstruction_mtime" not in existing_columns:
        connection.execute("ALTER TABLE comparisons ADD COLUMN reconstruction_mtime TEXT")
    if "full_z_centroid" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_z_centroid REAL")
    if "full_y_centroid" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_y_centroid REAL")
    if "full_x_centroid" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_x_centroid REAL")
    if "full_z_min" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_z_min INTEGER")
    if "full_z_max" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_z_max INTEGER")
    if "full_y_min" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_y_min INTEGER")
    if "full_y_max" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_y_max INTEGER")
    if "full_x_min" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_x_min INTEGER")
    if "full_x_max" not in event_columns:
        connection.execute("ALTER TABLE events ADD COLUMN full_x_max INTEGER")
    connection.commit()
    return connection


def file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def insert_run(
    connection: sqlite3.Connection,
    reference_path: Path,
    reference_dataset_root: Path,
    reference_recon_file: Path,
    dataset_path: str,
    args: argparse.Namespace,
    baseline_sigma: float,
    threshold_value: float,
    crop_offsets: tuple[int, int, int],
) -> int:
    crop_z_start, crop_y_start, crop_x_start = crop_offsets
    cursor = connection.execute(
        """
        INSERT INTO runs (
            reference_path,
            reference_dataset_root,
            reference_reconstruction_file,
            dataset_path,
            start_number,
            stop_number,
            baseline_sigma,
            threshold_sigma,
            threshold_value,
            min_event_size,
            merge_gap,
            crop_z_start,
            crop_y_start,
            crop_x_start,
            max_events
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(reference_path),
            str(reference_dataset_root),
            str(reference_recon_file),
            dataset_path,
            args.start_number,
            args.stop_number,
            baseline_sigma,
            args.threshold_sigma,
            threshold_value,
            args.min_event_size,
            args.merge_gap,
            crop_z_start,
            crop_y_start,
            crop_x_start,
            args.max_events,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def insert_comparison_result(
    connection: sqlite3.Connection,
    run_id: int,
    sequence_number: int,
    previous_sequence_number: int,
    previous_dataset_root: Path,
    previous_recon_file: Path,
    dataset_root: Path,
    recon_file: Path,
    events: list[Event3D],
    stored_events: list[Event3D],
    max_abs_diff: float,
    crop_offsets: tuple[int, int, int],
) -> None:
    crop_z_start, crop_y_start, crop_x_start = crop_offsets
    cursor = connection.execute(
        """
        INSERT INTO comparisons (
            run_id,
            sequence_number,
            previous_sequence_number,
            previous_dataset_root,
            previous_reconstruction_file,
            previous_reconstruction_mtime,
            dataset_root,
            reconstruction_file,
            reconstruction_mtime,
            detected_event_count,
            stored_event_count,
            max_abs_diff
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            sequence_number,
            previous_sequence_number,
            str(previous_dataset_root),
            str(previous_recon_file),
            file_mtime_iso(previous_recon_file),
            str(dataset_root),
            str(recon_file),
            file_mtime_iso(recon_file),
            len(events),
            len(stored_events),
            max_abs_diff,
        ),
    )
    comparison_id = int(cursor.lastrowid)

    for rank, event in enumerate(stored_events, start=1):
        z_centroid, y_centroid, x_centroid = event_centroid(event)
        full_z_centroid = z_centroid + crop_z_start
        full_y_centroid = y_centroid + crop_y_start
        full_x_centroid = x_centroid + crop_x_start
        connection.execute(
            """
            INSERT INTO events (
                comparison_id,
                event_rank,
                voxel_count,
                peak_abs_diff,
                peak_signed_diff,
                mean_abs_diff,
                mean_signed_diff,
                z_centroid,
                y_centroid,
                x_centroid,
                z_min,
                z_max,
                y_min,
                y_max,
                x_min,
                x_max,
                full_z_centroid,
                full_y_centroid,
                full_x_centroid,
                full_z_min,
                full_z_max,
                full_y_min,
                full_y_max,
                full_x_min,
                full_x_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_id,
                rank,
                event.voxel_count,
                event.peak_abs_diff,
                event.peak_signed_diff,
                event.sum_abs_diff / max(event.voxel_count, 1),
                event.sum_signed_diff / max(event.voxel_count, 1),
                z_centroid,
                y_centroid,
                x_centroid,
                event.z_min,
                event.z_max,
                event.y_min,
                event.y_max,
                event.x_min,
                event.x_max,
                full_z_centroid,
                full_y_centroid,
                full_x_centroid,
                event.z_min + crop_z_start,
                event.z_max + crop_z_start,
                event.y_min + crop_y_start,
                event.y_max + crop_y_start,
                event.x_min + crop_x_start,
                event.x_max + crop_x_start,
            ),
        )
    connection.commit()


def export_events_csv(connection: sqlite3.Connection, run_id: int, output_db: Path) -> Path:
    output_csv = output_db.with_suffix(".csv")
    rows = connection.execute(
        """
        SELECT
            c.sequence_number,
            c.previous_sequence_number,
            c.dataset_root,
            c.previous_dataset_root,
            c.reconstruction_file,
            c.previous_reconstruction_file,
            c.reconstruction_mtime,
            c.previous_reconstruction_mtime,
            e.event_rank,
            e.voxel_count,
            e.peak_abs_diff,
            e.peak_signed_diff,
            e.mean_abs_diff,
            e.mean_signed_diff,
            e.z_centroid,
            e.y_centroid,
            e.x_centroid,
            e.z_min,
            e.z_max,
            e.y_min,
            e.y_max,
            e.x_min,
            e.x_max,
            e.full_z_centroid,
            e.full_y_centroid,
            e.full_x_centroid,
            e.full_z_min,
            e.full_z_max,
            e.full_y_min,
            e.full_y_max,
            e.full_x_min,
            e.full_x_max
        FROM events e
        JOIN comparisons c ON c.id = e.comparison_id
        WHERE c.run_id = ?
        ORDER BY c.sequence_number, e.event_rank
        """,
        (run_id,),
    )
    fieldnames = [
        "sequence_number",
        "previous_sequence_number",
        "dataset_root",
        "previous_dataset_root",
        "reconstruction_file",
        "previous_reconstruction_file",
        "reconstruction_mtime",
        "previous_reconstruction_mtime",
        "event_rank",
        "voxel_count",
        "peak_abs_diff",
        "peak_signed_diff",
        "mean_abs_diff",
        "mean_signed_diff",
        "z_centroid",
        "y_centroid",
        "x_centroid",
        "z_min",
        "z_max",
        "y_min",
        "y_max",
        "x_min",
        "x_max",
        "full_z_centroid",
        "full_y_centroid",
        "full_x_centroid",
        "full_z_min",
        "full_z_max",
        "full_y_min",
        "full_y_max",
        "full_x_min",
        "full_x_max",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return output_csv


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.threshold_sigma <= 0:
        LOGGER.error("--threshold-sigma must be > 0")
        return 1
    if args.absolute_threshold is not None and args.absolute_threshold <= 0:
        LOGGER.error("--absolute-threshold must be > 0")
        return 1
    if args.preview_diff_noise_floor is not None and args.preview_diff_noise_floor < 0:
        LOGGER.error("--preview-diff-noise-floor must be >= 0")
        return 1
    if args.preview_diff_floor_fraction < 0:
        LOGGER.error("--preview-diff-floor-fraction must be >= 0")
        return 1
    if args.min_event_size <= 0:
        LOGGER.error("--min-event-size must be > 0")
        return 1
    if args.merge_gap < 0:
        LOGGER.error("--merge-gap must be >= 0")
        return 1
    if args.max_events <= 0:
        LOGGER.error("--max-events must be > 0")
        return 1
    if args.jobs <= 0:
        LOGGER.error("--jobs must be > 0")
        return 1
    if args.gif_downsample <= 0:
        LOGGER.error("--gif-downsample must be > 0")
        return 1
    if args.gif_fps <= 0:
        LOGGER.error("--gif-fps must be > 0")
        return 1
    if args.gif_only and args.preview:
        LOGGER.error("--gif-only cannot be used with --preview")
        return 1

    reference_path = Path(args.reference_path).expanduser()
    output_db_arg = Path(args.output_db).expanduser()

    try:
        reference_dataset_root, reference_recon_file = resolve_reconstruction_target(reference_path, args.dataset_path)
        if output_db_arg.is_absolute():
            output_db = output_db_arg.resolve()
        else:
            output_db = (reference_dataset_root.parent / output_db_arg).resolve()
        dataset_path = resolve_volume_dataset(reference_recon_file, args.dataset_path)
        crop_offsets = crop_offsets_for_volume(
            reference_recon_file,
            dataset_path,
            args.crop_z,
            args.crop_y,
            args.crop_x,
        )
        if args.preview:
            (
                preview_sequence,
                preview_dataset_root,
                preview_recon_file,
                preview_previous_sequence,
                preview_previous_dataset_root,
                preview_previous_recon_file,
            ) = resolve_preview_comparison(
                reference_dataset_root,
                args.start_number,
                args.stop_number,
                dataset_path,
                args.preview_sequence,
            )
            LOGGER.info("Series anchor dataset: %s", reference_dataset_root)
            LOGGER.info("Series anchor reconstruction: %s", reference_recon_file)
            LOGGER.info("Dataset path: %s", dataset_path)
            LOGGER.info("Event detection workers: %d", args.jobs)
            LOGGER.info(
                "Preview comparison selected: #%04d %s minus #%04d %s",
                preview_sequence,
                preview_dataset_root,
                preview_previous_sequence,
                preview_previous_dataset_root,
            )
            if args.absolute_threshold is not None:
                baseline_sigma = 0.0
                threshold_value = float(args.absolute_threshold)
                LOGGER.info("Baseline noise sigma: skipped because --absolute-threshold was provided")
                LOGGER.info("Detection threshold: %.6g (absolute override)", threshold_value)
            else:
                baseline_sigma = estimate_baseline_sigma(
                    preview_previous_recon_file,
                    preview_recon_file,
                    dataset_path,
                    args.noise_target_samples,
                    args.crop_z,
                    args.crop_y,
                    args.crop_x,
                )
                threshold_value = baseline_sigma * args.threshold_sigma
                LOGGER.info(
                    "Preview baseline noise sigma from #%04d minus #%04d: %.6g",
                    preview_sequence,
                    preview_previous_sequence,
                    baseline_sigma,
                )
                LOGGER.info("Detection threshold: %.6g", threshold_value)
            preview_z = choose_preview_z(
                preview_previous_recon_file,
                preview_recon_file,
                dataset_path,
                args.preview_z,
                args.crop_z,
                args.crop_y,
                args.crop_x,
            )
            LOGGER.info(
                "Showing preview for stepwise comparison #%04d %s minus #%04d %s at z=%s",
                preview_sequence,
                preview_dataset_root,
                preview_previous_sequence,
                preview_previous_dataset_root,
                preview_z,
            )
            show_detection_preview(
                preview_previous_recon_file,
                preview_recon_file,
                dataset_path,
                threshold_value,
                args.min_event_size,
                args.merge_gap,
                preview_z,
                preview_sequence,
                preview_previous_sequence,
                args,
            )
            LOGGER.info("Preview complete. No database or CSV written.")
            return 0

        comparisons = build_stepwise_comparisons(
            reference_dataset_root,
            args.start_number,
            args.stop_number,
            dataset_path,
        )
        if not comparisons:
            raise RuntimeError("No stepwise comparison reconstructions found in the requested range")

        LOGGER.info("Series anchor dataset: %s", reference_dataset_root)
        LOGGER.info("Series anchor reconstruction: %s", reference_recon_file)
        LOGGER.info("Dataset path: %s", dataset_path)
        LOGGER.info("Event detection workers: %d", args.jobs)
        LOGGER.info(
            "Processing sequence order: %s",
            ", ".join(f"#{sequence_number:04d}" for sequence_number, *_rest in comparisons),
        )

        first_sequence, first_dataset_root, first_recon_file, first_previous_sequence, first_previous_dataset_root, first_previous_recon_file = comparisons[0]
        LOGGER.info(
            "First stepwise comparison for baseline noise: #%04d %s minus #%04d %s",
            first_sequence,
            first_dataset_root,
            first_previous_sequence,
            first_previous_dataset_root,
        )

        center = parse_orthogonal_center(
            args.orthogonal_center,
            volume_shape(first_recon_file, dataset_path, args.crop_z, args.crop_y, args.crop_x),
        )
        requested_planes = [part.strip().lower() for part in args.gif_planes.split(",") if part.strip()]
        valid_planes = {"xy", "xz", "yz"}
        if any(plane not in valid_planes for plane in requested_planes):
            raise RuntimeError(f"--gif-planes must contain only {sorted(valid_planes)}")

        if args.absolute_threshold is not None:
            baseline_sigma = 0.0
            threshold_value = float(args.absolute_threshold)
            LOGGER.info("Baseline noise sigma: skipped because --absolute-threshold was provided")
            LOGGER.info("Detection threshold: %.6g (absolute override)", threshold_value)
        else:
            baseline_sigma = estimate_baseline_sigma(
                first_previous_recon_file,
                first_recon_file,
                dataset_path,
                args.noise_target_samples,
                args.crop_z,
                args.crop_y,
                args.crop_x,
            )
            threshold_value = baseline_sigma * args.threshold_sigma
            LOGGER.info("Baseline noise sigma: %.6g", baseline_sigma)
            LOGGER.info("Detection threshold: %.6g", threshold_value)

        if args.gif_only:
            gif_threshold_value = threshold_value if args.preview_diff_mode == "suppressed" else None
            gif_paths = save_timeseries_gifs(
                comparisons,
                dataset_path,
                output_db,
                center,
                requested_planes,
                args.gif_mode,
                args.gif_downsample,
                args.gif_fps,
                args.gif_labels,
                args.gif_colormap,
                args.gif_diff_colormap,
                args.diff_display_min,
                args.diff_display_max,
                args.preview_diff_mode,
                args.preview_diff_noise_floor,
                args.preview_diff_floor_fraction,
                gif_threshold_value,
                args.crop_z,
                args.crop_y,
                args.crop_x,
                args.jobs,
            )
            for gif_path in gif_paths:
                LOGGER.info("GIF written to %s", gif_path)
            LOGGER.info("GIF export complete. No database or CSV written.")
            return 0

        connection = initialize_database(output_db)
        try:
            run_id = insert_run(
                connection,
                reference_path.resolve(),
                reference_dataset_root,
                reference_recon_file,
                dataset_path,
                args,
                baseline_sigma,
                threshold_value,
                crop_offsets,
            )

            for comparison in comparisons:
                (
                    sequence_number,
                    dataset_root,
                    recon_file,
                    previous_sequence,
                    previous_dataset_root,
                    previous_recon_file,
                ) = comparison
                LOGGER.info(
                    "Processing stepwise comparison #%04d: %s minus #%04d: %s",
                    sequence_number,
                    dataset_root,
                    previous_sequence,
                    previous_dataset_root,
                )
                if args.jobs > 1:
                    LOGGER.info(
                        "Running slice detection for #%04d with %d workers",
                        sequence_number,
                        min(args.jobs, os.cpu_count() or args.jobs),
                    )
                _comparison, events, max_abs_diff = process_comparison_task(
                    comparison,
                    dataset_path,
                    threshold_value,
                    args.min_event_size,
                    args.merge_gap,
                    args.jobs,
                    args.crop_z,
                    args.crop_y,
                    args.crop_x,
                )
                stored_events = events[: args.max_events]
                insert_comparison_result(
                    connection,
                    run_id,
                    sequence_number,
                    previous_sequence,
                    previous_dataset_root,
                    previous_recon_file,
                    dataset_root,
                    recon_file,
                    events,
                    stored_events,
                    max_abs_diff,
                    crop_offsets,
                )
                LOGGER.info(
                    "Recorded %s events (%s stored) for #%04d, max abs diff %.6g",
                    len(events),
                    len(stored_events),
                    sequence_number,
                    max_abs_diff,
                )
            output_csv = export_events_csv(connection, run_id, output_db)
            LOGGER.info("CSV summary written to %s", output_csv)
            if args.save_gifs:
                gif_threshold_value = threshold_value if args.preview_diff_mode == "suppressed" else None
                gif_paths = save_timeseries_gifs(
                    comparisons,
                    dataset_path,
                    output_db,
                    center,
                    requested_planes,
                    args.gif_mode,
                    args.gif_downsample,
                    args.gif_fps,
                    args.gif_labels,
                    args.gif_colormap,
                    args.gif_diff_colormap,
                    args.diff_display_min,
                    args.diff_display_max,
                    args.preview_diff_mode,
                    args.preview_diff_noise_floor,
                    args.preview_diff_floor_fraction,
                    gif_threshold_value,
                    args.crop_z,
                    args.crop_y,
                    args.crop_x,
                    args.jobs,
                )
                for gif_path in gif_paths:
                    LOGGER.info("GIF written to %s", gif_path)
        finally:
            connection.close()
    except Exception as exc:
        log_exception_summary("Event tracking failed", exc)
        return 1

    LOGGER.info("Event tracking complete. Results written to %s", output_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
