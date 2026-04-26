#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_STATS_TARGET_SAMPLES = 1_000_000
DEFAULT_SUGGESTED_OUTPUT_VOXELS = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop, downsample, rescale, and compress a reconstructed 3D volume into a smaller HDF5 file."
    )
    parser.add_argument(
        "input_path",
        help="Input dataset directory or reconstruction HDF5 file.",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=None,
        help="First sequence number to include when compressing a numbered series. Default uses the first discovered member.",
    )
    parser.add_argument(
        "--stop-number",
        type=int,
        default=None,
        help="Last sequence number to include when compressing a numbered series. Default uses the last discovered member.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where compressed output files will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename for single-file mode. Default: <input_stem>_compressed.hdf5",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show orthogonal preview slices after applying crop/downsample/clipping/masking.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show the preview and exit without writing an output file.",
    )
    parser.add_argument(
        "--preview-center",
        default=None,
        help="Comma-separated z,y,x center indices in output-space for preview slices. Default: center of transformed volume.",
    )
    parser.add_argument(
        "--preview-sequence",
        type=int,
        default=None,
        help="Sequence number to preview within a numbered series. Default uses the first selected member.",
    )
    parser.add_argument(
        "--preview-colormap",
        default="gray",
        help="Matplotlib colormap for preview images. Default: gray.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the 3D reconstruction volume.",
    )
    parser.add_argument(
        "--output-dataset-path",
        default=None,
        help="Dataset path inside the output file. Default matches the input dataset path.",
    )
    parser.add_argument(
        "--crop-z",
        default=None,
        help="Crop along Z as start:stop.",
    )
    parser.add_argument(
        "--crop-y",
        default=None,
        help="Crop along Y as start:stop.",
    )
    parser.add_argument(
        "--crop-x",
        default=None,
        help="Crop along X as start:stop.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Uniform downsampling factor across all axes. Default: 1.",
    )
    parser.add_argument(
        "--downsample-mode",
        choices=("average", "subsample"),
        default="average",
        help="Downsampling mode. `average` performs block averaging, `subsample` keeps every Nth voxel. Default: average.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of reconstruction files to compress in parallel. Default: 1.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Lower intensity bound before output conversion.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Upper intensity bound before output conversion.",
    )
    parser.add_argument(
        "--to-uint8",
        action="store_true",
        help="Convert the output to uint8 using clip-min/clip-max as the scaling window.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Set voxels with absolute value below this threshold to zero before writing.",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression filter. Default: gzip.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level from 0 to 9. Default: 4.",
    )
    parser.add_argument(
        "--chunk-z",
        type=int,
        default=None,
        help="Optional chunk size along Z. Default chooses automatically.",
    )
    parser.add_argument(
        "--chunk-y",
        type=int,
        default=None,
        help="Optional chunk size along Y. Default chooses automatically.",
    )
    parser.add_argument(
        "--chunk-x",
        type=int,
        default=None,
        help="Optional chunk size along X. Default chooses automatically.",
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


def find_candidate_datasets(h5_file: h5py.File) -> list[str]:
    candidates: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim != 3:
            return
        if obj.dtype.kind not in "uif":
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
            if dataset.ndim != 3:
                raise RuntimeError(f"{dataset_path} is not a 3D dataset")
            return dataset_path

        candidates = find_candidate_datasets(h5_file)
        if not candidates:
            raise RuntimeError(f"No numeric 3D datasets found in {input_path}")
        return candidates[0]


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
    base_dir = dataset_root / "reconstructed_volumes"
    if not base_dir.is_dir():
        return []
    return sorted((path for path in base_dir.rglob("*.hdf5") if path.is_file()), key=lambda path: path.stat().st_mtime)


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


def list_series_reconstructions(
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
    return results


def select_series_reconstructions(
    all_reconstructions: list[tuple[int, Path, Path]],
    start_number: int | None,
    stop_number: int | None,
) -> list[tuple[int, Path, Path]]:
    if not all_reconstructions:
        return []
    low = min(item[0] for item in all_reconstructions) if start_number is None else start_number
    high = max(item[0] for item in all_reconstructions) if stop_number is None else stop_number
    if low > high:
        low, high = high, low
    return [item for item in all_reconstructions if low <= item[0] <= high]


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


def output_dtype(args: argparse.Namespace, source_dtype: np.dtype) -> np.dtype:
    if args.to_uint8:
        return np.dtype(np.uint8)
    return np.dtype(source_dtype)


def compute_output_shape(
    source_shape: tuple[int, int, int],
    crop_z: tuple[int, int],
    crop_y: tuple[int, int],
    crop_x: tuple[int, int],
    downsample: int,
) -> tuple[int, int, int]:
    cropped_shape = (
        crop_z[1] - crop_z[0],
        crop_y[1] - crop_y[0],
        crop_x[1] - crop_x[0],
    )
    return tuple(int(math.ceil(size / downsample)) for size in cropped_shape)


def downsample_average_2d(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(data)
    data = np.asarray(data, dtype=np.float32)
    out_y = int(math.ceil(data.shape[0] / factor))
    out_x = int(math.ceil(data.shape[1] / factor))
    output = np.empty((out_y, out_x), dtype=np.float32)
    for out_y_index in range(out_y):
        y0 = out_y_index * factor
        y1 = min(y0 + factor, data.shape[0])
        for out_x_index in range(out_x):
            x0 = out_x_index * factor
            x1 = min(x0 + factor, data.shape[1])
            output[out_y_index, out_x_index] = float(np.mean(data[y0:y1, x0:x1], dtype=np.float32))
    return output


def downsample_average_3d(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(data)
    data = np.asarray(data, dtype=np.float32)
    out_z = int(math.ceil(data.shape[0] / factor))
    out_y = int(math.ceil(data.shape[1] / factor))
    out_x = int(math.ceil(data.shape[2] / factor))
    output = np.empty((out_z, out_y, out_x), dtype=np.float32)
    for out_z_index in range(out_z):
        z0 = out_z_index * factor
        z1 = min(z0 + factor, data.shape[0])
        output[out_z_index] = downsample_average_2d(
            np.mean(data[z0:z1], axis=0, dtype=np.float32),
            factor,
        )
    return output


def downsample_slice(data: np.ndarray, factor: int, mode: str) -> np.ndarray:
    if factor <= 1:
        return np.asarray(data)
    if mode == "subsample":
        return np.asarray(data[::factor, ::factor])
    return downsample_average_2d(data, factor)


def downsample_volume(data: np.ndarray, factor: int, mode: str) -> np.ndarray:
    if factor <= 1:
        return np.asarray(data)
    if mode == "subsample":
        return np.asarray(data[::factor, ::factor, ::factor])
    return downsample_average_3d(data, factor)


def choose_sampling_step(shape: tuple[int, int, int], target_sample_count: int) -> int:
    total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
    if target_sample_count <= 0:
        return 1
    step = int(math.ceil((total_voxels / target_sample_count) ** (1.0 / 3.0)))
    return max(step, 1)


def choose_chunk_shape(shape: tuple[int, int, int], args: argparse.Namespace) -> tuple[int, int, int]:
    if args.chunk_z is not None or args.chunk_y is not None or args.chunk_x is not None:
        return (
            args.chunk_z or shape[0],
            args.chunk_y or min(shape[1], 256),
            args.chunk_x or min(shape[2], 256),
        )
    return (
        max(1, min(shape[0], 8)),
        max(1, min(shape[1], 256)),
        max(1, min(shape[2], 256)),
    )


def prepare_output_data(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    prepared = np.asarray(data, dtype=np.float32)

    if args.clip_min is not None or args.clip_max is not None:
        clip_min = args.clip_min if args.clip_min is not None else float(np.min(prepared))
        clip_max = args.clip_max if args.clip_max is not None else float(np.max(prepared))
        prepared = np.clip(prepared, clip_min, clip_max)
    else:
        clip_min = float(np.min(prepared))
        clip_max = float(np.max(prepared))

    if args.mask_threshold is not None:
        prepared = prepared.copy()
        prepared[np.abs(prepared) < args.mask_threshold] = 0.0

    if args.to_uint8:
        if args.clip_min is None or args.clip_max is None:
            raise RuntimeError("--to-uint8 requires both --clip-min and --clip-max")
        if args.clip_min >= args.clip_max:
            raise RuntimeError("--clip-min must be less than --clip-max for uint8 conversion")
        scaled = (prepared - args.clip_min) / (args.clip_max - args.clip_min)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    return prepared.astype(data.dtype, copy=False)


def estimate_transformed_statistics(
    source_dataset: h5py.Dataset,
    crop_z: tuple[int, int],
    crop_y: tuple[int, int],
    crop_x: tuple[int, int],
    downsample: int,
    target_sample_count: int = DEFAULT_STATS_TARGET_SAMPLES,
) -> dict[str, float | tuple[int, int, int]]:
    output_shape = compute_output_shape(source_dataset.shape, crop_z, crop_y, crop_x, downsample)
    sampling_step = choose_sampling_step(output_shape, target_sample_count)
    source_step = downsample * sampling_step
    samples: list[np.ndarray] = []

    for source_z in range(crop_z[0], crop_z[1], source_step):
        sampled = np.asarray(
            source_dataset[
                source_z,
                crop_y[0] : crop_y[1] : source_step,
                crop_x[0] : crop_x[1] : source_step,
            ],
            dtype=np.float32,
        )
        if sampled.size:
            samples.append(sampled.reshape(-1))

    if not samples:
        raise RuntimeError("No samples available for transformed preview statistics")

    sample = np.concatenate(samples).astype(np.float32, copy=False)
    percentiles = np.percentile(sample, [0.1, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5, 99.9])
    median = float(percentiles[4])
    mad = float(np.median(np.abs(sample - median)))
    robust_sigma = mad / 0.6744897501960817 if mad > 0 else float(np.std(sample))

    return {
        "sample_count": int(sample.size),
        "sampling_step": sampling_step,
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "p0_1": float(percentiles[0]),
        "p0_5": float(percentiles[1]),
        "p1": float(percentiles[2]),
        "p5": float(percentiles[3]),
        "p50": median,
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "p99_5": float(percentiles[7]),
        "p99_9": float(percentiles[8]),
        "robust_sigma": float(robust_sigma),
    }


def estimate_output_size_bytes(shape: tuple[int, int, int], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


def format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def scaled_shape_for_downsample(shape: tuple[int, int, int], downsample: int) -> tuple[int, int, int]:
    return tuple(int(math.ceil(size / downsample)) for size in shape)


def log_preview_suggestions(
    stats: dict[str, float | tuple[int, int, int]],
    target_shape: tuple[int, int, int],
    target_dtype: np.dtype,
    downsample: int,
) -> None:
    output_voxels = int(np.prod(target_shape, dtype=np.int64))
    if output_voxels <= DEFAULT_SUGGESTED_OUTPUT_VOXELS:
        suggested_downsample = downsample
    else:
        ratio = (output_voxels / DEFAULT_SUGGESTED_OUTPUT_VOXELS) ** (1.0 / 3.0)
        suggested_downsample = max(2, int(math.ceil(ratio)))
        suggested_downsample = max(suggested_downsample, downsample)
    suggested_mask_threshold = max(0.0, 3.0 * float(stats["robust_sigma"]))
    current_size = estimate_output_size_bytes(target_shape, target_dtype)
    uint8_size = estimate_output_size_bytes(target_shape, np.uint8)
    downsample2_shape = scaled_shape_for_downsample(target_shape, 2)
    downsample4_shape = scaled_shape_for_downsample(target_shape, 4)
    downsample2_size = estimate_output_size_bytes(downsample2_shape, target_dtype)
    downsample4_size = estimate_output_size_bytes(downsample4_shape, target_dtype)

    LOGGER.info(
        "Preview stats (sampled %s voxels, step %s): min=%g max=%g p0.5=%g p1=%g p5=%g p50=%g p95=%g p99=%g p99.5=%g p99.9=%g sigma~=%g",
        stats["sample_count"],
        stats["sampling_step"],
        stats["min"],
        stats["max"],
        stats["p0_5"],
        stats["p1"],
        stats["p5"],
        stats["p50"],
        stats["p95"],
        stats["p99"],
        stats["p99_5"],
        stats["p99_9"],
        stats["robust_sigma"],
    )
    LOGGER.info(
        "Suggested flags: --clip-min %.6g --clip-max %.6g --mask-threshold %.6g",
        float(stats["p0_5"]),
        float(stats["p99_5"]),
        suggested_mask_threshold,
    )
    LOGGER.info(
        "Estimated output size: current dtype %s, uint8 %s%s",
        format_bytes(current_size),
        format_bytes(uint8_size),
        "" if suggested_downsample == downsample else f" | consider --downsample {suggested_downsample}",
    )
    LOGGER.info(
        "Downsample size guide: --downsample 2 -> shape=%s, size=%s | --downsample 4 -> shape=%s, size=%s",
        downsample2_shape,
        format_bytes(downsample2_size),
        downsample4_shape,
        format_bytes(downsample4_size),
    )
    LOGGER.info(
        "Compression tradeoff guide: keep --compression gzip --compression-level 4 for balanced speed/size | "
        "try --compression-level 6 for smaller files with slower writes | try --compression lzf for faster writes with larger files"
    )


def ensure_parent_groups(h5_file: h5py.File, dataset_path: str) -> None:
    parent = Path(dataset_path).parent
    if str(parent) in {"", "."}:
        return
    h5_file.require_group(str(parent))


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def build_output_path(
    input_path: Path,
    output_dir: Path,
    output_name: str | None,
    dataset_root: Path | None = None,
) -> Path:
    base_name = dataset_root.name if dataset_root is not None and is_dataset_directory(dataset_root) else input_path.stem
    filename = output_name or f"{base_name}_compressed.hdf5"
    return output_dir / filename


def parse_preview_center(raw_center: str | None, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if raw_center:
        values = [int(part.strip()) for part in raw_center.split(",") if part.strip()]
        if len(values) != 3:
            raise RuntimeError("--preview-center must provide exactly 3 indices as z,y,x")
        center = tuple(values)
    else:
        center = tuple(size // 2 for size in shape)

    for axis, (index, axis_size) in enumerate(zip(center, shape)):
        if index < 0 or index >= axis_size:
            raise RuntimeError(f"Preview center index {index} is out of range for axis {axis} with size {axis_size}")
    return center


def target_to_source_index(index: int, crop_range: tuple[int, int], downsample: int) -> int:
    return crop_range[0] + index * downsample


def build_preview_views(
    source_dataset: h5py.Dataset,
    crop_z: tuple[int, int],
    crop_y: tuple[int, int],
    crop_x: tuple[int, int],
    downsample: int,
    downsample_mode: str,
    center: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    z_index, y_index, x_index = center
    z0 = target_to_source_index(z_index, crop_z, downsample)
    z1 = min(z0 + downsample, crop_z[1])
    y0 = target_to_source_index(y_index, crop_y, downsample)
    y1 = min(y0 + downsample, crop_y[1])
    x0 = target_to_source_index(x_index, crop_x, downsample)
    x1 = min(x0 + downsample, crop_x[1])

    xy_source = np.asarray(
        source_dataset[z0:z1, crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]],
        dtype=np.float32,
    )
    if xy_source.ndim == 3:
        xy_source = np.mean(xy_source, axis=0, dtype=np.float32)
    xy = downsample_slice(xy_source, downsample, downsample_mode)

    xz_source = np.asarray(
        source_dataset[crop_z[0] : crop_z[1], y0:y1, crop_x[0] : crop_x[1]],
        dtype=np.float32,
    )
    if xz_source.ndim == 3:
        xz_source = np.mean(xz_source, axis=1, dtype=np.float32)
    xz = downsample_slice(xz_source, downsample, downsample_mode)

    yz_source = np.asarray(
        source_dataset[crop_z[0] : crop_z[1], crop_y[0] : crop_y[1], x0:x1],
        dtype=np.float32,
    )
    if yz_source.ndim == 3:
        yz_source = np.mean(yz_source, axis=2, dtype=np.float32)
    yz = downsample_slice(yz_source, downsample, downsample_mode)

    return [
        ("XY", prepare_output_data(xy, args)),
        ("XZ", prepare_output_data(xz, args)),
        ("YZ", prepare_output_data(yz, args)),
    ]


def show_preview(
    views: list[tuple[str, np.ndarray]],
    center: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.atleast_1d(axes).ravel()
    for ax, (plane_name, image) in zip(axes, views):
        artist = ax.imshow(image, cmap=args.preview_colormap)
        ax.set_title(f"{plane_name} @ {center}")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    axes[len(views)].axis("off")
    axes[0].figure.suptitle(
        f"Preview of transformed volume\nshape={target_shape}, center={center}, dtype={views[0][1].dtype}"
    )
    plt.tight_layout()
    plt.show()


def format_progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def consume_progress_events(progress_queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            event = progress_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        event_type = event[0]
        if event_type == "start":
            _kind, sequence_number, total_slices = event
            LOGGER.info(
                "Starting compression for #%04d with %d output slices %s",
                sequence_number,
                total_slices,
                format_progress_bar(0, total_slices),
            )
        elif event_type == "progress":
            _kind, sequence_number, completed_slices, total_slices = event
            LOGGER.info(
                "Compression progress #%04d: %s %d/%d slices",
                sequence_number,
                format_progress_bar(completed_slices, total_slices),
                completed_slices,
                total_slices,
            )


def compress_reconstruction_task(
    sequence_number: int,
    input_path: str,
    output_path: str,
    dataset_path: str,
    output_dataset_path: str,
    crop_z_raw: str | None,
    crop_y_raw: str | None,
    crop_x_raw: str | None,
    downsample: int,
    downsample_mode: str,
    clip_min: float | None,
    clip_max: float | None,
    to_uint8: bool,
    mask_threshold: float | None,
    compression: str,
    compression_level: int,
    chunk_z: int | None,
    chunk_y: int | None,
    chunk_x: int | None,
    progress_queue=None,
) -> str:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    worker_args = argparse.Namespace(
        clip_min=clip_min,
        clip_max=clip_max,
        to_uint8=to_uint8,
        mask_threshold=mask_threshold,
        compression=compression,
        compression_level=compression_level,
        chunk_z=chunk_z,
        chunk_y=chunk_y,
        chunk_x=chunk_x,
    )

    with h5py.File(input_path_obj, "r") as input_h5:
        source_dataset = read_dataset(input_h5, dataset_path)
        source_shape = tuple(int(v) for v in source_dataset.shape)
        crop_z = parse_crop_range(crop_z_raw, source_shape[0], "Z")
        crop_y = parse_crop_range(crop_y_raw, source_shape[1], "Y")
        crop_x = parse_crop_range(crop_x_raw, source_shape[2], "X")
        target_shape = compute_output_shape(source_shape, crop_z, crop_y, crop_x, downsample)
        target_dtype = output_dtype(worker_args, source_dataset.dtype)
        chunk_shape = choose_chunk_shape(target_shape, worker_args)
        hdf5_compression = None if compression == "none" else compression
        compression_opts = compression_level if hdf5_compression == "gzip" else None

        with h5py.File(output_path_obj, "w") as output_h5:
            copy_attrs(input_h5.attrs, output_h5.attrs)
            ensure_parent_groups(output_h5, output_dataset_path)
            output_dataset = output_h5.create_dataset(
                output_dataset_path,
                shape=target_shape,
                dtype=target_dtype,
                chunks=chunk_shape,
                compression=hdf5_compression,
                compression_opts=compression_opts,
            )
            copy_attrs(source_dataset.attrs, output_dataset.attrs)

            total_slices = target_shape[0]
            milestones = {
                max(1, int(math.ceil(total_slices * fraction / 100.0)))
                for fraction in range(1, 101)
            }
            last_progress_time = time.monotonic()
            if progress_queue is not None:
                progress_queue.put(("start", sequence_number, total_slices))

            if downsample_mode == "subsample":
                source_z_iterable = range(crop_z[0], crop_z[1], downsample)
            else:
                source_z_iterable = range(crop_z[0], crop_z[1], downsample)

            for output_z, source_z in enumerate(source_z_iterable, start=1):
                if downsample_mode == "subsample":
                    slice_data = np.asarray(
                        source_dataset[
                            source_z,
                            crop_y[0] : crop_y[1] : downsample,
                            crop_x[0] : crop_x[1] : downsample,
                        ]
                    )
                else:
                    source_z_end = min(source_z + downsample, crop_z[1])
                    block = np.asarray(
                        source_dataset[
                            source_z:source_z_end,
                            crop_y[0] : crop_y[1],
                            crop_x[0] : crop_x[1],
                        ],
                        dtype=np.float32,
                    )
                    slice_data = downsample_volume(block, downsample, "average")[0]
                prepared_slice = prepare_output_data(slice_data, worker_args)
                output_dataset[output_z - 1] = prepared_slice
                now = time.monotonic()
                if output_z in milestones or output_z == total_slices or (now - last_progress_time) >= 60.0:
                    if progress_queue is not None:
                        progress_queue.put(("progress", sequence_number, output_z, total_slices))
                    last_progress_time = now

    return str(output_path_obj)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else None

    if not input_path.exists():
        LOGGER.error("Input path not found: %s", input_path)
        return 1
    if not args.preview_only and output_dir is None:
        LOGGER.error("--output-dir is required unless --preview-only is used.")
        return 1
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is not None and (args.start_number is not None or args.stop_number is not None):
        LOGGER.error("--output-name cannot be used together with --start-number/--stop-number series selection.")
        return 1
    if args.downsample < 1:
        LOGGER.error("--downsample must be >= 1")
        return 1
    if args.jobs < 1:
        LOGGER.error("--jobs must be >= 1")
        return 1
    if args.compression == "gzip" and not (0 <= args.compression_level <= 9):
        LOGGER.error("--compression-level must be between 0 and 9 for gzip")
        return 1

    try:
        reference_dataset_root, reference_recon_file = resolve_reconstruction_target(input_path, args.dataset_path)
        dataset_path = resolve_volume_dataset(reference_recon_file, args.dataset_path)
        output_dataset_path = args.output_dataset_path or dataset_path

        if is_dataset_directory(reference_dataset_root):
            all_reconstructions = list_series_reconstructions(reference_dataset_root, args.dataset_path)
        else:
            all_reconstructions = [(0, reference_dataset_root, reference_recon_file)]
        if not all_reconstructions:
            raise RuntimeError(f"No reconstruction series members found for {reference_dataset_root}")

        for sequence_number, dataset_root, recon_file in all_reconstructions:
            LOGGER.info(
                "Admitted series member #%04d: dataset=%s | recon=%s",
                sequence_number,
                dataset_root,
                recon_file,
            )

        selected_reconstructions = select_series_reconstructions(
            all_reconstructions,
            args.start_number,
            args.stop_number,
        )
        if not selected_reconstructions:
            raise RuntimeError("No reconstruction series members found in the requested range")
        if len(selected_reconstructions) > 1 and args.output_name is not None:
            raise RuntimeError("--output-name can only be used when exactly one reconstruction is selected")

        preview_target = selected_reconstructions[0]
        if args.preview_sequence is not None:
            matching = [item for item in selected_reconstructions if item[0] == args.preview_sequence]
            if not matching:
                raise RuntimeError(f"No selected reconstruction found for preview sequence #{args.preview_sequence:04d}")
            preview_target = matching[0]

        LOGGER.info("Series anchor dataset: %s", reference_dataset_root)
        LOGGER.info("Series anchor reconstruction: %s", reference_recon_file)
        LOGGER.info(
            "Processing sequence order: %s",
            ", ".join(f"#{sequence_number:04d}" for sequence_number, *_rest in selected_reconstructions),
        )
        if output_dir is not None:
            LOGGER.info("Output directory: %s", output_dir)
        LOGGER.info("Input dataset: %s", dataset_path)

        preview_sequence, preview_dataset_root, preview_input_path = preview_target

        with h5py.File(preview_input_path, "r") as preview_h5:
            source_dataset = read_dataset(preview_h5, dataset_path)
            source_shape = tuple(int(v) for v in source_dataset.shape)
            crop_z = parse_crop_range(args.crop_z, source_shape[0], "Z")
            crop_y = parse_crop_range(args.crop_y, source_shape[1], "Y")
            crop_x = parse_crop_range(args.crop_x, source_shape[2], "X")
            target_shape = compute_output_shape(source_shape, crop_z, crop_y, crop_x, args.downsample)
            target_dtype = output_dtype(args, source_dataset.dtype)
            chunk_shape = choose_chunk_shape(target_shape, args)
            preview_center = parse_preview_center(args.preview_center, target_shape)

            LOGGER.info("Preview/input file: %s", preview_input_path)
            LOGGER.info("Preview sequence: #%04d", preview_sequence)
            LOGGER.info("Input shape: %s", source_shape)
            LOGGER.info("Crop Z: %s", crop_z)
            LOGGER.info("Crop Y: %s", crop_y)
            LOGGER.info("Crop X: %s", crop_x)
            LOGGER.info("Downsample: %s", args.downsample)
            LOGGER.info("Downsample mode: %s", args.downsample_mode)
            LOGGER.info("Output shape: %s", target_shape)
            LOGGER.info("Output dtype: %s", target_dtype)
            LOGGER.info("Compression: %s", args.compression)
            LOGGER.info("Chunk shape: %s", chunk_shape)
            LOGGER.info("Preview center: %s", preview_center)

            if args.preview or args.preview_only:
                preview_stats = estimate_transformed_statistics(
                    source_dataset,
                    crop_z,
                    crop_y,
                    crop_x,
                    args.downsample,
                )
                log_preview_suggestions(
                    preview_stats,
                    target_shape,
                    target_dtype,
                    args.downsample,
                )
                preview_views = build_preview_views(
                    source_dataset,
                    crop_z,
                    crop_y,
                    crop_x,
                    args.downsample,
                    args.downsample_mode,
                    preview_center,
                    args,
                )
                show_preview(preview_views, preview_center, target_shape, args)
                if args.preview_only:
                    LOGGER.info("Preview-only mode: no output files written.")
                    return 0

        compression = None if args.compression == "none" else args.compression
        tasks = []
        for sequence_number, dataset_root, current_input_path in selected_reconstructions:
            current_output_path = build_output_path(current_input_path, output_dir, args.output_name, dataset_root)
            LOGGER.info("Queueing #%04d from %s", sequence_number, current_input_path)
            LOGGER.info("Output file: %s", current_output_path)
            tasks.append(
                (
                    sequence_number,
                    str(current_input_path),
                    str(current_output_path),
                    dataset_path,
                    output_dataset_path,
                    args.crop_z,
                    args.crop_y,
                    args.crop_x,
                    args.downsample,
                    args.downsample_mode,
                    args.clip_min,
                    args.clip_max,
                    args.to_uint8,
                    args.mask_threshold,
                    args.compression,
                    args.compression_level,
                    args.chunk_z,
                    args.chunk_y,
                    args.chunk_x,
                )
            )

        if args.jobs > 1 and len(tasks) > 1:
            max_workers = min(args.jobs, len(tasks), os.cpu_count() or args.jobs)
            LOGGER.info("Running series compression in parallel with %d workers", max_workers)
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue()
            stop_event = threading.Event()
            consumer_thread = threading.Thread(
                target=consume_progress_events,
                args=(progress_queue, stop_event),
                daemon=True,
            )
            consumer_thread.start()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(compress_reconstruction_task, *task, progress_queue): task[0]
                    for task in tasks
                }
                completed_files = 0
                total_files = len(tasks)
                for future in as_completed(future_map):
                    sequence_number = future_map[future]
                    output_path_str = future.result()
                    completed_files += 1
                    LOGGER.info("Compressed reconstruction #%04d written to %s", sequence_number, output_path_str)
                    LOGGER.info(
                        "Overall series progress: %s %d/%d files",
                        format_progress_bar(completed_files, total_files),
                        completed_files,
                        total_files,
                    )
            stop_event.set()
            consumer_thread.join(timeout=2.0)
            manager.shutdown()
        else:
            for task in tasks:
                sequence_number = task[0]
                output_path_str = compress_reconstruction_task(*task, None)
                LOGGER.info("Compressed reconstruction #%04d written to %s", sequence_number, output_path_str)
        return 0
    except Exception as exc:
        log_exception_summary("Compression failed", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
