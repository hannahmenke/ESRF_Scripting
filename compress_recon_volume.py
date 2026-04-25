#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop, downsample, rescale, and compress a reconstructed 3D volume into a smaller HDF5 file."
    )
    parser.add_argument(
        "input_path",
        help="Input reconstruction HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the compressed output file will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. Default: <input_stem>_compressed.hdf5",
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
    prepared = np.asarray(data, dtype=np.float32, copy=False)

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


def ensure_parent_groups(h5_file: h5py.File, dataset_path: str) -> None:
    parent = Path(dataset_path).parent
    if str(parent) in {"", "."}:
        return
    h5_file.require_group(str(parent))


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def build_output_path(input_path: Path, output_dir: Path, output_name: str | None) -> Path:
    filename = output_name or f"{input_path.stem}_compressed.hdf5"
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
    center: tuple[int, int, int],
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    z_index, y_index, x_index = center
    source_z = target_to_source_index(z_index, crop_z, downsample)
    source_y = target_to_source_index(y_index, crop_y, downsample)
    source_x = target_to_source_index(x_index, crop_x, downsample)

    xy = np.asarray(
        source_dataset[source_z, crop_y[0] : crop_y[1] : downsample, crop_x[0] : crop_x[1] : downsample]
    )
    xz = np.asarray(
        source_dataset[crop_z[0] : crop_z[1] : downsample, source_y, crop_x[0] : crop_x[1] : downsample]
    )
    yz = np.asarray(
        source_dataset[crop_z[0] : crop_z[1] : downsample, crop_y[0] : crop_y[1] : downsample, source_x]
    )

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


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else None
    output_path = build_output_path(input_path, output_dir, args.output_name) if output_dir is not None else None

    if not input_path.is_file():
        LOGGER.error("Input file not found: %s", input_path)
        return 1
    if not args.preview_only and output_dir is None:
        LOGGER.error("--output-dir is required unless --preview-only is used.")
        return 1
    if output_dir is not None and not output_dir.is_dir():
        LOGGER.error("Output directory not found: %s", output_dir)
        return 1
    if args.downsample < 1:
        LOGGER.error("--downsample must be >= 1")
        return 1
    if args.compression == "gzip" and not (0 <= args.compression_level <= 9):
        LOGGER.error("--compression-level must be between 0 and 9 for gzip")
        return 1

    try:
        dataset_path = resolve_volume_dataset(input_path, args.dataset_path)
        output_dataset_path = args.output_dataset_path or dataset_path

        with h5py.File(input_path, "r") as input_h5:
            source_dataset = read_dataset(input_h5, dataset_path)
            source_shape = tuple(int(v) for v in source_dataset.shape)
            crop_z = parse_crop_range(args.crop_z, source_shape[0], "Z")
            crop_y = parse_crop_range(args.crop_y, source_shape[1], "Y")
            crop_x = parse_crop_range(args.crop_x, source_shape[2], "X")
            target_shape = compute_output_shape(source_shape, crop_z, crop_y, crop_x, args.downsample)
            target_dtype = output_dtype(args, source_dataset.dtype)
            chunk_shape = choose_chunk_shape(target_shape, args)
            preview_center = parse_preview_center(args.preview_center, target_shape)

            LOGGER.info("Input file: %s", input_path)
            if output_dir is not None:
                LOGGER.info("Output directory: %s", output_dir)
            if output_path is not None:
                LOGGER.info("Output file: %s", output_path)
            LOGGER.info("Input dataset: %s", dataset_path)
            LOGGER.info("Input shape: %s", source_shape)
            LOGGER.info("Crop Z: %s", crop_z)
            LOGGER.info("Crop Y: %s", crop_y)
            LOGGER.info("Crop X: %s", crop_x)
            LOGGER.info("Downsample: %s", args.downsample)
            LOGGER.info("Output shape: %s", target_shape)
            LOGGER.info("Output dtype: %s", target_dtype)
            LOGGER.info("Compression: %s", args.compression)
            LOGGER.info("Chunk shape: %s", chunk_shape)
            LOGGER.info("Preview center: %s", preview_center)

            if args.preview or args.preview_only:
                preview_views = build_preview_views(
                    source_dataset,
                    crop_z,
                    crop_y,
                    crop_x,
                    args.downsample,
                    preview_center,
                    args,
                )
                show_preview(preview_views, preview_center, target_shape, args)
                if args.preview_only:
                    LOGGER.info("Preview-only mode: no output file written.")
                    return 0

            compression = None if args.compression == "none" else args.compression
            compression_opts = args.compression_level if compression == "gzip" else None

            with h5py.File(output_path, "w") as output_h5:
                copy_attrs(input_h5.attrs, output_h5.attrs)
                ensure_parent_groups(output_h5, output_dataset_path)
                output_dataset = output_h5.create_dataset(
                    output_dataset_path,
                    shape=target_shape,
                    dtype=target_dtype,
                    chunks=chunk_shape,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                copy_attrs(source_dataset.attrs, output_dataset.attrs)

                for output_z, source_z in enumerate(range(crop_z[0], crop_z[1], args.downsample)):
                    slice_data = np.asarray(
                        source_dataset[source_z, crop_y[0] : crop_y[1] : args.downsample, crop_x[0] : crop_x[1] : args.downsample]
                    )
                    prepared_slice = prepare_output_data(slice_data, args)
                    output_dataset[output_z] = prepared_slice

        LOGGER.info("Compressed reconstruction written to %s", output_path)
        return 0
    except Exception as exc:
        log_exception_summary("Compression failed", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
