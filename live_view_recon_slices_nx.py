#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


POLL_INTERVAL = 30.0
DEFAULT_FIGSIZE = (18, 10)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live view of reconstructed volumes stored inside tomography dataset directories."
    )
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Reference/baseline dataset directory or reconstruction HDF5 file.",
    )
    parser.add_argument(
        "--comparison-path",
        default=None,
        help="Optional explicit comparison/current reconstruction HDF5 file or dataset directory to show instead of auto-following.",
    )
    parser.add_argument(
        "--difference-path",
        default=None,
        help="Deprecated. Use --reference-path for the baseline and --comparison-path for the current reconstruction.",
    )
    parser.add_argument(
        "--show-difference",
        action="store_true",
        help="Display current-minus-reference slices below the current slices.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the reconstruction volume.",
    )
    parser.add_argument(
        "--orthogonal",
        action="store_true",
        help="Show one orthogonal XY/XZ/YZ slice triplet instead of multiple slices along one axis.",
    )
    parser.add_argument(
        "--orthogonal-center",
        default=None,
        help="Comma-separated center indices for orthogonal views as axis0,axis1,axis2. Defaults to the volume center.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Axis along which to extract slices. Default: 0.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=4,
        help="Number of evenly spaced slices to show if --slices is not given. Default: 4.",
    )
    parser.add_argument(
        "--slices",
        default=None,
        help="Comma-separated slice indices to show, e.g. '100,300,500'.",
    )
    parser.add_argument(
        "--position-mode",
        choices=("same", "all"),
        default="same",
        help="Auto-follow only the reference position (`same`) or all positions (`all`). Default: same.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL,
        help="Seconds between checks for a newer reconstruction. Default: 30.",
    )
    parser.add_argument(
        "--colormap",
        default="gray",
        help="Matplotlib colormap for the current reconstruction images. Default: gray.",
    )
    parser.add_argument(
        "--difference-colormap",
        default=None,
        help="Matplotlib colormap for difference images. Default uses --colormap.",
    )
    parser.add_argument(
        "--hot-cold",
        action="store_true",
        help="Use a diverging hot/cold colormap for difference images.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Display downsampling factor. 1 keeps full resolution. Default: 2.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster display mode: downsample by 2 and, in orthogonal mode, skip the slow YZ view.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Render once and keep the window open without polling for updates.",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Crop displayed images after downsampling as y_start:y_stop,x_start:x_stop.",
    )
    parser.add_argument(
        "--crop-x",
        default=None,
        help="Crop displayed images along X after downsampling as start:stop.",
    )
    parser.add_argument(
        "--crop-y",
        default=None,
        help="Crop displayed images along Y after downsampling as start:stop.",
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


def prompt_path(prompt: str, allow_empty: bool = False) -> Path | None:
    while True:
        raw = input(prompt).strip()
        if not raw and allow_empty:
            return None
        path = Path(raw).expanduser()
        if path.exists():
            return path
        LOGGER.warning("Path not found: %s", path)


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


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    return image[::factor, ::factor]


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


def parse_crop_spec(
    crop: str | None,
    crop_x: str | None,
    crop_y: str | None,
) -> tuple[str | None, str | None]:
    parsed_crop_y = crop_y
    parsed_crop_x = crop_x
    if crop is not None:
        parts = [part.strip() for part in crop.split(",", 1)]
        if len(parts) != 2:
            raise RuntimeError("--crop must be formatted as y_start:y_stop,x_start:x_stop")
        if crop_y is not None or crop_x is not None:
            raise RuntimeError("Use either --crop or --crop-x/--crop-y, not both")
        parsed_crop_y, parsed_crop_x = parts
    return parsed_crop_y, parsed_crop_x


def crop_image(image: np.ndarray, crop_x: str | None, crop_y: str | None) -> np.ndarray:
    y_start, y_stop = parse_crop_range(crop_y, image.shape[0], "Y")
    x_start, x_stop = parse_crop_range(crop_x, image.shape[1], "X")
    return image[y_start:y_stop, x_start:x_stop]


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


def parse_slice_indices(raw_slices: str | None, axis_size: int, num_slices: int) -> list[int]:
    if raw_slices:
        values: list[int] = []
        for part in raw_slices.split(","):
            part = part.strip()
            if not part:
                continue
            values.append(int(part))
        if not values:
            raise RuntimeError("No valid slice indices were provided.")
        indices = values
    else:
        if num_slices <= 0:
            raise RuntimeError("--num-slices must be >= 1")
        if num_slices == 1:
            indices = [axis_size // 2]
        else:
            indices = np.linspace(0, axis_size - 1, num_slices, dtype=int).tolist()

    for index in indices:
        if index < 0 or index >= axis_size:
            raise RuntimeError(f"Slice index {index} is out of range for axis size {axis_size}")
    return indices


def parse_orthogonal_center(raw_center: str | None, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if raw_center:
        values = [int(part.strip()) for part in raw_center.split(",") if part.strip()]
        if len(values) != 3:
            raise RuntimeError("--orthogonal-center must provide exactly 3 indices: axis0,axis1,axis2")
        center = tuple(values)
    else:
        center = tuple(size // 2 for size in shape)

    for axis, (index, axis_size) in enumerate(zip(center, shape)):
        if index < 0 or index >= axis_size:
            raise RuntimeError(f"Orthogonal center index {index} is out of range for axis {axis} with size {axis_size}")
    return center


def clamp_orthogonal_center(center: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(max(index, 0), axis_size - 1) for index, axis_size in zip(center, shape))


def clamp_slice_indices(indices: list[int], axis_size: int) -> list[int]:
    return [min(max(index, 0), axis_size - 1) for index in indices]


def extract_slice(volume: h5py.Dataset, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        image = volume[index, :, :]
    elif axis == 1:
        image = volume[:, index, :]
    else:
        image = volume[:, :, index]
    return np.asarray(image, dtype=np.float32)


def extract_slice_downsampled(volume: h5py.Dataset, axis: int, index: int, downsample: int) -> np.ndarray:
    if downsample <= 1:
        return extract_slice(volume, axis, index)
    if axis == 0:
        image = volume[index, ::downsample, ::downsample]
    elif axis == 1:
        image = volume[::downsample, index, ::downsample]
    else:
        image = volume[::downsample, ::downsample, index]
    return np.asarray(image, dtype=np.float32)


def orthogonal_axes(fast: bool) -> list[int]:
    return [0, 1] if fast else [0, 1, 2]


def is_dataset_directory(path: Path) -> bool:
    return path.is_dir() and (path / "projections").is_dir()


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


def dataset_position_name(dataset_root: Path, collection_dir: Path) -> str:
    series_name = dataset_series_name(dataset_root)
    prefix = f"{collection_dir.name}_"
    if series_name.startswith(prefix):
        return series_name[len(prefix):]
    return series_name


def dataset_sequence_number(dataset_root: Path) -> int:
    match = re.search(r"_(\d+)$", dataset_root.name)
    if match is None:
        return 0
    return int(match.group(1))


def candidate_reconstruction_files(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base_name in ("reconstructed_volumes", "reconstructed_slices"):
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


def verify_complete_volume_read(volume: h5py.Dataset) -> None:
    if not is_valid_volume_dataset(volume):
        raise RuntimeError(f"Dataset is not a valid 3D reconstruction volume: shape={volume.shape}")

    z_size, y_size, x_size = (int(size) for size in volume.shape)
    sample_indices = lambda size: sorted({0, size // 2, size - 1})

    for z_index in sample_indices(z_size):
        _ = np.asarray(volume[z_index, :, :], dtype=np.float32)
    for y_index in sample_indices(y_size):
        _ = np.asarray(volume[:, y_index, :], dtype=np.float32)
    for x_index in sample_indices(x_size):
        _ = np.asarray(volume[:, :, x_index], dtype=np.float32)


def is_readable_reconstruction_file(path: Path, dataset_path: str | None = None) -> bool:
    try:
        resolved_dataset_path = resolve_volume_dataset(path, dataset_path)
        with h5py.File(path, "r") as h5_file:
            volume = read_dataset(h5_file, resolved_dataset_path)
            verify_complete_volume_read(volume)
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

    readable_candidates = [path for path in valid_candidates if is_readable_reconstruction_file(path, dataset_path)]
    if not readable_candidates:
        raise RuntimeError(f"No readable reconstruction volume HDF5 files found in {dataset_root}")

    LOGGER.debug(
        "Reconstruction candidates for %s: %s",
        dataset_root,
        [path.name for path in readable_candidates],
    )
    return readable_candidates[-1]


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


def latest_reconstruction_target(
    collection_dir: Path,
    exclude: Path | None = None,
    position_name: str | None = None,
    dataset_path: str | None = None,
) -> tuple[Path, Path] | None:
    best_target: tuple[Path, Path] | None = None
    best_sequence = -1
    best_mtime = float("-inf")

    for dataset_dir in sorted(path for path in collection_dir.iterdir() if is_dataset_directory(path)):
        if exclude is not None and dataset_dir == exclude:
            continue
        if position_name is not None and dataset_position_name(dataset_dir, collection_dir) != position_name:
            continue
        try:
            recon_file = find_latest_reconstruction_file(dataset_dir, dataset_path)
        except Exception:
            continue
        sequence_number = dataset_sequence_number(dataset_dir)
        recon_mtime = recon_file.stat().st_mtime
        if sequence_number > best_sequence or (sequence_number == best_sequence and recon_mtime > best_mtime):
            best_sequence = sequence_number
            best_mtime = recon_mtime
            best_target = (dataset_dir, recon_file)

    return best_target


def resolve_display_target(
    reference_dataset_root: Path,
    reference_recon_file: Path,
    second_path: Path | None,
    position_mode: str,
    dataset_path: str | None = None,
) -> tuple[Path, Path, bool]:
    if second_path is not None:
        dataset_root, recon_file = resolve_reconstruction_target(second_path, dataset_path)
        return dataset_root, recon_file, True

    return reference_dataset_root, reference_recon_file, True


def load_volume_metadata(
    recon_file: Path,
    orthogonal: bool,
    orthogonal_center: str | None,
    axis: int,
    slices: str | None,
    num_slices: int,
    dataset_path: str | None,
) -> tuple[str, tuple[int, ...], list[int], tuple[int, int, int] | None]:
    resolved_dataset_path = resolve_volume_dataset(recon_file, dataset_path)
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, resolved_dataset_path)
        volume_shape = tuple(int(v) for v in volume.shape)
        if orthogonal or orthogonal_center is not None:
            center = parse_orthogonal_center(orthogonal_center, volume_shape)
            return resolved_dataset_path, volume_shape, [], center

        axis_size = int(volume.shape[axis])
        slice_indices = parse_slice_indices(slices, axis_size, num_slices)
        return resolved_dataset_path, volume_shape, slice_indices, None


def load_volume_slices(
    recon_file: Path,
    orthogonal: bool,
    orthogonal_center: tuple[int, int, int] | None,
    axis: int,
    slice_indices: list[int],
    dataset_path: str | None,
    downsample: int,
    fast: bool,
) -> tuple[str, list[np.ndarray]]:
    resolved_dataset_path = resolve_volume_dataset(recon_file, dataset_path)
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, resolved_dataset_path)
        if orthogonal:
            if orthogonal_center is None:
                raise RuntimeError("Orthogonal center is required in orthogonal mode")
            images = [
                extract_slice_downsampled(volume, axis_id, orthogonal_center[axis_id], downsample)
                for axis_id in orthogonal_axes(fast)
            ]
        else:
            images = [
                extract_slice_downsampled(volume, axis, slice_index, downsample)
                for slice_index in slice_indices
            ]
    return resolved_dataset_path, images


def volume_shape(recon_file: Path, dataset_path: str | None) -> tuple[int, int, int]:
    resolved_dataset_path = resolve_volume_dataset(recon_file, dataset_path)
    with h5py.File(recon_file, "r") as h5_file:
        volume = read_dataset(h5_file, resolved_dataset_path)
        return tuple(int(v) for v in volume.shape)


class VolumeCache:
    def __init__(self, dataset_path: str | None = None) -> None:
        self.dataset_path = dataset_path
        self._path: Path | None = None
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._resolved_dataset_path: str | None = None

    def load(
        self,
        path: Path,
        orthogonal: bool,
        orthogonal_center: tuple[int, int, int] | None,
        axis: int,
        slice_indices: list[int],
        downsample: int,
        fast: bool,
    ) -> tuple[str, list[np.ndarray]]:
        if self._path != path:
            self.close()
            self._file = h5py.File(path, "r")
            self._resolved_dataset_path = resolve_volume_dataset(path, self.dataset_path)
            self._dataset = read_dataset(self._file, self._resolved_dataset_path)
            self._path = path

        if self._dataset is None or self._resolved_dataset_path is None:
            raise RuntimeError("Dataset cache is not initialized")

        if orthogonal:
            if orthogonal_center is None:
                raise RuntimeError("Orthogonal center is required in orthogonal mode")
            images = [
                extract_slice_downsampled(self._dataset, axis_id, orthogonal_center[axis_id], downsample)
                for axis_id in orthogonal_axes(fast)
            ]
        else:
            images = [
                extract_slice_downsampled(self._dataset, axis, slice_index, downsample)
                for slice_index in slice_indices
            ]
        return self._resolved_dataset_path, images

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._dataset = None
        self._path = None
        self._resolved_dataset_path = None


def update_display(
    axes: list,
    image_artists: list,
    slice_indices: list[int],
    current_images: list[np.ndarray],
    axis_index: int,
    title: str,
    colormap: str,
    difference_colormap: str,
    orthogonal: bool,
    orthogonal_center: tuple[int, int, int] | None,
    difference_images: list[np.ndarray] | None,
    fast: bool,
    crop_x: str | None,
    crop_y: str | None,
) -> None:
    labels: list[str] = []
    axis_labels: list[tuple[str, str]] = []
    if orthogonal:
        plane_labels = {0: "XY", 1: "XZ", 2: "YZ"}
        labels = [f"{plane_labels[axis_id]} @ {orthogonal_center}" for axis_id in orthogonal_axes(fast)]
        plane_axis_labels = {0: ("X", "Y"), 1: ("X", "Z"), 2: ("Y", "Z")}
        axis_labels = [plane_axis_labels[axis_id] for axis_id in orthogonal_axes(fast)]
    else:
        labels = [f"Axis {axis_index} slice {slice_index}" for slice_index in slice_indices]
        if axis_index == 0:
            axis_labels = [("X", "Y")] * len(slice_indices)
        elif axis_index == 1:
            axis_labels = [("X", "Z")] * len(slice_indices)
        else:
            axis_labels = [("Y", "Z")] * len(slice_indices)

    panel_images: list[np.ndarray] = []
    panel_labels: list[str] = []
    panel_colormaps: list[str] = []
    panel_axis_labels: list[tuple[str, str]] = []
    for label, image in zip(labels, current_images):
        panel_labels.append(label)
        panel_images.append(crop_image(image, crop_x, crop_y))
        panel_colormaps.append(colormap)
        panel_axis_labels.append(axis_labels[len(panel_axis_labels)])
    if difference_images is not None:
        for label, diff_image, axis_label in zip(labels, difference_images, axis_labels):
            panel_labels.append(f"{label} difference")
            panel_images.append(crop_image(diff_image, crop_x, crop_y))
            panel_colormaps.append(difference_colormap)
            panel_axis_labels.append(axis_label)

    for ax, artist, label, image, panel_cmap, panel_axis_label in zip(
        axes, image_artists, panel_labels, panel_images, panel_colormaps, panel_axis_labels
    ):
        if artist is None:
            artist = ax.imshow(image, cmap=panel_cmap)
            ax._live_artist = artist
        else:
            artist.set_data(image)
            artist.set_cmap(panel_cmap)
            data_min = float(np.min(image))
            data_max = float(np.max(image))
            if data_min == data_max:
                data_min -= 0.5
                data_max += 0.5
            artist.set_clim(vmin=data_min, vmax=data_max)
        ax.set_title(label)
        ax.set_xlabel(panel_axis_label[0])
        ax.set_ylabel(panel_axis_label[1])

    if axes:
        axes[0].figure.suptitle(title)
        axes[0].figure.canvas.draw_idle()
    plt.draw()
    plt.pause(0.001)


def make_display_title(
    reference_dataset_root: Path,
    reference_recon_file: Path,
    current_dataset_root: Path,
    current_recon_file: Path,
) -> str:
    return (
        f"reference: {reference_dataset_root.name} | {reference_recon_file.name}\n"
        f"current: {current_dataset_root.name} | {current_recon_file.name}"
    )


def center_crop_to_shape(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise RuntimeError(f"Invalid crop target shape ({rows}, {cols})")
    if image.shape[0] < rows or image.shape[1] < cols:
        raise RuntimeError(f"Cannot crop image shape {image.shape} to ({rows}, {cols})")

    row_start = (image.shape[0] - rows) // 2
    col_start = (image.shape[1] - cols) // 2
    return image[row_start : row_start + rows, col_start : col_start + cols]


def align_image_pairs(
    current_images: list[np.ndarray],
    reference_images: list[np.ndarray] | None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    if reference_images is None:
        return current_images, None

    aligned_current_images: list[np.ndarray] = []
    aligned_reference_images: list[np.ndarray] = []
    for current, reference in zip(current_images, reference_images):
        common_rows = min(current.shape[0], reference.shape[0])
        common_cols = min(current.shape[1], reference.shape[1])
        if common_rows <= 0 or common_cols <= 0:
            raise RuntimeError(
                f"Cannot align empty overlapping slice shapes: "
                f"current={current.shape}, reference={reference.shape}"
            )
        aligned_current_images.append(center_crop_to_shape(current, common_rows, common_cols))
        aligned_reference_images.append(center_crop_to_shape(reference, common_rows, common_cols))

    return aligned_current_images, aligned_reference_images


def compute_difference_images(
    current_images: list[np.ndarray],
    reference_images: list[np.ndarray] | None,
) -> list[np.ndarray] | None:
    aligned_current_images, aligned_reference_images = align_image_pairs(current_images, reference_images)
    if aligned_reference_images is None:
        return None
    difference_images: list[np.ndarray] = []
    for current, reference in zip(aligned_current_images, aligned_reference_images):
        difference_images.append(current - reference)
    return difference_images


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    reference_path = (
        Path(args.reference_path).expanduser()
        if args.reference_path is not None
        else prompt_path("Reference dataset/reconstruction path: ")
    )
    comparison_path = (
        Path(args.comparison_path).expanduser()
        if args.comparison_path is not None
        else (
            None
            if args.reference_path is not None
            else prompt_path(
                "Comparison dataset/reconstruction path (leave empty to auto-follow latest recon): ",
                allow_empty=True,
            )
        )
    )
    difference_path = (
        Path(args.difference_path).expanduser()
        if args.difference_path is not None
        else None
    )

    if reference_path is None:
        LOGGER.error("Reference path is required.")
        return 1
    if difference_path is not None:
        LOGGER.error(
            "`--difference-path` is no longer supported. "
            "Use `--reference-path` for the baseline, `--comparison-path` for the current reconstruction, "
            "and `--show-difference` to display current minus reference."
        )
        return 1
    if not args.show_difference:
        LOGGER.info("Difference display is enabled by default.")
        args.show_difference = True
    if args.fast:
        args.downsample = max(args.downsample, 2)
    if args.downsample < 1:
        LOGGER.error("Downsample factor must be >= 1.")
        return 1
    try:
        args.crop_y, args.crop_x = parse_crop_spec(args.crop, args.crop_x, args.crop_y)
    except Exception as exc:
        log_exception_summary("Invalid crop settings", exc)
        return 1
    if args.hot_cold:
        args.difference_colormap = "coolwarm"
    if args.difference_colormap is None:
        args.difference_colormap = args.colormap

    current_cache = VolumeCache(args.dataset_path)
    baseline_cache = VolumeCache(args.dataset_path)
    try:
        reference_dataset_root, reference_recon_file = resolve_reconstruction_target(reference_path, args.dataset_path)
        _, reference_shape, configured_slice_indices, configured_orthogonal_center = load_volume_metadata(
            reference_recon_file,
            args.orthogonal,
            args.orthogonal_center,
            args.axis,
            args.slices,
            args.num_slices,
            args.dataset_path,
        )
        baseline_dataset_root = None
        baseline_recon_file = None
        baseline_images = None

        current_dataset_root, current_recon_file, auto_follow = resolve_display_target(
            reference_dataset_root,
            reference_recon_file,
            comparison_path,
            args.position_mode,
            args.dataset_path,
        )
        current_shape = volume_shape(current_recon_file, args.dataset_path)
        if args.orthogonal or args.orthogonal_center is not None:
            orthogonal_center = clamp_orthogonal_center(configured_orthogonal_center, current_shape)
            slice_indices = []
        else:
            orthogonal_center = None
            slice_indices = clamp_slice_indices(configured_slice_indices, current_shape[args.axis])
        dataset_path, current_images = current_cache.load(
            current_recon_file,
            args.orthogonal or args.orthogonal_center is not None,
            orthogonal_center,
            args.axis,
            slice_indices,
            args.downsample,
            args.fast,
        )
        display_images = current_images
        difference_images = None
        if args.show_difference:
            baseline_dataset_root, baseline_recon_file = reference_dataset_root, reference_recon_file
            _, baseline_images = baseline_cache.load(
                baseline_recon_file,
                args.orthogonal or args.orthogonal_center is not None,
                orthogonal_center,
                args.axis,
                slice_indices,
                args.downsample,
                args.fast,
            )
            display_images, aligned_baseline_images = align_image_pairs(current_images, baseline_images)
            difference_images = compute_difference_images(display_images, aligned_baseline_images)
    except Exception as exc:
        current_cache.close()
        baseline_cache.close()
        log_exception_summary("Startup failed", exc)
        return 1

    base_panel_count = len(orthogonal_axes(args.fast)) if (args.orthogonal or args.orthogonal_center is not None) else len(slice_indices)
    if baseline_images is not None:
        rows = 2
        cols = base_panel_count
        display_count = base_panel_count * 2
    else:
        display_count = base_panel_count
        cols = min(2, display_count)
        rows = (display_count + cols - 1) // cols

    plt.ion()
    figsize = DEFAULT_FIGSIZE if baseline_images is not None else (max(DEFAULT_FIGSIZE[0], 7 * cols), max(DEFAULT_FIGSIZE[1], 5 * rows))
    fig, axes_array = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes_array).ravel().tolist()
    image_artists: list = []
    for index, ax in enumerate(axes[:display_count]):
        panel_cmap = args.colormap
        if baseline_images is not None and index >= base_panel_count:
            panel_cmap = args.difference_colormap
        image_artists.append(ax.imshow(np.zeros_like(current_images[0]), cmap=panel_cmap))
        fig.colorbar(image_artists[-1], ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[display_count:]:
        ax.axis("off")

    update_display(
        axes[:display_count],
        image_artists,
        slice_indices,
        display_images,
        args.axis,
        make_display_title(
            reference_dataset_root,
            reference_recon_file,
            current_dataset_root,
            current_recon_file,
        ),
        args.colormap,
        args.difference_colormap,
        args.orthogonal or args.orthogonal_center is not None,
        orthogonal_center,
        difference_images,
        args.fast,
        args.crop_x,
        args.crop_y,
    )

    if args.static:
        plt.ioff()
        LOGGER.info("Reference dataset: %s", reference_dataset_root)
        LOGGER.info("Reference reconstruction: %s", reference_recon_file)
        LOGGER.info("Reference volume shape: %s", reference_shape)
        LOGGER.info("Displayed dataset: %s", current_dataset_root)
        LOGGER.info("Displayed file: %s", current_recon_file)
        if baseline_recon_file is not None:
            LOGGER.info("Difference reference dataset: %s", baseline_dataset_root)
            LOGGER.info("Difference reference file: %s", baseline_recon_file)
        LOGGER.info("Dataset path: %s", dataset_path)
        LOGGER.info("Display downsample: %s", args.downsample)
        LOGGER.info("Fast mode: %s", args.fast)
        LOGGER.info("Crop X: %s", args.crop_x or "full")
        LOGGER.info("Crop Y: %s", args.crop_y or "full")
        if args.orthogonal or args.orthogonal_center is not None:
            LOGGER.info("Orthogonal center: %s", orthogonal_center)
        else:
            LOGGER.info("Axis: %s", args.axis)
            LOGGER.info("Slices: %s", slice_indices)
        LOGGER.info("Static mode: True")
        LOGGER.info("Show difference: %s", args.show_difference)
        LOGGER.info("Difference colormap: %s", args.difference_colormap)
        try:
            plt.show()
        finally:
            current_cache.close()
            baseline_cache.close()
        return 0

    LOGGER.info("Reference dataset: %s", reference_dataset_root)
    LOGGER.info("Reference reconstruction: %s", reference_recon_file)
    LOGGER.info("Reference volume shape: %s", reference_shape)
    LOGGER.info("Current volume shape: %s", current_shape)
    LOGGER.info("Starting display dataset: %s", current_dataset_root)
    LOGGER.info("Starting display file: %s", current_recon_file)
    if baseline_recon_file is not None:
        LOGGER.info("Difference reference dataset: %s", baseline_dataset_root)
        LOGGER.info("Difference reference file: %s", baseline_recon_file)
    LOGGER.info("Dataset path: %s", dataset_path)
    LOGGER.info("Display downsample: %s", args.downsample)
    LOGGER.info("Fast mode: %s", args.fast)
    LOGGER.info("Crop X: %s", args.crop_x or "full")
    LOGGER.info("Crop Y: %s", args.crop_y or "full")
    if args.orthogonal or args.orthogonal_center is not None:
        LOGGER.info("Orthogonal center: %s", orthogonal_center)
    else:
        LOGGER.info("Axis: %s", args.axis)
        LOGGER.info("Slices: %s", slice_indices)
    LOGGER.info("Position mode: %s", args.position_mode)
    LOGGER.info("Static mode: False")
    LOGGER.info("Show difference: %s", args.show_difference)
    LOGGER.info("Difference colormap: %s", args.difference_colormap)
    if auto_follow:
        LOGGER.info("Watching collection for newer reconstructions: %s", reference_dataset_root.parent)

    last_seen_file = current_recon_file
    last_seen_dataset_root = current_dataset_root

    try:
        while plt.fignum_exists(fig.number):
            if auto_follow:
                position_name = None
                if args.position_mode == "same":
                    position_name = dataset_position_name(reference_dataset_root, reference_dataset_root.parent)
                newest_target = latest_reconstruction_target(
                    reference_dataset_root.parent,
                    position_name=position_name,
                    dataset_path=args.dataset_path,
                )
                if newest_target is not None:
                    newest_dataset, latest_file = newest_target
                    if newest_dataset != current_dataset_root or latest_file != current_recon_file:
                        current_dataset_root = newest_dataset
                        current_recon_file = latest_file

            if current_recon_file != last_seen_file:
                try:
                    dataset_changed = current_dataset_root != last_seen_dataset_root
                    current_shape = volume_shape(current_recon_file, args.dataset_path)
                    if args.orthogonal or args.orthogonal_center is not None:
                        orthogonal_center = clamp_orthogonal_center(configured_orthogonal_center, current_shape)
                        slice_indices = []
                    else:
                        slice_indices = clamp_slice_indices(configured_slice_indices, current_shape[args.axis])
                    _, current_images = current_cache.load(
                        current_recon_file,
                        args.orthogonal or args.orthogonal_center is not None,
                        orthogonal_center,
                        args.axis,
                        slice_indices,
                        args.downsample,
                        args.fast,
                    )
                    display_images = current_images
                    difference_images = None
                    if args.show_difference and baseline_recon_file is not None:
                        _, baseline_images = baseline_cache.load(
                            baseline_recon_file,
                            args.orthogonal or args.orthogonal_center is not None,
                            orthogonal_center,
                            args.axis,
                            slice_indices,
                            args.downsample,
                            args.fast,
                        )
                        display_images, aligned_baseline_images = align_image_pairs(current_images, baseline_images)
                        difference_images = compute_difference_images(display_images, aligned_baseline_images)
                    update_display(
                        axes[:display_count],
                        image_artists,
                        slice_indices,
                        display_images,
                        args.axis,
                        make_display_title(
                            reference_dataset_root,
                            reference_recon_file,
                            current_dataset_root,
                            current_recon_file,
                        ),
                        args.colormap,
                        args.difference_colormap,
                        args.orthogonal or args.orthogonal_center is not None,
                        orthogonal_center,
                        difference_images,
                        args.fast,
                        args.crop_x,
                        args.crop_y,
                    )
                    if dataset_changed:
                        LOGGER.info("Switched to new reconstruction dataset: %s", current_dataset_root)
                    LOGGER.info("Current volume shape: %s", current_shape)
                    LOGGER.info("Updated reconstruction dataset: %s", current_dataset_root)
                    LOGGER.info("Updated reconstruction file: %s", current_recon_file)
                    last_seen_dataset_root = current_dataset_root
                    last_seen_file = current_recon_file
                except Exception as exc:
                    log_exception_summary(f"Failed to update from {current_recon_file}", exc)

            plt.pause(args.poll_interval)
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
    finally:
        current_cache.close()
        baseline_cache.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
