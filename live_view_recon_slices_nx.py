#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


POLL_INTERVAL = 2.0
DEFAULT_FIGSIZE = (18, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live view of reconstructed volumes stored inside tomography dataset directories."
    )
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Reference dataset directory or reconstruction HDF5 file.",
    )
    parser.add_argument(
        "--comparison-path",
        default=None,
        help="Optional explicit reconstruction HDF5 file or dataset directory to show instead of auto-following.",
    )
    parser.add_argument(
        "--difference-path",
        default=None,
        help="Optional baseline reconstruction HDF5 file or dataset directory used to display current-minus-baseline slices.",
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
        help="Seconds between checks for a newer reconstruction.",
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
    return parser.parse_args()


def prompt_path(prompt: str, allow_empty: bool = False) -> Path | None:
    while True:
        raw = input(prompt).strip()
        if not raw and allow_empty:
            return None
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"Path not found: {path}")


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


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    return image[::factor, ::factor]


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


def find_latest_reconstruction_file(dataset_root: Path) -> Path:
    candidates = candidate_reconstruction_files(dataset_root)
    if not candidates:
        raise RuntimeError(f"No reconstruction HDF5 files found in {dataset_root}")

    valid_candidates = [path for path in candidates if is_reconstruction_file(path)]
    if not valid_candidates:
        raise RuntimeError(f"No valid reconstruction volume HDF5 files found in {dataset_root}")

    return valid_candidates[-1]


def resolve_reconstruction_target(raw_path: Path) -> tuple[Path, Path]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path not found: {path}")

    if path.is_file() and path.suffix in {".hdf5", ".h5"}:
        return resolve_dataset_root(path), path

    dataset_root = resolve_dataset_root(path)
    if not is_dataset_directory(dataset_root):
        raise RuntimeError(f"{path} does not resolve to a tomography dataset directory")
    return dataset_root, find_latest_reconstruction_file(dataset_root)


def latest_reconstruction_dataset(
    collection_dir: Path,
    exclude: Path | None = None,
    position_name: str | None = None,
) -> Path | None:
    dataset_dirs = sorted(
        (path for path in collection_dir.iterdir() if is_dataset_directory(path)),
        key=lambda path: path.stat().st_mtime,
    )
    for dataset_dir in reversed(dataset_dirs):
        if exclude is not None and dataset_dir == exclude:
            continue
        if position_name is not None and dataset_position_name(dataset_dir, collection_dir) != position_name:
            continue
        try:
            find_latest_reconstruction_file(dataset_dir)
        except Exception:
            continue
        return dataset_dir
    return None


def resolve_display_target(
    reference_dataset_root: Path,
    reference_recon_file: Path,
    second_path: Path | None,
    position_mode: str,
) -> tuple[Path, Path, bool]:
    if second_path is not None:
        dataset_root, recon_file = resolve_reconstruction_target(second_path)
        return dataset_root, recon_file, False

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
    images: list[np.ndarray],
    axis_index: int,
    title: str,
    colormap: str,
    difference_colormap: str,
    orthogonal: bool,
    orthogonal_center: tuple[int, int, int] | None,
    baseline_images: list[np.ndarray] | None,
    fast: bool,
) -> None:
    labels: list[str] = []
    if orthogonal:
        plane_labels = {0: "XY", 1: "XZ", 2: "YZ"}
        labels = [f"{plane_labels[axis_id]} @ {orthogonal_center}" for axis_id in orthogonal_axes(fast)]
    else:
        labels = [f"Axis {axis_index} slice {slice_index}" for slice_index in slice_indices]

    panel_images: list[np.ndarray] = []
    panel_labels: list[str] = []
    panel_colormaps: list[str] = []
    for label, image in zip(labels, images):
        panel_labels.append(label)
        panel_images.append(image)
        panel_colormaps.append(colormap)
    if baseline_images is not None:
        for label, image, baseline in zip(labels, images, baseline_images):
            panel_labels.append(f"{label} difference")
            panel_images.append(image - baseline)
            panel_colormaps.append(difference_colormap)

    for ax, artist, label, image, panel_cmap in zip(axes, image_artists, panel_labels, panel_images, panel_colormaps):
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
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if axes:
        axes[0].figure.suptitle(title)
        axes[0].figure.canvas.draw_idle()
    plt.draw()
    plt.pause(0.001)


def main() -> int:
    args = parse_args()

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
        print("Reference path is required.")
        return 1
    if args.fast:
        args.downsample = max(args.downsample, 2)
    if args.downsample < 1:
        print("Downsample factor must be >= 1.")
        return 1
    if args.hot_cold:
        args.difference_colormap = "coolwarm"
    if args.difference_colormap is None:
        args.difference_colormap = args.colormap

    current_cache = VolumeCache(args.dataset_path)
    baseline_cache = VolumeCache(args.dataset_path)
    try:
        reference_dataset_root, reference_recon_file = resolve_reconstruction_target(reference_path)
        _, reference_shape, slice_indices, orthogonal_center = load_volume_metadata(
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
        if difference_path is not None:
            baseline_dataset_root, baseline_recon_file = resolve_reconstruction_target(difference_path)
            _, baseline_images = baseline_cache.load(
                baseline_recon_file,
                args.orthogonal or args.orthogonal_center is not None,
                orthogonal_center,
                args.axis,
                slice_indices,
                args.downsample,
                args.fast,
            )

        current_dataset_root, current_recon_file, auto_follow = resolve_display_target(
            reference_dataset_root,
            reference_recon_file,
            comparison_path,
            args.position_mode,
        )
        dataset_path, current_images = current_cache.load(
            current_recon_file,
            args.orthogonal or args.orthogonal_center is not None,
            orthogonal_center,
            args.axis,
            slice_indices,
            args.downsample,
            args.fast,
        )
    except Exception as exc:
        current_cache.close()
        baseline_cache.close()
        print(f"Startup failed: {exc}")
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
        current_images,
        args.axis,
        f"{current_dataset_root.name} | {current_recon_file.name}",
        args.colormap,
        args.difference_colormap,
        args.orthogonal or args.orthogonal_center is not None,
        orthogonal_center,
        baseline_images,
        args.fast,
    )

    if args.static:
        plt.ioff()
        print(f"Reference dataset: {reference_dataset_root}")
        print(f"Reference reconstruction: {reference_recon_file}")
        print(f"Reference volume shape: {reference_shape}")
        print(f"Displayed dataset: {current_dataset_root}")
        print(f"Displayed file: {current_recon_file}")
        if baseline_recon_file is not None:
            print(f"Difference baseline dataset: {baseline_dataset_root}")
            print(f"Difference baseline file: {baseline_recon_file}")
        print(f"Dataset path: {dataset_path}")
        print(f"Display downsample: {args.downsample}")
        print(f"Fast mode: {args.fast}")
        if args.orthogonal or args.orthogonal_center is not None:
            print(f"Orthogonal center: {orthogonal_center}")
        else:
            print(f"Axis: {args.axis}")
            print(f"Slices: {slice_indices}")
        print("Static mode: True")
        print(f"Difference colormap: {args.difference_colormap}")
        try:
            plt.show()
        finally:
            current_cache.close()
            baseline_cache.close()
        return 0

    print(f"Reference dataset: {reference_dataset_root}")
    print(f"Reference reconstruction: {reference_recon_file}")
    print(f"Reference volume shape: {reference_shape}")
    print(f"Starting display dataset: {current_dataset_root}")
    print(f"Starting display file: {current_recon_file}")
    if baseline_recon_file is not None:
        print(f"Difference baseline dataset: {baseline_dataset_root}")
        print(f"Difference baseline file: {baseline_recon_file}")
    print(f"Dataset path: {dataset_path}")
    print(f"Display downsample: {args.downsample}")
    print(f"Fast mode: {args.fast}")
    if args.orthogonal or args.orthogonal_center is not None:
        print(f"Orthogonal center: {orthogonal_center}")
    else:
        print(f"Axis: {args.axis}")
        print(f"Slices: {slice_indices}")
    print(f"Position mode: {args.position_mode}")
    print("Static mode: False")
    print(f"Difference colormap: {args.difference_colormap}")
    if auto_follow:
        print(f"Watching collection for newer reconstructions: {reference_dataset_root.parent}")

    last_seen_file = current_recon_file
    last_seen_dataset_root = current_dataset_root

    try:
        while plt.fignum_exists(fig.number):
            if auto_follow:
                position_name = None
                if args.position_mode == "same":
                    position_name = dataset_position_name(reference_dataset_root, reference_dataset_root.parent)
                newest_dataset = latest_reconstruction_dataset(
                    reference_dataset_root.parent,
                    position_name=position_name,
                )
                if newest_dataset is not None:
                    latest_file = find_latest_reconstruction_file(newest_dataset)
                    if latest_file.stat().st_mtime > last_seen_file.stat().st_mtime:
                        current_dataset_root = newest_dataset
                        current_recon_file = latest_file

            if current_recon_file != last_seen_file:
                try:
                    dataset_changed = current_dataset_root != last_seen_dataset_root
                    _, current_images = current_cache.load(
                        current_recon_file,
                        args.orthogonal or args.orthogonal_center is not None,
                        orthogonal_center,
                        args.axis,
                        slice_indices,
                        args.downsample,
                        args.fast,
                    )
                    update_display(
                        axes[:display_count],
                        image_artists,
                        slice_indices,
                        current_images,
                        args.axis,
                        f"{current_dataset_root.name} | {current_recon_file.name}",
                        args.colormap,
                        args.difference_colormap,
                        args.orthogonal or args.orthogonal_center is not None,
                        orthogonal_center,
                        baseline_images,
                        args.fast,
                    )
                    if dataset_changed:
                        print(f"Switched to new reconstruction dataset: {current_dataset_root}")
                    print(f"Updated reconstruction dataset: {current_dataset_root}")
                    print(f"Updated reconstruction file: {current_recon_file}")
                    last_seen_dataset_root = current_dataset_root
                    last_seen_file = current_recon_file
                except Exception as exc:
                    print(f"Failed to update from {current_recon_file}: {exc}")

            plt.pause(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        current_cache.close()
        baseline_cache.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
