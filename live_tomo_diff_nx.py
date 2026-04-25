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
DEFAULT_FIGSIZE = (14, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live view of changes between tomography datasets stored as projection .nx files."
    )
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Reference dataset directory or projection .nx file.",
    )
    parser.add_argument(
        "--comparison-path",
        default=None,
        help="Comparison dataset directory or projection .nx file.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the projection stack. Default auto-detects the 3D image stack.",
    )
    parser.add_argument(
        "--projection-index",
        type=int,
        default=None,
        help="Zero-based projection number within the projection stack. Default: 0.",
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
        help="Seconds between checks for a newer tomography dataset.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Display downsampling factor. 1 keeps full resolution. Default: 1.",
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


def prompt_projection_index(default: int = 0) -> int:
    while True:
        raw = input(f"Projection number [default {default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Projection number must be an integer.")
            continue
        if value < 0:
            print("Projection number must be >= 0.")
            continue
        return value


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
        if obj.ndim < 3:
            return
        if obj.dtype.kind not in "uif":
            return

        interpretation = decode_scalar(obj.attrs.get("interpretation"))
        if interpretation == "image" or name.endswith("/data"):
            candidates.append(name)

    h5_file.visititems(visitor)
    candidates.sort()
    return candidates


def find_projection_file(dataset_root: Path) -> Path:
    projection_dir = dataset_root / "projections"
    if not projection_dir.is_dir():
        raise RuntimeError(f"No projections directory found in {dataset_root}")

    projection_files = sorted(projection_dir.glob("*.nx"))
    if not projection_files:
        raise RuntimeError(f"No projection .nx file found in {projection_dir}")
    return projection_files[0]


def resolve_dataset_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        return resolved
    if resolved.suffix == ".nx" and resolved.parent.name == "projections":
        return resolved.parent.parent
    return resolved.parent


def is_dataset_directory(path: Path) -> bool:
    return path.is_dir() and (path / "projections").is_dir()


def resolve_projection_file(raw_path: Path) -> tuple[Path, Path]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path not found: {path}")

    if path.is_file() and path.suffix == ".nx":
        dataset_root = resolve_dataset_root(path)
        return dataset_root, path

    dataset_root = resolve_dataset_root(path)
    if not is_dataset_directory(dataset_root):
        raise RuntimeError(f"{path} does not resolve to a dataset directory with projections/")

    return dataset_root, find_projection_file(dataset_root)


def find_image_dataset_path(projection_file: Path, dataset_path: str | None = None) -> str:
    with h5py.File(projection_file, "r") as h5_file:
        if dataset_path is not None:
            read_dataset(h5_file, dataset_path)
            return dataset_path

        candidates = find_candidate_datasets(h5_file)
        if not candidates:
            raise RuntimeError(f"No numeric image stack found in {projection_file}")
        return candidates[0]


def projection_count(projection_file: Path, dataset_path: str | None = None) -> int:
    resolved_dataset_path = find_image_dataset_path(projection_file, dataset_path)
    with h5py.File(projection_file, "r") as h5_file:
        dataset = read_dataset(h5_file, resolved_dataset_path)
        if dataset.ndim < 3:
            raise RuntimeError(f"Dataset {resolved_dataset_path} in {projection_file} is not an image stack")
        return int(dataset.shape[0])


def load_projection(
    projection_file: Path,
    projection_index: int,
    dataset_path: str | None = None,
    downsample: int = 1,
) -> np.ndarray:
    if projection_index < 0:
        raise RuntimeError("Projection index must be >= 0")

    resolved_dataset_path = find_image_dataset_path(projection_file, dataset_path)
    with h5py.File(projection_file, "r") as h5_file:
        dataset = read_dataset(h5_file, resolved_dataset_path)
        frame_count = int(dataset.shape[0])
        if projection_index >= frame_count:
            raise RuntimeError(
                f"Projection index {projection_index} is out of range for {projection_file} "
                f"(available: 0..{frame_count - 1})"
            )
        image = np.asarray(dataset[projection_index, ::downsample, ::downsample], dtype=np.float32)
        return image


def dataset_series_name(dataset_root: Path) -> str:
    return re.sub(r"_\d{4}$", "", dataset_root.name)


def dataset_position_name(dataset_root: Path, collection_dir: Path) -> str:
    series_name = dataset_series_name(dataset_root)
    prefix = f"{collection_dir.name}_"
    if series_name.startswith(prefix):
        return series_name[len(prefix):]
    return series_name


def latest_projection_dataset(
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
            find_projection_file(dataset_dir)
        except Exception:
            continue
        return dataset_dir

    return None


def resolve_second_target(
    reference_dataset_root: Path,
    second_path: Path | None,
    position_mode: str,
) -> tuple[Path, Path, bool]:
    if second_path is not None:
        dataset_root, projection_file = resolve_projection_file(second_path)
        return dataset_root, projection_file, False

    position_name = None
    if position_mode == "same":
        position_name = dataset_position_name(reference_dataset_root, reference_dataset_root.parent)

    latest_dataset = latest_projection_dataset(
        reference_dataset_root.parent,
        exclude=reference_dataset_root,
        position_name=position_name,
    )
    if latest_dataset is None:
        raise RuntimeError(f"No comparison tomography dataset found in {reference_dataset_root.parent}")
    return latest_dataset, find_projection_file(latest_dataset), True


def update_display(
    image_artist,
    axis,
    diff_image: np.ndarray,
    first_dataset_root: Path,
    second_dataset_root: Path,
) -> None:
    image_artist.set_data(diff_image)

    diff_min = float(np.min(diff_image))
    diff_max = float(np.max(diff_image))
    if diff_min == diff_max:
        diff_min -= 0.5
        diff_max += 0.5
    image_artist.set_clim(vmin=diff_min, vmax=diff_max)

    axis.set_title(f"{second_dataset_root.name} - {first_dataset_root.name}")
    plt.draw()
    plt.pause(0.001)


def main() -> int:
    args = parse_args()

    reference_path = (
        Path(args.reference_path).expanduser()
        if args.reference_path is not None
        else prompt_path("Reference dataset/projection path: ")
    )
    comparison_path = (
        Path(args.comparison_path).expanduser()
        if args.comparison_path is not None
        else (
            None
            if args.reference_path is not None
            else prompt_path(
                "Comparison dataset/projection path (leave empty to auto-follow latest tomo): ",
                allow_empty=True,
            )
        )
    )
    projection_index = (
        args.projection_index
        if args.projection_index is not None
        else prompt_projection_index()
    )

    if reference_path is None:
        print("Reference path is required.")
        return 1
    if projection_index < 0:
        print("Projection index must be >= 0.")
        return 1
    if args.downsample < 1:
        print("Downsample factor must be >= 1.")
        return 1

    try:
        reference_dataset_root, reference_projection_file = resolve_projection_file(reference_path)
        first_count = projection_count(reference_projection_file, args.dataset_path)
        first_image = load_projection(reference_projection_file, projection_index, args.dataset_path, args.downsample)

        current_dataset_root, current_projection_file, auto_follow = resolve_second_target(
            reference_dataset_root,
            comparison_path,
            args.position_mode,
        )
        second_count = projection_count(current_projection_file, args.dataset_path)
        second_image = load_projection(current_projection_file, projection_index, args.dataset_path, args.downsample)
    except Exception as exc:
        print(f"Startup failed: {exc}")
        return 1

    diff_image = second_image - first_image

    plt.ion()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    image_artist = ax.imshow(diff_image, cmap="gray")
    colorbar = fig.colorbar(image_artist, ax=ax)
    colorbar.set_label("Difference")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    update_display(image_artist, ax, diff_image, reference_dataset_root, current_dataset_root)

    print(f"Reference dataset: {reference_dataset_root}")
    print(f"Reference projections file: {reference_projection_file}")
    print(f"Reference projections available: {first_count}")
    print(f"Starting comparison dataset: {current_dataset_root}")
    print(f"Starting comparison file: {current_projection_file}")
    print(f"Comparison projections available: {second_count}")
    print(f"Using projection index: {projection_index}")
    print(f"Position mode: {args.position_mode}")
    print(f"Display downsample: {args.downsample}")
    if auto_follow:
        print(f"Watching collection for newer tomography datasets: {reference_dataset_root.parent}")

    last_seen_dataset = current_dataset_root

    try:
        while plt.fignum_exists(fig.number):
            if auto_follow:
                position_name = None
                if args.position_mode == "same":
                    position_name = dataset_position_name(reference_dataset_root, reference_dataset_root.parent)
                newest_dataset = latest_projection_dataset(
                    reference_dataset_root.parent,
                    exclude=reference_dataset_root,
                    position_name=position_name,
                )
                if newest_dataset is not None and newest_dataset != last_seen_dataset:
                    current_dataset_root = newest_dataset
                    current_projection_file = find_projection_file(current_dataset_root)

            if current_dataset_root != last_seen_dataset:
                try:
                    second_count = projection_count(current_projection_file, args.dataset_path)
                    second_image = load_projection(
                        current_projection_file,
                        projection_index,
                        args.dataset_path,
                        args.downsample,
                    )
                    diff_image = second_image - first_image
                    update_display(
                        image_artist,
                        ax,
                        diff_image,
                        reference_dataset_root,
                        current_dataset_root,
                    )
                    print(f"Updated comparison dataset: {current_dataset_root}")
                    print(f"Updated comparison file: {current_projection_file}")
                    print(f"Comparison projections available: {second_count}")
                    last_seen_dataset = current_dataset_root
                except Exception as exc:
                    print(f"Failed to update from {current_dataset_root}: {exc}")

            plt.pause(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
