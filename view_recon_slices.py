#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a reconstructed volume and display a few slices."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to the reconstruction HDF5 file.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the 3D reconstruction volume.",
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
        "--colormap",
        default="gray",
        help="Matplotlib colormap. Default: gray.",
    )
    return parser.parse_args()


def prompt_path(prompt: str) -> Path:
    while True:
        raw = input(prompt).strip()
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        print(f"File not found: {path}")


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


def extract_slice(volume: h5py.Dataset, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        image = volume[index, :, :]
    elif axis == 1:
        image = volume[:, index, :]
    else:
        image = volume[:, :, index]
    return np.asarray(image, dtype=np.float32)


def main() -> int:
    args = parse_args()

    input_path = (
        Path(args.input_path).expanduser()
        if args.input_path is not None
        else prompt_path("Reconstruction HDF5 file: ")
    )

    if not input_path.is_file():
        print(f"File not found: {input_path}")
        return 1

    try:
        dataset_path = resolve_volume_dataset(input_path, args.dataset_path)
        with h5py.File(input_path, "r") as h5_file:
            volume = read_dataset(h5_file, dataset_path)
            axis_size = int(volume.shape[args.axis])
            slice_indices = parse_slice_indices(args.slices, axis_size, args.num_slices)

            print(f"File: {input_path}")
            print(f"Dataset: {dataset_path}")
            print(f"Volume shape: {volume.shape}")
            print(f"Axis: {args.axis}")
            print(f"Slices: {slice_indices}")

            cols = min(2, len(slice_indices))
            rows = (len(slice_indices) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
            axes = np.atleast_1d(axes).ravel()

            for ax, slice_index in zip(axes, slice_indices):
                image = extract_slice(volume, args.axis, slice_index)
                artist = ax.imshow(image, cmap=args.colormap)
                ax.set_title(f"Axis {args.axis} slice {slice_index}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)

            for ax in axes[len(slice_indices):]:
                ax.axis("off")

            fig.suptitle(input_path.name)
            fig.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"Failed to open reconstruction: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
