#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


POLL_INTERVAL = 2.0
IMAGE_KEY_PROJECTION = 0
IMAGE_KEY_FLAT = 1
IMAGE_KEY_DARK = 2
DEFAULT_FIGSIZE = (14, 8)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live view of changes between the first radiogram of tomography scans."
    )
    parser.add_argument(
        "--reference-path",
        default=None,
        help="Reference dataset directory, dataset .h5, or detector .h5 file.",
    )
    parser.add_argument(
        "--comparison-path",
        default=None,
        help="Comparison dataset directory, dataset .h5, or detector .h5 file.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Exact HDF5 dataset path for the radiogram stack, if known.",
    )
    parser.add_argument(
        "--projection-index",
        type=int,
        default=None,
        help="Zero-based projection number within the full projection scan. Default: 0.",
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


def prompt_projection_index(default: int = 0) -> int:
    while True:
        raw = input(f"Projection number [default {default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            LOGGER.warning("Projection number must be an integer.")
            continue
        if value < 0:
            LOGGER.warning("Projection number must be >= 0.")
            continue
        return value


def prompt_position_mode(default: str = "same") -> str:
    while True:
        raw = input(f"Position mode [default {default}, same/all]: ").strip().lower()
        if not raw:
            return default
        if raw in {"same", "all"}:
            return raw
        LOGGER.warning("Position mode must be 'same' or 'all'.")


def decode_scalar(value) -> str | int | float | None:
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
        if obj.ndim < 3:
            return
        if obj.dtype.kind not in "uif":
            return

        interpretation = decode_scalar(obj.attrs.get("interpretation"))
        if interpretation == "image":
            candidates.append(name)
            return

        if name.endswith("/data"):
            candidates.append(name)

    h5_file.visititems(visitor)
    candidates.sort()
    return candidates


def scan_block_files(scan_dir: Path) -> list[Path]:
    return sorted(scan_dir.glob("*.h5"))


def find_image_dataset_path(scan_file: Path, dataset_path: str | None = None) -> str:
    with h5py.File(scan_file, "r") as h5_file:
        if dataset_path is not None:
            read_dataset(h5_file, dataset_path)
            return dataset_path

        candidates = find_candidate_datasets(h5_file)
        if not candidates:
            raise RuntimeError(f"No numeric image stack found in {scan_file}")
        return candidates[0]


def scan_projection_count(scan_dir: Path, dataset_path: str | None = None) -> int:
    total = 0
    files = scan_block_files(scan_dir)
    if not files:
        raise RuntimeError(f"No detector files found in {scan_dir}")

    resolved_dataset_path = find_image_dataset_path(files[0], dataset_path)
    for scan_file in files:
        with h5py.File(scan_file, "r") as h5_file:
            dataset = read_dataset(h5_file, resolved_dataset_path)
            if dataset.ndim < 3:
                raise RuntimeError(f"Dataset {resolved_dataset_path} in {scan_file} is not an image stack")
            total += int(dataset.shape[0])
    return total


def load_projection_radiogram(
    scan_dir: Path,
    projection_index: int,
    dataset_path: str | None = None,
) -> np.ndarray:
    if projection_index < 0:
        raise RuntimeError("Projection index must be >= 0")

    files = scan_block_files(scan_dir)
    if not files:
        raise RuntimeError(f"No detector files found in {scan_dir}")

    resolved_dataset_path = find_image_dataset_path(files[0], dataset_path)
    remaining_index = projection_index

    for scan_file in files:
        with h5py.File(scan_file, "r") as h5_file:
            dataset = read_dataset(h5_file, resolved_dataset_path)
            frame_count = int(dataset.shape[0])
            if remaining_index < frame_count:
                return np.asarray(dataset[remaining_index], dtype=np.float32)
            remaining_index -= frame_count

    total = scan_projection_count(scan_dir, resolved_dataset_path)
    raise RuntimeError(
        f"Projection index {projection_index} is out of range for {scan_dir} "
        f"(available: 0..{total - 1})"
    )


def resolve_dataset_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        return resolved

    if resolved.parent.name.startswith("scan"):
        return resolved.parent.parent

    return resolved.parent


def read_image_key(scan_file: Path) -> int | None:
    with h5py.File(scan_file, "r") as h5_file:
        for candidate in (
            "entry_0000/instrument/pcobi10lid19det1/header/image_key",
            "entry/instrument/detector/header/image_key",
        ):
            if candidate in h5_file:
                value = decode_scalar(h5_file[candidate][()])
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def classify_scan(scan_file: Path) -> str:
    image_key = read_image_key(scan_file)
    if image_key == IMAGE_KEY_PROJECTION:
        return "projection"
    if image_key == IMAGE_KEY_FLAT:
        return "flat"
    if image_key == IMAGE_KEY_DARK:
        return "dark"
    return "unknown"


def first_h5_in_scan_dir(scan_dir: Path) -> Path | None:
    files = scan_block_files(scan_dir)
    return files[0] if files else None


def find_projection_scan(dataset_root: Path) -> Path:
    projection_candidates: list[tuple[int, Path]] = []

    for scan_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("scan")):
        first_scan_file = first_h5_in_scan_dir(scan_dir)
        if first_scan_file is None:
            continue

        scan_kind = classify_scan(first_scan_file)
        if scan_kind == "projection":
            try:
                order = int(scan_dir.name.removeprefix("scan"))
            except ValueError:
                order = -1
            projection_candidates.append((order, scan_dir))

    if not projection_candidates:
        raise RuntimeError(f"No projection scan found in {dataset_root}")

    projection_candidates.sort()
    return projection_candidates[-1][1]


def find_dataset_master(dataset_root: Path) -> Path | None:
    expected = dataset_root / f"{dataset_root.name}.h5"
    if expected.is_file():
        return expected

    masters = sorted(
        path
        for path in dataset_root.glob("*.h5")
        if not path.parent.name.startswith("scan")
    )
    return masters[0] if masters else None


def dataset_series_name(dataset_root: Path) -> str:
    name = dataset_root.name
    return re.sub(r"_\d{4}$", "", name)


def dataset_position_name(dataset_root: Path, collection_dir: Path) -> str:
    series_name = dataset_series_name(dataset_root)
    prefix = f"{collection_dir.name}_"
    if series_name.startswith(prefix):
        return series_name[len(prefix):]
    return series_name


def is_dataset_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(child.is_dir() and child.name.startswith("scan") for child in path.iterdir()):
        return True
    return find_dataset_master(path) is not None


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
            find_projection_scan(dataset_dir)
        except Exception:
            continue
        return dataset_dir

    return None


def resolve_input_target(raw_path: Path) -> tuple[Path, Path]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path not found: {path}")

    if path.is_file() and path.parent.name.startswith("scan"):
        scan_kind = classify_scan(path)
        if scan_kind != "projection":
            raise RuntimeError(f"{path} is a {scan_kind} scan, not a projection scan")
        return resolve_dataset_root(path), path.parent

    dataset_root = resolve_dataset_root(path)
    if not is_dataset_directory(dataset_root):
        raise RuntimeError(f"{path} does not resolve to a tomography dataset directory")

    return dataset_root, find_projection_scan(dataset_root)


def resolve_second_target(
    reference_dataset_root: Path,
    second_path: Path | None,
    position_mode: str,
) -> tuple[Path, Path, bool]:
    if second_path is not None:
        dataset_root, scan_path = resolve_input_target(second_path)
        return dataset_root, scan_path, False

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

    return latest_dataset, find_projection_scan(latest_dataset), True


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
    configure_logging(args.log_level)

    reference_path = (
        Path(args.reference_path).expanduser()
        if args.reference_path is not None
        else prompt_path("Reference dataset/scan path: ")
    )
    comparison_path = (
        Path(args.comparison_path).expanduser()
        if args.comparison_path is not None
        else (
            None
            if args.reference_path is not None
            else prompt_path(
                "Comparison dataset/scan path (leave empty to auto-follow latest tomo): ",
                allow_empty=True,
            )
        )
    )
    projection_index = (
        args.projection_index
        if args.projection_index is not None
        else prompt_projection_index()
    )
    position_mode = (
        args.position_mode
        if args.comparison_path is not None or args.reference_path is not None
        else prompt_position_mode()
    )

    if reference_path is None:
        LOGGER.error("Reference path is required.")
        return 1
    if projection_index < 0:
        LOGGER.error("Projection index must be >= 0.")
        return 1

    try:
        reference_dataset_root, reference_scan = resolve_input_target(reference_path)
        first_count = scan_projection_count(reference_scan, args.dataset_path)
        first_image = load_projection_radiogram(reference_scan, projection_index, args.dataset_path)

        current_dataset_root, current_scan, auto_follow = resolve_second_target(
            reference_dataset_root,
            comparison_path,
            position_mode,
        )
        second_count = scan_projection_count(current_scan, args.dataset_path)
        second_image = load_projection_radiogram(current_scan, projection_index, args.dataset_path)
    except Exception as exc:
        log_exception_summary("Startup failed", exc)
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

    LOGGER.info("Reference dataset: %s", reference_dataset_root)
    LOGGER.info("Reference projection scan: %s", reference_scan)
    LOGGER.info("Reference projections available: %s", first_count)
    LOGGER.info("Starting comparison dataset: %s", current_dataset_root)
    LOGGER.info("Starting comparison scan: %s", current_scan)
    LOGGER.info("Comparison projections available: %s", second_count)
    LOGGER.info("Using projection index: %s", projection_index)
    LOGGER.info("Position mode: %s", position_mode)
    if auto_follow:
        LOGGER.info("Watching collection for newer tomography datasets: %s", reference_dataset_root.parent)

    last_seen_dataset = current_dataset_root

    try:
        while plt.fignum_exists(fig.number):
            if auto_follow:
                position_name = None
                if position_mode == "same":
                    position_name = dataset_position_name(reference_dataset_root, reference_dataset_root.parent)
                newest_dataset = latest_projection_dataset(
                    reference_dataset_root.parent,
                    exclude=reference_dataset_root,
                    position_name=position_name,
                )
                if newest_dataset is not None and newest_dataset != last_seen_dataset:
                    current_dataset_root = newest_dataset
                    current_scan = find_projection_scan(current_dataset_root)

            if current_dataset_root != last_seen_dataset:
                try:
                    second_count = scan_projection_count(current_scan, args.dataset_path)
                    second_image = load_projection_radiogram(current_scan, projection_index, args.dataset_path)
                    diff_image = second_image - first_image
                    update_display(
                        image_artist,
                        ax,
                        diff_image,
                        reference_dataset_root,
                        current_dataset_root,
                    )
                    LOGGER.info("Updated comparison dataset: %s", current_dataset_root)
                    LOGGER.info("Updated comparison scan: %s", current_scan)
                    LOGGER.info("Comparison projections available: %s", second_count)
                    last_seen_dataset = current_dataset_root
                except Exception as exc:
                    log_exception_summary(f"Failed to update from {current_dataset_root}", exc)

            plt.pause(args.poll_interval)
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
