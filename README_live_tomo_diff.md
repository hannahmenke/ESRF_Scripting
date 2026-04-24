# `live_tomo_diff.py`

## What it is

This script monitors changes between tomography datasets by subtracting one chosen projection radiogram of a tomo from the same projection radiogram of another.

It is designed for the dataset layout produced by `series_of_tomo` in `id19macros_improved.py`.

## What it does

When started, the script:

1. takes a reference dataset path
2. optionally takes a comparison dataset path
3. resolves the projection scan automatically from the HDF5 metadata
4. resolves a chosen projection number across all detector block files in the scan
5. ignores flat and dark scans by checking `image_key`
6. displays `comparison - reference` as an image
7. if no comparison path is given, watches the collection directory for the newest tomography dataset
8. can auto-follow only the same position as the reference, or all positions

## Accepted inputs

You can give any of these as reference or comparison input:

- a dataset directory such as `011_test/011_test_first_position`
- a dataset master file such as `011_test/011_test_first_position/011_test_first_position.h5`
- a detector scan file such as `.../scan0004/pcobi10lid19det1_0000.h5`

If a detector file is passed and it is a flat or dark, the script rejects it.

## How to run

Interactive:

```bash
python3 live_tomo_diff.py
```

Non-interactive:

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position \
  --comparison-path 011_test/011_test_first_position_0002
```

Auto-follow latest tomo:

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position
```

Optional arguments:

```bash
python3 live_tomo_diff.py --poll-interval 1.0
python3 live_tomo_diff.py --dataset-path entry_0000/instrument/pcobi10lid19det1/data
python3 live_tomo_diff.py --projection-index 237
python3 live_tomo_diff.py --position-mode all
```

## HDF5 behavior

- Projection / flat / dark classification comes from the detector file `image_key`.
- Projection numbering is zero-based and spans the full projection scan, even when the data are split into multiple detector files of 100 frames each.
- `--position-mode same` restricts auto-follow to the same position label as the reference dataset, for example `first_position`.
- `--position-mode all` allows auto-follow to pick the newest projection dataset from any position in the collection.
- If `--dataset-path` is not given, the script searches the scan file for an image stack dataset.
- For the `011_test` real dataset, it resolves:
  - `scan0004` in `011_test_first_position`
  - `scan0002` in later `011_test_first_position_*` datasets

## Dependencies

- `python3`
- `numpy`
- `h5py`
- `matplotlib`

## Current caveats

- The same projection index must exist in both the reference and comparison projection scans.
- Automatic dataset discovery assumes each tomo lives in its own dataset directory with `scan*` subdirectories and position names embedded in the dataset directory name.
