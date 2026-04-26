# `live_tomo_diff.py`

## Overview

`live_tomo_diff.py` displays projection-radiogram differences between two tomography datasets.

It is designed for dataset layouts produced by `series_of_tomo` in `id19macros_improved.py`, where each dataset contains `scan*` directories and detector HDF5 files.

The display shows:

```text
comparison - reference
```

for one selected projection index.

## Input Modes

Both `--reference-path` and `--comparison-path` may point to:

- a dataset directory
- a dataset master HDF5 file
- a detector HDF5 file inside a `scan*` directory

If `--comparison-path` is omitted, the script auto-follows the newest compatible tomography dataset in the same collection directory.

## Processing Model

The script:

1. resolves the reference dataset and its projection scan
2. resolves the comparison dataset explicitly or by auto-follow
3. finds the image-stack dataset inside the detector file unless `--dataset-path` is provided
4. maps one zero-based projection index across all detector block files in the scan
5. ignores flat and dark scans using `image_key`
6. displays the difference image in a live Matplotlib window

## Common Workflows

### Interactive mode

```bash
python3 live_tomo_diff.py
```

### Compare two explicit datasets

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position \
  --comparison-path 011_test/011_test_first_position_0002
```

### Auto-follow the newest tomography dataset

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position
```

### Use an explicit projection index and display scaling

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position \
  --projection-index 237 \
  --display-min -500 \
  --display-max 500
```

### Use a diverging colormap

```bash
python3 live_tomo_diff.py \
  --reference-path 011_test/011_test_first_position \
  --hot-cold
```

## Key Options

- `--projection-index`
  - zero-based projection index across the full scan
- `--position-mode same|all`
  - `same` restricts auto-follow to the same position label as the reference
  - `all` allows auto-follow across all positions in the collection
- `--downsample`
  - reduces display resolution at read time
- `--fast`
  - forces display downsampling to at least `4`
- `--display-min` / `--display-max`
  - fix the display range instead of using percentile scaling
- `--dataset-path`
  - use an explicit internal HDF5 dataset path

## HDF5 Behavior

- Projection / flat / dark classification comes from the detector file `image_key`.
- Projection numbering is zero-based across the full projection scan, even when data are split across multiple detector files.
- If `--dataset-path` is not given, the script searches the file for a suitable image-stack dataset.
- If a detector file is passed explicitly and it belongs to a flat or dark scan, the script rejects it.

## Dependencies

- `python3`
- `numpy`
- `h5py`
- `matplotlib`

## Caveats

- The same projection index must exist in both the reference and comparison scans.
- Automatic dataset discovery assumes one tomography dataset per dataset directory with `scan*` subdirectories and position labels embedded in the dataset name.
