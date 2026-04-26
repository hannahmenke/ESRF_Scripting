# `view_recon_slices.py`

## Overview

`view_recon_slices.py` opens one reconstructed 3D HDF5 volume and displays either:

- several slices along one axis
- one orthogonal `XY/XZ/YZ` triplet

It is intended as a lightweight inspection tool for a single reconstruction file.

## Processing Model

The script:

1. opens one reconstruction HDF5 file
2. resolves the internal 3D dataset automatically unless `--dataset-path` is provided
3. extracts either axis-aligned slices or one orthogonal slice triplet
4. displays the result in a Matplotlib figure

## Common Workflows

### Interactive prompt mode

```bash
python3 view_recon_slices.py
```

### Open one reconstruction directly

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5
```

### Show evenly spaced slices

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --axis 0 \
  --num-slices 4
```

### Show explicit slices

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --axis 1 \
  --slices 100,300,500
```

### Show one orthogonal triplet through the volume center

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --orthogonal
```

### Show one orthogonal triplet through a chosen point

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --orthogonal-center 500,1000,1000
```

### Use an explicit internal dataset path

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --dataset-path /entry/data/data
```

## Key Options

- `--dataset-path`
  - explicit HDF5 dataset path
- `--axis 0|1|2`
  - axis along which slices are extracted
- `--num-slices`
  - number of evenly spaced slices
- `--slices`
  - explicit comma-separated slice indices
- `--orthogonal`
  - show one `XY/XZ/YZ` triplet instead of axis-aligned slices
- `--orthogonal-center z,y,x`
  - explicit orthogonal center
- `--colormap`
  - Matplotlib colormap, default `gray`

## Dependencies

- `python3`
- `numpy`
- `h5py`
- `matplotlib`

## Notes

- Slice indices are zero-based.
- If `--dataset-path` is not supplied, the script searches the file for a suitable 3D numeric dataset.
- `--log-level DEBUG` enables full stack traces for file and dataset-resolution failures.
