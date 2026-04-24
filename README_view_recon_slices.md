# `view_recon_slices.py`

## What it is

This script opens a reconstructed 3D volume from an HDF5 file and displays a few slices.

## What it does

When started, the script:

1. opens a reconstruction `.h5` file
2. finds a likely 3D dataset automatically, or uses `--dataset-path`
3. extracts slices along the chosen axis
4. displays the slices in a matplotlib figure

## How to run

Interactive:

```bash
python3 view_recon_slices.py
```

Direct path:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5
```

Full explicit example:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --dataset-path /entry/data/data \
  --axis 1 \
  --slices 100,300,500
```

Choose axis and evenly spaced slices:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --axis 0 --num-slices 4
```

Choose exact slices:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --slices 100,300,500
```

Specify the exact internal dataset:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --dataset-path /entry/data/data
```

## Options

- `--axis 0|1|2`: axis along which to slice
- `--num-slices N`: number of evenly spaced slices to display
- `--slices a,b,c`: explicit slice indices
- `--dataset-path PATH`: exact HDF5 dataset path
- `--colormap NAME`: matplotlib colormap, default `gray`

## Dependencies

- `python3`
- `numpy`
- `h5py`
- `matplotlib`
