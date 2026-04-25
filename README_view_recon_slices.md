# `view_recon_slices.py`

## What it is

This script opens a reconstructed 3D volume from an HDF5 file and displays a few slices.

## What it does

When started, the script:

1. opens a reconstruction `.h5` file
2. finds a likely 3D dataset automatically, or uses `--dataset-path`
3. extracts slices along the chosen axis, or shows an orthogonal XY/XZ/YZ triplet
4. displays the result in a matplotlib figure

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

Show orthogonal slices through the volume center:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --orthogonal
```

Show orthogonal slices through a chosen point:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 \
  --orthogonal-center 500,1000,1000
```

Specify the exact internal dataset:

```bash
python3 view_recon_slices.py /path/to/reconstruction.h5 --dataset-path /entry/data/data
```

## Options

- `--axis 0|1|2`: axis along which to slice
- `--num-slices N`: number of evenly spaced slices to display
- `--slices a,b,c`: explicit slice indices
- `--orthogonal`: show one XY/XZ/YZ triplet through the volume center
- `--orthogonal-center a,b,c`: center indices for orthogonal views as axis0,axis1,axis2
- `--dataset-path PATH`: exact HDF5 dataset path
- `--colormap NAME`: matplotlib colormap, default `gray`

## Dependencies

- `python3`
- `numpy`
- `h5py`
- `matplotlib`
