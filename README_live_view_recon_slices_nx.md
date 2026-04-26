# `live_view_recon_slices_nx.py`

## Overview

`live_view_recon_slices_nx.py` displays reconstructed slices from tomography datasets and can auto-follow newer reconstructions in the same numbered series.

It is intended for dataset layouts where reconstruction outputs are stored under paths such as:

```text
dataset_name/reconstructed_volumes/**/*.hdf5
dataset_name/reconstructed_slices/**/*.hdf5
```

The viewer can display:

- multiple slices along one axis
- one orthogonal `XY/XZ/YZ` triplet
- the current reconstruction alone
- the current reconstruction together with `current - reference`

## Input Modes

`--reference-path` may point to:

- a dataset directory
- a reconstruction HDF5 file

`--comparison-path` may point to:

- a dataset directory
- a reconstruction HDF5 file

If `--comparison-path` is omitted, the script auto-follows a newer matching reconstruction in the same collection directory.

## Processing Model

The script:

1. resolves the reference reconstruction
2. resolves the current/comparison reconstruction explicitly or by auto-follow
3. loads the reconstruction volume from the chosen HDF5 dataset
4. extracts either axis-aligned slices or one orthogonal slice triplet
5. optionally computes and displays `current - reference`
6. updates the display as newer reconstructions appear unless `--static` is used

## Common Workflows

### Auto-follow the latest reconstruction with the same position

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --axis 0 \
  --num-slices 4 \
  --position-mode same \
  --downsample 2
```

### Watch a specific comparison reconstruction

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002/reconstructed_volumes/cast_volume/039_Estaillades_WW_Drainage_first_position_0002_0000pag_db0500_vol.hdf5 \
  --slices 0,1184,2367
```

### Show orthogonal slices through the volume center

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal
```

### Render once without live polling

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --static
```

### Show current reconstruction with difference images

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --show-difference \
  --hot-cold
```

## Display Controls

- `--axis 0|1|2`
  - axis for slice extraction
- `--num-slices`
  - number of evenly spaced slices when `--slices` is not provided
- `--slices`
  - explicit slice indices
- `--orthogonal`
  - show one `XY/XZ/YZ` triplet
- `--orthogonal-center z,y,x`
  - explicit orthogonal center
- `--downsample`
  - reduce display resolution at read time
- `--fast`
  - forces `--downsample 2` and skips the slower `YZ` view in orthogonal mode
- `--crop`, `--crop-x`, `--crop-y`
  - crop displayed images after downsampling

## Auto-follow Behavior

- `--position-mode same`
  - restricts auto-follow to datasets matching the same position label as the reference
- `--position-mode all`
  - allows auto-follow across all positions in the same collection
- `--static`
  - disables polling and renders once

## Notes

- The script prefers the newest file under `reconstructed_volumes/`, and falls back to `reconstructed_slices/` if needed.
- `--reference-path` is always treated as the baseline reconstruction.
- An explicit `--comparison-path` sets the starting current reconstruction, but live polling continues unless `--static` is used.
- The default polling interval is 30 seconds.
- `--difference-path` is deprecated and rejected.
- `--log-level DEBUG` enables full stack traces for startup and update failures.
