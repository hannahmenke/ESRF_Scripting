# `live_view_recon_slices_nx.py`

## What it is

This script live-displays slices from reconstruction HDF5 files stored inside tomography dataset directories with the `projections/*.nx` layout.

It is intended for datasets where reconstruction outputs appear under folders such as:

```text
dataset_name/reconstructed_volumes/**/*.hdf5
dataset_name/reconstructed_slices/**/*.hdf5
```

## What it does

When started, the script:

1. resolves a reference dataset directory or reconstruction HDF5 file
2. finds the latest reconstruction `.hdf5` file inside that dataset
3. extracts a few slices from the reconstruction volume
4. can alternatively show one orthogonal XY/XZ/YZ slice triplet
5. displays them in a live matplotlib window
6. if no comparison path is given, watches the collection for newer reconstructions
7. can optionally show current-minus-baseline difference slices next to the live current slices

## How to run

Auto-follow the latest reconstruction with the same position:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --axis 0 \
  --num-slices 4 \
  --position-mode same
```

Watch a specific reconstruction file directly:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position_0002/reconstructed_volumes/cast_volume/039_Estaillades_WW_Drainage_first_position_0002_0000pag_db0500_vol.hdf5 \
  --slices 0,1184,2367
```

Use an explicit internal dataset path:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path /path/to/reconstruction.hdf5 \
  --dataset-path /entry0000/reconstruction/results/data \
  --axis 1 \
  --slices 500,1000,1500
```

Show orthogonal slices through the reconstruction center:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal
```

Show orthogonal slices through a chosen point:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal-center 1184,1650,1650
```

Show the current reconstruction next to its difference from an earlier reconstruction:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --difference-path 0_39/039_Estaillades_WW_Drainage_first_position_0002/reconstructed_volumes/cast_volume/039_Estaillades_WW_Drainage_first_position_0002_0000pag_db0500_vol.hdf5 \
  --orthogonal
```

## Notes

- The script prefers the newest file under `reconstructed_volumes/`, and falls back to `reconstructed_slices/` if needed.
- `--position-mode same` restricts auto-follow to datasets matching the same position label as the reference dataset.
- `--position-mode all` allows auto-follow across all positions in the same collection directory.
- Slice indices are zero-based along the chosen axis.
- `--orthogonal` shows one XY/XZ/YZ triplet through the volume center.
- `--orthogonal-center a,b,c` selects the orthogonal intersection point as axis0,axis1,axis2.
- `--difference-path` adds a second panel set showing `current - baseline`.
