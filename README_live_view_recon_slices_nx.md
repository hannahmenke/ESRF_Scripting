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

1. resolves a reference/baseline dataset directory or reconstruction HDF5 file
2. resolves a comparison/current reconstruction from `--comparison-path` or by auto-following a newer dataset
3. extracts a few slices from the comparison volume
4. can alternatively show one orthogonal XY/XZ/YZ slice triplet
5. displays them in a live matplotlib window
6. can optionally show current-minus-reference difference slices next to the live current slices
7. can alternatively run in static mode and display a single comparison reconstruction without polling

## How to run

Auto-follow the latest reconstruction with the same position:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --axis 0 \
  --num-slices 4 \
  --position-mode same \
  --downsample 2
```

Watch a specific comparison reconstruction file directly:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002/reconstructed_volumes/cast_volume/039_Estaillades_WW_Drainage_first_position_0002_0000pag_db0500_vol.hdf5 \
  --slices 0,1184,2367
```

Use an explicit internal dataset path:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path /path/to/reference_reconstruction.hdf5 \
  --comparison-path /path/to/current_reconstruction.hdf5 \
  --dataset-path /entry0000/reconstruction/results/data \
  --axis 1 \
  --slices 500,1000,1500
```

Show orthogonal slices through the reconstruction center:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal
```

Show one reconstruction once without live updates:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --static
```

Use the fast orthogonal mode:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --fast
```

Show orthogonal slices through a chosen point:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal-center 1184,1650,1650
```

Show the current reconstruction next to its difference from the reference:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --show-difference
```

Show the difference images with a hot/cold colormap:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --show-difference \
  --hot-cold
```

Crop the displayed view to a smaller region:

```bash
python3 live_view_recon_slices_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --orthogonal \
  --crop 200:1200,300:1800
```

## Notes

- The script prefers the newest file under `reconstructed_volumes/`, and falls back to `reconstructed_slices/` if needed.
- `--reference-path` is always the baseline/reference reconstruction.
- `--comparison-path` is the current reconstruction to display; if omitted, the script auto-follows a newer matching dataset.
- In auto-follow mode, the display starts from the reference reconstruction and switches only when a newer matching reconstruction appears.
- If `--comparison-path` is provided explicitly, the script automatically shows `current - reference`.
- `--position-mode same` restricts auto-follow to datasets matching the same position label as the reference dataset.
- `--position-mode all` allows auto-follow across all positions in the same collection directory.
- If a dataset or reconstruction path that you explicitly pass does not contain a valid reconstruction volume, the script stops with a clear error.
- In auto-follow mode, newer matching datasets are ignored until they contain a valid reconstruction volume.
- Slice indices are zero-based along the chosen axis.
- `--orthogonal` shows one XY/XZ/YZ triplet through the volume center.
- `--orthogonal-center a,b,c` selects the orthogonal intersection point as axis0,axis1,axis2.
- `--show-difference` adds a second panel set showing `current - reference`.
- `--show-difference` is mainly useful in auto-follow mode; with an explicit `--comparison-path`, the difference row is enabled automatically.
- When `--show-difference` is used, the figure is arranged with current slices on the first row and difference slices on the second row.
- `--difference-path` is deprecated and rejected; use `--reference-path`, `--comparison-path`, and `--show-difference` instead.
- `--hot-cold` switches the difference images to the `coolwarm` colormap.
- `--difference-colormap` lets you choose any matplotlib colormap explicitly for the difference images.
- `--downsample` reduces the displayed slice size at read time and is the easiest way to speed up large recon views.
- `--crop y0:y1,x0:x1` crops the displayed images after downsampling.
- `--crop-x` and `--crop-y` let you crop only one displayed axis.
- Cropping uses the displayed panel axes: `XY` uses `X/Y`, `XZ` uses `X/Z`, and `YZ` uses `Y/Z`.
- `--fast` forces `--downsample 2` and, in orthogonal mode, shows only `XY` and `XZ`.
- `--static` renders once and keeps the window open without polling for newer reconstructions.
- `--log-level DEBUG` enables full stack traces for startup or live-update failures.
