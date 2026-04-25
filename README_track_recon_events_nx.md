# `track_recon_events_nx.py`

## What it is

This script scans a numbered reconstruction time series in the NX-style dataset layout and records large difference events between each reconstruction and the immediately previous valid reconstruction in the same series.

It is intended for datasets where each time point lives in its own dataset directory and reconstructed volumes appear under:

```text
dataset_name/reconstructed_volumes/**/*.hdf5
```

## What it does

When started, the script:

1. resolves one fixed `--reference-path`
2. finds comparison reconstructions in the same numbered series between `--start-number` and `--stop-number`
3. estimates a baseline noise level from the first stepwise comparison
4. sets a detection threshold as `baseline_sigma * --threshold-sigma`, or uses `--absolute-threshold` if provided
5. scans each comparison volume against the immediately previous valid reconstruction
6. finds discontiguous above-threshold events
7. records up to 100 events per comparison in an SQLite database, including 3D bounding boxes
8. writes a flat CSV summary next to the SQLite database for quick inspection
9. can optionally write orthogonal timeseries GIFs with step numbers burned into each frame

## How to run

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --output-db recon_events.sqlite
```

Use a higher threshold:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --threshold-sigma 8
```

Use an explicit absolute threshold instead of sigma-based thresholding:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --absolute-threshold 250
```

Save orthogonal stepwise GIFs:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --save-gifs \
  --gif-labels \
  --gif-planes xz,yz \
  --gif-mode diff \
  --orthogonal-center 1184,1650,1650
```

Use an explicit internal dataset path:

```bash
python3 track_recon_events_nx.py \
  --reference-path /path/to/reference_dataset \
  --start-number 2 \
  --stop-number 20 \
  --dataset-path /entry0000/reconstruction/results/data
```

## Database contents

The SQLite database contains:

- `runs`: one row per tracker run
- `comparisons`: one row per processed comparison reconstruction
- `events`: one row per recorded event

Each event row stores:

- event rank within that comparison
- voxel count
- peak absolute difference
- signed peak difference
- mean absolute difference
- signed mean difference
- event centroid as `z/y/x`
- 3D bounding box as `z_min/z_max`, `y_min/y_max`, `x_min/x_max`

Each comparison row also stores:

- current sequence number
- previous sequence number
- current dataset / recon path
- previous dataset / recon path
- current recon file modification timestamp
- previous recon file modification timestamp
- total detected events
- stored events
- maximum absolute difference in that step

## Notes

- Sequence numbers are taken from dataset suffixes like `_0002`; an unsuffixed dataset counts as `0`.
- The `--reference-path` is used to identify the series and anchor the first valid stepwise comparison.
- The first valid stepwise comparison in the requested range is used to estimate the baseline noise.
- `--absolute-threshold` overrides sigma-based thresholding when you want direct control of what counts as an event.
- A CSV summary is written next to the SQLite database using the same base filename.
- The stored timestamps currently come from the reconstruction files' filesystem modification times in UTC.
- `--gif-labels` adds an optional corner annotation like `#0007 prev #0006` to each GIF frame.
- Events are detected slice-by-slice and merged across adjacent slices into 3D bounding boxes.
- At most `--max-events` events are stored per comparison image, even if more are detected.
- `--min-event-size` filters out very small connected regions.
- `--log-level DEBUG` enables full stack traces for failures.
