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
7. merges nearby slice components into 3D events across adjacent `z` slices
8. records up to 100 events per comparison in an SQLite database, including cropped and full-volume coordinates
9. writes a flat CSV summary next to the SQLite database for quick inspection
10. can optionally write orthogonal timeseries GIFs with step numbers burned into each frame

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

Preview one stepwise comparison before writing any outputs:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --preview \
  --preview-sequence 7 \
  --absolute-threshold 250 \
  --min-event-size 1000
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
  --gif-mode both \
  --orthogonal-center 1184,1650,1650
```

Preview or GIF diff rendering can suppress low-amplitude noise using the same threshold-tied deadband:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --preview \
  --preview-diff-mode suppressed \
  --preview-diff-floor-fraction 0.5
```

Run only GIF export without writing SQLite or CSV outputs:

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --gif-only \
  --gif-mode both
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
- cropped event centroid as `z/y/x`
- cropped 3D bounding box as `z_min/z_max`, `y_min/y_max`, `x_min/x_max`
- full-volume centroid as `full_z/full_y/full_x`
- full-volume 3D bounding box as `full_z_min/full_z_max`, `full_y_min/full_y_max`, `full_x_min/full_x_max`

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
- `--preview` opens one selected stepwise comparison and exits without writing the SQLite or CSV outputs.
- The preview analyzes a local 5-slice stack centered on the shown `z` slice, but only displays the middle slice.
- A CSV summary is written next to the SQLite database using the same base filename.
- The stored timestamps currently come from the reconstruction files' filesystem modification times in UTC.
- `--gif-labels` adds an optional corner annotation like `#0007 prev #0006` to each GIF frame.
- `--gif-mode raw` writes side-by-side previous/current raw frames. `--gif-mode diff` writes only differences. `--gif-mode both` writes both sets.
- Events are detected slice-by-slice and merged across adjacent slices into 3D bounding boxes when their boxes touch or fall within `--merge-gap`.
- At most `--max-events` events are stored per comparison image, even if more are detected.
- The current defaults are tuned more aggressively than the original version: `--min-event-size 1000`, `--merge-gap 10`.
- `--min-event-size` filters out small 3D merged events after grouping.
- When cropping is active, the original `z_min/y_min/x_min` style columns are crop-relative and the `full_*` columns are full-volume coordinates.
- Relative `--output-db` paths are resolved relative to the reference collection directory.
- Preview and GIF diff rendering can use `--preview-diff-mode suppressed`, which applies a deadband around zero tied to the detection threshold by `--preview-diff-floor-fraction` unless `--preview-diff-noise-floor` is provided.
- `--log-level DEBUG` enables full stack traces for failures.
