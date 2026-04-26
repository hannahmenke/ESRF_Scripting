# `track_recon_events_nx.py`

## Overview

`track_recon_events_nx.py` scans a numbered reconstruction time series and records large stepwise difference events between each reconstruction and the immediately previous valid reconstruction in the same series.

It is intended for dataset layouts where each time point lives in its own dataset directory and reconstructed volumes are stored under:

```text
dataset_name/reconstructed_volumes/**/*.hdf5
```

The script produces:

- an SQLite database of comparisons and detected events
- a flat CSV summary
- optional orthogonal GIFs for quick visual review

## Processing Model

For each selected stepwise comparison, the script:

1. resolves the numbered series from `--reference-path`
2. selects members between `--start-number` and `--stop-number`
3. estimates baseline noise from the first valid comparison unless `--absolute-threshold` is provided
4. computes a detection threshold
5. thresholds each comparison volume against the previous valid reconstruction
6. groups slice components into 3D events across adjacent `z` slices
7. filters events by voxel count
8. records the largest events and writes summary outputs

## Core Defaults

- `--threshold-sigma 5.0`
- `--min-event-size 1000`
- `--merge-gap 10`
- `--max-events 100`
- `--jobs 1`

## Common Workflows

### Run event tracking

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --output-db recon_events.sqlite
```

### Use a stricter sigma threshold

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --threshold-sigma 8
```

### Use an explicit absolute threshold

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --absolute-threshold 250
```

### Preview one stepwise comparison

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --preview \
  --preview-sequence 7 \
  --absolute-threshold 250
```

### Save GIFs together with the database and CSV

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

### Export GIFs only

```bash
python3 track_recon_events_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --start-number 2 \
  --stop-number 20 \
  --gif-only \
  --gif-mode both
```

## Preview and GIF Rendering

Preview mode:

- analyzes a local 5-slice stack centered on the shown `z` index
- displays only the middle slice
- can suppress low-amplitude difference noise with a threshold-tied deadband

Relevant controls:

- `--preview-sequence`
- `--preview-z`
- `--preview-diff-mode raw|suppressed`
- `--preview-diff-floor-fraction`
- `--preview-diff-noise-floor`
- `--diff-display-min`
- `--diff-display-max`

GIF behavior:

- `--gif-mode raw`
  - writes side-by-side previous/current raw frames
- `--gif-mode diff`
  - writes difference frames
- `--gif-mode both`
  - writes both raw and diff GIF sets

GIF difference rendering uses the same suppression settings as the preview.

## Output Schema

The SQLite database contains:

- `runs`
- `comparisons`
- `events`

Each event row stores:

- event rank
- voxel count
- peak and mean signed/absolute difference
- cropped centroid and bounding box
- full-volume centroid and bounding box

The CSV summary is written next to the SQLite database with the same basename.

## Notes

- Sequence numbers come from dataset suffixes such as `_0002`; an unsuffixed dataset counts as `0`.
- The first valid stepwise comparison in range is used to estimate baseline noise when sigma-based thresholding is active.
- `--absolute-threshold` bypasses baseline estimation.
- `--min-event-size` filters merged 3D events after grouping.
- `--merge-gap` controls how far apart slice components may be in `x/y` and still be merged across adjacent `z` slices.
- Relative `--output-db` paths are resolved relative to the reference collection directory.
- When cropping is active, the original `z_min/y_min/x_min` style coordinates are crop-relative and the `full_*` columns are full-volume coordinates.
- `--log-level DEBUG` enables full stack traces.
