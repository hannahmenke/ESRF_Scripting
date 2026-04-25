# `live_tomo_diff_nx.py`

## What it is

This is a variant of `live_tomo_diff.py` for the `0_39` directory layout, where each tomography dataset contains a single projection stack stored in:

```text
dataset_name/projections/*.nx
```

Flats and darks are stored separately in `references/`, so the projection `.nx` file already contains only projection images.

## How to run

Compare two explicit datasets:

```bash
python3 live_tomo_diff_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --comparison-path 0_39/039_Estaillades_WW_Drainage_first_position_0002 \
  --projection-index 237
```

Auto-follow the latest dataset with the same position:

```bash
python3 live_tomo_diff_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --projection-index 237 \
  --position-mode same \
  --downsample 2
```

Use a hot/cold difference colormap:

```bash
python3 live_tomo_diff_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --projection-index 237 \
  --hot-cold
```

Use the faster display mode:

```bash
python3 live_tomo_diff_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position \
  --projection-index 237 \
  --fast
```

You can also pass the projection `.nx` file directly:

```bash
python3 live_tomo_diff_nx.py \
  --reference-path 0_39/039_Estaillades_WW_Drainage_first_position/projections/039_Estaillades_WW_Drainage_first_position.nx
```

## Notes

- The script looks for the projection stack inside the `.nx` file automatically.
- `--projection-index` is zero-based within the `.nx` projection stack.
- `--position-mode same` restricts auto-follow to datasets matching the same position label as the reference dataset.
- `--position-mode all` allows auto-follow across all positions in the same collection directory.
- `--downsample` reduces the displayed image size at read time and can make the viewer feel much faster on large images.
- `--fast` forces the display downsampling to at least `4`.
- `--hot-cold` switches the difference image to the `coolwarm` colormap.
- `--colormap` lets you choose any matplotlib colormap explicitly.
- `--log-level DEBUG` enables full stack traces for startup or live-update failures.
