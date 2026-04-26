# `compress_recon_volume.py`

## Overview

`compress_recon_volume.py` creates reduced HDF5 copies of reconstructed volumes for lighter inspection, sharing, or downstream analysis.

The script can operate on:

- a single reconstruction HDF5 file
- a dataset directory containing one reconstruction
- a numbered dataset series, writing one compressed output per series member

Supported transformations:

- `X/Y/Z` cropping
- uniform downsampling
- optional intensity clipping
- optional background masking
- optional `uint8` conversion
- chunked HDF5 compression

## Processing Model

For each selected reconstruction, the script:

1. resolves the input volume dataset
2. applies crop ranges
3. downsamples the volume
4. optionally clips intensities
5. optionally masks low-amplitude voxels
6. optionally converts the result to `uint8`
7. writes a new compressed HDF5 file

If a series is selected, each reconstruction is processed independently and written to its own output file.

## Input Modes

### Single-file mode

Pass one reconstruction HDF5 file:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory
```

### Series mode

Pass a dataset directory or one reconstruction file from a numbered series:

```bash
python3 compress_recon_volume.py 0_39/039_Estaillades_WW_Drainage_first_position \
  --output-dir /path/to/output_directory \
  --dataset-path /entry0000/reconstruction/results/data
```

Restrict the series to a sequence range:

```bash
python3 compress_recon_volume.py 0_39/039_Estaillades_WW_Drainage_first_position \
  --output-dir /path/to/output_directory \
  --start-number 2 \
  --stop-number 20
```

## Preview Mode

Preview mode shows orthogonal slices of the transformed output-space volume before writing files.

Preview without writing:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --preview-only \
  --crop-y 300:2600 \
  --crop-x 400:2800 \
  --downsample 2
```

Preview and still write outputs:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --preview \
  --preview-center 500,900,900
```

Preview one selected member from a numbered series:

```bash
python3 compress_recon_volume.py 0_39/039_Estaillades_WW_Drainage_first_position \
  --preview-only \
  --preview-sequence 7
```

During preview, the script also logs:

- robust intensity percentiles
- an estimated noise scale
- suggested `--clip-min` / `--clip-max`
- a suggested `--mask-threshold`
- estimated output size at the current settings
- estimated output sizes for `--downsample 2` and `--downsample 4`
- compression tradeoff guidance

## Common Workflows

### Crop and downsample

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --crop-y 300:2600 \
  --crop-x 400:2800 \
  --downsample 2
```

### Convert to `uint8`

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --clip-min 0 \
  --clip-max 3000 \
  --to-uint8
```

### Mask low-amplitude background

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --mask-threshold 25 \
  --compression lzf
```

### Compress a series in parallel

```bash
python3 compress_recon_volume.py 0_39/039_Estaillades_WW_Drainage_first_position \
  --output-dir /path/to/output_directory \
  --jobs 4
```

## Output Behavior

- `--output-dir` is required unless `--preview-only` is used.
- The output directory is created automatically if it does not exist.
- In single-file mode, the default output filename is:

```text
<input_stem>_compressed.hdf5
```

- In series mode, the default output filename is based on each dataset member name:

```text
<dataset_name>_compressed.hdf5
```

- `--output-name` is only valid when exactly one reconstruction is selected.
- By default, the output dataset path matches the input dataset path.

## Downsampling

Two downsampling modes are available:

- `average`
  - default
  - performs block averaging
  - appropriate for most visualization and reduced-resolution analysis use cases
- `subsample`
  - keeps every `N`th voxel
  - faster, but more prone to aliasing and missed small features

Example:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --downsample 2 \
  --downsample-mode subsample
```

## Intensity Operations

### Clipping

`--clip-min` and `--clip-max` limit the intensity range before optional conversion.

### Masking

`--mask-threshold` sets voxels to zero when:

```text
abs(value) < threshold
```

Masking is applied after clipping and before optional `uint8` conversion.

### `uint8` conversion

`--to-uint8` requires both `--clip-min` and `--clip-max`.

## Compression Guidance

Available compression filters:

- `gzip`
- `lzf`
- `none`

Practical tradeoffs:

- `--compression gzip --compression-level 4`
  - balanced default
- `--compression gzip --compression-level 6`
  - somewhat smaller files
  - slower writes
- `--compression lzf`
  - faster writes
  - larger files

## Parallel Execution

`--jobs` parallelizes across reconstruction files in a series, not across slices within a single HDF5 file.

This is the intended scaling model for batch compression because each worker writes its own output file independently.

## Notes

- `--preview-center` is specified in output-space coordinates after crop and downsample.
- The script copies file-level and dataset-level HDF5 attributes into the output file.
- If you still need quantitative intensities later, prefer preserving the source dtype and using crop/downsample/compression first before converting to `uint8`.
