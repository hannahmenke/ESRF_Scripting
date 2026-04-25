# `compress_recon_volume.py`

## What it is

This script creates a smaller HDF5 copy of a reconstruction volume by applying some combination of:

- `X/Y/Z` cropping
- downsampling
- intensity clipping
- optional `uint8` conversion
- optional background masking
- chunked HDF5 compression

It is intended for large reconstructed volumes where you want a lighter analysis or preview copy without touching the original file.

It also supports a preview mode so you can inspect the transformed volume before writing anything.

## How to run

Basic compression with gzip:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory
```

Crop and downsample:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --crop-y 300:2600 \
  --crop-x 400:2800 \
  --downsample 2
```

Convert to `uint8` with user-defined scaling:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --clip-min 0 \
  --clip-max 3000 \
  --to-uint8
```

Mask low-value background and use faster `lzf` compression:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --mask-threshold 25 \
  --compression lzf
```

Use an explicit internal dataset path:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --dataset-path /entry0000/reconstruction/results/data
```

Preview the transformed volume without writing an output file:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --preview-only \
  --crop-y 300:2600 \
  --crop-x 400:2800 \
  --downsample 2 \
  --clip-min 0 \
  --clip-max 3000 \
  --to-uint8
```

Preview first, then still write the output file:

```bash
python3 compress_recon_volume.py input_recon.hdf5 \
  --output-dir /path/to/output_directory \
  --preview \
  --preview-center 500,900,900
```

## Notes

- `--to-uint8` requires both `--clip-min` and `--clip-max`.
- `--output-dir` is required. The script writes the new file there, not into the directory where you happen to run the command.
- `--preview-only` lets you tune parameters without writing any file, so `--output-dir` is not required in that mode.
- `--preview-center` is specified in output-space coordinates after crop/downsample.
- Use `--output-name` if you want a specific filename. Otherwise the default is `<input_stem>_compressed.hdf5`.
- If you still need quantitative intensity values later, prefer staying in the original dtype and using crop/downsample/compression first.
- `gzip` gives better compression ratios; `lzf` is usually faster.
- The script copies file-level and dataset-level attributes into the output file.
- By default the output dataset path matches the input dataset path. Use `--output-dataset-path` if you want a different path inside the new file.
