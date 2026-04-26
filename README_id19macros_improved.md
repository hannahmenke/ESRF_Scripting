# `id19macros_improved.py`

## Overview

`id19macros_improved.py` is a BLISS macro collection for ESRF ID19 tomography and related beamline operations.

It groups together:

- beamline configuration helpers
- sample positioning helpers
- repeated tomography acquisition helpers
- high-resolution and low-resolution setup macros
- a small number of pulse-probe and multi-position acquisition helpers

This file is intended to be loaded inside the beamline control environment where the expected BLISS objects, motors, shutters, detectors, and scan macros already exist.

## Scope

The file is beamline-specific. It assumes the presence of global objects such as:

- `full_tomo`
- `mrfull_tomo`
- motion axes such as `yc`, `zc2`, `sshg`, `ssvg`, `pshg`, `psvg`
- beamline shutters and interlocks
- optical configuration devices

It is not designed to be executed as a standalone Python module.

## Main Macros

### `series_of_tomo(name, n_scans, dz=0)`

Creates a new collection and runs repeated tomography acquisitions.

Behavior:

1. creates a new collection
2. runs a dataset named `first_position`
3. optionally runs a second dataset after moving `sz` by `dz`
4. disables darks/flats after the first scan for the remainder of the series
5. restores the dark/flat flags when the macro exits, including on interruption

Notes:

- `n_scans = -1` runs continuously until interrupted
- `dz = 0` keeps acquisition at a single position
- `dz != 0` adds a second position for each cycle

### `move2LR(flag_halftomo=False)`

Configures the beamline for low-resolution tomography.

### `move2HR(flag_halftomo=True)`

Configures the beamline for high-resolution tomography.

### `moveSamplePos(...)`

Moves the sample to a position derived from image-space or voxel-space coordinates.

### `do_multiple_HR_scans(...)`

Runs a sequence of high-resolution scans across multiple ROI locations.

### `pp_shot(shot_name)`

Runs a pulse-probe style helper sequence with shutter checks and collection creation.

## Example Usage

```python
user.series_of_tomo("test_series", 5)
user.series_of_tomo("test_series", -1)
user.series_of_tomo("test_series", 10, dz=0.2)
user.move2HR()
user.move2LR(flag_halftomo=True)
```

## Operational Notes

- In `series_of_tomo`, darks and flats are enabled for the first scan and disabled afterward for throughput.
- `series_of_tomo(..., -1)` is intended for continuous acquisition until manual interruption.
- The dark/flat flags are restored in a `finally` block when the macro exits.
- Many helper macros encode fixed beamline positions and hardware assumptions.

## Requirements

- BLISS session with the expected global devices and macros already loaded
- Beamline-specific hardware names matching those used in this file

## Caveats

- The file contains beamline-specific constants and hard-coded device names.
- Some older sections still contain non-standard Unicode whitespace, which can break ordinary `python3 -m py_compile` checks outside the BLISS environment.
