# `id19macros_improved.py`

## What it is

This is a BLISS macro file for ESRF ID19 tomography and related beamline operations. It groups together:

- hardware positioning helpers
- shutter and laser helpers
- tomography acquisition helpers
- low-resolution and high-resolution scan configuration
- a few sample / ROI automation utilities

It is intended to be loaded inside the beamline control environment where objects such as `full_tomo`, `mrfull_tomo`, motors, shutters, and detectors already exist.

## Main functions

- `series_of_tomo(name, n_scans, dz=0)`
  Runs repeated tomography acquisitions in a new collection.
  If `dz != 0`, it also acquires a second position after moving `sz` by `dz`.
  `n_scans = -1` means run continuously until interrupted.

- `move2LR(flag_halftomo=False)`
  Configures the setup for low-resolution tomography.

- `move2HR(flag_halftomo=True)`
  Configures the setup for high-resolution tomography.

- `moveSamplePos(...)`
  Moves the sample to a position derived from voxel/image coordinates.

- `do_multiple_HR_scans(...)`
  Runs a series of HR scans over multiple ROI positions.

- `pp_shot(shot_name)`
  Helper for a pulse-probe style shot with shutter checks and collection creation.

## Notable behavior

- In `series_of_tomo`, darks and flats are enabled for the first scan and then disabled for subsequent scans.
- `series_of_tomo(..., -1)` now supports unlimited repeated scans.
- The dark/flat flags are restored to `True` in a `finally` block when the macro exits or is interrupted.

## Example usage

```python
user.series_of_tomo("test_series", 5)
user.series_of_tomo("test_series", -1)
user.series_of_tomo("test_series", 10, dz=0.2)
user.move2HR()
user.move2LR(flag_halftomo=True)
```

## Requirements

- BLISS environment with the expected global devices and macros already defined.
- Beamline-specific hardware names used in this file must exist in the active session.

## Current caveats

- The file contains beamline-specific assumptions and hard-coded hardware names.
- The file also contains some non-standard Unicode whitespace in older parts of the script, which can break normal `python3 -m py_compile` checks outside the BLISS environment.
