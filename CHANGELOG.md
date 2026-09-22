# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.0] - 2026-09-22

### Added
* Read nuclei data as published by the `looptrace` pipeline: drop the `B03_NUCLEI_SEGMENTATION` folder of an analysis into Napari.

### Changed
* The nuclear masks visualisation subfolder may be named `nuclear_masks_visualisation` (as now published by `looptrace`) or `_nuclear_masks_visualisation` (older `looptrace`); the former is preferred when both are present.
* A folder is read only when all three subfolders describe at least one field of view IN COMMON. Previously the three had merely to exist, so a folder whose subfolders covered different fields of view -- a run restricted with `selected_fovs`, or subfolders assembled by hand from different analyses -- was accepted and then failed inside `numpy.stack` on an empty list, naming neither fields of view nor the folder. It is now declined, with the data-file count per subfolder.
* A refusal names the subfolder that is missing rather than listing all three, and says so when it is `nuc_images`: for a while the common case will be an analysis folder produced before `looptrace` published that output, where everything else is present and correct.
* Napari's routing of a dropped folder to this reader is now tested through `npe2`'s own dispatch rather than only by calling `get_reader`. That routing is what makes the documented promise true -- the folder's NAME does not matter -- and it rests on `accepts_directories`, which no test previously exercised. `napari.yaml` now records why `filename_patterns` is vestigial here and must not be widened to `'*'`, which would offer this plugin for every file dropped into napari only for `get_reader` to decline it.
* A folder is scanned once during reader selection rather than four times: the per-field-of-view listing is computed once and both the usability check and its failure message are derived from it.

### Fixed
* `get_reader` declines a folder whose filenames give two names to one field of view (`P1.zarr` beside `P0001.zarr`, both parsing to 1) instead of raising out of reader selection, where an exception is a crash in napari rather than a decline that lets another plugin take the drop.

### Removed
* `NucleiDataSubfolders.relpaths` and `NucleiDataSubfolders.all_present_within`, both without callers once a refusal stopped listing all three subfolders.

## [v0.1.9] - 2025-11-04

### Changed
* Depend on latest (v0.6.1) version of `gertils`

## [v0.1.8] - 2025-04-01

### Changed
* Switch to Poetry for the project build

## [v0.1.7] - 2025-03-28

### Changed
* Pin version of `ruff` to latest (v0.7.4) which still will work with downstream project using Poetry v1.8.4 or earlier. 
See [Github discussion](https://github.com/astral-sh/ruff/issues/14681).

## [v0.1.6] - 2025-03-26

### Changed
* Pin version of `ruff` to latest (v0.8.0) which still will work with downstream project using Poetry v1.8.4 or earlier. 
See [Github discussion](https://github.com/astral-sh/ruff/issues/14681).

## [v0.1.5] - 2025-03-22

### Changed
* Bump version of `gertils` dependency to v0.6.0.

## [v0.1.4] - 2024-11-21

### Changed
* Use newest version of `gertils` (v0.5.1).

## [v0.1.3] - 2024-11-21

### Changed
* Bump up dependencies of `gertils` and `numpydoc_decorator`.
* Support Python 3.12.

## [v0.1.2] - 2024-05-03

### Added
* This release is strictly to add a lot of tests and to comply with formatting and linting checks.

## [v0.1.1] - 2024-04-19

### Changed
* Bump up version of `gertils`.

## [v0.1.0] - 2024-04-19
 
### Added
* This package, first release
