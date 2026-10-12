# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - 2026-10-13

### Changed
* Depend on `gertils` v0.7.0, whose `find_single_path_by_fov` reads a field of view from a file name more strictly. The names this plugin looks for are read as before.
* Drop support for Python 3.10, which gertils v0.7.0 no longer supports: nuclei-vis-napari requires Python 3.11 or 3.12, and no longer depends on `importlib_resources`.
* Allow napari 0.5 to 0.9 on macOS, as well as 0.4.19.post1, where it was pinned at 0.4.19.post1. napari 0.9 needs Python 3.11 or later. The plugin's own lock and macOS CI use napari 0.9.2; looptrace, which holds numpy below the 1.25 that napari 0.9 needs, keeps 0.4.19.
* The Nix shell uses nixpkgs 26.05, with Poetry 2, and `poetry.lock` is in Poetry 2's format.
* CI installs the newest Poetry, as it already did: the `poetry-version` the workflows set, `1.7.1`, is not an input of the action that installs Poetry, so it had no effect. Each CI job caches its environment separately, per Python.

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
* `get_reader` declines, rather than raising, when the folder cannot be examined -- an exception during napari's reader selection is a crash rather than a decline that hands the drop to the next plugin. Two cases: filenames giving two names to one field of view (`P1.zarr` beside `P0001.zarr`, both parsing to 1), and a folder the filesystem will not describe, whether a subfolder that exists but cannot be listed or a stale handle on a network mount. The second was reachable before this release too, at parse time.
* A folder that has all three subfolders but still cannot be read is refused at WARNING rather than DEBUG. Such a folder is almost certainly the one that was meant, and napari reports only "no reader available", which hid a cause the user could act on. Refusals of folders that are not nuclei data at all stay silent, since every plugin is asked about every drop.

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
