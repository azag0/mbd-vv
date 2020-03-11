This repository contains computational resources for the manuscript *Density-functional model for van der Waals interactions: Unifying many-body atomic approaches with nonlocal functionals*. It is structured as a Python package `mbdvv`. The DFT and MBD calculations are managed with [Caf](https://github.com/jhrmnn/mona).

## Requirements

- Python requirements are managed with [Poetry](https://poetry.eustace.io) and specified in the `pyproject.toml` and `poetry.lock` files. This includes the package [Pymbd](https://github.com/jhrmnn/libmbd) for calculating MBD energies.
-   [FHI-aims](https://aimsclub.fhi-berlin.mpg.de) for DFT calculations, commits `5e905e08` (2018-01-06), `b8a087dd` (2018-03-07), and `fbf4c4af5` (2019-02-08).
-   The data file `all-data.h5` from [10.1021/acs.jctc.7b01172](https://doi.org/10.1021/acs.jctc.7b01172), available at [10.6084/m9.figshare.5117167](https://doi.org/10.6084/m9.figshare.5117167).

## File organization

-   `mbdvv/`: The actual Python package, used in scripts and notebooks.
-   `data/`: Parsed raw data. These can be either generated from raw output files by running `scripts/figs.py`, or downloaded from [10.6084/m9.figshare.9943301](https://doi.org/10.6084/m9.figshare.9943301).
-   `notebooks/`: Jupyter and [KnitJ](https://github.com/jhrmnn/knitj) notebooks.
-   `patches/`: Patches of FHI-aims used for the calculations. They are to be applied against the commits stated on the first line of the patches.
-   `results/`: Processed numerical results in CSV format.
- `scripts/figs.py`: Scripts for generating figures for the manuscript.
- `vendor/`: Python dependencies managed as Git submodules.
