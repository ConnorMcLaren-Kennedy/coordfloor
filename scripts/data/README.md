# Data placement

This repository does not include the raw datasets.

The pipeline reads dataset locations from **`params/base.yaml`**. If your folders differ from the defaults below, edit the paths in that YAML before running preprocessing.

## Van Hooren dataset (primary)

Citation:
van Hooren, B., & Meijer, K. (2024). *Data in Brief*, 54, 110312. DOI: 10.1016/j.dib.2024.110312

### Expected paths (defaults)

`scripts/01_preprocess.py` uses these fields from `params/base.yaml`:

- `vanhooren.c3d_dir`: expected default
  - `data/raw/vanhooren/0. C3D files`
- `vanhooren.emg_map_xlsx`: expected default
  - `data/raw/vanhooren/Excel file with EMG channels.xlsx`

### Outputs written by preprocessing

Running `01_preprocess.py` creates (per subject):

- `data/processed/vanhooren/Sub_XX/level_strideparity/*_full.npz`
- `data/processed/vanhooren/Sub_XX/level_strideparity/*_matched.npz`
- `metadata/qc/vanhooren_level_strideparity_qc.csv`

The main pipeline uses the `*_matched.npz` files referenced by `outputs/unified_registry.csv`.

## Santuz dataset (validation)

Citation:
Santuz, A., et al. (2018). *Frontiers in Physiology*, 9, 1509. DOI: 10.3389/fphys.2018.01509

### What the validation script expects

`scripts/10_santuz_validation.py` reads the per-subject input files listed in:

- `outputs/unified_registry.csv` (example path pattern):
  - `data/processed/santuz/vanhooren_like/FILT_EMG_P0001_01_VH.dat`

Those `.dat` files must exist locally before running `10_santuz_validation.py`.

> Note: This repository does not currently include a Santuz→`vanhooren_like` conversion script. If you start from the raw Santuz release, you must create equivalent per-subject files yourself (or update the registry to point to your own processed outputs).
