# Deep-DiD Simulation and Reproduction Package

**Paper:** *A Deep-DiD Method to Estimate Heterogeneous Treatment Effects: Application to Content Creator Selection*, Marketing Science 45(2), 258–279 (2026), https://doi.org/10.1287/mksc.2023.0511
**Authors:** Yan Cheng (Shanghai University of Finance and Economics), Jingbo Wang (Chinese University of Hong Kong) and Xinyu Cao (Chinese University of Hong Kong), co-first authors; Zuo-Jun Max Shen; Yuhui Zhang
**Contact:** Yan Cheng, https://yanchengom.github.io; Jingbo Wang, jingbowang@cuhk.edu.hk; Xinyu Cao, xinyucao@cuhk.edu.hk

This repository contains the simulation code, pre-generated data, and plotting scripts used to reproduce the figures and tables in the associated paper. It is organized to allow users to regenerate simulation data, reproduce figures, and recreate the hyperparameter table.

## Repository structure
- `code/`: Simulation scripts that generate data used in figures and tables
  - `figure_2_across_beta_fe.py`
  - `figure_3a_across_N_T.py`
  - `figure_3b_across_N_D.py`
  - `figure_4_shift_in_fixed.py`
  - `figure_5a_beta_misspecified.py`
  - `figure_5b_fe_misspecified.py`
  - `figure_6_comparison_with_rf.py`
  - `table_1_hyperparameters.py`
- `simulated_data/`: JSON files (with .txt extension) produced by simulation scripts
- `figures/`: Generated PDF figures
- `table/`: Generated LaTeX/CSV tables
- `figure_script.py`: Plotting script that reads `simulated_data/` and produces PDFs in `figures/`

## Requirements and Conda environment
A Conda environment file `environment_windows.yml` is provided to manage dependencies. It is configured to install the latest PyTorch with CUDA support from the `pytorch` and `nvidia` channels.

To build the environment:

- Create from scratch:
```bash
conda env create -n deepdid -f environment_windows.yml
conda activate deepdid
```

- Update an existing environment:
```bash
conda env update -n deepdid -f environment_windows.yml --prune
conda activate deepdid
```

- CUDA notes: The environment is configured to use GPU by default. It installs `pytorch-cuda` to ensure compatibility. If you do not have a GPU or wish to run on CPU only, the scripts will automatically fallback to CPU if CUDA is unavailable.

## Device usage (CPU/GPU)
- Training/simulation scripts in `code/` automatically select GPU if available: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Plotting (`figure_script.py`) runs on CPU only (NumPy/Matplotlib).

## Regenerating simulation data (optional)
Each script in `code/` produces its corresponding input data file under `simulated_data/`. These scripts can be run individually; they are computationally heavier than plotting.

Examples:
```bash
# Activate the environment first
conda activate deepdid

# Data for Figure 2
python code/figure_2_across_beta_fe.py

# Data for Figure 3a
python code/figure_3a_across_N_T.py

# Data for Figure 6 (Random Forest vs DNN)
python code/figure_6_comparison_with_rf.py
```

Expected outputs are stored in `simulated_data/` as `.txt` files, for example:
- `data_2_across_beta_fe.txt`
- `data_3a_across_N_T.txt`
- `data_3b_across_N_D.txt`
- `data_4_shift_in_fixed.txt`
- `data_5a_beta_misspecified.txt`
- `data_5b_fe_misspecified.txt`
- `data_6_comparison_with_rf.txt`

## Reproducing the hyperparameter table
To reproduce the hyperparameter performance table and LaTeX outputs:
```bash
python code/table_1_hyperparameters.py
```
This writes CSV/LaTeX files under `table/`:
- `table_hyperparameters.csv`
- `table_hyperparameters.tex`
- `table_hyperparameters_nrmse.tex`

## Notes
- All paths are relative, so you can run from the repository root.
- GPU is optional; scripts automatically detect CUDA if available.
- Random seeds are set within scripts for reproducibility. However, please note that due to the inherent non-determinism of floating-point operations on GPUs (CUDA), results may vary slightly across different machines or runs, even with identical seeds.

## Troubleshooting
### OpenMP Error on Windows
If you encounter the error `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`, it indicates a conflict between multiple OpenMP runtimes (common when mixing Conda packages).

To resolve this, set the `KMP_DUPLICATE_LIB_OK` environment variable to `TRUE` before running the scripts.

**PowerShell:**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**Command Prompt (cmd):**
```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Citation
Please cite the paper:

Cheng, Y., Wang, J., Cao, X., Shen, Z.-J. M., and Zhang, Y. (2026). "A Deep-DiD Method to Estimate Heterogeneous Treatment Effects: Application to Content Creator Selection." *Marketing Science*, 45(2), 258-279. https://doi.org/10.1287/mksc.2023.0511

## License
MIT (see `LICENSE`).
