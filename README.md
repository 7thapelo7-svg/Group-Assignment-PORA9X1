# Group-Assignment-PORA9X1
Group Project Proposal - Systemic Risk-Driven Portfolio Selection

  Framework followed for the project 
  
  1. Replicate the KDE-based CoES minimisation framework as per Fung, T.C., Li, Y., Peng, L. and Qian, L., 2026. Statistical inference for systemic risk-driven portfolio selection. Journal of Econometrics, 253, p.106127.
  
 2.  The proposed Copula-EVT-based CoES minimisation as an improvement on the KDE CoES minimization framework

 3.  Evaluation Metrics these include (Risk and Return ,Downside Risk, Portfolio Stability , Computational Performance and Statistical Evaluation)

 4.  KDE vs Copula-EVT comparisons and conclusion

# Portfolio Tail Risk Reproduction

## Project overview
This repository reproduces the plots and tables shown in the analysis of **out‑of‑sample annualized portfolio loss given a weekly market downturn**. The code downloads market data, computes conditional tail risk estimates (CoES / ES) for several methods, runs portfolio rules, and produces the figures and summary tables used in the paper.

## Folder structure
**Top level**
- **`data/`** — raw and cached market data (CSV).  
- **`src/`** — analysis scripts and modules.  
- **`notebooks/`** — exploratory notebooks and quick checks.  
- **`results/figures/`** — generated plots (PNG, PDF).  
- **`results/tables/`** — generated tables (CSV, LaTeX).  
- **`configs/`** — configuration files (YAML) for reproducible parameter settings.  
- **`requirements.txt`** — pinned Python package list.  
- **`README.md`** — this file.

## Reproduce plots and tables step by step
1. **Create a clean environment**
   - Linux / macOS
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - Windows (PowerShell)
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```

2. **Download and cache market data**
   - Edit `configs/data_config.yaml` to set the ticker list and date range, or use the default config.
   - Run the data fetcher (it uses `yfinance` and caches CSVs in `data/`):
     ```bash
     python src/fetch_data.py --config configs/data_config.yaml
     ```
   - **Output**: CSV files in `data/` and a `data/cache_manifest.json`.

3. **Compute risk estimates and portfolio losses**
   - Run the main estimation script which computes CoES/ES for each method and stores raw results:
     ```bash
     python src/compute_tail_estimates.py --config configs/estimation_config.yaml
     ```
   - **Output**: intermediate results in `results/intermediate/` (NumPy / pickle files).

4. **Generate plots**
   - Create the figures used in the paper:
     ```bash
     python src/plot_results.py --outdir results/figures
     ```
   - **Output**: PNG and PDF files in `results/figures/`.

5. **Produce summary tables**
   - Generate the summary table (CSV and LaTeX):
     ```bash
     python src/generate_tables.py --outdir results/tables
     ```
   - **Output**: `results/tables/table1.csv` and `results/tables/table1.tex`.

6. **Reproduce everything in one command**
   - A convenience script runs all steps (download, compute, plot, table):
     ```bash
     bash scripts/run_all.sh
     ```
   - **Note**: this may take several minutes depending on the data range and methods selected.

## Scripts and key modules
- **`src/fetch_data.py`** — downloads tickers from `configs/data_config.yaml` and writes CSVs to `data/`. Uses caching to avoid repeated downloads.  
- **`src/compute_tail_estimates.py`** — computes KDE, Gaussian, Copula-LP (Gauss-EVT), nonparametric ES, and portfolio rules (MV, GMV, EW). Saves intermediate results.  
- **`src/plot_results.py`** — creates density plots, boxplots, and time series of tail estimates.  
- **`src/generate_tables.py`** — computes summary statistics (Mean, SD, RiskAdjMean, Skew, Kurtosis) and writes CSV/LaTeX tables.  
- **`scripts/run_all.sh`** — orchestrates the full pipeline.  
- **`configs/estimation_config.yaml`** — parameters such as \(g_m, g_p\), EVT thresholds, rolling window sizes, random seed.

## Reproducibility notes
- **Random seed**: set in `configs/estimation_config.yaml` as `random_seed` to ensure deterministic results for stochastic procedures.  
- **Caching**: downloaded data are cached in `data/` to avoid repeated network calls. Delete `data/` to force a fresh download.  
- **Runtime**: full pipeline runtime depends on the number of tickers and parameter grid; expect 5–30 minutes on a modern laptop for typical settings. Use `--parallel` flags in scripts where available to speed up computation.  
- **Validation**: run `notebooks/calibration_checks.ipynb` to inspect backtest hit rates and exceedance tests for CoES/ES forecasts.

## Troubleshooting
- If a package fails to install, upgrade `pip` and retry.  
- If `cvxpy` raises solver errors, install a supported solver (e.g., `OSQP`) or adjust solver options in `configs/estimation_config.yaml`.  
- If plots look different, confirm the `random_seed` and `matplotlib` backend in `configs/plot_config.yaml`.

## Contact
For questions about reproducing the analysis or to request alternative parameter settings, open an issue or contact the repository owner.

