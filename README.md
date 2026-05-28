# Doubly Robust Estimation for Correcting Selection Bias in AI Benchmark Evaluation

CS 321M Final Project — Antoine Maechler

**Repository:** https://github.com/antoinemaechler-X/dr-benchmark-bias

## Overview

AI model evaluation relies on benchmarks, but the matrix of (model × item) scores
is highly incomplete: not every model is tested on every item. This missingness is
plausibly **non-random** (MNAR): stronger or more prominent models tend to be evaluated
more thoroughly. Naive aggregation of observed scores can therefore produce biased
estimates of model capability.

We apply **doubly robust (DR) estimation** at the item level to correct for this bias,
using the [measurement-db](https://github.com/aims-foundation/measurement-db) dataset
(909 models × 16 benchmarks × ~104K items).

## Repository Structure

```
├── README.md                        # This file
├── DATA.md                          # Data provenance and download instructions
├── METHODOLOGY.md                   # Detailed methodology writeup
├── requirements.txt                 # Python dependencies (pinned versions)│
├── download_data.py                 # Step 1: Download raw data from HuggingFace
├── embed_items.ipynb                # Step 2: Generate item embeddings (Colab)
├── build_observation_matrices.py    # Step 3: Build observation/response matrices
├── collect_model_features.py        # Step 4: Collect model metadata features
├── predict_missingness.py           # Step 5: Predict item-level missingness
├── semi_synthetic_dr.py             # Step 6: Semi-synthetic DR experiments
├── held_out_model.py                # Step 7: Held-out model evaluation
├── additional_analyses.py           # Step 8: Real DR correction, bootstrap CIs
├── plot_results.py                  # Step 9: Plot focused DR results
├── plot_missingness_eda.py          # Step 10: EDA figures
├── load_embeddings.py               # Utility: load item embeddings
│
├── processed/                       # Pre-computed observation matrices (included)
├── figures/                         # All generated figures (included)
├── model_features.csv               # Pre-computed model features (included)
├── semi_synthetic_results.csv       # Pre-computed experiment results (included)
├── held_out_results.csv             # Pre-computed held-out results (included)
├── missingness_prediction_results.csv
├── real_dr_results.csv
└── held_out_scatter.csv
```

## Environment Setup

**Python 3.10+** is required (tested with 3.10; 3.11+ should also work).

```bash
# Clone the repository
git clone https://github.com/antoinemaechler-X/dr-benchmark-bias.git
cd dr-benchmark-bias

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pyarrow`
Data collection: `requests`, `datasets`, `huggingface-hub`
Embeddings (Colab only): `sentence-transformers`, `torch`

## Reproducing Results

### Quick Start (Recommended)

Most pre-computed intermediate files are included in the repo. To reproduce
the main analysis and all figures, you only need the item embeddings from
Google Drive:

1. **Download embeddings** from
   [Google Drive](https://drive.google.com/drive/folders/1QpXVTqALhEOWLRBqqW1u9Z9s8JW9SS93?usp=sharing)
   — place `item_embeddings.npy` and `item_embedding_meta.json` in the project root

2. **Run the analysis pipeline:**

```bash
# Step 5: Missingness prediction
python predict_missingness.py

# Step 6: Semi-synthetic DR experiments
python semi_synthetic_dr.py

# Step 7: Held-out model evaluation
python held_out_model.py

# Step 8: Additional analyses — real DR correction
python additional_analyses.py

# Step 9-10: Generate all figures
python plot_results.py
python plot_missingness_eda.py
```

3. **Verify outputs:** all CSV results appear in the project root, all figures
   in `figures/`.

### Full Pipeline (From Scratch)

See [DATA.md](DATA.md) for complete instructions. Requires access to the
gated `aims-foundation/torch-measure-data` HuggingFace dataset.

```bash
python download_data.py                 # Download raw parquet files
# Run embed_items.ipynb on Google Colab  # Generate embeddings
python build_observation_matrices.py    # Build processed matrices
python collect_model_features.py        # Collect model metadata (needs internet)
# Then run Steps 5-10 above
```

## Script → Output Mapping

| Script | Outputs | Paper Section |
|---|---|---|
| `predict_missingness.py` | `missingness_prediction_results.csv` | Table 1 (Missingness AUCs) |
| `semi_synthetic_dr.py` | `semi_synthetic_results.csv`, `figures/fig_dr_{bias,rmse,rank_corr}*.png` | Figures 3–5 (DR correction results) |
| `held_out_model.py` | `held_out_results.csv`, `figures/fig_held_out_*.png` | Figures 6–8 (Held-out evaluation) |
| `additional_analyses.py` | `real_dr_results.csv`, `held_out_scatter.csv`, `figures/fig_real_dr_*.png`, `figures/fig_dr_bias_ci.png`, `figures/fig_held_out_scatter.png` | Figure 9 (Real DR), supplementary |
| `plot_results.py` | `figures/fig_dr_{bias,rmse,rank_corr}_focused.png` | Figures 3–5 (focused 2-benchmark view) |
| `plot_missingness_eda.py` | `figures/fig_density_overview.png`, `fig_heatmaps.png`, `fig_mnar_evidence.png`, `fig_model_strength_coverage.png`, `fig_livecodebench_temporal.png`, `fig_missingness_summary_table.png` | Figures 1–2 (EDA) |

## Random Seeds

All stochastic processes use `random_state=42` for reproducibility:
- Cross-validation splits in `predict_missingness.py`
- MNAR mask generation in `semi_synthetic_dr.py` and `held_out_model.py`
- Bootstrap resampling in `additional_analyses.py`
- IRT model initialization

## Code Attribution

- **Data loading and schema**: adapted from the
  [measurement-db](https://github.com/aims-foundation/measurement-db) project
  (`aims-foundation/torch-measure-data`). See `data/DATA_FORMAT.md` for the
  parquet schema specification.
- **All analysis code** (DR estimation, IRT fitting, propensity modeling,
  semi-synthetic experiments): original implementation.
