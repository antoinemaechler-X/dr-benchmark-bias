# Data

## Quick Start (No Download Required)

The `processed/` directory and all CSV result files are included in this
repository. You can run the analysis and plotting scripts
(`predict_missingness.py`, `semi_synthetic_dr.py`, `held_out_model.py`,
`plot_results.py`, `plot_missingness_eda.py`) without downloading raw data,
**provided you also have the embedding and feature files** (see below).

## Pre-computed Intermediate Files (Included)

| File/Directory | Size | Description |
|---|---|---|
| `processed/` | 8.8 MB | Observation & response matrices per benchmark (`.npz`), item/model stats (`.csv`), ID mappings (`.json`) |
| `model_features.csv` | 94 KB | Collected model metadata (name-parsed features, Arena ELO, HF stats) |
| `missingness_prediction_results.csv` | 2 KB | Missingness model AUCs per benchmark |
| `semi_synthetic_results.csv` | 12 KB | Semi-synthetic DR experiment results |
| `held_out_results.csv` | 4 KB | Held-out model evaluation results |

## Large Files (Not in Repo)

These files are too large for Git and must be obtained separately if you want
to re-run the full pipeline from scratch:

| File | Size | Source | Needed By |
|---|---|---|---|
| `data/*.parquet` | 172 MB | measurement-db (see below) | `build_observation_matrices.py`, `collect_model_features.py` |
| `item_embeddings.npy` | 305 MB | Generated via `embed_items.ipynb` on Colab | `predict_missingness.py` |
| `item_embedding_meta.json` | 3.5 MB | Generated alongside embeddings | `predict_missingness.py` |

## Raw Data Source: measurement-db

The raw data comes from the **measurement-db** dataset, a structured collection
of AI model evaluation results across 16 benchmarks.

- **Repository**: `aims-foundation/torch-measure-data` on HuggingFace
- **Access**: Gated dataset — requires HuggingFace authentication and access approval
- **Format**: Parquet files (one per benchmark, plus registry files)

### Downloading Raw Data

1. Request access to `aims-foundation/torch-measure-data` on HuggingFace
2. Authenticate: `huggingface-cli login`
3. Run the download script:

```bash
python download_data.py
```

This downloads the 16 benchmark response files plus registry files
(`subjects.parquet`, `items.parquet`, `benchmarks.parquet`) into `data/`.

### Benchmarks Used (16)

AfriMed-QA, AgentDojo, AI2D, AndroidWorld, BFCL, Cybench,
Humanity's Last Exam, LiveCodeBench, MathArena, MathVista MINI,
MMBench V1.1, MMLU-Pro, MT-Bench, RewardBench, SWE-bench Verified,
UltraFeedback.

## Item Embeddings

The 768-dimensional item embeddings are generated using
`nomic-ai/nomic-embed-text-v1.5` via the `embed_items.ipynb` notebook
(designed for Google Colab with GPU). These embed ~103K benchmark items
and are used by `predict_missingness.py` to build item-level features.

**Download pre-computed embeddings:**
[Google Drive](https://drive.google.com/drive/folders/1QpXVTqALhEOWLRBqqW1u9Z9s8JW9SS93?usp=sharing)

Download `item_embeddings.npy` and `item_embedding_meta.json` and place
them in the project root directory.

**To regenerate from scratch:** run `embed_items.ipynb` on Google Colab
after placing `data/items.parquet` in the notebook's working directory.

## Full Pipeline Order

If reproducing everything from scratch:

```
1. download_data.py              → data/*.parquet
2. embed_items.ipynb (Colab)     → item_embeddings.npy, item_embedding_meta.json
3. build_observation_matrices.py → processed/
4. collect_model_features.py     → model_features.csv
5. predict_missingness.py        → missingness_prediction_results.csv
6. semi_synthetic_dr.py          → semi_synthetic_results.csv, figures/
7. held_out_model.py             → held_out_results.csv, figures/
8. plot_results.py               → figures/
9. plot_missingness_eda.py       → figures/
```

Steps 5–9 can run using only the pre-computed files included in this repo
(plus `item_embeddings.npy` and `item_embedding_meta.json` for step 5).
