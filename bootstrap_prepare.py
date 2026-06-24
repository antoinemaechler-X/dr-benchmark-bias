"""
Precompute deterministic per-benchmark bundles for the bootstrap.

Everything that does NOT depend on the random masking draw is computed once
here and saved to a small .npz, so the Modal containers never need the 319 MB
embeddings file and can start instantly.

Saved per benchmark (bootstrap_bundles/{bid}.npz):
  O_orig      (n_models, n_items)   original observation mask
  M           (n_models, n_items)   response matrix (may contain nan)
  X_model     (n_models, d_m)       model features
  X_item      (n_items,  d_i)       item features (stats + PCA of embeddings)
  M_irt_full  (n_models, n_items)   IRT imputation fit on full observed data
  z_fo        (n_obs,)              masking logit-scores, features-only (MAR)
  z_fs        (n_obs,)              masking logit-scores, features+score (MNAR)
  mu_true     (n_models,)           ground-truth per-model mean over O_orig=1
  model_ids   (n_models,)           model id strings
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semi_synthetic_dr as S

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / 'bootstrap_bundles'
OUT.mkdir(exist_ok=True)

BENCHMARKS = ['ai2d_test', 'mmbench_v11', 'agentdojo', 'bfcl', 'mtbench']


def main():
    t0 = time.time()
    print("Loading global data (model features + item embeddings)...")
    mf, emb, emb_idx = S.load_global()

    for bid in BENCHMARKS:
        tb = time.time()
        print(f"\n=== {bid} ===")
        O_orig, M, model_ids, item_ids, item_stats = S.load_benchmark(bid)
        n_models, n_items = O_orig.shape
        print(f"  {n_models}x{n_items}, density={O_orig.mean():.3f}")

        X_model, _ = S.build_model_features(mf, model_ids)
        X_item = S.build_item_features(item_stats, item_ids, emb, emb_idx)
        mu_true = S.compute_ground_truth(M, O_orig)

        print("  Fitting full IRT (for MNAR imputation)...")
        M_irt_full = S.fit_irt(M, O_orig)

        print("  Learning MAR masking model (features only)...")
        z_fo = S.learn_masking_model(X_model, X_item, O_orig, M, n_models, n_items,
                                     include_score=False, M_irt=M_irt_full)
        print("  Learning MNAR masking model (features + score)...")
        z_fs = S.learn_masking_model(X_model, X_item, O_orig, M, n_models, n_items,
                                     include_score=True, M_irt=M_irt_full)

        out_path = OUT / f'{bid}.npz'
        np.savez_compressed(
            out_path,
            O_orig=O_orig.astype(np.int8),
            M=M.astype(np.float32),
            X_model=X_model.astype(np.float32),
            X_item=X_item.astype(np.float32),
            M_irt_full=M_irt_full.astype(np.float32),
            z_fo=z_fo.astype(np.float32),
            z_fs=z_fs.astype(np.float32),
            mu_true=mu_true.astype(np.float32),
            model_ids=np.array(model_ids, dtype=object),
        )
        sz = out_path.stat().st_size / 1e6
        print(f"  Saved {out_path.name} ({sz:.1f} MB) in {time.time()-tb:.1f}s")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
