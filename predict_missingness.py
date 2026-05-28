"""
Task 1: Predict item-level missingness for all benchmarks with <100% density.
Logistic regression only. No subsampling. Chunked for large benchmarks.
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).resolve().parent
PROCESSED = PROJECT / 'processed'

MODEL_BOOL_COLS = [
    'is_instruct', 'is_multimodal', 'is_closed', 'is_swebench_agent',
    'is_reward_model', 'is_fc_variant', 'is_reasoning'
]
MODEL_NUM_COLS = ['log_param_count', 'arena_elo', 'hf_downloads', 'openrouter_context_length']
MODEL_CAT_COLS = ['provider', 'arch_family']

CHUNK_SIZE = 200_000
MATERIALIZE_THRESHOLD = 500_000


def load_all():
    mf = pd.read_csv(PROJECT / 'model_features.csv')
    emb = np.load(PROJECT / 'item_embeddings.npy')
    with open(PROJECT / 'item_embedding_meta.json') as f:
        emb_meta = json.load(f)
    emb_idx = {iid: i for i, iid in enumerate(emb_meta['item_ids'])}
    obs_summary = pd.read_csv(PROCESSED / 'observation_summary.csv')
    return mf, emb, emb_idx, obs_summary


def load_benchmark(benchmark_id):
    bdir = PROCESSED / benchmark_id
    O = np.load(bdir / 'observation_matrix.npz')['data']
    with open(bdir / 'model_ids.json') as f:
        model_ids = json.load(f)
    with open(bdir / 'item_ids.json') as f:
        item_ids = json.load(f)
    item_stats = pd.read_csv(bdir / 'item_stats.csv')
    return O, model_ids, item_ids, item_stats


def build_model_features(mf, model_ids):
    mf_sub = mf.set_index('subject_id').loc[model_ids].reset_index()
    X_bool = mf_sub[MODEL_BOOL_COLS].values.astype(float)
    feat_names = list(MODEL_BOOL_COLS)

    X_num = mf_sub[MODEL_NUM_COLS].copy()
    for c in MODEL_NUM_COLS:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med if pd.notna(med) else 0)
    X_num = X_num.values.astype(float)
    feat_names += list(MODEL_NUM_COLS)

    cat_frames = []
    for c in MODEL_CAT_COLS:
        dummies = pd.get_dummies(mf_sub[c], prefix=c, drop_first=True, dtype=float)
        cat_frames.append(dummies)
        feat_names += list(dummies.columns)
    X_cat = pd.concat(cat_frames, axis=1).values if cat_frames else np.empty((len(model_ids), 0))

    return np.hstack([X_bool, X_num, X_cat]), feat_names


def build_item_features(item_stats, item_ids, emb, emb_idx, n_pca=20):
    ist = item_stats.set_index('item_id').loc[item_ids]
    stat_cols = []
    stat_arrays = []
    for c in ['item_difficulty', 'item_discrimination', 'content_length']:
        if c in ist.columns:
            vals = ist[c].values.astype(float)
            vals = np.nan_to_num(vals, nan=np.nanmedian(vals) if np.any(np.isfinite(vals)) else 0)
            stat_arrays.append(vals)
            stat_cols.append(c)
    X_stats = np.column_stack(stat_arrays) if stat_arrays else np.empty((len(item_ids), 0))

    # Embeddings
    indices = [emb_idx.get(iid) for iid in item_ids]
    if all(idx is not None for idx in indices):
        item_emb = emb[indices]
    else:
        item_emb = np.zeros((len(item_ids), emb.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx is not None:
                item_emb[i] = emb[idx]

    actual_n_pca = min(n_pca, item_emb.shape[0] - 1, item_emb.shape[1])
    if actual_n_pca > 0:
        pca = PCA(n_components=actual_n_pca, random_state=42)
        X_emb = pca.fit_transform(item_emb)
        emb_names = [f'emb_pc{i}' for i in range(actual_n_pca)]
    else:
        X_emb = np.empty((len(item_ids), 0))
        emb_names = []

    return np.hstack([X_stats, X_emb]), stat_cols + emb_names


def make_pair_matrix(X_model, X_item, n_models, n_items):
    """Materialize full pair feature matrix. Only for <=500K pairs."""
    X = np.empty((n_models * n_items, X_model.shape[1] + X_item.shape[1]), dtype=np.float32)
    X[:, :X_model.shape[1]] = np.repeat(X_model, n_items, axis=0)
    X[:, X_model.shape[1]:] = np.tile(X_item, (n_models, 1))
    return X


def run_cv_logistic(X_model, X_item, y, n_models, n_items, col_slice=None):
    """Run 5-fold CV logistic regression. Chunked for large benchmarks."""
    n_total = n_models * n_items
    use_chunks = n_total > MATERIALIZE_THRESHOLD

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, briers = [], []
    last_clf = None
    last_scaler = None

    if not use_chunks:
        X = make_pair_matrix(X_model, X_item, n_models, n_items)
        if col_slice is not None:
            X = X[:, col_slice]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            clf.fit(X[train_idx], y[train_idx])
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], y_prob))
            briers.append(brier_score_loss(y[test_idx], y_prob))
            last_clf = clf
            last_scaler = scaler
    else:
        # Chunked: fit scaler on a sample first, then process chunks
        # Build scaler from a random sample of 200K pairs
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, min(200_000, n_total), replace=False)
        sample_mi = sample_idx // n_items
        sample_ii = sample_idx % n_items
        Xm_s = X_model[sample_mi]
        Xi_s = X_item[sample_ii]
        X_sample = np.hstack([Xm_s, Xi_s])
        if col_slice is not None:
            X_sample = X_sample[:, col_slice]
        scaler = StandardScaler()
        scaler.fit(X_sample)
        del X_sample, Xm_s, Xi_s

        for train_idx, test_idx in skf.split(np.zeros(n_total), y):
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            # Fit in chunks
            # For logistic regression we need all training data at once (no partial_fit)
            # So we materialize train set in chunks and collect
            train_chunks = []
            for start in range(0, len(train_idx), CHUNK_SIZE):
                chunk_idx = train_idx[start:start + CHUNK_SIZE]
                mi = chunk_idx // n_items
                ii = chunk_idx % n_items
                Xc = np.hstack([X_model[mi], X_item[ii]])
                if col_slice is not None:
                    Xc = Xc[:, col_slice]
                Xc = scaler.transform(Xc)
                Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
                train_chunks.append(Xc)
            X_train = np.vstack(train_chunks)
            del train_chunks
            clf.fit(X_train, y[train_idx])
            del X_train

            # Predict in chunks
            y_probs = []
            for start in range(0, len(test_idx), CHUNK_SIZE):
                chunk_idx = test_idx[start:start + CHUNK_SIZE]
                mi = chunk_idx // n_items
                ii = chunk_idx % n_items
                Xc = np.hstack([X_model[mi], X_item[ii]])
                if col_slice is not None:
                    Xc = Xc[:, col_slice]
                Xc = scaler.transform(Xc)
                Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
                y_probs.append(clf.predict_proba(Xc)[:, 1])
            y_prob = np.concatenate(y_probs)
            aucs.append(roc_auc_score(y[test_idx], y_prob))
            briers.append(brier_score_loss(y[test_idx], y_prob))
            last_clf = clf
            last_scaler = scaler

    return np.mean(aucs), np.std(aucs), np.mean(briers), last_clf, last_scaler


def main():
    t0 = time.time()
    print("Loading data...")
    mf, emb, emb_idx, obs_summary = load_all()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # 5 focus benchmarks with genuine item-level missingness
    FOCUS = ['ai2d_test', 'mmbench_v11', 'mmlupro', 'matharena', 'ultrafeedback']
    targets = obs_summary[obs_summary['benchmark_id'].isin(FOCUS)].sort_values('density', ascending=False)
    print(f"\nFocus benchmarks: {len(targets)}")
    for _, row in targets.iterrows():
        print(f"  {row['benchmark_id']:20s}: density={row['density']:.3f}, "
              f"{int(row['n_models'])}×{int(row['n_items'])}")

    all_results = []

    for _, brow in targets.iterrows():
        bid = brow['benchmark_id']
        O, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O.shape
        n_total = n_models * n_items
        n_missing = int((1 - O).sum())

        # Skip benchmarks with <10 missing cells
        if n_missing < 10:
            print(f"\n  Skipping {bid}: only {n_missing} missing cells (<10, not enough for prediction)")
            continue

        tb = time.time()
        X_model, model_feat_names = build_model_features(mf, model_ids)
        X_item, item_feat_names = build_item_features(item_stats, item_ids, emb, emb_idx)
        y = O.ravel()
        d_model = X_model.shape[1]
        all_feat_names = model_feat_names + item_feat_names

        # Full model: model + item features
        auc_mean, auc_std, brier, clf, scaler = run_cv_logistic(
            X_model, X_item, y, n_models, n_items)

        # Model-only ablation
        model_only_auc, _, _, _, _ = run_cv_logistic(
            X_model, X_item, y, n_models, n_items,
            col_slice=slice(0, d_model))

        # Item-only ablation
        item_only_auc, _, _, _, _ = run_cv_logistic(
            X_model, X_item, y, n_models, n_items,
            col_slice=slice(d_model, None))

        # Top 5 features by |coef| from last fold's fitted model
        top_feats = []
        if clf is not None and hasattr(clf, 'coef_'):
            coefs = clf.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[::-1][:5]
            for idx in top_idx:
                if idx < len(all_feat_names):
                    top_feats.append((all_feat_names[idx], float(coefs[idx])))

        elapsed = time.time() - tb
        print(f"  {bid:20s}: AUC={auc_mean:.4f}±{auc_std:.4f}  model={model_only_auc:.4f}  "
              f"item={item_only_auc:.4f}  ({elapsed:.1f}s)")

        res = {
            'benchmark_id': bid,
            'density': brow['density'],
            'n_pairs': n_total,
            'n_missing': n_missing,
            'logistic_auc_mean': auc_mean,
            'logistic_auc_std': auc_std,
            'logistic_brier': brier,
            'model_only_auc': model_only_auc,
            'item_only_auc': item_only_auc,
        }
        for i, (fname, fcoef) in enumerate(top_feats):
            res[f'top_feat_{i+1}_name'] = fname
            res[f'top_feat_{i+1}_coef'] = fcoef
        all_results.append(res)

    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT / 'missingness_prediction_results.csv', index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("  SUMMARY (sorted by density descending)")
    print(f"{'='*80}")
    summary_cols = ['benchmark_id', 'density', 'n_missing', 'logistic_auc_mean',
                    'model_only_auc', 'item_only_auc']
    print(df[summary_cols].to_string(index=False))
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print(f"Saved to missingness_prediction_results.csv")


if __name__ == '__main__':
    main()
