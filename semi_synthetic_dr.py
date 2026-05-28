"""
Semi-Synthetic MNAR Experiments v2
- Learn masking from REAL missingness patterns (not hand-tuned β)
- Two modes:
  - MAR: masking learned from features only → propensity CAN recover → DR works
  - MNAR: masking learned from features + score → propensity CANNOT fully recover → DR partial
- Sweep keep rates to show bias vs correction
- Compare: Naive, IRT, IPW, DR+IRT
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).resolve().parent
PROCESSED = PROJECT / 'processed'
FIGURES = PROJECT / 'figures'
FIGURES.mkdir(exist_ok=True)

BENCHMARKS = ['mmbench_v11', 'ai2d_test', 'mmlupro', 'matharena']
KEEP_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ['features_only', 'features_and_score']
CHUNK_SIZE = 200_000

MODEL_BOOL_COLS = [
    'is_instruct', 'is_multimodal', 'is_closed', 'is_swebench_agent',
    'is_reward_model', 'is_fc_variant', 'is_reasoning'
]
MODEL_NUM_COLS = ['log_param_count', 'arena_elo', 'hf_downloads', 'openrouter_context_length']
MODEL_CAT_COLS = ['provider', 'arch_family']


# ── Data loading ─────────────────────────────────────────────────────────

def load_global():
    mf = pd.read_csv(PROJECT / 'model_features.csv')
    emb = np.load(PROJECT / 'item_embeddings.npy')
    with open(PROJECT / 'item_embedding_meta.json') as f:
        meta = json.load(f)
    emb_idx = {iid: i for i, iid in enumerate(meta['item_ids'])}
    return mf, emb, emb_idx


def load_benchmark(bid):
    bdir = PROCESSED / bid
    O = np.load(bdir / 'observation_matrix.npz')['data']
    M = np.load(bdir / 'response_matrix.npz')['data']
    with open(bdir / 'model_ids.json') as f:
        model_ids = json.load(f)
    with open(bdir / 'item_ids.json') as f:
        item_ids = json.load(f)
    item_stats = pd.read_csv(bdir / 'item_stats.csv')
    return O, M, model_ids, item_ids, item_stats


# ── Feature construction ────────────────────────────────────────────────

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
    stat_arrays = []
    for c in ['item_difficulty', 'item_discrimination', 'content_length']:
        if c in ist.columns:
            vals = ist[c].values.astype(float)
            vals = np.nan_to_num(vals, nan=np.nanmedian(vals) if np.any(np.isfinite(vals)) else 0)
            stat_arrays.append(vals)
    X_stats = np.column_stack(stat_arrays) if stat_arrays else np.empty((len(item_ids), 0))

    indices = [emb_idx.get(iid) for iid in item_ids]
    item_emb = np.zeros((len(item_ids), emb.shape[1]), dtype=np.float32)
    for i, idx in enumerate(indices):
        if idx is not None:
            item_emb[i] = emb[idx]

    actual_n = min(n_pca, item_emb.shape[0] - 1, item_emb.shape[1])
    if actual_n > 0:
        pca = PCA(n_components=actual_n, random_state=42)
        X_emb = pca.fit_transform(item_emb)
    else:
        X_emb = np.empty((len(item_ids), 0))
    return np.hstack([X_stats, X_emb])


def build_pair_features(X_model, X_item, flat_indices, n_items, extra_col=None):
    """Build pair feature matrix for given flat indices. Optionally append extra column."""
    mi = flat_indices // n_items
    ii = flat_indices % n_items
    X = np.hstack([X_model[mi], X_item[ii]])
    if extra_col is not None:
        X = np.hstack([X, extra_col[flat_indices, None]])
    return X


# ── IRT (1PL Rasch) ─────────────────────────────────────────────────────

def fit_irt(M, O_mask, n_iter=200, lr=0.1, lam=0.01):
    """Fit 1PL IRT on entries where O_mask=1. Returns M_irt for all entries."""
    n_models, n_items = M.shape
    obs = O_mask == 1

    M_safe = np.where(obs, np.nan_to_num(M, nan=0.5), np.nan)
    model_means = np.nanmean(M_safe, axis=1)
    model_means = np.clip(np.where(np.isnan(model_means), 0.5, model_means), 0.01, 0.99)
    theta = logit(model_means)

    item_means = np.nanmean(M_safe, axis=0)
    item_means = np.clip(np.where(np.isnan(item_means), 0.5, item_means), 0.01, 0.99)
    beta_j = -logit(item_means)

    obs_i, obs_j = np.where(obs)
    y_obs = np.clip(np.nan_to_num(M[obs_i, obs_j], nan=0.5), 0.001, 0.999)

    effective_lr = lr * min(1.0, n_models / 30.0)

    for _ in range(n_iter):
        logits = np.clip(theta[obs_i] - beta_j[obs_j], -10, 10)
        p = sigmoid(logits)
        residual = y_obs - p

        grad_theta = np.zeros(n_models)
        np.add.at(grad_theta, obs_i, residual)
        grad_theta -= lam * theta

        grad_beta = np.zeros(n_items)
        np.add.at(grad_beta, obs_j, -residual)
        grad_beta -= lam * beta_j

        theta += effective_lr * np.clip(grad_theta, -5, 5)
        beta_j += effective_lr * np.clip(grad_beta, -5, 5)
        theta = np.clip(theta, -10, 10)
        beta_j = np.clip(beta_j, -10, 10)

    return sigmoid(theta[:, None] - beta_j[None, :])


# ── Learn masking from real missingness ─────────────────────────────────

def learn_masking_model(X_model, X_item, O_orig, M, n_models, n_items,
                        include_score=False, M_irt=None):
    """Learn P(O_orig=1 | features [+ score]) from real data.

    Returns logit scores z for O_orig=1 entries (higher z = more likely observed).
    The masking will preferentially drop entries with lower z.
    """
    n_total = n_models * n_items
    y = O_orig.ravel().astype(int)

    # For MNAR: create M_filled (real M where O=1, IRT prediction where O=0)
    extra_col = None
    if include_score:
        M_filled = np.where(O_orig == 1,
                            np.nan_to_num(M, nan=0.5),
                            M_irt).ravel().astype(np.float32)
        extra_col = M_filled

    # Fit scaler on sample
    rng = np.random.RandomState(42)
    sample_size = min(300_000, n_total)
    sample_idx = rng.choice(n_total, sample_size, replace=False)
    X_sample = build_pair_features(X_model, X_item, sample_idx, n_items,
                                   extra_col=extra_col)
    scaler = StandardScaler()
    scaler.fit(X_sample)
    del X_sample

    # Train logistic regression in chunks
    X_chunks, y_chunks = [], []
    for start in range(0, n_total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_total)
        idx = np.arange(start, end)
        Xc = build_pair_features(X_model, X_item, idx, n_items, extra_col=extra_col)
        Xc = scaler.transform(Xc)
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        X_chunks.append(Xc)
        y_chunks.append(y[start:end])
    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)
    del X_chunks, y_chunks

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_all, y_all)

    # Get predicted probabilities for ALL entries
    probs = clf.predict_proba(X_all)[:, 1]
    del X_all

    # Extract logit scores for O_orig=1 entries only
    orig_mask = O_orig.ravel() == 1
    z = logit(np.clip(probs[orig_mask], 1e-6, 1 - 1e-6))

    return z


def generate_masking(z, O_orig, keep_rate, rng):
    """Generate O_new by masking O_orig=1 entries following learned pattern.

    Entries with lower z (less expected to be observed) get masked first.
    """
    z_centered = z - z.mean()

    # Binary search for alpha to hit target keep_rate
    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + z_centered).mean()
        if rate < keep_rate:
            lo = mid
        else:
            hi = mid
        if abs(rate - keep_rate) < 1e-4:
            break
    alpha = (lo + hi) / 2

    p_keep = sigmoid(alpha + z_centered)
    keep = rng.random(len(z)) < p_keep

    O_new = O_orig.copy().ravel()
    orig_indices = np.where(O_orig.ravel() == 1)[0]
    O_new[orig_indices[~keep]] = 0
    O_new = O_new.reshape(O_orig.shape)

    return O_new, keep.mean()


# ── Propensity model ─────────────────────────────────────────────────────

def fit_propensity(X_model, X_item, O_orig, O_new, n_models, n_items):
    """Fit propensity: P(O_new=1 | features) among O_orig=1 entries.

    The propensity model NEVER sees the score M — only features.
    Returns pi_hat for O_orig=1 entries (flat array)."""
    orig_mask = O_orig.ravel() == 1
    orig_indices = np.where(orig_mask)[0]
    y_prop = O_new.ravel()[orig_indices]
    n_orig = len(orig_indices)

    # Build features for O_orig=1 entries
    X_chunks = []
    for start in range(0, n_orig, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_orig)
        idx = orig_indices[start:end]
        Xc = build_pair_features(X_model, X_item, idx, n_items)
        X_chunks.append(Xc)
    X_train = np.vstack(X_chunks)
    del X_chunks

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_prop)

    pi_flat = clf.predict_proba(X_train)[:, 1]
    pi_flat = np.clip(pi_flat, 0.05, 0.95)
    del X_train

    # Map back to full matrix for convenience
    pi_hat = np.ones((n_models, n_items), dtype=np.float32) * 0.5  # default
    pi_hat.ravel()[orig_indices] = pi_flat
    return pi_hat


# ── Estimators ───────────────────────────────────────────────────────────

def compute_ground_truth(M, O_orig):
    """Ground truth: mean score per model over O_orig=1 entries."""
    M_safe = np.where(O_orig == 1, np.nan_to_num(M, nan=0.0), np.nan)
    mu = np.nanmean(M_safe, axis=1)
    return mu


def estimator_naive(M, O_new, O_orig):
    """Mean of entries surviving in O_new (among O_orig=1)."""
    n_models = M.shape[0]
    mu = np.full(n_models, np.nan)
    for i in range(n_models):
        mask = (O_new[i] == 1) & (O_orig[i] == 1)
        if mask.sum() > 0:
            mu[i] = np.nanmean(M[i, mask])
    return mu


def estimator_irt(M_irt, O_orig):
    """IRT prediction averaged over O_orig=1 entries."""
    n_models = M_irt.shape[0]
    mu = np.full(n_models, np.nan)
    for i in range(n_models):
        mask = O_orig[i] == 1
        if mask.sum() > 0:
            mu[i] = M_irt[i, mask].mean()
    return mu


def estimator_ipw(M, O_new, O_orig, pi_hat):
    """Inverse propensity weighted mean."""
    n_models = M.shape[0]
    mu = np.full(n_models, np.nan)
    M_safe = np.nan_to_num(M, nan=0.0)
    for i in range(n_models):
        orig_mask = O_orig[i] == 1
        n_orig = orig_mask.sum()
        if n_orig == 0:
            continue
        new_obs = (O_new[i, orig_mask] == 1).astype(float)
        mu[i] = (new_obs * M_safe[i, orig_mask] / pi_hat[i, orig_mask]).sum() / n_orig
    return mu


def estimator_dr(M, M_irt, O_new, O_orig, pi_hat):
    """Doubly robust: IRT outcome + propensity correction."""
    n_models = M.shape[0]
    mu = np.full(n_models, np.nan)
    M_safe = np.nan_to_num(M, nan=0.0)
    for i in range(n_models):
        orig_mask = O_orig[i] == 1
        n_orig = orig_mask.sum()
        if n_orig == 0:
            continue
        new_obs = (O_new[i, orig_mask] == 1).astype(float)
        scores = M_safe[i, orig_mask]
        m_irt = M_irt[i, orig_mask]
        pi = pi_hat[i, orig_mask]
        # DR formula: R_hat + O/pi * (M - R_hat)
        mu[i] = (m_irt + new_obs / pi * (scores - m_irt)).sum() / n_orig
    return mu


def evaluate(mu_hat, mu_true):
    valid = ~(np.isnan(mu_hat) | np.isnan(mu_true))
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan
    mh, mt = mu_hat[valid], mu_true[valid]
    bias = (mh - mt).mean()
    rmse = np.sqrt(((mh - mt) ** 2).mean())
    rank_corr = spearmanr(mh, mt).correlation
    return bias, rmse, rank_corr


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading global data...")
    mf, emb, emb_idx = load_global()
    print(f"  Done in {time.time()-t0:.1f}s\n")

    all_results = []

    for b_idx, bid in enumerate(BENCHMARKS):
        tb = time.time()
        print(f"[{b_idx+1}/{len(BENCHMARKS)}] {bid}")
        O_orig, M, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O_orig.shape
        density = O_orig.mean()
        print(f"  {n_models}×{n_items}, density={density:.3f}")

        # Ground truth
        mu_true = compute_ground_truth(M, O_orig)

        # Build features (once per benchmark)
        X_model, _ = build_model_features(mf, model_ids)
        X_item = build_item_features(item_stats, item_ids, emb, emb_idx)

        # For MNAR mode: fit IRT on full observed data to impute M for O=0 entries
        print(f"  Fitting IRT on full observed data (for MNAR imputation)...")
        M_irt_full = fit_irt(M, O_orig)

        for mode in MODES:
            include_score = (mode == 'features_and_score')

            # Learn masking model from real missingness pattern (once per mode)
            print(f"  Learning {mode.upper()} masking model...")
            tm = time.time()
            z_scores = learn_masking_model(
                X_model, X_item, O_orig, M, n_models, n_items,
                include_score=include_score, M_irt=M_irt_full
            )
            print(f"    Masking model fitted in {time.time()-tm:.1f}s")
            print(f"    z range: [{z_scores.min():.2f}, {z_scores.max():.2f}], "
                  f"std={z_scores.std():.2f}")

            for keep_rate in KEEP_RATES:
                rng = np.random.RandomState(42 + int(keep_rate * 100))
                O_new, actual_rate = generate_masking(z_scores, O_orig, keep_rate, rng)

                # Fit IRT on masked data
                M_irt = fit_irt(M, O_new)

                # Fit propensity on masked data (features only, never M)
                pi_hat = fit_propensity(X_model, X_item, O_orig, O_new,
                                        n_models, n_items)

                # Compute estimators
                mu_naive = estimator_naive(M, O_new, O_orig)
                mu_irt = estimator_irt(M_irt, O_orig)
                mu_ipw = estimator_ipw(M, O_new, O_orig, pi_hat)
                mu_dr = estimator_dr(M, M_irt, O_new, O_orig, pi_hat)

                # Evaluate
                nb, nr, nc = evaluate(mu_naive, mu_true)
                ib, ir, ic = evaluate(mu_irt, mu_true)
                pb, pr, pc = evaluate(mu_ipw, mu_true)
                db, drr, dc = evaluate(mu_dr, mu_true)

                print(f"    keep={keep_rate:.1f}: naive={nb:+.4f}, irt={ib:+.4f}, "
                      f"ipw={pb:+.4f}, dr={db:+.4f}  (actual_rate={actual_rate:.3f})")

                all_results.append({
                    'benchmark_id': bid, 'mode': mode,
                    'keep_rate': keep_rate, 'actual_keep_rate': actual_rate,
                    'naive_bias': nb, 'naive_rmse': nr, 'naive_rank_corr': nc,
                    'irt_bias': ib, 'irt_rmse': ir, 'irt_rank_corr': ic,
                    'ipw_bias': pb, 'ipw_rmse': pr, 'ipw_rank_corr': pc,
                    'dr_bias': db, 'dr_rmse': drr, 'dr_rank_corr': dc,
                })

        print(f"  Done in {time.time()-tb:.1f}s\n")

    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT / 'semi_synthetic_results.csv', index=False)
    print(f"Saved semi_synthetic_results.csv ({len(df)} rows)")

    # ── Figures ───────────────────────────────────────────────────────────

    estimator_names = ['naive', 'irt', 'ipw', 'dr']
    colors = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
    labels_map = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
    markers = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}

    mode_titles = {'features_only': 'Masking from features only',
                   'features_and_score': 'Masking from features + score'}

    for metric, ylabel, zero_line in [
        ('bias', 'Bias (estimated − true)', True),
        ('rmse', 'RMSE', False),
        ('rank_corr', 'Spearman ρ', False),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey='row')

        for row, mode in enumerate(MODES):
            dfs = df[df['mode'] == mode]
            for col, bid in enumerate(BENCHMARKS):
                ax = axes[row, col]
                bdf = dfs[dfs['benchmark_id'] == bid]
                for est in estimator_names:
                    col_name = f'{est}_{metric}'
                    ax.plot(bdf['keep_rate'], bdf[col_name],
                            marker=markers[est], ms=5, color=colors[est],
                            label=labels_map[est], linewidth=1.5)
                if zero_line:
                    ax.axhline(0, color='gray', ls='--', lw=0.8)
                ax.set_xlabel('Keep rate')
                if row == 0:
                    ax.set_title(bid, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'{mode_titles[mode]}\n{ylabel}', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(KEEP_RATES)

        # Single legend at bottom
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', ncol=4,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fname = f'fig_dr_{metric}.png'
        fig.savefig(FIGURES / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figures/{fname}")

    # Summary table at keep_rate=0.7
    print(f"\n{'='*90}")
    print("  SUMMARY at keep_rate=0.7")
    print(f"{'='*90}")
    for mode in MODES:
        print(f"\n  --- {mode.upper()} ---")
        s = df[(df['keep_rate'] == 0.7) & (df['mode'] == mode)]
        cols = ['benchmark_id', 'naive_bias', 'irt_bias', 'ipw_bias', 'dr_bias',
                'naive_rmse', 'irt_rmse', 'ipw_rmse', 'dr_rmse']
        print(s[cols].to_string(index=False))

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
