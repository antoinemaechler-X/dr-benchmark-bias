"""
Held-Out Model Experiment (Leave-One-Out)

For each model (LOO):
  1. Train IRT, masking, propensity on remaining models (no data leakage)
  2. Held-out model gets MNAR-masked partial observations
  3. Estimate held-out model's full-benchmark score: Naive, IRT, IPW, DR
  4. Compare against ground truth

Tests whether DR+IRT improves score prediction for a newly-evaluated model.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from semi_synthetic_dr import (
    load_global, load_benchmark,
    build_model_features, build_item_features,
    PROJECT, FIGURES, CHUNK_SIZE,
)

warnings.filterwarnings('ignore')

BENCHMARKS = ['mmlupro', 'matharena']
KEEP_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ['features_only', 'features_and_score']
HOLD_OUT_FRAC = 0.20          # fraction of models to hold out
HOLD_OUT_MIN = 5              # minimum number of held-out models


# ── IRT ──────────────────────────────────────────────────────────────────

def fit_irt_params(M, O_mask, n_iter=200, lr=0.1, lam=0.01):
    """Fit 1PL IRT on O_mask=1 entries. Returns (theta, beta) vectors."""
    n_models, n_items = M.shape
    obs = O_mask == 1

    M_safe = np.where(obs, np.nan_to_num(M, nan=0.5), np.nan)
    model_means = np.nanmean(M_safe, axis=1)
    model_means = np.clip(np.where(np.isnan(model_means), 0.5, model_means), 0.01, 0.99)
    theta = logit(model_means)

    item_means = np.nanmean(M_safe, axis=0)
    item_means = np.clip(np.where(np.isnan(item_means), 0.5, item_means), 0.01, 0.99)
    beta = -logit(item_means)

    obs_i, obs_j = np.where(obs)
    y_obs = np.clip(np.nan_to_num(M[obs_i, obs_j], nan=0.5), 0.001, 0.999)
    eff_lr = lr * min(1.0, n_models / 30.0)

    for _ in range(n_iter):
        eta = np.clip(theta[obs_i] - beta[obs_j], -10, 10)
        p = sigmoid(eta)
        r = y_obs - p

        g_theta = np.zeros(n_models)
        np.add.at(g_theta, obs_i, r)
        g_theta -= lam * theta

        g_beta = np.zeros(n_items)
        np.add.at(g_beta, obs_j, -r)
        g_beta -= lam * beta

        theta += eff_lr * np.clip(g_theta, -5, 5)
        beta += eff_lr * np.clip(g_beta, -5, 5)
        theta = np.clip(theta, -10, 10)
        beta = np.clip(beta, -10, 10)

    return theta, beta


def estimate_theta_single(beta, y_obs, obs_items, n_iter=50, lam=0.01):
    """Newton-Raphson for a single model's theta given fixed beta."""
    if len(obs_items) == 0:
        return 0.0
    mean_y = np.clip(np.mean(y_obs), 0.01, 0.99)
    theta = logit(mean_y)
    b = beta[obs_items]

    for _ in range(n_iter):
        eta = np.clip(theta - b, -10, 10)
        p = sigmoid(eta)
        grad = (y_obs - p).sum() - lam * theta
        hess = -(p * (1 - p)).sum() - lam
        if abs(hess) < 1e-12:
            break
        step = grad / hess          # hess < 0, so this goes in the right direction
        theta -= step
        theta = np.clip(theta, -10, 10)
        if abs(step) < 1e-8:
            break
    return theta


# ── Masking model ────────────────────────────────────────────────────────

def _build_train_pairs(X_model_train, X_item, flat_indices, n_items, extra_flat=None):
    mi = flat_indices // n_items
    ii = flat_indices % n_items
    X = np.hstack([X_model_train[mi], X_item[ii]])
    if extra_flat is not None:
        X = np.hstack([X, extra_flat[flat_indices, None]])
    return X


def learn_masking_on_train(X_model_train, X_item, O_train, M_train, n_items,
                           include_score=False, M_irt_train=None):
    """Learn P(O=1|features[+score]) from training models only.
    Returns (clf, scaler, z_train) where z_train are logit-scores for O_train=1 entries."""
    n_train = X_model_train.shape[0]
    n_total = n_train * n_items
    y = O_train.ravel().astype(int)

    extra_flat = None
    if include_score:
        M_filled = np.where(O_train == 1,
                            np.nan_to_num(M_train, nan=0.5),
                            M_irt_train).ravel().astype(np.float32)
        extra_flat = M_filled

    # Fit scaler on sample
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, min(300_000, n_total), replace=False)
    scaler = StandardScaler()
    scaler.fit(_build_train_pairs(X_model_train, X_item, sample_idx, n_items, extra_flat))

    # Train in chunks
    X_chunks, y_chunks = [], []
    for start in range(0, n_total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_total)
        idx = np.arange(start, end)
        Xc = scaler.transform(
            _build_train_pairs(X_model_train, X_item, idx, n_items, extra_flat))
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        X_chunks.append(Xc)
        y_chunks.append(y[start:end])
    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_all, y_all)

    probs = clf.predict_proba(X_all)[:, 1]
    z_train = logit(np.clip(probs[O_train.ravel() == 1], 1e-6, 1 - 1e-6))

    del X_all, X_chunks, y_chunks
    return clf, scaler, z_train


def compute_z_held(clf, scaler, X_model_m, X_item, O_orig_m, M_m,
                   include_score=False):
    """Apply trained masking model to held-out model's O_orig=1 items."""
    obs_j = np.where(O_orig_m == 1)[0]
    if len(obs_j) == 0:
        return np.array([])
    X = np.hstack([np.tile(X_model_m, (len(obs_j), 1)), X_item[obs_j]])
    if include_score:
        X = np.hstack([X, np.nan_to_num(M_m[obs_j], nan=0.5).reshape(-1, 1)])
    X = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    probs = clf.predict_proba(X)[:, 1]
    return logit(np.clip(probs, 1e-6, 1 - 1e-6))


# ── Masking generation ───────────────────────────────────────────────────

def _calibrate_alpha(z_centered, keep_rate):
    """Binary-search for alpha so that mean(sigmoid(alpha + z_centered)) ≈ keep_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if sigmoid(mid + z_centered).mean() < keep_rate:
            lo = mid
        else:
            hi = mid
        if abs(sigmoid(mid + z_centered).mean() - keep_rate) < 1e-4:
            break
    return (lo + hi) / 2


def generate_masking_joint(z_train, z_held, O_train, O_orig_m,
                           keep_rate, rng_train, rng_held):
    """Generate O_new for training models + keep-mask for held-out model.
    Alpha is calibrated on training z; same mechanism applied to held-out."""
    z_mean = z_train.mean()
    z_train_c = z_train - z_mean
    z_held_c = z_held - z_mean

    alpha = _calibrate_alpha(z_train_c, keep_rate)

    # Training
    keep_train = rng_train.random(len(z_train)) < sigmoid(alpha + z_train_c)
    O_new_train = O_train.copy().ravel()
    orig_idx = np.where(O_train.ravel() == 1)[0]
    O_new_train[orig_idx[~keep_train]] = 0
    O_new_train = O_new_train.reshape(O_train.shape)

    # Held-out
    keep_held = rng_held.random(len(z_held)) < sigmoid(alpha + z_held_c)
    held_orig_j = np.where(O_orig_m == 1)[0]
    O_new_m = np.zeros_like(O_orig_m)
    O_new_m[held_orig_j[keep_held]] = 1

    return O_new_train, O_new_m, keep_train.mean()


# ── Propensity ───────────────────────────────────────────────────────────

def fit_propensity_on_train(X_model_train, X_item, O_train, O_new_train, n_items):
    """Fit propensity P(O_new=1 | O_orig=1, features) on training models.
    Returns (clf, scaler)."""
    orig_idx = np.where(O_train.ravel() == 1)[0]
    y_prop = O_new_train.ravel()[orig_idx]

    X_chunks = []
    for start in range(0, len(orig_idx), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(orig_idx))
        idx = orig_idx[start:end]
        mi = idx // n_items
        ii = idx % n_items
        X_chunks.append(np.hstack([X_model_train[mi], X_item[ii]]))
    X_tr = np.vstack(X_chunks)

    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_tr, y_prop)
    return clf, scaler


def apply_propensity_to_held(clf, scaler, X_model_m, X_item, O_orig_m):
    """Apply propensity model to held-out model's O_orig=1 items.
    Returns pi array aligned with O_orig=1 indices."""
    obs_j = np.where(O_orig_m == 1)[0]
    if len(obs_j) == 0:
        return np.array([])
    X = np.hstack([np.tile(X_model_m, (len(obs_j), 1)), X_item[obs_j]])
    X = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    pi = np.clip(clf.predict_proba(X)[:, 1], 0.05, 0.95)
    return pi


# ── Per-model estimators ────────────────────────────────────────────────

def held_out_estimators(M_m, O_orig_m, O_new_m, M_irt_m, pi_orig):
    """Compute 4 estimators for a single held-out model.

    Args:
        M_m:       raw scores (n_items,)
        O_orig_m:  original observation mask (n_items,)
        O_new_m:   additional-masked observation mask (n_items,)
        M_irt_m:   IRT predictions sigmoid(theta_m - beta) (n_items,)
        pi_orig:   propensity for O_orig=1 items, aligned with orig_j

    Returns dict of {estimator_name: mu_hat}.
    """
    orig_j = np.where(O_orig_m == 1)[0]
    n_orig = len(orig_j)
    if n_orig == 0:
        return {k: np.nan for k in ['naive', 'irt', 'ipw', 'dr']}

    M_orig = np.nan_to_num(M_m[orig_j], nan=0.0)
    O_new_orig = O_new_m[orig_j].astype(float)      # 1/0 aligned with orig_j
    irt_orig = M_irt_m[orig_j]

    # Naive: mean of surviving observations
    obs_new = np.where(O_new_m == 1)[0]
    naive = np.nanmean(M_m[obs_new]) if len(obs_new) > 0 else np.nan

    # IRT: mean IRT prediction over O_orig items
    irt = irt_orig.mean()

    # IPW: reweighted mean
    ipw = (O_new_orig * M_orig / pi_orig).sum() / n_orig

    # DR: IRT + propensity correction
    dr = (irt_orig + O_new_orig / pi_orig * (M_orig - irt_orig)).sum() / n_orig

    return {'naive': naive, 'irt': irt, 'ipw': ipw, 'dr': dr}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading global data...")
    mf, emb, emb_idx = load_global()
    print(f"  Done ({time.time()-t0:.1f}s)\n")

    all_results = []

    for bid in BENCHMARKS:
        tb = time.time()
        O_orig, M, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O_orig.shape
        print(f"{'='*60}")
        print(f"{bid}: {n_models} models × {n_items} items, density={O_orig.mean():.3f}")
        print(f"{'='*60}")

        X_model, _ = build_model_features(mf, model_ids)
        X_item = build_item_features(item_stats, item_ids, emb, emb_idx)

        # Ground truth per model
        mu_true = np.array([np.nanmean(M[i, O_orig[i] == 1])
                            for i in range(n_models)])

        # Sample held-out models
        rng_sample = np.random.RandomState(42)
        n_held = max(HOLD_OUT_MIN, int(np.ceil(n_models * HOLD_OUT_FRAC)))
        held_out_ids = rng_sample.choice(n_models, size=n_held, replace=False)
        held_out_ids.sort()
        print(f"  Holding out {n_held}/{n_models} models: {held_out_ids.tolist()}")

        # Collect per-model predictions
        preds = {(mode, kr): {e: [] for e in ['naive', 'irt', 'ipw', 'dr', 'true']}
                 for mode in MODES for kr in KEEP_RATES}

        for mi, m in enumerate(held_out_ids):
            tm = time.time()
            train_mask = np.ones(n_models, dtype=bool)
            train_mask[m] = False
            O_train = O_orig[train_mask]
            M_train = M[train_mask]
            X_model_train = X_model[train_mask]

            # 1) IRT on training O_orig → beta_k (item params)
            theta_train, beta_k = fit_irt_params(M_train, O_train)
            M_irt_train = sigmoid(theta_train[:, None] - beta_k[None, :])

            for mode in MODES:
                include_score = (mode == 'features_and_score')

                # 2) Learn masking from training models
                mask_clf, mask_scaler, z_train = learn_masking_on_train(
                    X_model_train, X_item, O_train, M_train, n_items,
                    include_score=include_score, M_irt_train=M_irt_train)

                # 3) Compute z for held-out model's O_orig=1 items
                z_held = compute_z_held(
                    mask_clf, mask_scaler, X_model[m], X_item,
                    O_orig[m], M[m], include_score=include_score)

                if len(z_held) == 0:
                    continue

                for keep_rate in KEEP_RATES:
                    rng_tr = np.random.RandomState(42 + int(keep_rate * 100))
                    rng_hd = np.random.RandomState(1000 + m * 10 + int(keep_rate * 100))

                    # 4) Generate O_new for training + held-out (shared alpha)
                    O_new_train, O_new_m, actual_rate = generate_masking_joint(
                        z_train, z_held, O_train, O_orig[m],
                        keep_rate, rng_tr, rng_hd)

                    n_obs_new = O_new_m.sum()
                    if n_obs_new < 3:
                        continue

                    # 5) Propensity: train on training models, apply to held-out
                    prop_clf, prop_scaler = fit_propensity_on_train(
                        X_model_train, X_item, O_train, O_new_train, n_items)
                    pi_orig = apply_propensity_to_held(
                        prop_clf, prop_scaler, X_model[m], X_item, O_orig[m])

                    # 6) Estimate theta_m from held-out O_new with fixed beta_k
                    obs_new_j = np.where(O_new_m == 1)[0]
                    y_obs = np.clip(np.nan_to_num(M[m, obs_new_j], nan=0.5),
                                    0.001, 0.999)
                    theta_m = estimate_theta_single(beta_k, y_obs, obs_new_j)
                    M_irt_m = sigmoid(theta_m - beta_k)

                    # 7) Compute estimators
                    ests = held_out_estimators(
                        M[m], O_orig[m], O_new_m, M_irt_m, pi_orig)

                    key = (mode, keep_rate)
                    for e in ['naive', 'irt', 'ipw', 'dr']:
                        preds[key][e].append(ests[e])
                    preds[key]['true'].append(mu_true[m])

            elapsed = time.time() - tm
            print(f"  [{mi+1}/{n_held}] model {m} ({elapsed:.1f}s)")

        # ── Aggregate ────────────────────────────────────────────────────
        for (mode, keep_rate), p in preds.items():
            mt = np.array(p['true'])
            if len(mt) < 2:
                continue
            row = {'benchmark_id': bid, 'mode': mode, 'keep_rate': keep_rate}
            for e in ['naive', 'irt', 'ipw', 'dr']:
                mh = np.array(p[e])
                valid = ~(np.isnan(mh) | np.isnan(mt))
                if valid.sum() < 2:
                    row[f'{e}_bias'] = np.nan
                    row[f'{e}_rmse'] = np.nan
                    row[f'{e}_rank_corr'] = np.nan
                else:
                    diff = mh[valid] - mt[valid]
                    row[f'{e}_bias'] = diff.mean()
                    row[f'{e}_rmse'] = np.sqrt((diff ** 2).mean())
                    row[f'{e}_rank_corr'] = spearmanr(mh[valid], mt[valid]).correlation
            all_results.append(row)

        print(f"  {bid} done in {time.time()-tb:.0f}s\n")

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT / 'held_out_results.csv', index=False)
    print(f"Saved held_out_results.csv ({len(df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────
    estimator_names = ['naive', 'irt', 'ipw', 'dr']
    colors = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
    labels_map = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
    markers = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}
    mode_titles = {'features_only': 'Masking: features only',
                   'features_and_score': 'Masking: features + score'}

    for metric, ylabel, zero_line in [
        ('bias', 'Bias (estimated − true)', True),
        ('rmse', 'RMSE', False),
        ('rank_corr', 'Spearman ρ', False),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey='row')
        for row, bid in enumerate(BENCHMARKS):
            bdf = df[df['benchmark_id'] == bid]
            for col, mode in enumerate(MODES):
                ax = axes[row, col]
                mdf = bdf[bdf['mode'] == mode]
                for est in estimator_names:
                    ax.plot(mdf['keep_rate'], mdf[f'{est}_{metric}'],
                            marker=markers[est], ms=6, color=colors[est],
                            label=labels_map[est], linewidth=1.8)
                if zero_line:
                    ax.axhline(0, color='gray', ls='--', lw=0.8)
                ax.set_xlabel('Keep rate', fontsize=10)
                if row == 0:
                    ax.set_title(mode_titles[mode], fontsize=11)
                if col == 0:
                    ax.set_ylabel(f'{bid}\n{ylabel}', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(KEEP_RATES)

        handles, lbls = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='lower center', ncol=4,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fname = f'fig_held_out_{metric}.png'
        fig.savefig(FIGURES / fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figures/{fname}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  HELD-OUT MODEL RESULTS at keep_rate=0.7")
    print(f"{'='*80}")
    for mode in MODES:
        print(f"\n  --- {mode} ---")
        s = df[(df['keep_rate'] == 0.7) & (df['mode'] == mode)]
        cols = ['benchmark_id'] + [f'{e}_{m}' for e in estimator_names
                                   for m in ['bias', 'rmse', 'rank_corr']]
        cols = [c for c in cols if c in s.columns]
        print(s[cols].to_string(index=False))

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
