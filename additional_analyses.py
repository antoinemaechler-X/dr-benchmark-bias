"""
Additional Analyses (all fast):
  1. Real DR correction on actual O_orig — corrected rankings
  2. Bootstrap CIs for Step 2 bias plots
  3. Held-out scatter plots (predicted vs true)
"""

import time, warnings
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
    load_global, load_benchmark, build_model_features, build_item_features,
    build_pair_features, fit_irt, learn_masking_model, generate_masking,
    fit_propensity, compute_ground_truth,
    estimator_naive, estimator_irt, estimator_ipw, estimator_dr,
    PROJECT, FIGURES, CHUNK_SIZE,
)
from held_out_model import (
    fit_irt_params, estimate_theta_single,
    learn_masking_on_train, compute_z_held,
    generate_masking_joint, fit_propensity_on_train,
    apply_propensity_to_held, held_out_estimators,
    HOLD_OUT_FRAC, HOLD_OUT_MIN,
)

warnings.filterwarnings('ignore')

BENCHMARKS = ['mmlupro', 'matharena']
KEEP_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ['features_only', 'features_and_score']
N_BOOT = 500


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    rng = np.random.RandomState(42)
    n = len(values)
    means = [values[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)]
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


# ── Part 1: Real DR on actual O_orig ─────────────────────────────────────

def real_dr_correction(mf, emb, emb_idx):
    print("\n" + "=" * 60)
    print("PART 1: Real DR Correction on Actual Data")
    print("=" * 60)

    all_rows = []

    for bid in BENCHMARKS:
        t = time.time()
        O_orig, M, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O_orig.shape
        X_model, _ = build_model_features(mf, model_ids)
        X_item = build_item_features(item_stats, item_ids, emb, emb_idx)

        # IRT on full O_orig
        M_irt = fit_irt(M, O_orig)

        # Propensity: P(O_orig=1 | features) over ALL pairs
        n_total = n_models * n_items
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_total, min(300_000, n_total), replace=False)
        scaler = StandardScaler()
        scaler.fit(build_pair_features(X_model, X_item, sample_idx, n_items))

        X_chunks, y_chunks = [], []
        for start in range(0, n_total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_total)
            idx = np.arange(start, end)
            Xc = scaler.transform(build_pair_features(X_model, X_item, idx, n_items))
            Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
            X_chunks.append(Xc)
            y_chunks.append(O_orig.ravel()[start:end])
        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks)

        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        clf.fit(X_all, y_all)
        pi_flat = np.clip(clf.predict_proba(X_all)[:, 1], 0.05, 0.95)
        pi_hat = pi_flat.reshape(n_models, n_items)
        del X_all, X_chunks

        M_safe = np.nan_to_num(M, nan=0.0)

        for i in range(n_models):
            orig = O_orig[i] == 1
            n_orig = orig.sum()
            if n_orig == 0:
                continue

            mu_naive = np.nanmean(M[i, orig])
            mu_irt = M_irt[i].mean()                             # over ALL items
            mu_dr = (M_irt[i] + O_orig[i] / pi_hat[i] * (M_safe[i] - M_irt[i])).mean()

            all_rows.append({
                'benchmark_id': bid, 'model_id': model_ids[i],
                'n_observed': int(n_orig), 'obs_rate': n_orig / n_items,
                'naive': mu_naive, 'irt': mu_irt, 'dr': mu_dr,
            })

        print(f"  {bid} ({time.time()-t:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(PROJECT / 'real_dr_results.csv', index=False)

    # Print summary + ranking changes
    for bid in BENCHMARKS:
        bdf = df[df['benchmark_id'] == bid].copy()
        bdf['naive_rank'] = bdf['naive'].rank(ascending=False)
        bdf['dr_rank'] = bdf['dr'].rank(ascending=False)
        bdf['rank_change'] = bdf['naive_rank'] - bdf['dr_rank']
        rho = spearmanr(bdf['naive'], bdf['dr']).correlation
        shift = (bdf['dr'] - bdf['naive']).mean()

        print(f"\n  {bid}: naive→DR mean shift = {shift:+.4f}, rank ρ = {rho:.3f}")
        movers = bdf.reindex(bdf['rank_change'].abs().sort_values(ascending=False).index)
        print(f"    Top rank movers:")
        for _, r in movers.head(5).iterrows():
            print(f"      {r['model_id'][:25]:25s}  obs={r['obs_rate']:.0%}  "
                  f"naive={r['naive']:.3f}  dr={r['dr']:.3f}  "
                  f"rank {int(r['naive_rank'])}→{int(r['dr_rank'])}")

    # Plot: observation rate vs DR correction magnitude
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for col, bid in enumerate(BENCHMARKS):
        ax = axes[col]
        bdf = df[df['benchmark_id'] == bid]
        correction = bdf['dr'] - bdf['naive']
        ax.scatter(bdf['obs_rate'], correction, s=40, alpha=0.7,
                   color='#1f77b4', edgecolors='white', linewidth=0.5)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Observation rate', fontsize=10)
        ax.set_ylabel('DR correction (DR − Naive)', fontsize=10)
        ax.set_title(bid, fontsize=11)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_real_dr_correction.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved figures/fig_real_dr_correction.png")

    return df


# ── Part 2: Bootstrap CIs for Step 2 ────────────────────────────────────

def step2_with_bootstrap(mf, emb, emb_idx):
    print("\n" + "=" * 60)
    print("PART 2: Step 2 Bias Plot with Bootstrap CIs")
    print("=" * 60)

    all_results = []

    for bid in BENCHMARKS:
        tb = time.time()
        O_orig, M, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O_orig.shape
        X_model, _ = build_model_features(mf, model_ids)
        X_item = build_item_features(item_stats, item_ids, emb, emb_idx)
        mu_true = compute_ground_truth(M, O_orig)
        M_irt_full = fit_irt(M, O_orig)

        for mode in MODES:
            include_score = (mode == 'features_and_score')
            z = learn_masking_model(X_model, X_item, O_orig, M, n_models, n_items,
                                    include_score=include_score, M_irt=M_irt_full)
            for keep_rate in KEEP_RATES:
                rng = np.random.RandomState(42 + int(keep_rate * 100))
                O_new, _ = generate_masking(z, O_orig, keep_rate, rng)
                M_irt = fit_irt(M, O_new)
                pi_hat = fit_propensity(X_model, X_item, O_orig, O_new, n_models, n_items)

                mus = {
                    'naive': estimator_naive(M, O_new, O_orig),
                    'irt':   estimator_irt(M_irt, O_orig),
                    'ipw':   estimator_ipw(M, O_new, O_orig, pi_hat),
                    'dr':    estimator_dr(M, M_irt, O_new, O_orig, pi_hat),
                }

                row = {'benchmark_id': bid, 'mode': mode, 'keep_rate': keep_rate}
                for name, mu_hat in mus.items():
                    valid = ~(np.isnan(mu_hat) | np.isnan(mu_true))
                    diff = mu_hat[valid] - mu_true[valid]
                    row[f'{name}_bias'] = diff.mean()
                    lo, hi = bootstrap_ci(diff)
                    row[f'{name}_bias_lo'] = lo
                    row[f'{name}_bias_hi'] = hi
                    row[f'{name}_rmse'] = np.sqrt((diff ** 2).mean())
                    row[f'{name}_rank_corr'] = spearmanr(mu_hat[valid], mu_true[valid]).correlation
                all_results.append(row)

        print(f"  {bid} ({time.time()-tb:.0f}s)")

    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT / 'step2_with_ci.csv', index=False)

    # Plot bias with CI bands
    colors = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
    labels_map = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
    markers = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}
    mode_titles = {'features_only': 'Masking: features only',
                   'features_and_score': 'Masking: features + score'}

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey='row')
    for row_i, bid in enumerate(BENCHMARKS):
        bdf = df[df['benchmark_id'] == bid]
        for col_i, mode in enumerate(MODES):
            ax = axes[row_i, col_i]
            mdf = bdf[bdf['mode'] == mode].sort_values('keep_rate')
            for est in ['naive', 'irt', 'ipw', 'dr']:
                kr = mdf['keep_rate'].values
                bias = mdf[f'{est}_bias'].values
                ax.plot(kr, bias, marker=markers[est], ms=6, color=colors[est],
                        label=labels_map[est], linewidth=1.8)
            ax.axhline(0, color='gray', ls='--', lw=0.8)
            ax.set_xlabel('Keep rate', fontsize=10)
            if row_i == 0:
                ax.set_title(mode_titles[mode], fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f'{bid}\nBias (estimated − true)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(KEEP_RATES)

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES / 'fig_dr_bias_ci.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figures/fig_dr_bias_ci.png")

    return df


# ── Part 3: Held-out scatter plots ──────────────────────────────────────

def held_out_scatter(mf, emb, emb_idx):
    print("\n" + "=" * 60)
    print("PART 3: Held-Out Scatter Plots")
    print("=" * 60)

    scatter_rows = []

    for bid in BENCHMARKS:
        tb = time.time()
        O_orig, M, model_ids, item_ids, item_stats = load_benchmark(bid)
        n_models, n_items = O_orig.shape
        X_model, _ = build_model_features(mf, model_ids)
        X_item = build_item_features(item_stats, item_ids, emb, emb_idx)
        mu_true = np.array([np.nanmean(M[i, O_orig[i] == 1]) for i in range(n_models)])

        rng_sample = np.random.RandomState(42)
        n_held = max(HOLD_OUT_MIN, int(np.ceil(n_models * HOLD_OUT_FRAC)))
        held_ids = rng_sample.choice(n_models, size=n_held, replace=False)
        held_ids.sort()

        for m in held_ids:
            train_mask = np.ones(n_models, dtype=bool)
            train_mask[m] = False
            O_train, M_train = O_orig[train_mask], M[train_mask]
            X_model_train = X_model[train_mask]

            theta_train, beta_k = fit_irt_params(M_train, O_train)
            M_irt_train = sigmoid(theta_train[:, None] - beta_k[None, :])

            # Only features_only, keep=0.7 for scatter
            mask_clf, mask_scaler, z_train = learn_masking_on_train(
                X_model_train, X_item, O_train, M_train, n_items,
                include_score=False, M_irt_train=M_irt_train)
            z_held = compute_z_held(mask_clf, mask_scaler, X_model[m], X_item,
                                    O_orig[m], M[m], include_score=False)
            if len(z_held) == 0:
                continue

            rng_tr = np.random.RandomState(42 + 70)
            rng_hd = np.random.RandomState(1000 + m * 10 + 70)
            O_new_train, O_new_m, _ = generate_masking_joint(
                z_train, z_held, O_train, O_orig[m], 0.7, rng_tr, rng_hd)
            if O_new_m.sum() < 3:
                continue

            prop_clf, prop_scaler = fit_propensity_on_train(
                X_model_train, X_item, O_train, O_new_train, n_items)
            pi_orig = apply_propensity_to_held(
                prop_clf, prop_scaler, X_model[m], X_item, O_orig[m])

            obs_j = np.where(O_new_m == 1)[0]
            y_obs = np.clip(np.nan_to_num(M[m, obs_j], nan=0.5), 0.001, 0.999)
            theta_m = estimate_theta_single(beta_k, y_obs, obs_j)
            M_irt_m = sigmoid(theta_m - beta_k)

            ests = held_out_estimators(M[m], O_orig[m], O_new_m, M_irt_m, pi_orig)
            scatter_rows.append({
                'benchmark_id': bid, 'model_idx': m, 'mu_true': mu_true[m],
                **ests,
            })

        print(f"  {bid}: {n_held} models ({time.time()-tb:.0f}s)")

    sdf = pd.DataFrame(scatter_rows)
    sdf.to_csv(PROJECT / 'held_out_scatter.csv', index=False)

    # Plot: absolute error comparison (naive error vs DR error per model)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for col, bid in enumerate(BENCHMARKS):
        ax = axes[col]
        bdf = sdf[sdf['benchmark_id'] == bid]
        naive_err = (bdf['naive'] - bdf['mu_true']).abs().values
        dr_err = (bdf['dr'] - bdf['mu_true']).abs().values

        mx = max(naive_err.max(), dr_err.max()) * 1.1
        ax.plot([0, mx], [0, mx], 'k--', lw=0.8, alpha=0.5)
        ax.scatter(naive_err, dr_err, s=60, alpha=0.7, color='#1f77b4',
                   edgecolors='white', linewidth=0.5)
        ax.fill_between([0, mx], [0, 0], [0, mx], color='#2ca02c', alpha=0.07)
        ax.set_xlabel('Naive absolute error', fontsize=10)
        ax.set_ylabel('DR+IRT absolute error', fontsize=10)
        ax.set_title(f'{bid} (features_only, keep=0.7)', fontsize=11)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        n_improved = (dr_err < naive_err).sum()
        ax.text(0.95, 0.05, f'DR better: {n_improved}/{len(bdf)}',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_held_out_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figures/fig_held_out_scatter.png")

    return sdf


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    mf, emb, emb_idx = load_global()

    real_dr_correction(mf, emb, emb_idx)
    step2_with_bootstrap(mf, emb, emb_idx)
    held_out_scatter(mf, emb, emb_idx)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
