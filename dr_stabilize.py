"""
(D) Tame the DR explosions on low-propensity models, WITHOUT touching any
existing results. Outputs are written under new dr_stabilize_* names.

The explosion comes from the DR correction term  O/pi * (score - m_irt):
when pi_hat is tiny the inverse weight is huge. We compare:
  clip05   baseline DR, propensity clipped to [0.05, 0.95]   (current method)
  clip10   DR, propensity clipped to [0.10, 0.90]
  clip20   DR, propensity clipped to [0.20, 0.80]
  snorm05  self-normalized (Hajek) DR, clip [0.05, 0.95]
  snorm10  self-normalized (Hajek) DR, clip [0.10, 0.90]

Self-normalized DR divides the correction by the sum of weights instead of n,
which bounds the influence of any single huge weight.

Runs locally on the precomputed bundles. Headline scenario: MNAR, multiple
keep-rates, N seeds. Reuses semi_synthetic_dr for masking / IRT.

Writes: dr_stabilize_results.csv     (per benchmark x strategy x keep: tail stats)
        dr_stabilize_permodel.csv     (per benchmark x strategy x model, keep=0.7)
        figures/fig_dr_stabilize.png
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
import semi_synthetic_dr as S

BUNDLES = PROJECT / 'bootstrap_bundles'
FIGURES = PROJECT / 'figures'

BENCHMARKS = ['mmlupro', 'matharena']
BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}
MODE = 'features_and_score'          # MNAR: where selection bias is real
KEEP_RATES = [0.5, 0.7, 0.9]
N_SEEDS = 50

STRATEGIES = {
    'clip05':  dict(clip=(0.05, 0.95), snorm=False),
    'clip10':  dict(clip=(0.10, 0.90), snorm=False),
    'clip20':  dict(clip=(0.20, 0.80), snorm=False),
    'snorm05': dict(clip=(0.05, 0.95), snorm=True),
    'snorm10': dict(clip=(0.10, 0.90), snorm=True),
}


def fit_propensity_raw(X_model, X_item, O_orig, O_new, n_models, n_items):
    """Same as S.fit_propensity but returns UNCLIPPED pi (so we can clip many
    ways downstream)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    orig_mask = O_orig.ravel() == 1
    orig_indices = np.where(orig_mask)[0]
    y_prop = O_new.ravel()[orig_indices]
    n_orig = len(orig_indices)

    X_chunks = []
    for start in range(0, n_orig, S.CHUNK_SIZE):
        end = min(start + S.CHUNK_SIZE, n_orig)
        idx = orig_indices[start:end]
        X_chunks.append(S.build_pair_features(X_model, X_item, idx, n_items))
    X_train = np.vstack(X_chunks)
    scaler = StandardScaler()
    X_train = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_prop)
    pi_flat = clf.predict_proba(X_train)[:, 1]

    pi_raw = np.full((n_models, n_items), 0.5, dtype=np.float32)
    pi_raw.ravel()[orig_indices] = pi_flat
    return pi_raw


def dr_estimate(M, M_irt, O_new, O_orig, pi_raw, clip, snorm):
    """Per-model DR under a given clip and (optional) self-normalization."""
    lo, hi = clip
    pi = np.clip(pi_raw, lo, hi)
    n_models = M.shape[0]
    mu = np.full(n_models, np.nan)
    M_safe = np.nan_to_num(M, nan=0.0)
    for i in range(n_models):
        orig = O_orig[i] == 1
        n_orig = orig.sum()
        if n_orig == 0:
            continue
        new_obs = (O_new[i, orig] == 1).astype(float)
        scores = M_safe[i, orig]
        m_irt = M_irt[i, orig]
        w = new_obs / pi[i, orig]
        corr = w * (scores - m_irt)
        if snorm:
            denom = w.sum()
            mu[i] = m_irt.mean() + (corr.sum() / denom if denom > 0 else 0.0)
        else:
            mu[i] = (m_irt + corr).sum() / n_orig
    return mu


def tail_stats(bias):
    b = bias[~np.isnan(bias)]
    ab = np.abs(b)
    return dict(grand_mean=b.mean(), mean_abs=ab.mean(), median_abs=np.median(ab),
                p95_abs=np.percentile(ab, 95), max_abs=ab.max(),
                n_explode=int((ab > 0.05).sum()), rmse=np.sqrt((b ** 2).mean()))


def main():
    t0 = time.time()
    results = []
    permodel_rows = []

    for bid in BENCHMARKS:
        tb = time.time()
        print(f"\n=== {bid} ===")
        b = np.load(BUNDLES / f'{bid}.npz', allow_pickle=True)
        O_orig = b['O_orig'].astype(np.int8)
        M = b['M'].astype(np.float32)
        X_model, X_item = b['X_model'], b['X_item']
        mu_true = b['mu_true']
        model_ids = [str(m) for m in b['model_ids']]
        z = b['z_fs']
        n_models, n_items = O_orig.shape

        # accumulate per-model bias across seeds: strat -> keep -> (sum, count)
        acc = {s: {kr: np.zeros(n_models) for kr in KEEP_RATES} for s in STRATEGIES}
        cnt = {s: {kr: np.zeros(n_models) for kr in KEEP_RATES} for s in STRATEGIES}
        # also baseline naive for reference
        naive_acc = {kr: np.zeros(n_models) for kr in KEEP_RATES}
        naive_cnt = {kr: np.zeros(n_models) for kr in KEEP_RATES}

        for seed in range(N_SEEDS):
            for kr in KEEP_RATES:
                rng = np.random.RandomState(seed * 1000 + int(round(kr * 100)))
                O_new, _ = S.generate_masking(z, O_orig, kr, rng)
                M_irt = S.fit_irt(M, O_new)
                pi_raw = fit_propensity_raw(X_model, X_item, O_orig, O_new,
                                            n_models, n_items)
                mu_naive = S.estimator_naive(M, O_new, O_orig)
                v = ~np.isnan(mu_naive)
                naive_acc[kr][v] += (mu_naive - mu_true)[v]
                naive_cnt[kr][v] += 1
                for s, cfg in STRATEGIES.items():
                    mu = dr_estimate(M, M_irt, O_new, O_orig, pi_raw,
                                     cfg['clip'], cfg['snorm'])
                    v = ~np.isnan(mu)
                    acc[s][kr][v] += (mu - mu_true)[v]
                    cnt[s][kr][v] += 1
            if (seed + 1) % 25 == 0:
                print(f"  {bid}: {seed+1}/{N_SEEDS} seeds ({time.time()-tb:.0f}s)")

        # per-model mean bias, then tail stats
        for kr in KEEP_RATES:
            nb = naive_acc[kr] / np.maximum(naive_cnt[kr], 1)
            nb[naive_cnt[kr] == 0] = np.nan
            st = tail_stats(nb)
            results.append({'benchmark': bid, 'strategy': 'naive', 'keep_rate': kr, **st})
            for s in STRATEGIES:
                pmb = acc[s][kr] / np.maximum(cnt[s][kr], 1)
                pmb[cnt[s][kr] == 0] = np.nan
                results.append({'benchmark': bid, 'strategy': s, 'keep_rate': kr,
                                **tail_stats(pmb)})
                if kr == 0.7:
                    for mid, val in zip(model_ids, pmb):
                        permodel_rows.append({'benchmark': bid, 'strategy': s,
                                              'model': mid, 'bias': val})
        print(f"  done in {time.time()-tb:.0f}s")

    res = pd.DataFrame(results)
    res.to_csv(PROJECT / 'dr_stabilize_results.csv', index=False)
    pd.DataFrame(permodel_rows).to_csv(PROJECT / 'dr_stabilize_permodel.csv', index=False)
    print(f"\nSaved dr_stabilize_results.csv and dr_stabilize_permodel.csv")

    # ── Figure: tail behaviour at keep=0.7 ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ['naive', 'clip05', 'clip10', 'clip20', 'snorm05', 'snorm10']
    for ax, bid in zip(axes, BENCHMARKS):
        s7 = res[(res.benchmark == bid) & (res.keep_rate == 0.7)].set_index('strategy')
        x = np.arange(len(order))
        ax.bar(x - 0.2, [s7.loc[o, 'median_abs'] for o in order], width=0.4,
               label='median |bias|', color='#1f77b4')
        ax.bar(x + 0.2, [s7.loc[o, 'max_abs'] for o in order], width=0.4,
               label='max |bias| (worst model)', color='#d62728', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('|per-model bias|')
        ax.set_title(f'{BENCH_TITLES[bid]} (MNAR, keep=0.7)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
    fig.suptitle('DR stabilization: median vs worst-case per-model |bias| '
                 f'({N_SEEDS} seeds)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES / 'fig_dr_stabilize.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # ── Console summary at keep=0.7 ──
    print(f"\n{'='*92}\n  DR stabilization @ keep=0.7  (MNAR, {N_SEEDS} seeds)\n{'='*92}")
    for bid in BENCHMARKS:
        print(f"\n  {BENCH_TITLES[bid]}")
        s7 = res[(res.benchmark == bid) & (res.keep_rate == 0.7)]
        print(f"    {'strategy':9s} {'grand':>8s} {'med|b|':>8s} {'mean|b|':>8s} "
              f"{'p95|b|':>8s} {'max|b|':>8s} {'#expl':>6s}")
        for o in order:
            r = s7[s7.strategy == o].iloc[0]
            print(f"    {o:9s} {r.grand_mean:+8.4f} {r.median_abs:8.4f} "
                  f"{r.mean_abs:8.4f} {r.p95_abs:8.4f} {r.max_abs:8.4f} {int(r.n_explode):6d}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
