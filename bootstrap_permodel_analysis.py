"""
Model-by-model breakdown of the bootstrap, to answer:
  - Does the small GRAND-MEAN bias hide larger per-model effects that cancel?
  - Is a ~0.02 correction "interesting"? -> look at the per-model distribution.

For each (benchmark, mode, keep_rate) we have bootstrap estimates of shape
(n_draws, n_models). Per model we average the bias (estimate - mu_true) over
draws, giving a stable per-model bias for each estimator.

Reads:  bootstrap_permodel.npz, bootstrap_bundles/{bid}.npz (for model names)
Writes: figures/fig_permodel_bias.png
        permodel_bias.csv
        permodel_top_offenders.csv
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'
BUNDLES = PROJECT / 'bootstrap_bundles'

BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}
COLORS = {'naive': '#d62728', 'dr': '#1f77b4'}
# Headline scenario: real selection bias (MNAR), moderate masking.
MODE = 'features_and_score'
KEEP = 0.7


def load_permodel():
    return np.load(PROJECT / 'bootstrap_permodel.npz', allow_pickle=True)


def model_names(bid):
    b = np.load(BUNDLES / f'{bid}.npz', allow_pickle=True)
    return [str(m) for m in b['model_ids']]


def per_model_table(d, bid):
    """Return DataFrame: one row per model with mean per-model bias (over draws)
    and its sd, for naive and DR."""
    pre = f'{bid}__{MODE}__{KEEP}'
    mu_true = d[f'{pre}__mu_true']           # (n_models,)
    names = model_names(bid)
    rows = []
    for est in ['naive', 'dr']:
        est_mat = d[f'{pre}__{est}']         # (n_draws, n_models)
        bias = est_mat - mu_true[None, :]    # per draw, per model
        rows.append((est, bias.mean(axis=0), bias.std(axis=0)))
    out = pd.DataFrame({'benchmark': bid, 'model': names, 'true': mu_true})
    for est, mean_b, sd_b in rows:
        out[f'{est}_bias'] = mean_b
        out[f'{est}_bias_sd'] = sd_b
    out['abs_naive'] = out.naive_bias.abs()
    out['abs_dr'] = out.dr_bias.abs()
    out['abs_reduction'] = out.abs_naive - out.abs_dr
    return out


def plot(tables):
    fig, axes = plt.subplots(len(tables), 2, figsize=(13, 5.2 * len(tables)))
    if len(tables) == 1:
        axes = axes[None, :]
    for r, (bid, t) in enumerate(tables.items()):
        # --- Left: distribution of per-model bias, naive vs DR ---
        ax = axes[r, 0]
        bins = np.linspace(min(t.naive_bias.min(), t.dr_bias.min()),
                           max(t.naive_bias.max(), t.dr_bias.max()), 25)
        ax.hist(t.naive_bias, bins=bins, alpha=0.55, color=COLORS['naive'],
                label='Naive')
        ax.hist(t.dr_bias, bins=bins, alpha=0.55, color=COLORS['dr'],
                label='DR+IRT')
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.axvline(t.naive_bias.mean(), color=COLORS['naive'], ls=':', lw=1.5)
        ax.axvline(t.dr_bias.mean(), color=COLORS['dr'], ls=':', lw=1.5)
        ax.set_xlabel('Per-model bias (estimated - true)')
        ax.set_ylabel('# models')
        ax.set_title(f'{BENCH_TITLES[bid]}: distribution of per-model bias\n'
                     f'(dotted = grand mean)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Right: naive vs DR per model, sorted by naive bias ---
        ax = axes[r, 1]
        ts = t.sort_values('naive_bias').reset_index(drop=True)
        x = np.arange(len(ts))
        ax.vlines(x, ts.naive_bias, ts.dr_bias, color='gray', alpha=0.4, lw=0.8)
        ax.scatter(x, ts.naive_bias, s=18, color=COLORS['naive'], label='Naive')
        ax.scatter(x, ts.dr_bias, s=18, color=COLORS['dr'], label='DR+IRT')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Model (sorted by naive bias)')
        ax.set_ylabel('Per-model bias')
        ax.set_title(f'{BENCH_TITLES[bid]}: per-model correction (each line = 1 model)',
                     fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Model-by-model bias  ({MODE}, keep={KEEP}, mean over 1000 draws)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES / 'fig_permodel_bias.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    d = load_permodel()
    tables = {bid: per_model_table(d, bid) for bid in ['mmlupro', 'matharena']}

    allt = pd.concat(tables.values(), ignore_index=True)
    allt.to_csv(PROJECT / 'permodel_bias.csv', index=False)
    print(f"Saved permodel_bias.csv ({len(allt)} models)")

    plot(tables)

    # Summary stats + worst offenders
    offenders = []
    print(f"\n{'='*82}\n  Per-model bias summary  ({MODE}, keep={KEEP})\n{'='*82}")
    for bid, t in tables.items():
        gm_n, gm_d = t.naive_bias.mean(), t.dr_bias.mean()
        print(f"\n  {BENCH_TITLES[bid]}  (n={len(t)} models)")
        print(f"    GRAND-MEAN bias:        naive={gm_n:+.4f}   dr={gm_d:+.4f}")
        print(f"    MEAN |per-model| bias:  naive={t.abs_naive.mean():.4f}   "
              f"dr={t.abs_dr.mean():.4f}")
        print(f"    MAX  |per-model| bias:  naive={t.abs_naive.max():.4f}   "
              f"dr={t.abs_dr.max():.4f}")
        frac = (t.abs_reduction > 0).mean()
        print(f"    DR reduces |bias| for:  {frac*100:.0f}% of models")
        print(f"    spread (sd) of per-model naive bias: {t.naive_bias.std():.4f} "
              f"(vs grand mean {gm_n:+.4f})")
        top = t.reindex(t.abs_naive.sort_values(ascending=False).index).head(5)
        print("    Top-5 most naive-biased models:")
        for _, r in top.iterrows():
            print(f"      {r.model[:42]:42s} naive={r.naive_bias:+.4f} -> "
                  f"dr={r.dr_bias:+.4f}")
            offenders.append({'benchmark': bid, 'model': r.model,
                              'naive_bias': r.naive_bias, 'dr_bias': r.dr_bias,
                              'true': r.true})
    pd.DataFrame(offenders).to_csv(PROJECT / 'permodel_top_offenders.csv', index=False)
    print("\nSaved permodel_top_offenders.csv")


if __name__ == '__main__':
    main()
