"""
(C) Is item missingness related to item difficulty?
i.e. are HARD items disproportionately unobserved (selection on difficulty)?

item_stats.item_difficulty = mean accuracy over the models that attempted the
item (NaN-skipped), so HIGHER = EASIER. We define hardness = 1 - accuracy and
observation rate = n_models_observed / n_models, then look at their relationship.

Caveats (printed): accuracy is measured only on observed entries, so for items
attempted by few models it is noisy and itself selection-affected.

Reads:  processed/{bid}/item_stats.csv, model_ids.json
Writes: figures/fig_missingness_vs_difficulty.png
        missingness_vs_difficulty.csv
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

PROJECT = Path(__file__).resolve().parent
PROCESSED = PROJECT / 'processed'
FIGURES = PROJECT / 'figures'

BENCHMARKS = ['mmlupro', 'matharena']
BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}


def load(bid):
    ist = pd.read_csv(PROCESSED / bid / 'item_stats.csv')
    with open(PROCESSED / bid / 'model_ids.json') as f:
        n_models = len(json.load(f))
    ist = ist.dropna(subset=['item_difficulty', 'n_models_observed']).copy()
    ist['accuracy'] = ist['item_difficulty']          # higher = easier
    ist['hardness'] = 1.0 - ist['accuracy']           # higher = harder
    ist['obs_rate'] = ist['n_models_observed'] / n_models
    ist['benchmark'] = bid
    return ist, n_models


def binned(x, y, n_bins=10):
    """Mean y within deciles of x (for a trend overlay)."""
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    centers, means, ses = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 3:
            centers.append(x[m].mean())
            means.append(y[m].mean())
            ses.append(y[m].std() / np.sqrt(m.sum()))
    return np.array(centers), np.array(means), np.array(ses)


def main():
    tables = {}
    for bid in BENCHMARKS:
        ist, n_models = load(bid)
        tables[bid] = (ist, n_models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    summary = []
    for ax, bid in zip(axes, BENCHMARKS):
        ist, n_models = tables[bid]
        x = ist['hardness'].values
        y = ist['obs_rate'].values
        rho, p = spearmanr(x, y)
        r, _ = pearsonr(x, y)

        ax.scatter(x, y, s=8, alpha=0.25, color='#1f77b4', edgecolors='none')
        cx, cy, cse = binned(x, y)
        ax.errorbar(cx, cy, yerr=cse, color='#d62728', lw=2, marker='o', ms=5,
                    capsize=3, label='decile mean')
        ax.set_xlabel('Item hardness  (1 - accuracy,  higher = harder)')
        ax.set_ylabel('Observation rate  (frac. of models that attempted item)')
        ax.set_title(f'{BENCH_TITLES[bid]}  (n={len(ist)} items, {n_models} models)\n'
                     f'Spearman rho = {rho:+.3f}  (p={p:.1e})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        summary.append({'benchmark': bid, 'n_items': len(ist), 'n_models': n_models,
                        'spearman_hardness_obs': rho, 'spearman_p': p,
                        'pearson_hardness_obs': r,
                        'mean_obs_rate': y.mean(),
                        'obs_rate_easyhalf': y[x < np.median(x)].mean(),
                        'obs_rate_hardhalf': y[x >= np.median(x)].mean()})

    fig.suptitle('Item missingness vs. difficulty: are hard items observed less?',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES / 'fig_missingness_vs_difficulty.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    sdf = pd.DataFrame(summary)
    sdf.to_csv(PROJECT / 'missingness_vs_difficulty.csv', index=False)
    print(f"Saved missingness_vs_difficulty.csv\n")
    print(f"{'='*78}\n  Missingness vs difficulty\n{'='*78}")
    for s in summary:
        direction = ("HARD items observed LESS" if s['spearman_hardness_obs'] < 0
                     else "HARD items observed MORE")
        print(f"\n  {BENCH_TITLES[s['benchmark']]}: Spearman(hardness, obs_rate) "
              f"= {s['spearman_hardness_obs']:+.3f} (p={s['spearman_p']:.1e})  -> {direction}")
        print(f"    obs rate: easy half = {s['obs_rate_easyhalf']:.3f}, "
              f"hard half = {s['obs_rate_hardhalf']:.3f}")


if __name__ == '__main__':
    main()
