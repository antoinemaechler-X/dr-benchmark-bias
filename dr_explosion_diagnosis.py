"""
Diagnostic for (D): the DR explosions are driven by tiny-sample models, not by
low propensity. Plot |per-model DR bias| against the model's observation count;
the catastrophic cases sit at very low n_obs, where a sample-size guard (fall
back to IRT-only) removes them.

Reads:  dr_stabilize_permodel.csv, bootstrap_bundles/{bid}.npz
Writes: figures/fig_dr_explosion_diagnosis.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'
BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}
GUARD = 10  # min observations to trust the DR correction

d_npz = np.load(PROJECT / 'bootstrap_permodel.npz', allow_pickle=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, bid in zip(axes, ['mmlupro', 'matharena']):
    b = np.load(PROJECT / 'bootstrap_bundles' / f'{bid}.npz', allow_pickle=True)
    n_obs = (b['O_orig'] == 1).sum(axis=1).astype(float)
    pre = f'{bid}__features_and_score__0.7'
    mu_true = d_npz[f'{pre}__mu_true']
    dr_abs = np.abs(np.nanmean(d_npz[f'{pre}__dr'] - mu_true[None, :], axis=0))
    irt_abs = np.abs(np.nanmean(d_npz[f'{pre}__irt'] - mu_true[None, :], axis=0))
    ax.scatter(n_obs, dr_abs, s=45, color='#1f77b4', alpha=0.75, label='DR+IRT')
    ax.scatter(n_obs, irt_abs, s=45, color='#2ca02c', marker='^', alpha=0.75,
               label='IRT-only')
    ax.axvline(GUARD, color='red', ls=':', lw=1.5,
               label=f'overlap threshold (n_obs={GUARD})')
    ax.axhline(0.05, color='gray', ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.02)
    ax.set_xlabel('# items the model was evaluated on (log scale)')
    ax.set_ylabel('|per-model bias|  (mean over 1000 seeds)')
    ax.set_title(f'{BENCH_TITLES[bid]}: error vs sample size', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle('Both DR and IRT explode on the same tiny-sample models '
             '-> a coverage (positivity) problem, not a DR pathology', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = FIGURES / 'fig_dr_explosion_diagnosis.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved {out}")
