"""
Apply the simple absolute coverage filter (n_obs >= MIN_OBS) and regenerate the
main bootstrap figures. This is NOT the relative %-of-items threshold (rejected);
it only removes models with too few observations to estimate a mean. Only
MathArena is affected (3 models with <10 obs); the other benchmarks are unchanged.

Per-model bootstrap arrays come from:
  bootstrap_permodel.npz      -> mmlupro, matharena
  bootstrap_permodel_v2.npz   -> livecodebench, ultrafeedback
  bootstrap_permodel_v3.npz   -> ai2d_test, mmbench_v11, agentdojo, bfcl, mtbench

Regenerates (originals first backed up to *_allmodels.png):
  fig_bootstrap_bias.png, fig_bootstrap_bias_meanci.png   (all 9 benchmarks)
  fig_permodel_bias.png                                   (all 9 benchmarks)
  fig_bootstrap_crossbench.png                            (all 9 benchmarks)
and the summary CSVs.
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIG = PROJECT / 'figures'
MIN_OBS = 10

D1 = np.load(PROJECT / 'bootstrap_permodel.npz', allow_pickle=True)
D2 = np.load(PROJECT / 'bootstrap_permodel_v2.npz', allow_pickle=True)
D3 = np.load(PROJECT / 'bootstrap_permodel_v3.npz', allow_pickle=True)
SRC = {'mmlupro': D1, 'matharena': D1, 'livecodebench': D2, 'ultrafeedback': D2,
       'ai2d_test': D3, 'mmbench_v11': D3, 'agentdojo': D3, 'bfcl': D3, 'mtbench': D3}

N_MODELS = {'ai2d_test': 254, 'mmbench_v11': 251, 'bfcl': 93, 'matharena': 88,
            'livecodebench': 70, 'mmlupro': 48, 'mtbench': 34, 'agentdojo': 28,
            'ultrafeedback': 17}
TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena',
          'livecodebench': 'LiveCodeBench', 'ultrafeedback': 'UltraFeedback',
          'ai2d_test': 'AI2D', 'mmbench_v11': 'MMBench',
          'agentdojo': 'AgentDojo', 'bfcl': 'BFCL', 'mtbench': 'MT-Bench'}
MODES = ['features_only', 'features_and_score']
KEEPS = [0.5, 0.6, 0.7, 0.8, 0.9]
ESTS = ['naive', 'irt', 'ipw', 'dr']
COLORS = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
LABELS = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
MARKERS = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}
MODE_T = {'features_only': 'MAR (features only)', 'features_and_score': 'MNAR (features + score)'}


def cov_mask(bid):
    b = np.load(PROJECT / 'bootstrap_bundles' / f'{bid}.npz', allow_pickle=True)
    return (b['O_orig'] == 1).sum(axis=1) >= MIN_OBS


MASK = {b: cov_mask(b) for b in N_MODELS}


def arr(bid, mode, kr, est):
    return SRC[bid][f'{bid}__{mode}__{kr}__{est}'][:, MASK[bid]]


def true(bid, mode, kr):
    return SRC[bid][f'{bid}__{mode}__{kr}__mu_true'][MASK[bid]]


def bias_seed(bid, mode, kr, est):
    return np.nanmean(arr(bid, mode, kr, est) - true(bid, mode, kr)[None, :], axis=1)


def summarize(benches):
    rows = []
    for bid in benches:
        ndrop = int((~MASK[bid]).sum())
        for mode in MODES:
            for kr in KEEPS:
                for est in ESTS:
                    v = bias_seed(bid, mode, kr, est)
                    se = v.std(ddof=1) / np.sqrt(len(v))
                    rows.append(dict(bid=bid, mode=mode, keep_rate=kr, estimator=est,
                        n_models=int(MASK[bid].sum()), n_dropped=ndrop,
                        mean_bias=v.mean(), lo95=np.percentile(v, 2.5),
                        hi95=np.percentile(v, 97.5),
                        ci_lo=v.mean()-1.96*se, ci_hi=v.mean()+1.96*se))
    return pd.DataFrame(rows)


def backup(name):
    p = FIG / name
    if p.exists():
        shutil.copy(p, FIG / name.replace('.png', '_allmodels.png'))


def plot_ci(summ, band, benches):
    lo, hi = ('lo95', 'hi95') if band == 'percentile' else ('ci_lo', 'ci_hi')
    fig, axes = plt.subplots(2, len(benches), figsize=(4*len(benches), 8), sharex=True)
    for r, mode in enumerate(MODES):
        for c, bid in enumerate(benches):
            ax = axes[r, c]
            s = summ[(summ.bid == bid) & (summ['mode'] == mode)]
            for est in ESTS:
                e = s[s.estimator == est].sort_values('keep_rate')
                ax.fill_between(e.keep_rate, e[lo], e[hi], color=COLORS[est], alpha=0.2, lw=0)
                ax.plot(e.keep_rate, e.mean_bias, marker=MARKERS[est], ms=5,
                        color=COLORS[est], label=LABELS[est], lw=1.8)
            ax.axhline(0, color='gray', ls='--', lw=0.8); ax.grid(True, alpha=0.3)
            if r == 0:
                nd = s.n_dropped.iloc[0]
                ax.set_title(f'{TITLES[bid]} ({s.n_models.iloc[0]} models'
                             + (f', {nd} dropped)' if nd else ')'), fontsize=11)
            if r == 1: ax.set_xlabel('Keep rate')
            if c == 0: ax.set_ylabel(f'{MODE_T[mode]}\nBias (est - true)', fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    tag = '2.5-97.5 percentile bands' if band == 'percentile' else 'mean +/- 1.96 SE'
    fig.suptitle(f'Bias with {tag}  (n_obs >= {MIN_OBS}, 1000 draws)', fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    name = 'fig_bootstrap_bias.png' if band == 'percentile' else 'fig_bootstrap_bias_meanci.png'
    backup(name); fig.savefig(FIG / name, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"Saved {name}")


def plot_permodel(benches):
    mode, kr = 'features_and_score', 0.7
    fig, axes = plt.subplots(len(benches), 2, figsize=(13, 5 * len(benches)))
    for r, bid in enumerate(benches):
        nb = np.nanmean(arr(bid, mode, kr, 'naive') - true(bid, mode, kr)[None, :], axis=0)
        db = np.nanmean(arr(bid, mode, kr, 'dr') - true(bid, mode, kr)[None, :], axis=0)
        ax = axes[r, 0]
        bins = np.linspace(min(nb.min(), db.min()), max(nb.max(), db.max()), 25)
        ax.hist(nb, bins=bins, alpha=0.55, color=COLORS['naive'], label='Naive')
        ax.hist(db, bins=bins, alpha=0.55, color=COLORS['dr'], label='DR+IRT')
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.axvline(nb.mean(), color=COLORS['naive'], ls=':', lw=1.5)
        ax.axvline(db.mean(), color=COLORS['dr'], ls=':', lw=1.5)
        ax.set_xlabel('Per-model bias (est - true)'); ax.set_ylabel('# models')
        ax.set_title(f'{TITLES[bid]}: per-model bias (n_obs >= {MIN_OBS})', fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[r, 1]
        o = np.argsort(nb); x = np.arange(len(o))
        ax.vlines(x, nb[o], db[o], color='gray', alpha=0.4, lw=0.8)
        ax.scatter(x, nb[o], s=18, color=COLORS['naive'], label='Naive')
        ax.scatter(x, db[o], s=18, color=COLORS['dr'], label='DR+IRT')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Model (sorted by naive bias)'); ax.set_ylabel('Per-model bias')
        ax.set_title(f'{TITLES[bid]}: per-model correction', fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Model-by-model bias (MNAR, keep=0.7, n_obs >= {MIN_OBS}, 1000 draws)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    backup('fig_permodel_bias.png')
    fig.savefig(FIG / 'fig_permodel_bias.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    print("Saved fig_permodel_bias.png")


def plot_crossbench(summ):
    order = sorted(N_MODELS, key=lambda b: -N_MODELS[b])
    def cell(bid, est):
        s = summ[(summ.bid == bid) & (summ['mode'] == 'features_and_score') & (summ.keep_rate == 0.7)]
        r = s[s.estimator == est].iloc[0]; return r.mean_bias, r.lo95, r.hi95
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 5.5))
    x = np.arange(len(order))
    for est, color, off, lab in [('naive', '#d62728', -0.13, 'Naive'), ('dr', '#1f77b4', 0.13, 'DR+IRT')]:
        m = [cell(b, est)[0] for b in order]
        lo = [cell(b, est)[0]-cell(b, est)[1] for b in order]
        hi = [cell(b, est)[2]-cell(b, est)[0] for b in order]
        axA.errorbar(x+off, m, yerr=[lo, hi], fmt='o', ms=7, capsize=4, color=color, label=lab)
    axA.axhline(0, color='gray', ls='--', lw=0.8); axA.set_xticks(x)
    axA.set_xticklabels([f'{TITLES[b]}\n(n={N_MODELS[b]})' for b in order], fontsize=9)
    axA.set_ylabel('Bias (est - true), MNAR, keep=0.7')
    axA.set_title('Per-benchmark bias: naive vs DR (95% CI)', fontsize=11)
    axA.legend(); axA.grid(True, alpha=0.3, axis='y')
    na = np.array([abs(cell(b, 'naive')[0]) for b in order])
    da = np.array([abs(cell(b, 'dr')[0]) for b in order])
    lim = max(na.max(), da.max())*1.15
    axB.plot([0, lim], [0, lim], color='gray', ls='--', lw=0.9, label='no change')
    axB.fill_between([0, lim], [0, 0], [0, lim], color='green', alpha=0.05)
    axB.scatter(na, da, s=80, color='#1f77b4', zorder=3)
    for b, x0, y0 in zip(order, na, da):
        axB.annotate(TITLES[b], (x0, y0), fontsize=9, xytext=(6, 4), textcoords='offset points')
    axB.set_xlim(0, lim); axB.set_ylim(0, lim)
    axB.set_xlabel('|naive bias| (selection strength)')
    axB.set_ylabel('|DR bias| (residual after correction)')
    axB.set_title('DR reduces bias most where selection is strongest\n(below the line = improvement)', fontsize=10)
    axB.legend(loc='upper left'); axB.grid(True, alpha=0.3)
    fig.suptitle('DR across 9 benchmarks: halves MNAR selection bias where it exists '
                 f'(n_obs >= {MIN_OBS})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    backup('fig_bootstrap_crossbench.png')
    fig.savefig(FIG / 'fig_bootstrap_crossbench.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    print("Saved fig_bootstrap_crossbench.png")


def main():
    nine = sorted(N_MODELS, key=lambda b: -N_MODELS[b])
    summ = summarize(nine)
    summ.to_csv(PROJECT / 'bootstrap_summary.csv', index=False)
    plot_ci(summ, 'percentile', nine)
    plot_ci(summ, 'meanci', nine)
    plot_permodel(nine)
    plot_crossbench(summ)
    summ[summ.estimator.isin(ESTS)].to_csv(PROJECT / 'bootstrap_crossbench_summary.csv', index=False)

    print(f"\nAll benchmarks MNAR keep=0.7 (n_obs>={MIN_OBS}):")
    for bid in nine:
        dropped = int((~MASK[bid]).sum())
        print(f"\n  {TITLES[bid]} (dropped {dropped}):")
        for est in ESTS:
            v = bias_seed(bid, 'features_and_score', 0.7, est)
            print(f"    {LABELS[est]:8s} {v.mean():+.4f} [{np.percentile(v,2.5):+.4f}, {np.percentile(v,97.5):+.4f}]")


if __name__ == '__main__':
    main()
