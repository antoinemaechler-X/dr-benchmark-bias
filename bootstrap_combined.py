"""
Combine the original (mmlupro, matharena) and v2 (livecodebench, ultrafeedback)
bootstraps into one cross-benchmark view.

Headline finding: DR removes ~half the MNAR selection bias on benchmarks with
real selection (MathArena, LiveCodeBench) and does no harm where there is little
selection to correct (MMLU-Pro, UltraFeedback).

Writes: figures/fig_bootstrap_crossbench.png, bootstrap_crossbench_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'

N_MODELS = {'ai2d_test': 254, 'mmbench_v11': 251, 'bfcl': 93, 'matharena': 88,
            'livecodebench': 70, 'mmlupro': 48, 'mtbench': 34, 'agentdojo': 28,
            'ultrafeedback': 17}
TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena',
          'livecodebench': 'LiveCodeBench', 'ultrafeedback': 'UltraFeedback',
          'ai2d_test': 'AI2D', 'mmbench_v11': 'MMBench',
          'agentdojo': 'AgentDojo', 'bfcl': 'BFCL', 'mtbench': 'MT-Bench'}
KEEP = 0.7


def load_all():
    d1 = pd.read_csv(PROJECT / 'bootstrap_results.csv')
    d2 = pd.read_csv(PROJECT / 'bootstrap_results_v2.csv')
    d3 = pd.read_csv(PROJECT / 'bootstrap_results_v3.csv')
    return pd.concat([d1, d2, d3], ignore_index=True)


def cell(df, bid, mode, est):
    s = df[(df.bid == bid) & (df['mode'] == mode) & (df.keep_rate == KEEP)]
    v = s[f'{est}_bias'].values
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    df = load_all()
    order = sorted(N_MODELS, key=lambda b: -N_MODELS[b])

    rows = []
    for bid in order:
        for mode in ['features_only', 'features_and_score']:
            for est in ['naive', 'irt', 'ipw', 'dr']:
                m, lo, hi = cell(df, bid, mode, est)
                rows.append({'benchmark': bid, 'n_models': N_MODELS[bid],
                             'mode': mode, 'estimator': est,
                             'mean_bias': m, 'lo95': lo, 'hi95': hi})
    summ = pd.DataFrame(rows)
    summ.to_csv(PROJECT / 'bootstrap_crossbench_summary.csv', index=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 5.5))

    # Panel A: naive vs DR per benchmark, MNAR, keep=0.7
    x = np.arange(len(order))
    for est, color, off, lab in [('naive', '#d62728', -0.13, 'Naive'),
                                 ('dr', '#1f77b4', 0.13, 'DR+IRT')]:
        means, los, his = [], [], []
        for bid in order:
            m, lo, hi = cell(df, bid, 'features_and_score', est)
            means.append(m); los.append(m - lo); his.append(hi - m)
        axA.errorbar(x + off, means, yerr=[los, his], fmt='o', ms=7, capsize=4,
                     color=color, label=lab)
    axA.axhline(0, color='gray', ls='--', lw=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels([f'{TITLES[b]}\n(n={N_MODELS[b]})' for b in order],
                        fontsize=9)
    axA.set_ylabel('Bias (estimated - true), MNAR, keep=0.7')
    axA.set_title('Per-benchmark bias: naive vs DR (95% CI)', fontsize=11)
    axA.legend(); axA.grid(True, alpha=0.3, axis='y')

    # Panel B: |DR bias| vs |naive bias| -> below the diagonal = DR improves
    naive_ab = np.array([abs(cell(df, b, 'features_and_score', 'naive')[0]) for b in order])
    dr_ab = np.array([abs(cell(df, b, 'features_and_score', 'dr')[0]) for b in order])
    lim = max(naive_ab.max(), dr_ab.max()) * 1.15
    axB.plot([0, lim], [0, lim], color='gray', ls='--', lw=0.9, label='no change')
    axB.fill_between([0, lim], [0, 0], [0, lim], color='green', alpha=0.05)
    axB.scatter(naive_ab, dr_ab, s=80, color='#1f77b4', zorder=3)
    for b, x0, y0 in zip(order, naive_ab, dr_ab):
        axB.annotate(TITLES[b], (x0, y0), fontsize=9, xytext=(6, 4),
                     textcoords='offset points')
    axB.set_xlim(0, lim); axB.set_ylim(0, lim)
    axB.set_xlabel('|naive bias|  (selection strength)')
    axB.set_ylabel('|DR bias|  (residual after correction)')
    axB.set_title('DR reduces bias most where selection is strongest\n'
                  '(points below the line = improvement)', fontsize=10)
    axB.legend(loc='upper left'); axB.grid(True, alpha=0.3)

    fig.suptitle('DR across 9 benchmarks: halves MNAR selection bias where it '
                 'exists, harmless where it does not', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / 'fig_bootstrap_crossbench.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

    # console: full estimator table at MNAR keep=0.7
    print(f"\n{'='*86}\n  MNAR, keep=0.7  (mean bias [95% CI])\n{'='*86}")
    print(f"  {'benchmark':14s}{'n':>5s}  {'naive':>18s}{'IRT':>18s}{'IPW':>18s}{'DR+IRT':>18s}")
    for bid in order:
        def fmt(est):
            m, lo, hi = cell(df, bid, 'features_and_score', est)
            return f'{m:+.3f}[{lo:+.3f},{hi:+.3f}]'
        print(f"  {TITLES[bid]:14s}{N_MODELS[bid]:5d}  "
              f"{fmt('naive'):>18s}{fmt('irt'):>18s}{fmt('ipw'):>18s}{fmt('dr'):>18s}")


if __name__ == '__main__':
    main()
