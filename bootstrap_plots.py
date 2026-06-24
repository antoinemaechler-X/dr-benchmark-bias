"""
Turn the bootstrap output into the Figure-1 upgrade: mean bias with a 95%
confidence band (2.5-97.5 percentile over masking draws) for each estimator,
plus a summary table at keep_rate=0.7.

Reads:  bootstrap_results.csv
Writes: figures/fig_bootstrap_bias.png
        bootstrap_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'
FIGURES.mkdir(exist_ok=True)

BENCHMARKS = ['mmlupro', 'matharena']
MODES = ['features_only', 'features_and_score']
ESTIMATORS = ['naive', 'irt', 'ipw', 'dr']

COLORS = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
LABELS = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
MARKERS = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}
MODE_TITLES = {'features_only': 'MAR (masking from features only)',
               'features_and_score': 'MNAR (masking from features + score)'}
BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}


def summarize(df):
    """Per (bid, mode, keep_rate, estimator): mean, lo, hi over seeds."""
    rows = []
    for bid in BENCHMARKS:
        for mode in MODES:
            sub = df[(df.bid == bid) & (df['mode'] == mode)]
            for kr, g in sub.groupby('keep_rate'):
                for est in ESTIMATORS:
                    v = g[f'{est}_bias'].values
                    se = v.std(ddof=1) / np.sqrt(len(v))
                    rows.append({
                        'bid': bid, 'mode': mode, 'keep_rate': kr,
                        'estimator': est, 'n_draws': len(v),
                        'mean_bias': v.mean(), 'std_bias': v.std(), 'se_bias': se,
                        # percentile band over draws (spread of a single experiment)
                        'lo95': np.percentile(v, 2.5),
                        'hi95': np.percentile(v, 97.5),
                        # CI on the MEAN bias (precision of the estimate, shrinks ~1/sqrt(n))
                        'ci_lo': v.mean() - 1.96 * se,
                        'ci_hi': v.mean() + 1.96 * se,
                    })
    return pd.DataFrame(rows)


def plot_bias_ci(summ, band='percentile'):
    """band='percentile' -> 2.5/97.5 spread over draws;
       band='meanci'     -> 95% CI on the mean bias (mean +- 1.96 SE)."""
    lo_col, hi_col = ('lo95', 'hi95') if band == 'percentile' else ('ci_lo', 'ci_hi')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, mode in enumerate(MODES):
        for col, bid in enumerate(BENCHMARKS):
            ax = axes[row, col]
            s = summ[(summ.bid == bid) & (summ['mode'] == mode)]
            for est in ESTIMATORS:
                e = s[s.estimator == est].sort_values('keep_rate')
                ax.fill_between(e.keep_rate, e[lo_col], e[hi_col],
                                color=COLORS[est], alpha=0.20, linewidth=0)
                ax.plot(e.keep_rate, e.mean_bias, marker=MARKERS[est], ms=5,
                        color=COLORS[est], label=LABELS[est], linewidth=1.8)
            ax.axhline(0, color='gray', ls='--', lw=0.8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(BENCH_TITLES[bid], fontsize=12)
            if row == 1:
                ax.set_xlabel('Keep rate')
            if col == 0:
                ax.set_ylabel(f'{MODE_TITLES[mode]}\nBias (estimated - true)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    if band == 'percentile':
        title = 'Bias with 2.5-97.5 percentile bands over 1000 masking draws'
        out = FIGURES / 'fig_bootstrap_bias.png'
    else:
        title = 'Mean bias with 95% CI (mean +/- 1.96 SE, 1000 masking draws)'
        out = FIGURES / 'fig_bootstrap_bias_meanci.png'
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def main():
    df = pd.read_csv(PROJECT / 'bootstrap_results.csv')
    print(f"Loaded {len(df)} rows, {df.seed.nunique()} unique seeds")
    summ = summarize(df)
    summ.to_csv(PROJECT / 'bootstrap_summary.csv', index=False)
    print(f"Saved bootstrap_summary.csv ({len(summ)} rows)")
    plot_bias_ci(summ, band='percentile')
    plot_bias_ci(summ, band='meanci')

    # Headline table at keep_rate=0.7
    print(f"\n{'='*78}\n  Bias (mean [95% CI]) at keep_rate=0.7\n{'='*78}")
    for bid in BENCHMARKS:
        for mode in MODES:
            print(f"\n  {BENCH_TITLES[bid]} - {MODE_TITLES[mode]}")
            s = summ[(summ.bid == bid) & (summ['mode'] == mode) &
                     (summ.keep_rate == 0.7)]
            for est in ESTIMATORS:
                r = s[s.estimator == est].iloc[0]
                print(f"    {LABELS[est]:8s}  {r.mean_bias:+.4f}  "
                      f"[{r.lo95:+.4f}, {r.hi95:+.4f}]  (sd={r.std_bias:.4f})")


if __name__ == '__main__':
    main()
