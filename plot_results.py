"""Plot semi-synthetic DR results for mmlupro and matharena only.

Reads semi_synthetic_results.csv and produces 2x2 grids (benchmark × masking mode)
for each metric (bias, RMSE, rank correlation).

Outputs:
    figures/fig_dr_bias_focused.png
    figures/fig_dr_rmse_focused.png
    figures/fig_dr_rank_corr_focused.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'
df = pd.read_csv(PROJECT / 'semi_synthetic_results.csv')

BENCHMARKS = ['mmlupro', 'matharena']
MODES = ['features_only', 'features_and_score']
mode_titles = {'features_only': 'Masking from features only',
               'features_and_score': 'Masking from features + score'}

estimator_names = ['naive', 'irt', 'ipw', 'dr']
colors = {'naive': '#d62728', 'irt': '#2ca02c', 'ipw': '#ff7f0e', 'dr': '#1f77b4'}
labels_map = {'naive': 'Naive', 'irt': 'IRT', 'ipw': 'IPW', 'dr': 'DR+IRT'}
markers = {'naive': 's', 'irt': '^', 'ipw': 'D', 'dr': 'o'}

for metric, ylabel, zero_line in [
    ('bias', 'Bias (estimated − true)', True),
    ('rmse', 'RMSE', False),
    ('rank_corr', 'Spearman ρ', False),
]:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey='row')
    for row, bid in enumerate(BENCHMARKS):
        bdf_all = df[df['benchmark_id'] == bid]
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            bdf = bdf_all[bdf_all['mode'] == mode]
            for est in estimator_names:
                col_name = f'{est}_{metric}'
                ax.plot(bdf['keep_rate'], bdf[col_name],
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
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f'fig_dr_{metric}_focused.png'
    fig.savefig(FIGURES / fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figures/{fname}")
