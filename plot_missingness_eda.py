"""
Missingness EDA Visualizations for measurement-db.
Produces 6 publication-quality figures for the paper's EDA subsection.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10})

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, 'processed')
FIGURES = os.path.join(BASE, 'figures')
os.makedirs(FIGURES, exist_ok=True)

plt.style.use('default')

# ── Load observation summary ──
obs_summary = pd.read_csv(os.path.join(PROCESSED, 'observation_summary.csv'))

def load_benchmark(bid):
    """Load observation matrix, item_stats, model_stats for a benchmark."""
    bdir = os.path.join(PROCESSED, bid)
    O = np.load(os.path.join(bdir, 'observation_matrix.npz'))['data']
    item_stats = pd.read_csv(os.path.join(bdir, 'item_stats.csv'))
    model_stats = pd.read_csv(os.path.join(bdir, 'model_stats.csv'))
    with open(os.path.join(bdir, 'item_ids.json')) as f:
        item_ids = json.load(f)
    with open(os.path.join(bdir, 'model_ids.json')) as f:
        model_ids = json.load(f)
    return O, item_stats, model_stats, item_ids, model_ids


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Density overview bar chart
# ═══════════════════════════════════════════════════════════════════
def fig1_density_overview():
    df = obs_summary.sort_values('density', ascending=True).reset_index(drop=True)

    colors = []
    for d in df['density']:
        if d >= 1.0:
            colors.append('#2ca02c')  # green
        elif d >= 0.85:
            colors.append('#1f77b4')  # blue
        elif d >= 0.30:
            colors.append('#ff7f0e')  # orange
        else:
            colors.append('#d62728')  # red

    fig, ax = plt.subplots(figsize=(6.75, 4.5))
    y = np.arange(len(df))
    bars = ax.barh(y, df['density'] * 100, color=colors, edgecolor='none', height=0.7)

    for i, row in df.iterrows():
        pct = f"{row['density']*100:.1f}%"
        dims = f"{int(row['n_models'])}×{int(row['n_items'])}"
        label = f" {pct}  ({dims})"
        ax.text(row['density'] * 100 + 0.5, i, label, va='center', fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(df['benchmark_id'], fontsize=8)
    ax.set_xlabel('Observation Density (%)', fontsize=10)
    ax.set_title('Observation Density Across Benchmarks', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(os.path.join(FIGURES, 'fig_density_overview.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_density_overview.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Observation matrix heatmaps (all 16 benchmarks)
# ═══════════════════════════════════════════════════════════════════
def fig2_heatmaps():
    # Sort benchmarks by density ascending (most missing first)
    sorted_df = obs_summary.sort_values('density', ascending=True)
    benchmarks = sorted_df['benchmark_id'].tolist()

    fig, axes = plt.subplots(4, 4, figsize=(6.75, 7.5))

    for ax, bid in zip(axes.flat, benchmarks):
        O, item_stats, model_stats, _, _ = load_benchmark(bid)
        density = obs_summary.loc[obs_summary['benchmark_id'] == bid, 'density'].values[0]

        # Sort rows by number of items observed (descending)
        row_obs = O.sum(axis=1)
        row_order = np.argsort(-row_obs)
        # Sort columns by number of models observed (descending)
        col_obs = O.sum(axis=0)
        col_order = np.argsort(-col_obs)

        O_sorted = O[np.ix_(row_order, col_order)]

        ax.imshow(O_sorted, cmap='Greys', aspect='auto', interpolation='none')
        ax.set_title(f'{bid} ({density:.0%})', fontsize=6.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(h_pad=0.8, w_pad=0.4)
    plt.savefig(os.path.join(FIGURES, 'fig_heatmaps.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: MNAR evidence — item difficulty vs observation rate
# ═══════════════════════════════════════════════════════════════════
def fig3_mnar_evidence():
    benchmarks = ['ai2d_test', 'livecodebench', 'mmlupro', 'matharena']
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.5))

    for ax, bid in zip(axes.flat, benchmarks):
        _, item_stats, model_stats, _, _ = load_benchmark(bid)
        n_models = obs_summary.loc[obs_summary['benchmark_id'] == bid, 'n_models'].values[0]

        x = item_stats['item_difficulty'].values
        y = item_stats['n_models_observed'].values / n_models  # observation rate

        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]

        ax.scatter(x_clean, y_clean, s=4, alpha=0.4, color='#1f77b4', edgecolors='none')

        # Spearman correlation
        rho, pval = stats.spearmanr(x_clean, y_clean)
        pstr = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
        ax.text(0.03, 0.03, f"ρ={rho:.3f}, {pstr}", transform=ax.transAxes,
                fontsize=7, va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.8, edgecolor='gray'))

        # Trend line
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1, alpha=0.7)

        ax.set_title(bid, fontsize=9)
        ax.set_xlabel('Item Difficulty', fontsize=8)
        ax.set_ylabel('Item Obs. Rate', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig_mnar_evidence.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_mnar_evidence.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Model strength vs coverage
# ═══════════════════════════════════════════════════════════════════
def fig4_model_strength_coverage():
    benchmarks = ['ai2d_test', 'livecodebench', 'hle', 'matharena']
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.5))

    for ax, bid in zip(axes.flat, benchmarks):
        _, _, model_stats, _, _ = load_benchmark(bid)

        x = model_stats['model_mean'].values
        y = model_stats['model_obs_rate'].values

        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        names = model_stats.loc[mask, 'display_name'].values

        ax.scatter(x_clean, y_clean, s=10, alpha=0.5, color='#2ca02c', edgecolors='none')

        # Spearman correlation
        rho, pval = stats.spearmanr(x_clean, y_clean)
        pstr = f"p<0.001" if pval < 0.001 else f"p={pval:.3f}"
        ax.text(0.03, 0.03, f"ρ={rho:.3f}, {pstr}", transform=ax.transAxes,
                fontsize=7, va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.8, edgecolor='gray'))

        # Label outliers (lowest obs rate)
        n_label = min(5, len(x_clean))
        outlier_idx = np.argsort(y_clean)[:n_label]
        for idx in outlier_idx:
            name = names[idx][:20]
            ax.annotate(name, (x_clean[idx], y_clean[idx]),
                       fontsize=5, alpha=0.8,
                       xytext=(5, 3), textcoords='offset points')

        ax.set_title(bid, fontsize=9)
        ax.set_xlabel('Model Mean Score', fontsize=8)
        ax.set_ylabel('Model Obs. Rate', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig_model_strength_coverage.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_model_strength_coverage.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: LiveCodeBench temporal missingness
# ═══════════════════════════════════════════════════════════════════
def fig5_livecodebench_temporal():
    O, item_stats, model_stats, item_ids, model_ids = load_benchmark('livecodebench')

    # Sort models by number of items observed (descending)
    row_obs = O.sum(axis=1)
    row_order = np.argsort(-row_obs)
    # Keep items in original order (potential temporal ordering)
    O_sorted = O[row_order, :]

    fig, ax = plt.subplots(figsize=(3.25, 3.5))
    ax.imshow(O_sorted, cmap='Greys', aspect='auto', interpolation='none')
    ax.set_xlabel('Items (original order)', fontsize=8)
    ax.set_ylabel('Models (sorted by coverage)', fontsize=8)
    ax.set_title('LiveCodeBench: Temporal\nMissingness Pattern', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

    plt.savefig(os.path.join(FIGURES, 'fig_livecodebench_temporal.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_livecodebench_temporal.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Missingness summary table
# ═══════════════════════════════════════════════════════════════════
def fig6_summary_table():
    # Filter to benchmarks with <100% density
    df = obs_summary[obs_summary['density'] < 1.0].copy()
    df = df.sort_values('density', ascending=True).reset_index(drop=True)

    # Compute ρ(difficulty, coverage) from item_stats if needed
    rows = []
    for _, row in df.iterrows():
        bid = row['benchmark_id']
        n_models = row['n_models']
        _, item_stats, model_stats, _, _ = load_benchmark(bid)

        # ρ(strength, coverage) from observation_summary
        rho_sc = row['corr_strength_coverage']

        # ρ(difficulty, coverage) — compute from item_stats
        obs_rate = item_stats['n_models_observed'] / n_models
        diff = item_stats['item_difficulty']
        mask = np.isfinite(diff) & np.isfinite(obs_rate)
        if mask.sum() > 2:
            rho_dc, _ = stats.spearmanr(diff[mask], obs_rate[mask])
        else:
            rho_dc = np.nan

        rows.append({
            'Benchmark': bid,
            'Density': f"{row['density']:.1%}",
            'Models': int(row['n_models']),
            'Items': int(row['n_items']),
            'ρ(str,cov)': f"{rho_sc:.2f}" if pd.notna(rho_sc) else '—',
            'ρ(diff,cov)': f"{rho_dc:.2f}" if pd.notna(rho_dc) else '—',
        })

    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.axis('off')

    col_labels = table_df.columns.tolist()
    cell_text = table_df.values.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#d9e2f3')
        table[0, j].set_text_props(fontweight='bold')

    # Alternate row colors
    for i in range(len(cell_text)):
        for j in range(len(col_labels)):
            if i % 2 == 1:
                table[i + 1, j].set_facecolor('#f2f2f2')

    ax.set_title('Missingness Statistics (benchmarks with <100% density)',
                 fontsize=8, fontweight='bold', pad=10)

    plt.savefig(os.path.join(FIGURES, 'fig_missingness_summary_table.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ fig_missingness_summary_table.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating missingness EDA figures...")
    fig1_density_overview()
    fig2_heatmaps()
    fig3_mnar_evidence()
    fig4_model_strength_coverage()
    fig5_livecodebench_temporal()
    fig6_summary_table()

    # Verification: list all PNGs with sizes
    print("\n── Generated figures ──")
    for f in sorted(os.listdir(FIGURES)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(FIGURES, f))
            print(f"  {f:45s}  {size/1024:.0f} KB")
