"""
(D, principled version) Shrinkage DR: instead of a hard sample-size cutoff,
blend the DR estimate with the lower-variance IRT estimate, leaning on IRT when
a model has few observations.

    mu_shrunk_i = (1 - lambda_i) * mu_IRT_i + lambda_i * mu_DR_i
    lambda_i    = n_i / (n_i + alpha)          # n_i = # items model i was evaluated on

alpha is the only knob (a "pseudo-count" / regularization strength). We select
it by minimizing semi-synthetic RMSE-to-truth, and we do it HONESTLY: alpha is
tuned on half the bootstrap seeds and all metrics are reported on the held-out
half. Special cases: alpha=0 -> vanilla DR;  alpha=inf -> IRT-only.

Uses the existing 1000-draw bootstrap (bootstrap_permodel.npz) -> no new compute.

Writes: dr_shrinkage_results.csv, figures/fig_dr_shrinkage.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'figures'

BENCHMARKS = ['mmlupro', 'matharena']
BENCH_TITLES = {'mmlupro': 'MMLU-Pro', 'matharena': 'MathArena'}
MODE = 'features_and_score'
KEEP = 0.7
ALPHAS = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 1e9])


def load(bid):
    d = np.load(PROJECT / 'bootstrap_permodel.npz', allow_pickle=True)
    pre = f'{bid}__{MODE}__{KEEP}'
    mu_irt = d[f'{pre}__irt']        # (S, n)
    mu_dr = d[f'{pre}__dr']          # (S, n)
    mu_true = d[f'{pre}__mu_true']   # (n,)
    b = np.load(PROJECT / 'bootstrap_bundles' / f'{bid}.npz', allow_pickle=True)
    n_obs = (b['O_orig'] == 1).sum(axis=1).astype(float)   # (n,)
    return mu_irt, mu_dr, mu_true, n_obs


def metrics(mu, mu_true):
    """mu: (S, n). Returns RMSE-to-truth (over all seed,model), and per-model
    mean-bias-based worst/median |bias|."""
    bias = mu - mu_true[None, :]
    rmse = np.sqrt(np.nanmean(bias ** 2))
    permodel = np.nanmean(bias, axis=0)              # (n,)
    ab = np.abs(permodel)
    return rmse, np.median(ab), ab.max(), int((ab > 0.05).sum())


def shrunk(mu_irt, mu_dr, n_obs, alpha):
    lam = n_obs / (n_obs + alpha)                    # (n,)
    return (1 - lam)[None, :] * mu_irt + lam[None, :] * mu_dr


def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, bid in zip(axes, BENCHMARKS):
        mu_irt, mu_dr, mu_true, n_obs = load(bid)
        S = mu_irt.shape[0]
        tune, test = slice(0, S // 2), slice(S // 2, S)

        # ---- select alpha on the TUNE half by minimizing RMSE ----
        tune_rmse = [metrics(shrunk(mu_irt[tune], mu_dr[tune], n_obs, a), mu_true)[0]
                     for a in ALPHAS]
        a_star = ALPHAS[int(np.argmin(tune_rmse))]

        # ---- full alpha curve on TEST half (honest) ----
        test_curve = [metrics(shrunk(mu_irt[test], mu_dr[test], n_obs, a), mu_true)
                      for a in ALPHAS]
        test_rmse = [c[0] for c in test_curve]
        test_max = [c[2] for c in test_curve]

        for a, (rm, med, mx, ne) in zip(ALPHAS, test_curve):
            rows.append({'benchmark': bid, 'alpha': a, 'rmse_test': rm,
                         'median_abs': med, 'max_abs': mx, 'n_explode': ne,
                         'selected': a == a_star})

        # reference points
        dr_m = metrics(mu_dr[test], mu_true)       # alpha=0
        irt_m = metrics(mu_irt[test], mu_true)     # alpha=inf
        star_m = metrics(shrunk(mu_irt[test], mu_dr[test], n_obs, a_star), mu_true)

        x = np.where(ALPHAS >= 1e9, ALPHAS.max() * 0 + 1e4, ALPHAS)  # plot inf at 1e4
        x = [a if a < 1e9 else 1e4 for a in ALPHAS]
        ax.plot(x, test_rmse, 'o-', color='#1f77b4', label='RMSE (held-out)')
        ax.axvline(a_star if a_star < 1e9 else 1e4, color='green', ls='--',
                   label=f'selected alpha*={a_star:.0f}')
        ax.set_xscale('symlog')
        ax.set_xlabel('shrinkage alpha  (0 = vanilla DR, large = IRT-only)')
        ax.set_ylabel('RMSE to truth (held-out seeds)')
        ax.set_title(f'{BENCH_TITLES[bid]}', fontsize=11)
        ax2 = ax.twinx()
        ax2.plot(x, test_max, 's--', color='#d62728', alpha=0.7,
                 label='worst |bias|')
        ax2.set_ylabel('worst-model |bias|', color='#d62728')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', fontsize=8)

        print(f"\n{BENCH_TITLES[bid]} (held-out seeds):")
        print(f"  vanilla DR (alpha=0):  RMSE={dr_m[0]:.4f}  median|b|={dr_m[1]:.4f}  "
              f"max|b|={dr_m[2]:.4f}  #expl={dr_m[3]}")
        print(f"  shrinkage  (alpha*={a_star:.0f}): RMSE={star_m[0]:.4f}  median|b|={star_m[1]:.4f}  "
              f"max|b|={star_m[2]:.4f}  #expl={star_m[3]}")
        print(f"  IRT-only   (alpha=inf):RMSE={irt_m[0]:.4f}  median|b|={irt_m[1]:.4f}  "
              f"max|b|={irt_m[2]:.4f}  #expl={irt_m[3]}")

    fig.suptitle('Shrinkage DR: risk-minimizing blend of DR and IRT '
                 '(alpha tuned on train seeds, shown on held-out)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES / 'fig_dr_shrinkage.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}")
    pd.DataFrame(rows).to_csv(PROJECT / 'dr_shrinkage_results.csv', index=False)
    print("Saved dr_shrinkage_results.csv")


if __name__ == '__main__':
    main()
