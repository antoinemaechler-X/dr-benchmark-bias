"""
Bootstrap the additional benchmarks (livecodebench, ultrafeedback) on
Modal, DETACHED so the laptop can be closed. Results are written to a Modal
Volume in the cloud and downloaded afterwards.

Launch (returns immediately; safe to close laptop):
    modal run --detach bootstrap_modal_v2.py --iters 1000

Download results when ready:
    modal volume get dr-bootstrap-vol bootstrap_results_v2.csv  ./
    modal volume get dr-bootstrap-vol bootstrap_permodel_v2.npz ./

Check progress any time:
    modal app list          # shows dr-bootstrap-v2 running
"""

import modal

app = modal.App("dr-bootstrap-v2")
vol = modal.Volume.from_name("dr-bootstrap-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.1",
                 "pandas==2.2.2", "matplotlib==3.9.0")
    .add_local_file("semi_synthetic_dr.py", "/root/semi_synthetic_dr.py")
    .add_local_dir("bootstrap_bundles", "/root/bootstrap_bundles")
)

KEEP_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ["features_only", "features_and_score"]
# per-benchmark seed chunk size (tuned so chunks take comparable wall time)
BENCH_CHUNK = {"livecodebench": 20, "ultrafeedback": 5}


@app.function(image=image, timeout=3600, cpu=1.0, retries=2)
def run_chunk(task):
    import sys
    import numpy as np
    sys.path.insert(0, "/root")
    import semi_synthetic_dr as S

    bid, mode, seeds = task["bid"], task["mode"], task["seeds"]
    b = np.load(f"/root/bootstrap_bundles/{bid}.npz", allow_pickle=True)
    O_orig = b["O_orig"].astype(np.int8)
    M = b["M"].astype(np.float32)
    X_model, X_item = b["X_model"], b["X_item"]
    mu_true = b["mu_true"]
    z = b["z_fo"] if mode == "features_only" else b["z_fs"]
    n_models, n_items = O_orig.shape

    out = []
    for seed in seeds:
        for kr in KEEP_RATES:
            rng = np.random.RandomState(int(seed) * 1000 + int(round(kr * 100)))
            O_new, actual = S.generate_masking(z, O_orig, kr, rng)
            M_irt = S.fit_irt(M, O_new)
            pi = S.fit_propensity(X_model, X_item, O_orig, O_new, n_models, n_items)
            mus = {
                "naive": S.estimator_naive(M, O_new, O_orig),
                "irt":   S.estimator_irt(M_irt, O_orig),
                "ipw":   S.estimator_ipw(M, O_new, O_orig, pi),
                "dr":    S.estimator_dr(M, M_irt, O_new, O_orig, pi),
            }
            rec = {"bid": bid, "mode": mode, "keep_rate": kr,
                   "seed": int(seed), "actual_keep_rate": float(actual)}
            for name, mu in mus.items():
                bias, rmse, rc = S.evaluate(mu, mu_true)
                rec[f"{name}_bias"] = bias
                rec[f"{name}_rmse"] = rmse
                rec[f"{name}_rankcorr"] = rc
                rec[f"{name}_mu"] = mu.astype(np.float32)
            rec["mu_true"] = mu_true.astype(np.float32)
            out.append(rec)
    return out


@app.function(image=image, timeout=14400, volumes={"/out": vol})
def orchestrate(iters, benchmarks):
    """Runs in the cloud: fans out chunks, aggregates, writes to the Volume."""
    import numpy as np
    import pandas as pd

    tasks = []
    for bid in benchmarks:
        cs = BENCH_CHUNK[bid]
        for mode in MODES:
            for start in range(0, iters, cs):
                seeds = list(range(start, min(start + cs, iters)))
                tasks.append({"bid": bid, "mode": mode, "seeds": seeds})
    print(f"Dispatching {len(tasks)} chunks for {benchmarks} x {iters} iters...")

    all_recs = []
    for res in run_chunk.map(tasks):
        all_recs.extend(res)
    print(f"Collected {len(all_recs)} records.")

    agg_cols = ["bid", "mode", "keep_rate", "seed", "actual_keep_rate"]
    metric_cols = [f"{n}_{m}" for n in ["naive", "irt", "ipw", "dr"]
                   for m in ["bias", "rmse", "rankcorr"]]
    agg_rows, permodel = [], {}
    for r in all_recs:
        agg_rows.append({**{c: r[c] for c in agg_cols},
                         **{c: r[c] for c in metric_cols}})
        key = f"{r['bid']}|{r['mode']}|{r['keep_rate']}"
        d = permodel.setdefault(key, {"seed": [], "mu_true": r["mu_true"],
                                      "naive": [], "irt": [], "ipw": [], "dr": []})
        d["seed"].append(r["seed"])
        for name in ["naive", "irt", "ipw", "dr"]:
            d[name].append(r[f"{name}_mu"])

    df = pd.DataFrame(agg_rows).sort_values(["bid", "mode", "keep_rate", "seed"])
    df.to_csv("/out/bootstrap_results_v2.csv", index=False)

    npz = {}
    for key, d in permodel.items():
        order = np.argsort(d["seed"])
        safe = key.replace("|", "__")
        npz[f"{safe}__seed"] = np.array(d["seed"])[order]
        npz[f"{safe}__mu_true"] = d["mu_true"]
        for name in ["naive", "irt", "ipw", "dr"]:
            npz[f"{safe}__{name}"] = np.stack(d[name])[order]
    np.savez_compressed("/out/bootstrap_permodel_v2.npz", **npz)
    vol.commit()
    print(f"Wrote bootstrap_results_v2.csv ({len(df)} rows) and "
          f"bootstrap_permodel_v2.npz ({len(permodel)} keys) to the Volume.")
    return len(df)


@app.local_entrypoint()
def main(iters: int = 1000):
    benchmarks = ["livecodebench", "ultrafeedback"]
    call = orchestrate.spawn(iters, benchmarks)
    print(f"Launched orchestrate (id={call.object_id}) detached.")
    print("Safe to close the laptop. When done, download with:")
    print("  modal volume get dr-bootstrap-vol bootstrap_results_v2.csv  ./")
    print("  modal volume get dr-bootstrap-vol bootstrap_permodel_v2.npz ./")
