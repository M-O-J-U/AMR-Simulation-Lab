"""
Multi-Seed GNN vs Random Forest Comparison.

Built after discovering that a single GNN training run and a single RF
training run are not stable enough to support a "GNN beats RF" (or vice
versa) claim: two nominally identical DEFAULT_CONFIG runs produced GNN
AUROC 0.9599 and 0.9292 (a 0.031 swing), and RF AUROC 0.9344 and 0.9548
(a 0.020 swing) — enough movement to flip which model wins, from a single
unreplicated observation of each.

Root causes of that variance, found and fixed in the same investigation:
  1. GNN training had no seed control at all (model init, dropout,
     minibatch order) — see _train_core's new torch_seed parameter.
  2. RF/LR's training subsample was drawn via the GLOBAL, unseeded
     np.random.choice(), so even with RF's own random_state fixed, the
     DATA it saw varied every run — see run_baselines'/run_ablation's
     new subsample_seed parameter.

Both are now individually reproducible given a fixed seed. This module
answers the real question, which neither fix alone answers: given
GENUINE run-to-run variance (different seeds, same config), do the GNN's
and RF's AUROC distributions actually differ, or do they overlap enough
that "GNN beats RF" is not a defensible claim?
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import GENE_INDEX, N_GENES
from ai.gnn_trainer import DEFAULT_CONFIG, split_dataset, _train_core, collect_all_data
from ai.baselines import flatten_dataset

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


REDUCED_MULTISEED_CONFIG = {
    **DEFAULT_CONFIG,
    "seeds_per_scenario": 2,
    "epochs":             30,
    "patience":           8,
}


def _rf_auroc_macro(X_tr, y_tr, X_te, y_te, random_state: int) -> float:
    """Train one RF (one random_state) per gene, one-vs-rest, return macro AUROC."""
    aurocs = []
    for g in range(N_GENES):
        y_g = y_tr[:, g]
        if y_g.sum() < 5:
            continue
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=random_state,
        )
        rf.fit(X_tr, y_g)
        probs = rf.predict_proba(X_te)[:, 1]
        if y_te[:, g].sum() == 0:
            continue
        try:
            aurocs.append(roc_auc_score(y_te[:, g], probs))
        except ValueError:
            continue
    return float(np.mean(aurocs)) if aurocs else float("nan")


def run_multiseed_comparison(
    config:          Optional[dict] = None,
    n_seeds:         int = 5,
    max_train_samples: int = 100_000,
    verbose:         bool = True,
) -> Dict[str, object]:
    """
    Train the GNN n_seeds times (different torch_seed, same data/split)
    and RF n_seeds times (different random_state + subsample_seed, same
    data/split), report both AUROC distributions, and test whether they
    differ significantly (Mann-Whitney U — appropriate for small N,
    doesn't assume normality).

    Returns dict with per-seed results, summary statistics, and the
    statistical test result.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Multi-Seed GNN vs Random Forest Comparison")
        print(f"  n_seeds={n_seeds} | epochs={config['epochs']} | "
              f"patience={config['patience']}")
        print(f"{'='*60}\n")

    print("Collecting simulation data (shared across all seeds)...")
    t0 = time.time()
    all_pairs = collect_all_data(config, logger=None)
    if len(all_pairs) < 10:
        raise RuntimeError(f"Only {len(all_pairs)} graph pairs collected.")
    print(f"  Collected {len(all_pairs)} pairs in {time.time()-t0:.1f}s\n")

    # Same split for every seed — isolates model-training variance from
    # data-split variance, which is exactly the variable we're studying.
    tr_ds, val_ds, te_ds = split_dataset(all_pairs, config)
    print(f"Split: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test "
          f"(split_seed={config.get('split_seed', 42)}, identical for every seed)\n")

    X_tr, y_tr = flatten_dataset(tr_ds)
    X_te, y_te = flatten_dataset(te_ds)

    ckpt_dir = Path(config["checkpoint_dir"]) / "_multiseed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gnn_aurocs: List[float] = []
    rf_aurocs:  List[float] = []
    per_seed_results = []

    print(f"{'Seed':>6} | {'GNN AUROC':>10} | {'RF AUROC':>10} | {'GNN time(s)':>12} | {'RF time(s)':>11}")
    print("-" * 60)

    for seed_idx in range(n_seeds):
        torch_seed     = 1000 + seed_idx
        rf_seed        = 2000 + seed_idx
        subsample_seed = 3000 + seed_idx

        t_gnn = time.time()
        ckpt_path = str(ckpt_dir / f"model_seed{seed_idx}.pt")
        _, gnn_test_metrics, _ = _train_core(
            tr_ds, val_ds, te_ds, config, ckpt_path,
            verbose=False, print_steps=False, torch_seed=torch_seed,
        )
        gnn_auroc = gnn_test_metrics["auroc_macro"]
        gnn_elapsed = time.time() - t_gnn

        t_rf = time.time()
        rng = np.random.RandomState(subsample_seed)
        if X_tr.shape[0] > max_train_samples:
            idx = rng.choice(X_tr.shape[0], max_train_samples, replace=False)
            X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]
        else:
            X_tr_sub, y_tr_sub = X_tr, y_tr
        rf_auroc = _rf_auroc_macro(X_tr_sub, y_tr_sub, X_te, y_te, random_state=rf_seed)
        rf_elapsed = time.time() - t_rf

        gnn_aurocs.append(gnn_auroc)
        rf_aurocs.append(rf_auroc)
        per_seed_results.append({
            "seed_idx": seed_idx, "torch_seed": torch_seed,
            "rf_seed": rf_seed, "subsample_seed": subsample_seed,
            "gnn_auroc": gnn_auroc, "rf_auroc": rf_auroc,
            "gnn_elapsed_s": gnn_elapsed, "rf_elapsed_s": rf_elapsed,
        })

        print(f"{seed_idx:>6} | {gnn_auroc:>10.4f} | {rf_auroc:>10.4f} | "
              f"{gnn_elapsed:>12.1f} | {rf_elapsed:>11.1f}")

    gnn_arr = np.array(gnn_aurocs)
    rf_arr  = np.array(rf_aurocs)

    summary = {
        "gnn_mean": float(gnn_arr.mean()), "gnn_std": float(gnn_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
        "gnn_min":  float(gnn_arr.min()),  "gnn_max":  float(gnn_arr.max()),
        "rf_mean":  float(rf_arr.mean()),  "rf_std":  float(rf_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
        "rf_min":   float(rf_arr.min()),   "rf_max":   float(rf_arr.max()),
        "mean_diff": float(gnn_arr.mean() - rf_arr.mean()),
    }

    if SKLEARN_OK and n_seeds >= 3:
        try:
            u_stat, p_value = mannwhitneyu(gnn_arr, rf_arr, alternative="two-sided")
            summary["mannwhitney_u"] = float(u_stat)
            summary["mannwhitney_p"] = float(p_value)
        except ValueError:
            summary["mannwhitney_u"] = None
            summary["mannwhitney_p"] = None
    else:
        summary["mannwhitney_u"] = None
        summary["mannwhitney_p"] = None

    ranges_overlap = not (gnn_arr.min() > rf_arr.max() or rf_arr.min() > gnn_arr.max())

    print(f"\n{'='*60}")
    print(f"  GNN: mean={summary['gnn_mean']:.4f} std={summary['gnn_std']:.4f} "
          f"range=[{summary['gnn_min']:.4f}, {summary['gnn_max']:.4f}]")
    print(f"  RF:  mean={summary['rf_mean']:.4f} std={summary['rf_std']:.4f} "
          f"range=[{summary['rf_min']:.4f}, {summary['rf_max']:.4f}]")
    print(f"  Mean difference (GNN - RF): {summary['mean_diff']:+.4f}")
    print(f"  Ranges overlap: {ranges_overlap}")
    if summary["mannwhitney_p"] is not None:
        print(f"  Mann-Whitney U p-value: {summary['mannwhitney_p']:.4f}")
        if summary["mannwhitney_p"] < 0.05:
            print(f"  -> Difference IS statistically significant at p<0.05")
        else:
            print(f"  -> Difference is NOT statistically significant at p<0.05 -- "
                  f"do not claim one model beats the other from this data")
    else:
        print(f"  n_seeds < 3 -- no statistical test performed. Increase n_seeds "
              f"for a defensible significance claim.")
    print(f"{'='*60}\n")

    results = {
        "per_seed":  per_seed_results,
        "summary":   summary,
        "n_seeds":   n_seeds,
        "config":    config,
        "ranges_overlap": ranges_overlap,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = str(Path(config["checkpoint_dir"]) / "multiseed_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-seed GNN vs RF comparison")
    parser.add_argument("--full",    action="store_true",
                       help="Use DEFAULT_CONFIG (60 epochs, 3 seeds/scenario) instead of reduced scope")
    parser.add_argument("--n-seeds", type=int, default=5,
                       help="Number of independent training seeds per model (default 5)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy() if args.full else REDUCED_MULTISEED_CONFIG.copy()
    if not args.full:
        print("NOTE: running REDUCED_MULTISEED_CONFIG (2 seeds, 30 epochs). "
              "Use --full for submission-grade numbers.")

    run_multiseed_comparison(config=cfg, n_seeds=args.n_seeds)