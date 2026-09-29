"""
Baseline Comparison for AMRResistanceGNN.

Trains three baselines on the same features and reports AUROC, AUPRC, F1
so the paper can show: GNN > ML baselines > random.

Baselines:
  1. Frequency baseline — predicts transfer probability = gene's acquisition_prob
     from CARD (no learning, pure domain knowledge)
  2. Logistic Regression — linear model on flattened node+edge features
  3. Random Forest — ensemble on same features, captures non-linearity
     but no graph structure

This is standard practice for GNN papers. Without this, reviewers reject.
"""

import json
import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import (
    AMRGraphDataset, collect_training_snapshots,
    GENE_INDEX, N_GENES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
)
from ai.gnn_trainer import split_dataset
from ai.gnn_trainer import DEFAULT_CONFIG, compute_metrics
from data.card_loader import RESISTANCE_GENES

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("scikit-learn not installed. Run: pip install scikit-learn")


def calibrated_macro_f1(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute macro F1/precision/recall at each gene's own F1-optimal
    threshold, averaged over genes with real test-set support — the same
    methodology threshold_calibration.py uses for the GNN.

    This MUST be applied identically to every model in the comparison
    table (frequency baseline, LR, RF, GNN). Using gnn_trainer.compute_metrics()'s
    fixed default threshold (0.35) for baselines while calibrating only the
    GNN would make the GNN look artificially superior on F1 — not because
    it performs better, but because it's the only model given a threshold
    appropriate to this dataset's ~0.01% positive rate. Fairness requires
    the same threshold-selection procedure for every model being compared.
    """
    from ai.threshold_calibration import find_optimal_thresholds
    import io, contextlib

    # find_optimal_thresholds prints a per-gene table; suppress it here
    # since this helper is called 4x per comparison run and the printed
    # detail is redundant with what run_baselines already prints.
    with contextlib.redirect_stdout(io.StringIO()):
        per_gene = find_optimal_thresholds(y_true, y_prob, "f1")

    genes_with_support = [g for g, v in per_gene.items() if v["support"] > 0]
    if not genes_with_support:
        return {"f1_macro": float("nan"), "precision_macro": float("nan"),
                "recall_macro": float("nan"), "n_genes_with_support": 0}

    return {
        "f1_macro":        float(np.mean([per_gene[g]["f1"]        for g in genes_with_support])),
        "precision_macro": float(np.mean([per_gene[g]["precision"] for g in genes_with_support])),
        "recall_macro":    float(np.mean([per_gene[g]["recall"]    for g in genes_with_support])),
        "n_genes_with_support": len(genes_with_support),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLATTEN GRAPH DATA TO TABULAR FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def flatten_dataset(ds: AMRGraphDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert graph dataset to tabular (X, y) for sklearn baselines.

    For each directed edge (i→j):
      X = [node_features_i | node_features_j | edge_features]
          shape: (NODE_FEATURE_DIM * 2 + EDGE_FEATURE_DIM,) = 75 dims
          (35*2 + 5, post leakage-remediation feature set)

    y = gene_labels (E, N_GENES)
    """
    import torch
    X_list, y_list = [], []

    for data in ds:
        n_edges    = data.edge_index.shape[1]
        if n_edges == 0:
            continue

        x          = data.x.numpy()           # (N, 35)
        edge_index = data.edge_index.numpy()  # (2, E)
        edge_attr  = data.edge_attr.numpy()   # (E, 5)
        y          = data.y.numpy()           # (E, 10)

        src = edge_index[0]
        dst = edge_index[1]

        # Concatenate [h_src | h_dst | edge_features]
        X_edges = np.concatenate([x[src], x[dst], edge_attr], axis=1)

        X_list.append(X_edges)
        y_list.append(y)

    if not X_list:
        return np.zeros((0, NODE_FEATURE_DIM*2 + EDGE_FEATURE_DIM)), np.zeros((0, N_GENES))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyBaseline:
    """
    Predicts gene transfer probability = acquisition_prob from CARD.
    No learning — pure domain knowledge baseline.
    This is the minimum bar: if GNN doesn't beat this, it's useless.
    """

    def __init__(self):
        # CARD acquisition probabilities (per-step transfer rates)
        self.gene_probs = {
            gene: RESISTANCE_GENES[gene].acquisition_prob
            for gene in GENE_INDEX
            if gene in RESISTANCE_GENES
        }

    def predict(self, n_edges: int) -> np.ndarray:
        """Returns (n_edges, N_GENES) probability array."""
        probs = np.zeros((n_edges, N_GENES), dtype=np.float32)
        for g_idx, gene in enumerate(GENE_INDEX):
            probs[:, g_idx] = self.gene_probs.get(gene, 0.01)
        return probs

    def name(self): return "FrequencyBaseline (CARD acquisition_prob)"


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(
    tr_ds: AMRGraphDataset,
    te_ds: AMRGraphDataset,
    max_train_samples: int = 100_000,
    results_path: str = "ai/checkpoints/baseline_results.json",
    subsample_seed: int = 42,
    model_seed: int = 42,
    include_gnn: bool = True,
    rf_params: Optional[dict] = None,
    models: Optional[tuple] = None,
) -> Dict[str, Dict]:
    """
    Train all baselines on tr_ds, evaluate on te_ds.
    Returns dict: {model_name: metrics_dict}

    subsample_seed: FIXED (found while investigating cross-run AUROC
    variance — Random Forest's AUROC moved from 0.9344 to 0.9548 between
    two DEFAULT_CONFIG runs despite RandomForestClassifier's own
    random_state being hardcoded to 42 in both). Root cause: the training
    subsample below used np.random.choice() against the GLOBAL, unseeded
    numpy RNG, so even with identical underlying data and an identical
    model random_state, RF (and LR) were trained on a genuinely different
    random 100k-edge subsample every run. This is the same class of bug
    as the original split_dataset() global-random issue, just in
    numpy.random instead of Python's random module, in a file that fix
    never touched. Now uses a local seeded np.random.RandomState instance
    so the SAME subsample is drawn every time, given the same input data
    and subsample_seed — required for the GNN-vs-baseline comparison to
    be reproducible, not just internally self-consistent.
    """
    if not SKLEARN_OK:
        return {}

    results = {}

    print("\n=== Baseline Comparison ===")
    print("Flattening datasets...")
    t0 = time.time()

    X_tr, y_tr = flatten_dataset(tr_ds)
    X_te, y_te = flatten_dataset(te_ds)

    print(f"  Train: {X_tr.shape[0]:,} edges | Test: {X_te.shape[0]:,} edges")
    print(f"  Feature dim: {X_tr.shape[1]}")
    print(f"  Positive rate: train={y_tr.mean():.4f}, test={y_te.mean():.4f}")

    # Subsample training if too large (LR/RF can't handle 700k samples fast)
    rng = np.random.RandomState(subsample_seed)
    if X_tr.shape[0] > max_train_samples:
        idx = rng.choice(X_tr.shape[0], max_train_samples, replace=False)
        X_tr_sub = X_tr[idx]
        y_tr_sub = y_tr[idx]
        print(f"  Subsampled to {max_train_samples:,} for sklearn baselines "
              f"(subsample_seed={subsample_seed})")
    else:
        X_tr_sub = X_tr
        y_tr_sub = y_tr

    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sub)
    X_te_scaled = scaler.transform(X_te)

    if models is None or "frequency_baseline" in models:
        # ── 1. Frequency baseline ────────────────────────────────────────────────
        print("\n[1/4] Frequency Baseline (CARD acquisition probs)...")
        freq = FrequencyBaseline()
        freq_probs = freq.predict(X_te.shape[0])
        freq_metrics = compute_metrics(y_te, freq_probs)
        freq_calib   = calibrated_macro_f1(y_te, freq_probs)
        results["frequency_baseline"] = {
            "model": freq.name(),
            "auroc_macro": freq_metrics["auroc_macro"],
            "auprc_macro": freq_metrics["auprc_macro"],
            "f1_macro":        freq_calib["f1_macro"],
            "precision_macro": freq_calib["precision_macro"],
            "recall_macro":    freq_calib["recall_macro"],
            "per_gene_auroc": {g: freq_metrics.get(f"auroc_{g}", float("nan"))
                               for g in GENE_INDEX},
        }
        print(f"  AUROC={freq_metrics['auroc_macro']:.4f} | "
              f"AUPRC={freq_metrics['auprc_macro']:.4f} | "
              f"F1(calibrated)={freq_calib['f1_macro']:.4f}")

    if models is None or "logistic_regression" in models:
        # ── 2. Logistic Regression ───────────────────────────────────────────────
        print("\n[2/4] Logistic Regression (per-gene, one-vs-rest)...")
        lr_all_probs = np.zeros((X_te.shape[0], N_GENES), dtype=np.float32)

        for g_idx, gene in enumerate(GENE_INDEX):
            y_g = y_tr_sub[:, g_idx]
            if y_g.sum() < 5:  # skip genes with very few positives in train
                lr_all_probs[:, g_idx] = 0.001
                continue
            lr = LogisticRegression(
                max_iter=500, C=1.0,
                class_weight="balanced",
                solver="saga", n_jobs=-1,
                random_state=model_seed,   # saga shuffles; was unseeded before 2026-09-29
            )
            lr.fit(X_tr_scaled, y_g)
            lr_all_probs[:, g_idx] = lr.predict_proba(X_te_scaled)[:, 1]

        lr_metrics = compute_metrics(y_te, lr_all_probs)
        lr_calib   = calibrated_macro_f1(y_te, lr_all_probs)
        results["logistic_regression"] = {
            "model": "Logistic Regression (balanced class weight)",
            "auroc_macro": lr_metrics["auroc_macro"],
            "auprc_macro": lr_metrics["auprc_macro"],
            "f1_macro":        lr_calib["f1_macro"],
            "precision_macro": lr_calib["precision_macro"],
            "recall_macro":    lr_calib["recall_macro"],
            "per_gene_auroc": {g: lr_metrics.get(f"auroc_{g}", float("nan"))
                               for g in GENE_INDEX},
        }
        print(f"  AUROC={lr_metrics['auroc_macro']:.4f} | "
              f"AUPRC={lr_metrics['auprc_macro']:.4f} | "
              f"F1(calibrated)={lr_calib['f1_macro']:.4f}")

    if models is None or "random_forest" in models:
        # ── 3. Random Forest ─────────────────────────────────────────────────────
        print("\n[3/4] Random Forest (100 trees, balanced subsample)...")
        rf_all_probs = np.zeros((X_te.shape[0], N_GENES), dtype=np.float32)

        for g_idx, gene in enumerate(GENE_INDEX):
            y_g = y_tr_sub[:, g_idx]
            if y_g.sum() < 5:
                rf_all_probs[:, g_idx] = 0.001
                continue
            rf = RandomForestClassifier(
                **{"n_estimators": 100, "max_depth": 8, **(rf_params or {})},
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=model_seed,   # default 42 = previous behaviour
            )
            rf.fit(X_tr_sub, y_g)  # RF doesn't need scaling
            rf_all_probs[:, g_idx] = rf.predict_proba(X_te)[:, 1]

        rf_metrics = compute_metrics(y_te, rf_all_probs)
        rf_calib   = calibrated_macro_f1(y_te, rf_all_probs)
        results["random_forest"] = {
            "model": "Random Forest (100 trees, max_depth=8, balanced_subsample)",
            "auroc_macro": rf_metrics["auroc_macro"],
            "auprc_macro": rf_metrics["auprc_macro"],
            "f1_macro":        rf_calib["f1_macro"],
            "precision_macro": rf_calib["precision_macro"],
            "recall_macro":    rf_calib["recall_macro"],
            "per_gene_auroc": {g: rf_metrics.get(f"auroc_{g}", float("nan"))
                               for g in GENE_INDEX},
        }
        print(f"  AUROC={rf_metrics['auroc_macro']:.4f} | "
              f"AUPRC={rf_metrics['auprc_macro']:.4f} | "
              f"F1(calibrated)={rf_calib['f1_macro']:.4f}")

    if not include_gnn:
        # Multi-seed runs (ai/reseeded_results.py) evaluate the GNN themselves,
        # per seed, on the same test set — no single training_results.json.
        return results

    # ── 4. GNN results (from checkpoint) ─────────────────────────────────────
    # FIXED post-JBHI-03955-2026 remediation: this block previously
    # hardcoded 0.9934/0.0602/0.0885 as fallback defaults AND explicitly
    # overwrote any real computed auroc_macro below 0.95 with those same
    # hardcoded literals (see git history / archived version), with a
    # comment claiming this was "known results from your full training
    # run." That silently discarded honest results and substituted
    # fabricated ones — the direct source of the evaluation-record
    # inconsistency flagged in editorial review. There is no fallback
    # here anymore: if training_results.json is missing, malformed, or
    # doesn't contain the expected keys, this now fails loudly rather
    # than inventing numbers.
    gnn_results_path = "ai/checkpoints/training_results.json"
    if not os.path.exists(gnn_results_path):
        raise FileNotFoundError(
            f"{gnn_results_path} not found. Run GNN training first "
            f"(ai/gnn_trainer.py) before running baseline comparison — "
            f"there is no fallback GNN result to compare against."
        )

    with open(gnn_results_path) as f:
        gnn_data = json.load(f)
    gnn_test = gnn_data.get("test_metrics")
    if gnn_test is None:
        raise ValueError(
            f"{gnn_results_path} does not contain a 'test_metrics' key. "
            f"Cannot report a GNN comparison result without it. "
            f"Contents found: {list(gnn_data.keys())}"
        )
    required_keys = ["auroc_macro", "auprc_macro"]
    missing = [k for k in required_keys if k not in gnn_test]
    if missing:
        raise ValueError(
            f"{gnn_results_path}['test_metrics'] is missing required "
            f"keys: {missing}. Refusing to substitute default values "
            f"for missing evaluation metrics."
        )

    # F1/precision/recall are intentionally NOT read from
    # training_results.json here. gnn_trainer.compute_metrics() binarizes
    # predictions at a hardcoded default threshold (0.35), which is
    # meaningless at this dataset's positive rate (~1e-4) — it forces
    # every gene's predicted-positive count to zero, giving f1_macro=0.0
    # exactly, a degenerate artifact rather than a real evaluation. The
    # correctly threshold-calibrated F1 (via per-gene F1-maximizing
    # thresholds, see ai/threshold_calibration.py) is required instead.
    calib_path = "ai/checkpoints/calibration_results.json"
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"{calib_path} not found. Run threshold calibration "
            f"(ai/threshold_calibration.py) before baseline comparison — "
            f"F1 at the naive default threshold (0.35) is degenerate at "
            f"this dataset's positive rate and must not be reported."
        )
    with open(calib_path) as f:
        calib_data = json.load(f)
    calib_summary = calib_data.get("summary")
    if calib_summary is None or "macro_f1_at_recommended_threshold" not in calib_summary:
        raise ValueError(
            f"{calib_path} does not contain summary.macro_f1_at_recommended_threshold. "
            f"Cannot report a non-degenerate F1 without it."
        )

    results["gnn_ours"] = {
        "model": "AMRResistanceGNN (GAT, 3 layers, 4 heads, 128-dim) — OURS",
        "auroc_macro": gnn_test["auroc_macro"],
        "auprc_macro": gnn_test["auprc_macro"],
        "f1_macro":       calib_summary["macro_f1_at_recommended_threshold"],
        "precision_macro":calib_summary["macro_precision_at_recommended_threshold"],
        "recall_macro":   calib_summary["macro_recall_at_recommended_threshold"],
        "f1_note": (
            "F1/precision/recall computed at the F1-optimal calibrated "
            "threshold (see threshold_calibration.py), NOT at a fixed "
            "default threshold — the latter is degenerate (F1=0.0) at "
            "this dataset's ~0.01% positive rate."
        ),
        "source_files": {
            "auroc_auprc": gnn_results_path,
            "f1_precision_recall": calib_path,
        },
        "per_gene_auroc": {
            g: gnn_test.get(f"auroc_{g}", float("nan"))
            for g in GENE_INDEX
        },
    }

    elapsed = time.time() - t0
    print(f"\n=== COMPARISON TABLE (for paper Table 4) ===")
    print(f"{'Model':<50} {'AUROC':>7} {'AUPRC':>7} {'F1':>7}")
    print("-" * 70)
    order = ["frequency_baseline", "logistic_regression", "random_forest", "gnn_ours"]
    for key in order:
        if key in results:
            r = results[key]
            marker = " [BEST]" if key == "gnn_ours" else ""
            short = r["model"][:48]
            print(f"{short:<50} {r['auroc_macro']:>7.4f} {r['auprc_macro']:>7.4f} {r['f1_macro']:>7.4f}{marker}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    Path("ai/checkpoints").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    tr_ds: AMRGraphDataset,
    te_ds: AMRGraphDataset,
    subsample_seed: int = 42,
) -> Dict[str, float]:
    """
    Ablation study: what happens when we remove each feature group?
    This answers reviewer question: "Which features matter most?"

    Strategy: train Logistic Regression with feature groups masked out.
    This approximates ablation without the cost of retraining the full GNN.

    NOTE: this previously accepted a full_gnn_auroc parameter defaulting
    to the fabricated 0.9934 literal. It was never actually used in the
    function body (dead code), but was removed entirely rather than left
    as an unused landmine that could silently reintroduce the fabricated
    number if wired into a future delta-vs-GNN calculation.

    subsample_seed: same fix as run_baselines() above — the 50k-sample
    subsampling below previously used the global unseeded np.random,
    making this ablation's LR training data different every run.
    """
    if not SKLEARN_OK:
        return {}

    print("\n=== Ablation Study (feature group importance) ===")

    X_tr, y_tr = flatten_dataset(tr_ds)
    X_te, y_te = flatten_dataset(te_ds)

    # Subsample for speed
    rng = np.random.RandomState(subsample_seed)
    if X_tr.shape[0] > 50_000:
        idx = rng.choice(X_tr.shape[0], 50_000, replace=False)
        X_tr = X_tr[idx]; y_tr = y_tr[idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Feature group slices (flattened 75-dim vector: [35 src | 35 dst | 5 edge])
    # Updated post feature-leakage remediation (JBHI-03955-2026).
    # Note: "No behavioral" now removes in_biofilm + is_persister only (2 dims).
    # sos_active was removed from the feature set entirely (leakage fix) so
    # it no longer appears as an ablation target.
    GROUPS = {
        "All features (full model)": None,
        "No genomic genes":     list(range(0,10))  + list(range(35,45)),
        "No physiological":     list(range(10,15)) + list(range(45,50)),
        "No behavioral":        list(range(15,17)) + list(range(50,52)),
        "No spatial position":  list(range(17,19)) + list(range(52,54)),
        "No antibiotic exposure":list(range(29,35))+ list(range(64,70)),
        "No edge features":     list(range(70,75)),
    }

    ablation_results = {}
    print(f"{'Feature group removed':<40} {'AUROC':>7} {'vs Full':>8}")
    print("-" * 58)

    for name, mask_cols in GROUPS.items():
        X_tr_abl = X_tr_s.copy()
        X_te_abl = X_te_s.copy()

        if mask_cols:
            X_tr_abl[:, mask_cols] = 0.0
            X_te_abl[:, mask_cols] = 0.0

        # Train LR per gene
        all_probs = np.zeros((X_te.shape[0], N_GENES), dtype=np.float32)
        for g_idx in range(N_GENES):
            y_g = y_tr[:, g_idx]
            if y_g.sum() < 5:
                all_probs[:, g_idx] = 0.001
                continue
            lr = LogisticRegression(max_iter=300, class_weight="balanced",
                                    solver="saga", n_jobs=-1)
            lr.fit(X_tr_abl, y_g)
            all_probs[:, g_idx] = lr.predict_proba(X_te_abl)[:, 1]

        m = compute_metrics(y_te, all_probs)
        auroc = m["auroc_macro"]
        delta = auroc - ablation_results.get("All features (full model)", auroc)
        ablation_results[name] = auroc

        delta_str = f"{delta:+.4f}" if mask_cols else "baseline"
        print(f"{name:<40} {auroc:>7.4f} {delta_str:>8}")

    return ablation_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_all_comparisons(config: dict = None) -> dict:
    """Run baselines + ablation and save all results."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print("Collecting data for baseline comparison...")
    all_pairs = []
    from ai.feature_engineering import collect_training_snapshots
    for scenario in config["scenarios"]:
        for seed in range(config["seeds_per_scenario"]):
            pairs = collect_training_snapshots(
                n_steps=config["steps_per_run"],
                scenario=scenario,
                seed=seed + 100,
                snapshot_interval=config["snapshot_interval"],
            )
            all_pairs.extend(pairs)

    from ai.gnn_trainer import split_dataset
    tr_ds, val_ds, te_ds = split_dataset(all_pairs, config)
    print(f"Dataset: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test graphs")

    baseline_results = run_baselines(tr_ds, te_ds)
    ablation_results = run_ablation(tr_ds, te_ds)

    combined = {
        "baselines": baseline_results,
        "ablation":  ablation_results,
    }
    with open("ai/checkpoints/full_comparison.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    return combined


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    from ai.gnn_trainer import QUICK_CONFIG, DEFAULT_CONFIG
    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    run_all_comparisons(cfg)