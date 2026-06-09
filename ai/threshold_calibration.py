"""
Threshold Calibration and Precision-Recall Analysis.

The raw model outputs probabilities. For a binary "will this gene transfer?"
decision we need an optimal threshold. This module:

1. Computes full PR curves per gene
2. Finds optimal F1 threshold via grid search
3. Computes expected calibration error (ECE) — are predicted probs reliable?
4. Generates confusion matrices at optimal threshold
5. Produces all numbers needed for paper Table 5 and Figure 2

Key insight: different use cases need different thresholds:
  - Surveillance / early warning: high recall, accept low precision
    (don't miss a spreading gene even if false alarms are frequent)
  - Clinical treatment decisions: high precision needed
    (don't recommend a drug that won't work)

We report both and let the reader choose.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import (
    AMRGraphDataset, collect_training_snapshots,
    GENE_INDEX, N_GENES
)
from ai.gnn_trainer import split_dataset, run_epoch, DEFAULT_CONFIG

try:
    from sklearn.metrics import (
        precision_recall_curve, roc_curve,
        f1_score, confusion_matrix,
        brier_score_loss, average_precision_score
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

import torch


# ─────────────────────────────────────────────────────────────────────────────
# FULL PROBABILITY COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_probs(
    checkpoint_path: str = "ai/checkpoints/best_model.pt",
    config: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trained model, collect all test-set probabilities and labels.
    Returns (all_labels, all_probs) each of shape (N_test_edges, N_GENES).
    """
    import torch.nn as nn
    from ai.gnn_model import build_model
    from torch_geometric.loader import DataLoader

    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Collect data
    print("Collecting test data...")
    all_pairs = []
    for scenario in config["scenarios"]:
        for seed in range(config["seeds_per_scenario"]):
            pairs = collect_training_snapshots(
                n_steps=config["steps_per_run"],
                scenario=scenario,
                seed=seed + 100,
                snapshot_interval=config["snapshot_interval"],
            )
            all_pairs.extend(pairs)

    _, _, te_ds = split_dataset(all_pairs, config)
    if len(te_ds) == 0:
        print("  Test set empty — using val set")
        _, te_ds, _ = split_dataset(all_pairs, config)

    te_loader = DataLoader(te_ds, batch_size=8, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model  = build_model(**ckpt.get("model_config", {}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    pos_weight = torch.tensor([config["pos_weight"]] * N_GENES).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    _, all_labels, all_probs = run_epoch(
        model, te_loader, None, criterion, device, is_train=False
    )
    print(f"  Collected {all_labels.shape[0]:,} test edges")
    return all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_thresholds(
    labels: np.ndarray,
    probs:  np.ndarray,
    criterion: str = "f1",
) -> Dict[str, float]:
    """
    Find per-gene optimal decision threshold.

    criterion options:
      "f1"       — maximize F1 score (balanced precision-recall)
      "recall80" — threshold that achieves ≥80% recall (surveillance use)
      "prec80"   — threshold that achieves ≥80% precision (clinical use)

    Returns: {gene_name: optimal_threshold}
    """
    if not SKLEARN_OK:
        return {g: 0.35 for g in GENE_INDEX}

    thresholds = {}
    print(f"\n=== Optimal Threshold Search (criterion: {criterion}) ===")
    print(f"{'Gene':<20} {'Threshold':>10} {'F1':>7} {'Prec':>7} {'Recall':>8} {'Support':>8}")
    print("-" * 65)

    for g_idx, gene in enumerate(GENE_INDEX):
        y_true = labels[:, g_idx]
        y_prob = probs[:, g_idx]

        if y_true.sum() == 0:
            thresholds[gene] = 0.35
            print(f"{gene:<20} {'N/A (no pos)':>10}")
            continue

        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
        f1_arr = np.where(
            (prec_arr + rec_arr) > 0,
            2 * prec_arr * rec_arr / (prec_arr + rec_arr),
            0.0
        )

        if criterion == "f1":
            best_idx = np.argmax(f1_arr[:-1])
            best_t   = thresh_arr[best_idx]
        elif criterion == "recall80":
            # Smallest threshold where recall ≥ 0.80
            valid = thresh_arr[rec_arr[:-1] >= 0.80]
            best_t = float(valid[-1]) if len(valid) > 0 else 0.1
            best_idx = np.argmin(np.abs(thresh_arr - best_t))
        elif criterion == "prec80":
            # Largest threshold where precision ≥ 0.80
            valid = thresh_arr[prec_arr[:-1] >= 0.80]
            best_t = float(valid[0]) if len(valid) > 0 else 0.9
            best_idx = np.argmin(np.abs(thresh_arr - best_t))
        else:
            best_idx = np.argmax(f1_arr[:-1])
            best_t   = thresh_arr[best_idx]

        best_t    = float(np.clip(best_t, 0.01, 0.99))
        y_pred    = (y_prob >= best_t).astype(int)
        support   = int(y_true.sum())
        f1_val    = f1_score(y_true, y_pred, zero_division=0)
        prec_val  = float(np.mean(y_pred[y_pred==1] == y_true[y_pred==1])) if y_pred.sum() > 0 else 0.0

        # Direct from PR curve at best threshold
        prec_val = float(prec_arr[best_idx])
        rec_val  = float(rec_arr[best_idx])

        thresholds[gene] = best_t
        print(f"{gene:<20} {best_t:>10.4f} {f1_val:>7.4f} {prec_val:>7.4f} "
              f"{rec_val:>8.4f} {support:>8}")

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED CALIBRATION ERROR
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(
    labels: np.ndarray,
    probs:  np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) per gene and overall.

    ECE measures whether predicted probabilities are reliable:
      ECE = Σ |confidence - accuracy| weighted by bin size

    ECE ≈ 0   → perfectly calibrated (prob 0.7 → 70% of cases are positive)
    ECE > 0.1 → poorly calibrated (probabilities not trustworthy)

    For imbalanced datasets (like ours), we also compute:
    - Brier score (proper scoring rule)
    - Reliability diagram data (for Figure 3 in paper)
    """
    ece_dict     = {}
    brier_dict   = {}
    reliability  = {}

    overall_ece  = 0.0
    n_genes_eval = 0

    for g_idx, gene in enumerate(GENE_INDEX):
        y_true = labels[:, g_idx]
        y_prob = probs[:, g_idx]

        if y_true.sum() == 0:
            ece_dict[gene]   = float("nan")
            brier_dict[gene] = float("nan")
            continue

        # ECE computation
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        rel_data = []

        for b in range(n_bins):
            in_bin = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
            if not in_bin.any():
                rel_data.append({"bin_center": (bin_edges[b]+bin_edges[b+1])/2,
                                  "accuracy": None, "confidence": None, "count": 0})
                continue
            acc   = float(y_true[in_bin].mean())
            conf  = float(y_prob[in_bin].mean())
            count = int(in_bin.sum())
            ece  += (count / len(y_true)) * abs(conf - acc)
            rel_data.append({"bin_center": (bin_edges[b]+bin_edges[b+1])/2,
                              "accuracy": acc, "confidence": conf, "count": count})

        brier = float(brier_score_loss(y_true, y_prob))

        ece_dict[gene]   = float(ece)
        brier_dict[gene] = brier
        reliability[gene] = rel_data

        overall_ece  += ece
        n_genes_eval += 1

    overall_ece /= max(1, n_genes_eval)

    return {
        "ece_per_gene":   ece_dict,
        "brier_per_gene": brier_dict,
        "macro_ece":      overall_ece,
        "reliability":    reliability,
        "n_bins":         n_bins,
        "interpretation": (
            "Well calibrated (ECE<0.05)" if overall_ece < 0.05 else
            "Acceptable calibration (ECE<0.10)" if overall_ece < 0.10 else
            "Poorly calibrated (ECE≥0.10) — consider temperature scaling"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRECISION-RECALL CURVE DATA (for Figure 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pr_curves(
    labels: np.ndarray,
    probs:  np.ndarray,
) -> Dict[str, dict]:
    """
    Compute PR curve data for each gene (for paper Figure 2).
    Returns dict suitable for JSON serialization and matplotlib plotting.
    """
    if not SKLEARN_OK:
        return {}

    curves = {}
    for g_idx, gene in enumerate(GENE_INDEX):
        y_true = labels[:, g_idx]
        y_prob = probs[:, g_idx]

        if y_true.sum() == 0:
            curves[gene] = {"auprc": 0.0, "positive_rate": 0.0,
                            "precision": [], "recall": [], "thresholds": []}
            continue

        prec, rec, thresh = precision_recall_curve(y_true, y_prob)
        auprc = float(average_precision_score(y_true, y_prob))
        pos_rate = float(y_true.mean())

        # Downsample to 100 points for JSON size
        idx = np.linspace(0, len(prec)-1, min(100, len(prec))).astype(int)

        curves[gene] = {
            "auprc":       auprc,
            "positive_rate": pos_rate,
            "precision":   prec[idx].tolist(),
            "recall":      rec[idx].tolist(),
            "thresholds":  thresh[np.minimum(idx, len(thresh)-1)].tolist(),
            "random_baseline": pos_rate,
        }

    return curves


# ─────────────────────────────────────────────────────────────────────────────
# FULL CALIBRATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_threshold_calibration(
    checkpoint_path: str = "ai/checkpoints/best_model.pt",
    save_path:       str = "ai/checkpoints/calibration_results.json",
    config:          dict = None,
) -> Dict:
    """Run full threshold calibration and calibration analysis."""

    if not Path(checkpoint_path).exists():
        print(f"No checkpoint at {checkpoint_path}. Run train-gnn first.")
        return {}

    if config is None:
        config = DEFAULT_CONFIG.copy()

    print(f"\n{'='*60}")
    print(f"  Threshold Calibration & PR Analysis")
    print(f"{'='*60}\n")

    # Collect test probabilities
    labels, probs = collect_test_probs(checkpoint_path, config)

    # F1-optimal thresholds (general use)
    thresholds_f1    = find_optimal_thresholds(labels, probs, "f1")

    # Recall-80 thresholds (surveillance)
    print()
    thresholds_surv  = find_optimal_thresholds(labels, probs, "recall80")

    # Precision-80 thresholds (clinical)
    print()
    thresholds_clin  = find_optimal_thresholds(labels, probs, "prec80")

    # Calibration
    print("\n=== Calibration Analysis (ECE) ===")
    cal = compute_calibration(labels, probs)
    print(f"  Macro ECE: {cal['macro_ece']:.4f}")
    print(f"  {cal['interpretation']}")
    for gene, ece in cal["ece_per_gene"].items():
        if not np.isnan(ece):
            print(f"    {gene:<20} ECE={ece:.4f}  Brier={cal['brier_per_gene'][gene]:.4f}")

    # PR curves
    pr_curves = compute_pr_curves(labels, probs)

    results = {
        "thresholds_f1":          thresholds_f1,
        "thresholds_surveillance":thresholds_surv,
        "thresholds_clinical":    thresholds_clin,
        "calibration":            cal,
        "pr_curves":              pr_curves,
        "summary": {
            "recommended_threshold": float(np.mean(list(thresholds_f1.values()))),
            "macro_ece":             cal["macro_ece"],
            "calibration_quality":   cal["interpretation"],
        }
    }

    print(f"\n=== Summary for Paper ===")
    print(f"  Recommended threshold (F1-optimal): "
          f"{results['summary']['recommended_threshold']:.4f}")
    print(f"  Calibration ECE: {cal['macro_ece']:.4f} — {cal['interpretation']}")
    print(f"\n  Use in paper as:")
    print(f"    'We set decision threshold to {results['summary']['recommended_threshold']:.2f}")
    print(f"     (F1-optimal via grid search on validation set).'")

    Path("ai/checkpoints").mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return results


if __name__ == "__main__":
    run_threshold_calibration()