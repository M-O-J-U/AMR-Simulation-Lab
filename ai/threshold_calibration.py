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
                biology=config.get("biology", "paper_v1"),
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
) -> Dict[str, dict]:
    """
    Find per-gene optimal decision threshold.

    criterion options:
      "f1"       — maximize F1 score (balanced precision-recall)
      "recall80" — threshold that achieves ≥80% recall (surveillance use)
      "prec80"   — threshold that achieves ≥80% precision (clinical use)

    Returns: {gene_name: {"threshold": float, "f1": float, "precision": float,
                          "recall": float, "support": int}}

    NOTE (post JBHI-03955-2026 remediation): under extreme class imbalance
    (positive support as low as 3-50 out of 100k+ test edges), real F1
    values are often < 0.001. Previously this function printed F1 rounded
    to 4 decimal places, which silently displayed genuinely non-zero F1 as
    "0.0000" — misleading, since a reviewer reading "F1=0.0000" reasonably
    concludes the model produces zero correct positive predictions, which
    was not true. Display precision below is now adaptive, and the full
    numeric values (not just the threshold) are returned so callers can
    report accurate figures rather than re-deriving them from a rounded
    printout.
    """
    if not SKLEARN_OK:
        return {g: {"threshold": 0.35, "f1": None, "precision": None,
                    "recall": None, "support": 0} for g in GENE_INDEX}

    results = {}
    print(f"\n=== Optimal Threshold Search (criterion: {criterion}) ===")
    print(f"{'Gene':<20} {'Threshold':>10} {'F1':>10} {'Prec':>10} {'Recall':>8} {'Support':>8}")
    print("-" * 70)

    for g_idx, gene in enumerate(GENE_INDEX):
        y_true = labels[:, g_idx]
        y_prob = probs[:, g_idx]

        if y_true.sum() == 0:
            results[gene] = {"threshold": 0.35, "f1": None, "precision": None,
                             "recall": None, "support": 0}
            print(f"{gene:<20} {'N/A (no pos)':>10}")
            continue

        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
        # The last point of a sklearn PR curve is always (precision=1, recall=0)
        # by construction (or occasionally 0/0), which triggers a spurious
        # divide warning and is already excluded via f1_arr[:-1] below —
        # suppress the warning rather than let it imply a real numeric issue.
        with np.errstate(invalid="ignore", divide="ignore"):
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

        raw_best_t = float(thresh_arr[best_idx])
        # The clip floor must be well below the actual positive rate, not
        # an arbitrary round number. A fixed floor of 0.01 (used prior to
        # this fix) is ~100x the true positive rate in this dataset
        # (~1e-4), which forced EVERY gene's threshold to round up to a
        # value where the model predicts zero positives for everyone —
        # producing an unusable "recommended threshold" with P=R=F1=0
        # exactly, not approximately. The floor here is instead derived
        # from the data itself: two orders of magnitude below the
        # empirical positive rate, so genuinely low but real optimal
        # thresholds under extreme class imbalance are not silently
        # clamped away.
        min_floor  = max(1e-6, float(y_true.mean()) / 100.0)
        best_t     = float(np.clip(raw_best_t, min_floor, 0.99))

        # CRITICAL: recompute y_pred, precision, recall, and F1 all from the
        # SAME (clipped) threshold. Previously precision/recall were read
        # from prec_arr[best_idx]/rec_arr[best_idx] — i.e. at raw_best_t,
        # the UNCLIPPED threshold — while F1 was computed from y_pred at
        # best_t, the CLIPPED threshold. When raw_best_t fell below 0.01
        # (routine under this level of class imbalance, where the true
        # F1-maximizing threshold is often far below 0.01), this silently
        # reported precision/recall for a different, more permissive
        # threshold than the one actually used for the F1 score and for
        # the recommended decision threshold — an internal inconsistency
        # of exactly the kind flagged in JBHI-03955-2026 editorial review.
        y_pred    = (y_prob >= best_t).astype(int)
        support   = int(y_true.sum())
        f1_val    = float(f1_score(y_true, y_pred, zero_division=0))
        tp        = int(((y_pred == 1) & (y_true == 1)).sum())
        pred_pos  = int((y_pred == 1).sum())
        prec_val  = float(tp / pred_pos) if pred_pos > 0 else 0.0
        rec_val   = float(tp / support)  if support  > 0 else 0.0

        results[gene] = {
            "threshold": best_t,
            "raw_unclipped_threshold": raw_best_t,
            "threshold_was_clipped": abs(best_t - raw_best_t) > 1e-9,
            "f1": f1_val, "precision": prec_val, "recall": rec_val,
            "support": support,
        }

        # Adaptive display precision: use scientific notation for values
        # small enough that %.4f would misleadingly show 0.0000.
        f1_str   = f"{f1_val:.2e}"   if 0 < f1_val   < 0.0001 else f"{f1_val:.4f}"
        prec_str = f"{prec_val:.2e}" if 0 < prec_val < 0.0001 else f"{prec_val:.4f}"
        print(f"{gene:<20} {best_t:>10.4f} {f1_str:>10} {prec_str:>10} "
              f"{rec_val:>8.4f} {support:>8}")

    return results


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

    # Genes with actual test-set positives (support > 0) — everything else
    # is a placeholder entry (threshold=0.35, f1/precision/recall=None) and
    # must be excluded from any macro average or it silently corrupts it.
    genes_with_support = [g for g, v in thresholds_f1.items() if v["support"] > 0]

    macro_threshold = float(np.mean([thresholds_f1[g]["threshold"] for g in genes_with_support])) \
        if genes_with_support else float("nan")
    macro_f1        = float(np.mean([thresholds_f1[g]["f1"]        for g in genes_with_support])) \
        if genes_with_support else float("nan")
    macro_precision = float(np.mean([thresholds_f1[g]["precision"] for g in genes_with_support])) \
        if genes_with_support else float("nan")
    macro_recall    = float(np.mean([thresholds_f1[g]["recall"]    for g in genes_with_support])) \
        if genes_with_support else float("nan")

    results = {
        "thresholds_f1":          thresholds_f1,
        "thresholds_surveillance":thresholds_surv,
        "thresholds_clinical":    thresholds_clin,
        "calibration":            cal,
        "pr_curves":              pr_curves,
        "n_genes_with_test_positives": len(genes_with_support),
        "genes_with_test_positives":   genes_with_support,
        "summary": {
            "recommended_threshold": macro_threshold,
            "macro_f1_at_recommended_threshold":        macro_f1,
            "macro_precision_at_recommended_threshold": macro_precision,
            "macro_recall_at_recommended_threshold":    macro_recall,
            "macro_ece":             cal["macro_ece"],
            "calibration_quality":   cal["interpretation"],
        }
    }

    print(f"\n=== Summary for Paper ===")
    print(f"  Genes with test-set positives: {len(genes_with_support)}/{N_GENES} "
          f"({', '.join(genes_with_support)})")
    print(f"  Recommended threshold (F1-optimal, averaged over genes with "
          f"support): {macro_threshold:.4f}")
    print(f"  Macro F1 at recommended threshold:        {macro_f1:.6f}")
    print(f"  Macro precision at recommended threshold: {macro_precision:.6f}")
    print(f"  Macro recall at recommended threshold:    {macro_recall:.6f}")
    print(f"  Calibration ECE: {cal['macro_ece']:.4f} — {cal['interpretation']}")
    print(f"\n  Use in paper as:")
    thresh_str = f"{macro_threshold:.2e}" if 0 < macro_threshold < 0.001 else f"{macro_threshold:.4f}"
    print(f"    'We set decision threshold to {thresh_str} (F1-optimal,")
    print(f"     averaged over the {len(genes_with_support)} genes with test-set")
    print(f"     positives), yielding macro F1={macro_f1:.4f}, precision=")
    print(f"     {macro_precision:.4f}, recall={macro_recall:.4f}.'")

    Path("ai/checkpoints").mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return results


if __name__ == "__main__":
    run_threshold_calibration()