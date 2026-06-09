"""
GNN Training Loop for AMR Resistance Gene Transfer Prediction.

Training strategy:
  - Data: simulation snapshots collected across multiple scenarios and seeds
  - Split: 70% train / 15% val / 15% test (stratified by scenario)
  - Loss: Binary Cross-Entropy with class weights (HGT events are rare — imbalanced)
  - Optimizer: AdamW with cosine annealing LR schedule
  - Early stopping: patience=10 on validation AUROC
  - Checkpointing: saves best model by val AUROC
  - Metrics: AUROC, AUPRC, F1, precision, recall per gene

Why AUROC over accuracy:
  HGT events are rare (<2% of edges per step). Accuracy is meaningless
  for imbalanced classification — a model predicting "never transfer"
  gets 98% accuracy. AUROC measures discriminative power regardless of threshold.

Why per-gene metrics:
  Different genes have different transfer rates. blaTEM-1 transfers very
  frequently (high acquisition_prob). blaNDM-1 is rarer. A single aggregate
  metric hides these differences — we need per-gene insight for the paper.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ai.feature_engineering import (
    collect_training_snapshots, AMRGraphDataset,
    GENE_INDEX, N_GENES, get_dims
)
from ai.gnn_model import AMRResistanceGNN, build_model
from simulation.sim_logger import SimLogger, LogLevel

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "scenarios":          ["ecoli_cipro", "klebsiella_carbapenem",
                           "pakistan_crisis", "xdr_acinetobacter"],
    "seeds_per_scenario": 3,
    "steps_per_run":      80,
    "snapshot_interval":  3,

    # Model
    "hidden_dim":    128,
    "edge_enc_dim":  64,
    "n_layers":      3,
    "heads":         4,
    "dropout":       0.15,

    # Training
    "epochs":         60,
    "batch_size":     8,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "pos_weight":     15.0,    # HGT events are rare → upweight positive class
    "grad_clip":      1.0,
    "patience":       12,      # early stopping patience (epochs)
    "min_delta":      0.001,   # minimum improvement to reset patience counter

    # Splits
    "train_frac":    0.70,
    "val_frac":      0.15,
    # test = 1 - train - val = 0.15

    # Paths
    "checkpoint_dir": "ai/checkpoints",
    "log_dir":        "logs",
}


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_labels: np.ndarray,   # (N_samples, N_GENES)
    all_probs:  np.ndarray,   # (N_samples, N_GENES)
    threshold:  float = 0.35,
) -> Dict[str, float]:
    """
    Compute per-gene and aggregate metrics.
    Handles the case where some genes have no positive examples.
    """
    metrics = {}
    aurocs, auprcs, f1s = [], [], []

    for g_idx, gene in enumerate(GENE_INDEX):
        labels_g = all_labels[:, g_idx]
        probs_g  = all_probs[:, g_idx]

        if labels_g.sum() == 0:
            # No positive examples for this gene in this batch
            metrics[f"auroc_{gene}"] = float("nan")
            metrics[f"auprc_{gene}"] = float("nan")
            metrics[f"f1_{gene}"]    = float("nan")
            continue

        try:
            auroc = roc_auc_score(labels_g, probs_g)
            auprc = average_precision_score(labels_g, probs_g)
            preds_bin = (probs_g >= threshold).astype(int)
            f1    = f1_score(labels_g, preds_bin, zero_division=0)
        except Exception:
            auroc, auprc, f1 = float("nan"), float("nan"), float("nan")

        metrics[f"auroc_{gene}"] = auroc
        metrics[f"auprc_{gene}"] = auprc
        metrics[f"f1_{gene}"]    = f1
        aurocs.append(auroc)
        auprcs.append(auprc)
        f1s.append(f1)

    # Aggregate (macro average, ignoring NaN)
    valid_aurocs = [v for v in aurocs if not np.isnan(v)]
    valid_auprcs = [v for v in auprcs if not np.isnan(v)]
    valid_f1s    = [v for v in f1s    if not np.isnan(v)]

    metrics["auroc_macro"] = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0
    metrics["auprc_macro"] = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
    metrics["f1_macro"]    = float(np.mean(valid_f1s))    if valid_f1s    else 0.0

    # Overall positive rate (how imbalanced is the data)
    metrics["positive_rate"] = float(all_labels.mean())
    metrics["n_samples"]     = int(all_labels.shape[0])

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:       AMRResistanceGNN,
    loader:      "DataLoader",
    optimizer:   Optional[AdamW],
    criterion:   nn.BCEWithLogitsLoss,
    device:      torch.device,
    is_train:    bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run one full epoch (train or eval).
    Returns: (mean_loss, all_labels, all_probs) for metric computation.
    """
    model.train(is_train)
    total_loss  = 0.0
    n_batches   = 0
    all_labels  = []
    all_probs   = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)

            if batch.y is None or batch.y.shape[0] == 0:
                continue

            logits = model(batch)             # (E, N_GENES)
            labels = batch.y                  # (E, N_GENES)

            # Skip if shapes mismatch (can happen at graph boundaries)
            if logits.shape[0] != labels.shape[0]:
                continue

            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         DEFAULT_CONFIG["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labs  = labels.detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labs)

    mean_loss  = total_loss / max(1, n_batches)
    all_probs  = np.concatenate(all_probs,  axis=0) if all_probs  else np.zeros((1, N_GENES))
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((1, N_GENES))

    return mean_loss, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_data(config: dict, logger: Optional[SimLogger] = None) -> list:
    """
    Run simulations across all configured scenarios and seeds.
    Returns list of (graph_t0, graph_t1) training pairs.
    """
    all_pairs = []
    total_runs = len(config["scenarios"]) * config["seeds_per_scenario"]
    run_num = 0

    for scenario in config["scenarios"]:
        for seed in range(config["seeds_per_scenario"]):
            run_num += 1
            t_start = time.time()

            if logger:
                logger.log(LogLevel.MILESTONE,
                    f"Collecting data [{run_num}/{total_runs}] — "
                    f"scenario: {scenario}, seed: {seed}")

            pairs = collect_training_snapshots(
                n_steps=config["steps_per_run"],
                scenario=scenario,
                seed=seed + 100,   # offset to avoid overlap with validation seeds
                snapshot_interval=config["snapshot_interval"],
            )
            all_pairs.extend(pairs)
            elapsed = time.time() - t_start

            msg = (f"  → {len(pairs)} graph pairs collected "
                   f"({elapsed:.1f}s) | running total: {len(all_pairs)}")
            print(msg)
            if logger:
                logger.log(LogLevel.STEP, msg)

    return all_pairs


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    pairs: list, config: dict
) -> Tuple["AMRGraphDataset", "AMRGraphDataset", "AMRGraphDataset"]:
    """Randomly split graph pairs into train/val/test datasets."""
    import random
    random.shuffle(pairs)
    n      = len(pairs)
    n_tr   = int(n * config["train_frac"])
    n_val  = int(n * config["val_frac"])

    tr_pairs  = pairs[:n_tr]
    val_pairs = pairs[n_tr : n_tr + n_val]
    te_pairs  = pairs[n_tr + n_val:]

    return (
        AMRGraphDataset(tr_pairs),
        AMRGraphDataset(val_pairs),
        AMRGraphDataset(te_pairs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:       AMRResistanceGNN,
    optimizer:   AdamW,
    epoch:       int,
    metrics:     dict,
    config:      dict,
    path:        str,
):
    """Save model checkpoint with all metadata needed to reproduce."""
    ckpt = {
        "epoch":        epoch,
        "metrics":      metrics,
        "config":       config,
        "model_config": {
            "hidden_dim":   config["hidden_dim"],
            "edge_enc_dim": config["edge_enc_dim"],
            "n_layers":     config["n_layers"],
            "heads":        config["heads"],
            "dropout":      config["dropout"],
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "architecture":    model.architecture_summary(),
        "gene_names":      GENE_INDEX,
        "feature_dims":    get_dims(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str) -> Tuple["AMRResistanceGNN", dict]:
    """Load checkpoint and return (model, metrics)."""
    from ai.gnn_model import build_model
    ckpt  = torch.load(path, map_location="cpu")
    model = build_model(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config:    Optional[dict] = None,
    verbose:   bool = True,
) -> Tuple["AMRResistanceGNN", dict]:
    """
    Full training pipeline:
      1. Collect simulation data
      2. Build datasets and loaders
      3. Train GNN with early stopping
      4. Evaluate on test set
      5. Save checkpoint + full metrics JSON

    Returns: (trained_model, test_metrics)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = SimLogger.start_session(
        session_name="gnn_training",
        log_dir=config["log_dir"],
        step_interval=1,
        print_to_terminal=False,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AMR GNN Training")
        print(f"  Device : {device}")
        print(f"  Epochs : {config['epochs']}")
        print(f"  Scenarios: {config['scenarios']}")
        print(f"{'='*60}\n")

    logger.log(LogLevel.MILESTONE,
        f"GNN training started | device={device} | "
        f"scenarios={config['scenarios']} | epochs={config['epochs']}")

    # ── 1. Collect data ───────────────────────────────────────────────────────
    print("Step 1/5: Collecting simulation snapshots...")
    all_pairs = collect_all_data(config, logger)
    if len(all_pairs) < 10:
        raise RuntimeError(
            f"Only {len(all_pairs)} graph pairs collected — "
            "need at least 10. Increase steps_per_run or seeds_per_scenario."
        )
    print(f"  Total pairs: {len(all_pairs)}\n")
    logger.log(LogLevel.MILESTONE, f"Data collected: {len(all_pairs)} graph pairs")

    # ── 2. Build datasets ─────────────────────────────────────────────────────
    print("Step 2/5: Building train/val/test splits...")
    tr_ds, val_ds, te_ds = split_dataset(all_pairs, config)
    print(f"  Train: {len(tr_ds)} | Val: {len(val_ds)} | Test: {len(te_ds)}\n")

    if len(tr_ds) == 0:
        raise RuntimeError("Training dataset is empty after split. Collect more data.")
    if len(val_ds) == 0:
        print("  WARNING: validation set empty — reusing train set for validation")
        val_ds = tr_ds
    if len(te_ds) == 0:
        print("  WARNING: test set empty — reusing val set for test")
        te_ds = val_ds

    def _make_loader(ds, shuffle):
        """Create DataLoader — silently returns None for empty datasets."""
        if len(ds) == 0:
            return []
        return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)

    tr_loader  = _make_loader(tr_ds,  shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    te_loader  = _make_loader(te_ds,  shuffle=False)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("Step 3/5: Building GNN model...")
    model = build_model(
        hidden_dim=config["hidden_dim"],
        edge_enc_dim=config["edge_enc_dim"],
        n_layers=config["n_layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)

    arch = model.architecture_summary()
    print(f"  Trainable parameters: {arch['trainable_params']:,}")
    print(f"  Architecture: {arch['architecture']}\n")
    logger.log(LogLevel.MILESTONE,
        f"Model built | params={arch['trainable_params']:,} | "
        f"arch={arch['architecture']}")

    # Loss: BCEWithLogitsLoss with positive class weighting
    pos_weight = torch.tensor([config["pos_weight"]] * N_GENES,
                               dtype=torch.float).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = AdamW(model.parameters(),
                       lr=config["lr"],
                       weight_decay=config["weight_decay"])
    scheduler  = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print("Step 4/5: Training...")
    print(f"{'Epoch':>6} | {'Loss_tr':>8} | {'Loss_val':>8} | "
          f"{'AUROC_val':>10} | {'AUPRC_val':>10} | {'F1_val':>8} | {'LR':>8}")
    print("-" * 72)

    best_auroc     = 0.0
    patience_count = 0
    history        = []
    best_ckpt_path = str(ckpt_dir / "best_model.pt")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        tr_loss,  tr_labels,  tr_probs  = run_epoch(
            model, tr_loader,  optimizer, criterion, device, is_train=True)
        val_loss, val_labels, val_probs = run_epoch(
            model, val_loader, None,      criterion, device, is_train=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = compute_metrics(val_labels, val_probs)
        val_auroc   = val_metrics["auroc_macro"]
        val_auprc   = val_metrics["auprc_macro"]
        val_f1      = val_metrics["f1_macro"]

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {tr_loss:>8.4f} | {val_loss:>8.4f} | "
              f"{val_auroc:>10.4f} | {val_auprc:>10.4f} | {val_f1:>8.4f} | "
              f"{current_lr:>8.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "val_loss": val_loss,
            "val_auroc": val_auroc, "val_auprc": val_auprc,
            "val_f1": val_f1, "lr": current_lr, "elapsed_s": elapsed,
        })

        logger.log(LogLevel.STEP,
            f"Epoch {epoch}/{config['epochs']} | "
            f"loss={tr_loss:.4f}/{val_loss:.4f} | "
            f"AUROC={val_auroc:.4f} | AUPRC={val_auprc:.4f} | F1={val_f1:.4f}",
            step=epoch)

        # Checkpoint if best
        if val_auroc > best_auroc + config["min_delta"]:
            best_auroc     = val_auroc
            patience_count = 0
            save_checkpoint(model, optimizer, epoch,
                            val_metrics, config, best_ckpt_path)
            print(f"  ★ New best AUROC: {best_auroc:.4f} → saved checkpoint")
            logger.log(LogLevel.MILESTONE,
                f"New best model | AUROC={best_auroc:.4f} | "
                f"epoch={epoch} → {best_ckpt_path}", step=epoch)
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {config['patience']} epochs)")
                logger.log(LogLevel.MILESTONE,
                    f"Early stopping at epoch {epoch} | "
                    f"best AUROC={best_auroc:.4f}", step=epoch)
                break

    # ── 5. Test evaluation ───────────────────────────────────────────────────
    print(f"\nStep 5/5: Evaluating best model on test set...")

    # Load best checkpoint
    best_model, _ = load_checkpoint(best_ckpt_path)
    best_model = best_model.to(device)

    _, te_labels, te_probs = run_epoch(
        best_model, te_loader, None, criterion, device, is_train=False)
    test_metrics = compute_metrics(te_labels, te_probs)

    print(f"\n  Test AUROC (macro): {test_metrics['auroc_macro']:.4f}")
    print(f"  Test AUPRC (macro): {test_metrics['auprc_macro']:.4f}")
    print(f"  Test F1    (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Positive rate:      {test_metrics['positive_rate']:.4f}")
    print(f"\n  Per-gene AUROC:")
    for gene in GENE_INDEX:
        v = test_metrics.get(f"auroc_{gene}", float("nan"))
        bar = "█" * int(v * 20) if not np.isnan(v) else "N/A"
        print(f"    {gene:<20} {bar:<22} {v:.3f}" if not np.isnan(v) else f"    {gene:<20} (no positives in test set)")

    # Save full results
    results_path = str(ckpt_dir / "training_results.json")
    results = {
        "test_metrics":  test_metrics,
        "best_val_auroc": best_auroc,
        "history":        history,
        "config":         config,
        "architecture":   best_model.architecture_summary(),
        "gene_names":     GENE_INDEX,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path}")
    print(f"  Best model:    {best_ckpt_path}")
    print(f"{'='*60}\n")

    logger.log(LogLevel.MILESTONE,
        f"Training complete | test_AUROC={test_metrics['auroc_macro']:.4f} | "
        f"test_AUPRC={test_metrics['auprc_macro']:.4f} | "
        f"results={results_path}")
    logger.log_session_summary(0, config["epochs"], {})
    logger.close()

    return best_model, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TRAINING (reduced config for testing)
# ─────────────────────────────────────────────────────────────────────────────

QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    "scenarios":          ["ecoli_cipro", "klebsiella_carbapenem"],
    "seeds_per_scenario": 2,
    "steps_per_run":      40,
    "epochs":             20,
    "patience":           5,
    "batch_size":         4,
}


def quick_train() -> Tuple["AMRResistanceGNN", dict]:
    """Fast training run for testing/CI."""
    return train(config=QUICK_CONFIG, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AMR GNN")
    parser.add_argument("--quick",   action="store_true", help="Quick run (reduced data/epochs)")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    args = parser.parse_args()

    cfg = QUICK_CONFIG.copy() if args.quick else DEFAULT_CONFIG.copy()
    if args.epochs:    cfg["epochs"]    = args.epochs
    if args.scenarios: cfg["scenarios"] = args.scenarios

    model, metrics = train(config=cfg)
    print(f"Done. Final test AUROC: {metrics['auroc_macro']:.4f}")