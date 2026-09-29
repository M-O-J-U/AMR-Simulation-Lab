"""
GNN-Native Ablation Study.

Unlike ai/baselines.py's run_ablation() (which masks feature columns and
trains a cheap Logistic Regression proxy per condition, then reports LR's
AUROC drop as a stand-in for feature importance), this module retrains
the ACTUAL AMRResistanceGNN architecture from scratch for each masked
condition and reports the real model's test AUROC drop.

Why this matters and why the LR-proxy version is not sufficient on its
own: a linear model's sensitivity to a masked feature does not
necessarily reflect a graph attention network's sensitivity to the same
feature, since the GNN can route information through message passing in
ways an LR proxy structurally cannot. Prior LR-proxy ablation on this
codebase showed "No behavioral" (biofilm/persister) IMPROVING AUROC
(+0.0316) and "No genomic genes" dominating all other groups by a wide
margin (-0.1173) — if the paper's discussion section claims the GNN's
advantage over flat baselines comes from spatial/behavioral graph
structure, the LR-proxy ablation actively contradicts that claim. This
module answers the question directly on the real architecture instead of
inferring it from a linear stand-in.

Cost: this retrains a full AMRResistanceGNN once per condition (9
conditions: all-features baseline + 8 masked groups). At full
DEFAULT_CONFIG scale (60 epochs, early stopping, 3 seeds/scenario) this
is 9x the cost of one normal training run. A REDUCED_ABLATION_CONFIG is
provided for faster, honestly-labeled reduced-scope runs, following the
same "flag reduced scope explicitly" discipline used throughout this
codebase's remediation.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import (
    collect_training_snapshots, AMRGraphDataset,
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GENE_INDEX, N_GENES,
)
from ai.gnn_trainer import (
    DEFAULT_CONFIG, split_dataset, _train_core, collect_all_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP SLICES
#
# These MUST match extract_node_features()/extract_edge_features()'s
# actual column order in ai/feature_engineering.py exactly, and MUST stay
# in sync with NodeEncoder's GENE_SLICE/PHYSIO_SLICE/etc. in
# ai/gnn_model.py. If those extraction functions are ever reordered
# without updating both this dict and NodeEncoder's slices, ablation
# results here will silently mask the wrong columns.
# ─────────────────────────────────────────────────────────────────────────────

NODE_GROUPS: Dict[str, Optional[slice]] = {
    "All features (full model)": None,
    "No genomic genes":          slice(0,  10),
    "No physiological":          slice(10, 15),
    "No behavioral":             slice(15, 17),
    "No spatial position":       slice(17, 19),
    "No population":             slice(19, 22),
    "No species":                slice(22, 27),
    "No gram stain":              slice(27, 29),
    "No antibiotic exposure":    slice(29, 35),
}

# All 5 edge features are masked together as one condition, matching the
# granularity of ai/baselines.py's LR-proxy "No edge features" category
# for direct comparability between the two ablation methodologies.
EDGE_GROUP_SLICE = slice(0, 5)


# ─────────────────────────────────────────────────────────────────────────────
# REDUCED-SCOPE CONFIG (explicit, not hidden)
# ─────────────────────────────────────────────────────────────────────────────

REDUCED_ABLATION_CONFIG = {
    **DEFAULT_CONFIG,
    "seeds_per_scenario": 2,
    "epochs":             30,
    "patience":           8,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET MASKING
# ─────────────────────────────────────────────────────────────────────────────

def mask_dataset(
    ds,
    node_slice: Optional[slice] = None,
    edge_slice: Optional[slice] = None,
) -> list:
    """
    Return a NEW list of PyG Data objects with the specified node-feature
    and/or edge-feature columns zeroed, WITHOUT mutating the original
    dataset's tensors (same discipline as split_dataset() — mutating a
    caller's data in place is exactly the kind of silent bug this whole
    remediation effort has been rooting out).

    node_slice/edge_slice = None means "don't mask this dimension".
    Both None returns unmodified clones (used for the "all features"
    baseline condition, so every condition — including the baseline —
    goes through the identical clone-and-rebuild path).
    """
    from torch_geometric.data import Data

    masked = []
    for data in ds:
        x         = data.x.clone()
        edge_attr = data.edge_attr.clone()

        if node_slice is not None:
            x[:, node_slice] = 0.0
        if edge_slice is not None:
            edge_attr[:, edge_slice] = 0.0

        new_data = Data(
            x=x, edge_index=data.edge_index.clone(),
            edge_attr=edge_attr, y=data.y.clone(),
        )
        if hasattr(data, "metadata"):
            new_data.metadata = data.metadata
        masked.append(new_data)

    return masked


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ABLATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_gnn_ablation(
    config:            Optional[dict] = None,
    include_edge_group: bool = True,
    verbose:           bool = True,
) -> Dict[str, dict]:
    """
    Retrain the actual AMRResistanceGNN once per feature-group-masked
    condition and report real test AUROC/AUPRC/F1 for each, with delta
    vs. the "all features" baseline condition.

    Every condition uses the SAME train/val/test split (same split_seed,
    same underlying collected data) so the comparison is apples-to-apples
    — this reuses the split_dataset() seeding fix from earlier in this
    codebase's remediation; without it, each condition would silently
    evaluate against a different held-out set, invalidating the whole
    comparison the same way the original baselines.py bug did.

    Returns: {condition_name: {auroc, auprc, f1, delta_auroc, n_epochs_trained}}
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    ckpt_dir = Path(config["checkpoint_dir"]) / "_ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GNN-Native Ablation Study")
        print(f"  Retraining the actual AMRResistanceGNN per condition")
        print(f"  Epochs: {config['epochs']} | Patience: {config['patience']} | "
              f"Seeds/scenario: {config['seeds_per_scenario']}")
        print(f"{'='*60}\n")

    print("Collecting simulation data (shared across all conditions)...")
    t0 = time.time()
    all_pairs = collect_all_data(config, logger=None)
    if len(all_pairs) < 10:
        raise RuntimeError(
            f"Only {len(all_pairs)} graph pairs collected — need at least 10."
        )
    print(f"  Collected {len(all_pairs)} pairs in {time.time()-t0:.1f}s\n")

    # Same split_seed (from config, default 42) used for every condition —
    # this is the critical piece that makes the comparison valid.
    tr_ds, val_ds, te_ds = split_dataset(all_pairs, config)
    print(f"Split: {len(tr_ds)} train / {len(val_ds)} val / {len(te_ds)} test "
          f"(split_seed={config.get('split_seed', 42)}, identical for every condition)\n")

    conditions: List[Tuple[str, Optional[slice], Optional[slice]]] = [
        (name, node_slice, None) for name, node_slice in NODE_GROUPS.items()
    ]
    if include_edge_group:
        conditions.append(("No edge features", None, EDGE_GROUP_SLICE))

    results: Dict[str, dict] = {}
    baseline_auroc: Optional[float] = None

    print(f"{'Condition':<32} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'ΔAUROC':>9} {'Epochs':>7} {'Time(s)':>8}")
    print("-" * 92)

    for cond_idx, (name, node_slice, edge_slice) in enumerate(conditions):
        t_cond = time.time()

        masked_tr  = mask_dataset(tr_ds,  node_slice, edge_slice)
        masked_val = mask_dataset(val_ds, node_slice, edge_slice)
        masked_te  = mask_dataset(te_ds,  node_slice, edge_slice)

        safe_name  = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        ckpt_path  = str(ckpt_dir / f"model_{safe_name}.pt")

        _, test_metrics, history = _train_core(
            masked_tr, masked_val, masked_te, config, ckpt_path,
            verbose=False, print_steps=False,
        )

        auroc = test_metrics["auroc_macro"]
        auprc = test_metrics["auprc_macro"]
        f1    = test_metrics["f1_macro"]
        n_epochs = len(history)
        elapsed  = time.time() - t_cond

        if baseline_auroc is None:
            baseline_auroc = auroc
        delta = auroc - baseline_auroc

        delta_str = f"{delta:+.4f}" if cond_idx > 0 else "baseline"
        print(f"{name:<32} {auroc:>8.4f} {auprc:>8.4f} {f1:>8.4f} "
              f"{delta_str:>9} {n_epochs:>7} {elapsed:>8.1f}")

        results[name] = {
            "auroc_macro": auroc, "auprc_macro": auprc, "f1_macro": f1,
            "delta_auroc_vs_full": delta,
            "n_epochs_trained": n_epochs,
            "elapsed_s": elapsed,
            "checkpoint": ckpt_path,
        }

    results_path = str(Path(config["checkpoint_dir"]) / "gnn_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "results": results,
            "config": config,
            "methodology": (
                "GNN-native: each row is a FULL AMRResistanceGNN retrained "
                "from scratch with the named feature group zeroed in both "
                "node and/or edge features, evaluated on the SAME held-out "
                "test set (same split_seed) as every other condition. This "
                "is NOT the LR-proxy ablation in ai/baselines.py — compare "
                "the two explicitly rather than treating either alone as "
                "definitive."
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)

    print(f"\n  Saved: {results_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN-native ablation study")
    parser.add_argument("--full", action="store_true",
                       help="Use DEFAULT_CONFIG (60 epochs, 3 seeds) instead of reduced scope")
    parser.add_argument("--no-edge", action="store_true",
                       help="Skip the edge-features-masked condition")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy() if args.full else REDUCED_ABLATION_CONFIG.copy()
    if not args.full:
        print("NOTE: running REDUCED_ABLATION_CONFIG (2 seeds, 30 epochs). "
              "Use --full for submission-grade numbers (9x the training cost).")

    run_gnn_ablation(config=cfg, include_edge_group=not args.no_edge)