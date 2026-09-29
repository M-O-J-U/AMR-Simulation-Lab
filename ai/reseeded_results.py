"""
Reseeded results for the new paper: every headline number as mean ± SD over
model seeds, from the corrected (post-leakage-fix, post-seeding-fix) pipeline.

    python -m ai.reseeded_results                 # full: 5 model seeds, 3 ablation seeds
    python -m ai.reseeded_results --quick         # smoke test (tiny config)

What is fixed vs varied
  - DATA is fixed: DEFAULT_CONFIG simulation runs (4 scenarios x data seeds
    100-102, 80 steps), deterministic across processes since the 2026-09-29
    seeding fixes; one train/val/test split (split_seed 42). A SHA-256 of all
    graph tensors is recorded so the dataset is identifiable.
  - MODEL SEEDS vary: s = 0..N-1. GNN torch_seed = 1000+s; RF and LR
    random_state = s; sklearn training subsample seed = s. The frequency
    baseline is deterministic (SD 0).
  - GNN training on CUDA is not guaranteed bitwise-reproducible (PyG scatter
    ops use atomics). The ablation's "All features" condition re-trains each
    seed with the same torch_seed as the main run, and the difference is
    reported as a measured reproducibility check.

Outputs: ai/checkpoints/reseeded/results.json, summary.md, models/gnn_seed*.pt
The older single-run files in ai/checkpoints/*.json are NOT modified.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import GENE_INDEX, N_GENES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from ai.gnn_trainer import (DEFAULT_CONFIG, collect_all_data, split_dataset, _train_core,
                            run_epoch, compute_metrics)
from ai.baselines import run_baselines, calibrated_macro_f1
from ai.threshold_calibration import compute_calibration
from ai.gnn_ablation import mask_dataset, NODE_GROUPS, EDGE_GROUP_SLICE

OUT_DIR = Path("ai/checkpoints/reseeded")

QUICK_CONFIG = {**DEFAULT_CONFIG, "scenarios": ["pakistan_crisis"], "seeds_per_scenario": 1,
                "steps_per_run": 30, "epochs": 2, "patience": 2}


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finite(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def agg(xs) -> dict:
    """mean ± sample SD (ddof=1) over seeds, ignoring NaN (e.g. genes with
    no test positives); n = number of finite values."""
    v = _finite(xs)
    if not v:
        return {"mean": None, "sd": None, "n": 0, "min": None, "max": None, "values": list(xs)}
    return {"mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n": len(v), "min": float(min(v)), "max": float(max(v)), "values": list(xs)}


def dataset_fingerprint(pairs) -> dict:
    h = hashlib.sha256()
    n_edges = n_pos = 0
    for g0, _ in pairs:
        for k in ("node_features", "edge_index", "edge_features", "gene_labels"):
            a = g0[k]
            h.update(k.encode()); h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        n_edges += int(g0["edge_index"].shape[1])
        n_pos += int(np.asarray(g0["gene_labels"]).sum())
    return {"sha256": h.hexdigest(), "n_graph_pairs": len(pairs),
            "n_edges": n_edges, "n_edge_gene_positives": n_pos}


def evaluate_gnn(model, te_ds, config, device) -> dict:
    from torch_geometric.loader import DataLoader
    loader = DataLoader(te_ds, batch_size=config["batch_size"], shuffle=False)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config["pos_weight"]] * N_GENES).to(device))
    model.to(device).eval()
    _, labels, probs = run_epoch(model, loader, None, crit, device, is_train=False)
    m = compute_metrics(labels, probs)
    f1 = calibrated_macro_f1(labels, probs)
    cal = compute_calibration(labels, probs)
    return {
        "auroc_macro": m["auroc_macro"], "auprc_macro": m["auprc_macro"],
        "f1_macro": f1["f1_macro"], "precision_macro": f1["precision_macro"],
        "recall_macro": f1["recall_macro"],
        "ece": cal["macro_ece"],
        "per_gene_auroc": {g: m.get(f"auroc_{g}", float("nan")) for g in GENE_INDEX},
        "n_test_edges": int(labels.shape[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, model_seeds, ablation_seeds, out_dir: Path = OUT_DIR) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

    log(f"collecting data: {config['scenarios']} x {config['seeds_per_scenario']} data seeds, "
        f"{config['steps_per_run']} steps")
    pairs = collect_all_data(config)
    fp = dataset_fingerprint(pairs)
    log(f"dataset: {fp}")
    tr, val, te = split_dataset(pairs, config)
    log(f"split: train {len(tr)} / val {len(val)} / test {len(te)} graphs (split_seed "
        f"{config.get('split_seed', 42)})")

    per_seed = []
    for s in model_seeds:
        log(f"--- model seed {s}: GNN (torch_seed {1000+s})")
        ck = str(out_dir / "models" / f"gnn_seed{s}.pt")
        model, _, hist = _train_core(tr, val, te, config, ck, device=device, verbose=False,
                                     print_steps=False, torch_seed=1000 + s)
        gnn = evaluate_gnn(model, te, config, device)
        gnn["epochs_trained"] = len(hist)
        log(f"    GNN AUROC {gnn['auroc_macro']:.4f} AUPRC {gnn['auprc_macro']:.4f}")
        log(f"--- model seed {s}: baselines (RF/LR random_state {s}, subsample_seed {s})")
        bl = run_baselines(tr, te, results_path=str(tmp / "bl.json"),
                           subsample_seed=s, model_seed=s, include_gnn=False)
        per_seed.append({"seed": s, "gnn": gnn, **{k: v for k, v in bl.items()}})
        log(f"    RF {bl['random_forest']['auroc_macro']:.4f} | "
            f"LR {bl['logistic_regression']['auroc_macro']:.4f} | "
            f"freq {bl['frequency_baseline']['auroc_macro']:.4f}")

    models = ["gnn", "random_forest", "logistic_regression", "frequency_baseline"]
    metrics = ["auroc_macro", "auprc_macro", "f1_macro", "precision_macro", "recall_macro"]
    summary = {m: {k: agg([r[m].get(k) for r in per_seed]) for k in metrics} for m in models}
    summary["gnn"]["ece"] = agg([r["gnn"]["ece"] for r in per_seed])
    per_gene = {m: {g: agg([r[m]["per_gene_auroc"].get(g) for r in per_seed]) for g in GENE_INDEX}
                for m in models}
    diffs = [r["gnn"]["auroc_macro"] - r["random_forest"]["auroc_macro"] for r in per_seed]
    paired = {"gnn_minus_rf_auroc": agg(diffs),
              "seeds_gnn_better": int(sum(d > 0 for d in diffs)), "n_seeds": len(diffs)}

    # ── Ablation ────────────────────────────────────────────────────────────
    conditions = [(n, sl, None) for n, sl in NODE_GROUPS.items()]
    conditions.append(("No edge features", None, EDGE_GROUP_SLICE))
    abl_raw = {n: [] for n, _, _ in conditions}
    for s in ablation_seeds:
        for name, ns, es in conditions:
            log(f"--- ablation seed {s}: {name}")
            mtr, mval, mte = (mask_dataset(d, ns, es) for d in (tr, val, te))
            model, _, _ = _train_core(mtr, mval, mte, config, str(tmp / "abl.pt"), device=device,
                                      verbose=False, print_steps=False, torch_seed=1000 + s)
            abl_raw[name].append(evaluate_gnn(model, mte, config, device)["auroc_macro"])
    base = abl_raw["All features (full model)"]
    ablation = {n: {"auroc": agg(v),
                    "delta_vs_all": agg([a - b for a, b in zip(v, base)])}
                for n, v in abl_raw.items()}

    # Measured GPU reproducibility: same data, same torch_seed, trained twice
    main_by_seed = {r["seed"]: r["gnn"]["auroc_macro"] for r in per_seed}
    repro = [abs(a - main_by_seed[s]) for s, a in zip(ablation_seeds, base) if s in main_by_seed]

    shutil.rmtree(tmp, ignore_errors=True)
    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "device": str(device), "torch": torch.__version__,
        "feature_dims": {"node": NODE_FEATURE_DIM, "edge": EDGE_FEATURE_DIM},
        "config": {k: v for k, v in config.items()},
        "model_seeds": list(model_seeds), "ablation_seeds": list(ablation_seeds),
        "dataset": fp,
        "summary": summary, "per_gene_auroc": per_gene, "paired_gnn_vs_rf": paired,
        "ablation": ablation,
        "reproducibility_check": {
            "what": "|AUROC(main run) - AUROC(ablation 'All features' rerun)|, same data and torch_seed",
            "abs_diffs": repro, "max_abs_diff": max(repro) if repro else None},
        "per_seed": per_seed,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    (out_dir / "summary.md").write_text(render_summary(results), encoding="utf-8")
    log(f"done -> {out_dir/'results.json'}, {out_dir/'summary.md'}")
    return results


def _ms(a, digits=4):
    return "n/a" if a["mean"] is None else f"{a['mean']:.{digits}f} ± {a['sd']:.{digits}f} (n={a['n']})"


def render_summary(r: dict) -> str:
    L = [f"# Reseeded results ({r['generated']})", "",
         f"Model seeds {r['model_seeds']}; ablation seeds {r['ablation_seeds']}; "
         f"device {r['device']}; torch {r['torch']}; features {r['feature_dims']}.",
         f"Dataset sha256 `{r['dataset']['sha256'][:16]}…`: {r['dataset']['n_graph_pairs']} graph "
         f"pairs, {r['dataset']['n_edges']:,} edges, {r['dataset']['n_edge_gene_positives']:,} "
         f"edge-gene positives.", "", "## Model comparison (test set, mean ± SD over seeds)", "",
         "| Model | AUROC | AUPRC | F1 (calibrated) |", "|---|---|---|---|"]
    names = {"gnn": "AMRResistanceGNN", "random_forest": "Random Forest",
             "logistic_regression": "Logistic Regression", "frequency_baseline": "Frequency baseline"}
    for m, label in names.items():
        s = r["summary"][m]
        L.append(f"| {label} | {_ms(s['auroc_macro'])} | {_ms(s['auprc_macro'])} | {_ms(s['f1_macro'])} |")
    p = r["paired_gnn_vs_rf"]
    L += ["", f"GNN − RF AUROC (paired by seed): {_ms(p['gnn_minus_rf_auroc'])}; "
              f"GNN better on {p['seeds_gnn_better']}/{p['n_seeds']} seeds.",
          f"GNN ECE: {_ms(r['summary']['gnn']['ece'])}.", "",
          "## Per-gene AUROC (GNN vs RF)", "", "| Gene | GNN | RF |", "|---|---|---|"]
    for g in r["per_gene_auroc"]["gnn"]:
        L.append(f"| {g} | {_ms(r['per_gene_auroc']['gnn'][g])} | {_ms(r['per_gene_auroc']['random_forest'][g])} |")
    L += ["", "## GNN feature-group ablation (retrained per condition)", "",
          "| Condition | AUROC | Δ vs all features (paired) |", "|---|---|---|"]
    for n, a in r["ablation"].items():
        L.append(f"| {n} | {_ms(a['auroc'])} | {_ms(a['delta_vs_all'])} |")
    rc = r["reproducibility_check"]
    L += ["", f"Reproducibility check ({rc['what']}): max |Δ| = {rc['max_abs_diff']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        run(QUICK_CONFIG, range(2), range(1), out_dir=Path("ai/checkpoints/_reseeded_quick"))
    else:
        run(DEFAULT_CONFIG, range(a.seeds), range(a.ablation_seeds))
