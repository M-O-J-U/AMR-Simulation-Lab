"""
Small hyperparameter sweep for the GNN and Random Forest, to test whether
"edge features hurt, GNN ≈ RF" survives retuning.

    python -m ai.hparam_sweep --part gnn     # GPU
    python -m ai.hparam_sweep --part rf      # CPU (can run alongside --part gnn)
    python -m ai.hparam_sweep --summarize

Protocol (fixed before looking at any results)
  - Same fixed dataset and split as ai/reseeded_results.py (DEFAULT_CONFIG,
    split_seed 42); dataset SHA-256 recorded.
  - Selection uses VALIDATION AUROC only (never test):
      GNN configs: mean over selection seeds 0, 1 (torch_seed 1000+s)
      RF configs:  selection seed 0 (random_state 0, subsample_seed 0)
  - Final: the selected config of each arm on TEST, model seeds 0-4, with the
    same seed mapping as the reported numbers (GNN torch_seed 1000+s; RF
    random_state s and subsample_seed s).
  - Arms:
      gnn_full        hidden {64,128} x lr {3e-4,1e-3} x GAT layers {2,3}
      gnn_no_edge     same grid, edge-feature vectors zeroed (message passing kept)
      graph_free      n_layers = 0: each edge scored from its two endpoint node
                      encodings + edge features, no message passing;
                      hidden {64,128} x lr {3e-4,1e-3}
      rf              n_estimators {100,300} x max_depth {8,16,None}
Outputs: ai/checkpoints/sweep/<biology>/{gnn,rf}.json and summary.md
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.gnn_trainer import DEFAULT_CONFIG, collect_all_data, split_dataset, _train_core
from ai.baselines import run_baselines
from ai.gnn_ablation import mask_dataset, EDGE_GROUP_SLICE
from ai.reseeded_results import evaluate_gnn, dataset_fingerprint, agg, add_headline

SELECT_SEEDS_GNN = [0, 1]
SELECT_SEEDS_RF = [0]
FINAL_SEEDS = [0, 1, 2, 3, 4]

GNN_GRID = [{"hidden_dim": h, "lr": lr, "n_layers": nl}
            for h, lr, nl in itertools.product([64, 128], [3e-4, 1e-3], [2, 3])]
FREE_GRID = [{"hidden_dim": h, "lr": lr, "n_layers": 0}
             for h, lr in itertools.product([64, 128], [3e-4, 1e-3])]
RF_GRID = [{"n_estimators": n, "max_depth": d}
           for n, d in itertools.product([100, 300], [8, 16, None])]


TAG = None      # set from --tag; output folder suffix
ARMS = None     # set from --arms; None = all GNN arms
METRIC = "auroc_macro"   # selection/report metric; --headline switches to the ai/eval_genes.py policy


def out_dir(config) -> Path:
    d = Path("ai/checkpoints/sweep") / (config.get("biology", "paper_v1") + (f"_{TAG}" if TAG else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(config):
    pairs = collect_all_data(config)
    tr, val, te = split_dataset(pairs, config)
    return dataset_fingerprint(pairs), tr, val, te


def run_gnn(config):
    t0 = time.time(); od = out_dir(config); tmp = od / "_tmp.pt"
    log = lambda *a: print(f"[gnn {time.time()-t0:7.1f}s]", *a, flush=True)
    fp, tr, val, te = load(config)
    log(f"dataset {fp['sha256'][:12]} | train {len(tr)} val {len(val)} test {len(te)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noedge = tuple(mask_dataset(d, None, EDGE_GROUP_SLICE) for d in (tr, val, te))
    arms = {"gnn_full": (GNN_GRID, (tr, val, te)), "gnn_no_edge": (GNN_GRID, noedge),
            "graph_free": (FREE_GRID, (tr, val, te)),
            "graph_free_no_edge": (FREE_GRID, noedge)}   # no message passing AND no edge features
    arms = {k: v for k, v in arms.items() if (ARMS is None and k != "graph_free_no_edge") or (ARMS and k in ARMS)}

    def train(hp, data, seed, eval_on):
        cfg = {**config, **hp}
        model, _, hist = _train_core(*data, cfg, str(tmp), device=device, verbose=False,
                                     print_steps=False, torch_seed=1000 + seed)
        ds = data[1] if eval_on == "val" else data[2]
        return evaluate_gnn(model, ds, cfg, device), len(hist)

    results = {"dataset": fp, "protocol": {"select_seeds": SELECT_SEEDS_GNN, "final_seeds": FINAL_SEEDS},
               "arms": {}}
    for arm, (grid, data) in arms.items():
        sel = []
        for hp in grid:
            vals = [train(hp, data, s, "val")[0][METRIC] for s in SELECT_SEEDS_GNN]
            sel.append({"hp": hp, "val_auroc": vals, "val_mean": float(np.mean(vals))})
            log(f"{arm} {hp} val AUROC {np.mean(vals):.4f} {[round(v, 4) for v in vals]}")
        best = max(sel, key=lambda r: r["val_mean"])
        finals = []
        for s in FINAL_SEEDS:
            m, ep = train(best["hp"], data, s, "test")
            finals.append({"seed": s, "epochs": ep, **{k: m.get(k) for k in
                           ("auroc_macro", "auprc_macro", "f1_macro", "headline_auroc", "headline_auprc")},
                           "per_gene_auroc": m["per_gene_auroc"]})
            log(f"{arm} FINAL seed {s}: test {METRIC} {m[METRIC]:.4f} (epochs {ep})")
        results["arms"][arm] = {
            "selection": sel, "best_hp": best["hp"],
            "test": {k: agg([f[k] for f in finals]) for k in
                     ("auroc_macro", "auprc_macro", "f1_macro", "headline_auroc", "headline_auprc")},
            "final_runs": finals}
        results["selection_metric"] = METRIC
        json.dump(results, open(od / "gnn.json", "w"), indent=2, default=str)
    tmp.unlink(missing_ok=True)
    log("done")


def run_rf(config):
    t0 = time.time(); od = out_dir(config)
    log = lambda *a: print(f"[rf  {time.time()-t0:7.1f}s]", *a, flush=True)
    fp, tr, val, te = load(config)
    log(f"dataset {fp['sha256'][:12]}")
    def fit(hp, s, ds):
        r = run_baselines(tr, ds, results_path=str(od / "_rf_tmp.json"), subsample_seed=s,
                          model_seed=s, include_gnn=False, rf_params=hp, models=("random_forest",))
        return add_headline(r["random_forest"])
    sel = []
    for hp in RF_GRID:
        vals = [fit(hp, s, val)[METRIC] for s in SELECT_SEEDS_RF]
        sel.append({"hp": hp, "val_auroc": vals, "val_mean": float(np.mean(vals))})
        log(f"rf {hp} val AUROC {np.mean(vals):.4f}")
    best = max(sel, key=lambda r: r["val_mean"])
    finals = []
    for s in FINAL_SEEDS:
        m = fit(best["hp"], s, te)
        finals.append({"seed": s, **{k: m.get(k) for k in
                       ("auroc_macro", "auprc_macro", "f1_macro", "headline_auroc", "headline_auprc")},
                       "per_gene_auroc": m["per_gene_auroc"]})
        log(f"rf FINAL seed {s}: test {METRIC} {m[METRIC]:.4f}")
    res = {"dataset": fp, "protocol": {"select_seeds": SELECT_SEEDS_RF, "final_seeds": FINAL_SEEDS},
           "selection": sel, "best_hp": best["hp"],
           "test": {k: agg([f[k] for f in finals]) for k in
                    ("auroc_macro", "auprc_macro", "f1_macro", "headline_auroc", "headline_auprc")},
           "final_runs": finals, "selection_metric": METRIC}
    json.dump(res, open(od / "rf.json", "w"), indent=2, default=str)
    (od / "_rf_tmp.json").unlink(missing_ok=True)
    log("done")


def summarize(config):
    od = out_dir(config)
    g = json.load(open(od / "gnn.json")); r = json.load(open(od / "rf.json"))
    assert g["dataset"]["sha256"] == r["dataset"]["sha256"], "GNN and RF sweeps used different data"
    metric = g.get("selection_metric", "auroc_macro")
    ms = lambda a: "n/a" if a["mean"] is None else f"{a['mean']:.4f} ± {a['sd']:.4f}"
    L = [f"# Hyperparameter sweep — {od.name}", "",
         f"Dataset sha256 `{g['dataset']['sha256'][:16]}…`. Selection on validation `{metric}`; "
         f"final = selected config on test, seeds {FINAL_SEEDS}.", "",
         f"| Arm | Selected hyperparameters | Test {metric} | Test AUPRC |", "|---|---|---|---|"]
    auprc = "headline_auprc" if metric == "headline_auroc" else "auprc_macro"
    arms = dict(g["arms"]); arms["rf"] = r
    for arm, a in arms.items():
        L.append(f"| {arm} | {a['best_hp']} | {ms(a['test'][metric])} | {ms(a['test'][auprc])} |")
    vals = {arm: [f[metric] for f in a["final_runs"]] for arm, a in arms.items()}
    L.append("")
    L.append("Per-seed: " + "; ".join(f"{arm} {[round(v, 4) for v in v_]}" for arm, v_ in vals.items()))
    names = list(vals)
    for i_ in range(len(names)):
        for j_ in range(i_ + 1, len(names)):
            d = [a - b for a, b in zip(vals[names[i_]], vals[names[j_]])]
            L.append(f"\n{names[i_]} − {names[j_]} (paired by seed): {ms(agg(d))}; "
                     f"first better on {sum(x > 0 for x in d)}/{len(d)} seeds.")
    (od / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["gnn", "rf"])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--biology", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--arms", default=None, help="comma-separated GNN arms")
    ap.add_argument("--headline", action="store_true",
                    help="select/report on the ai/eval_genes.py headline macro")
    a = ap.parse_args()
    TAG = a.tag
    ARMS = a.arms.split(",") if a.arms else None
    if a.headline:
        METRIC = "headline_auroc"
    cfg = dict(DEFAULT_CONFIG)
    if a.biology:
        cfg["biology"] = a.biology
    if a.summarize: summarize(cfg)
    elif a.part == "gnn": run_gnn(cfg)
    elif a.part == "rf": run_rf(cfg)
