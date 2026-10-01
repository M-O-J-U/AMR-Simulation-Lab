"""
Reseeded results for the new paper: every headline number as mean ± SD over
model seeds, from the corrected (post-leakage-fix, post-seeding-fix) pipeline.

    python -m ai.reseeded_results                          # defaults
    python -m ai.reseeded_results --biology lab_v2 \\
        --gnn-hparams '{"hidden_dim":128,"lr":0.001,"n_layers":2}' \\
        --rf-hparams '{"n_estimators":300,"max_depth":8}' \\
        --no-edge --graph-free --tag tuned_noedge           # tuned, edge features zeroed
    python -m ai.reseeded_results --quick                  # smoke test

What is fixed vs varied
  - DATA is fixed: DEFAULT_CONFIG simulation runs (scenarios_for(config) x
    data seeds 100-102), deterministic across processes since the 2026-09-29
    seeding fixes; one train/val/test split (split_seed 42). A SHA-256 of all
    graph tensors is recorded so the dataset is identifiable.
  - MODEL SEEDS vary: s = 0..N-1. GNN torch_seed = 1000+s; RF and LR
    random_state = s; sklearn training subsample seed = s. The frequency
    baseline is deterministic (SD 0).
  - --no-edge zeroes the 5 edge-feature columns for EVERY GNN run (main runs,
    graph-free arm and all ablation conditions); message passing is kept.
  - --graph-free adds an arm with n_layers = 0 (no message passing) on the
    same inputs, paired by seed, to measure what message passing adds.
  - Headline macros follow ai/eval_genes.py (mexAB-oprM excluded; vanA
    reported on its own labelled line); the raw all-gene macro is kept too.
  - GNN training on CUDA is not guaranteed bitwise-reproducible (PyG scatter
    ops use atomics). The ablation's "All features" condition re-trains each
    seed with the same torch_seed as the main run, and the difference is
    reported as a measured reproducibility check.

Outputs: ai/checkpoints/reseeded/<biology>[_<tag>]/{results.json, summary.md, models/}
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
                            run_epoch, compute_metrics, dosing_for, scenarios_for)
from ai.baselines import run_baselines, calibrated_macro_f1
from ai.threshold_calibration import compute_calibration
from ai.gnn_ablation import mask_dataset, NODE_GROUPS, EDGE_GROUP_SLICE
from ai.eval_genes import EXCLUDED, SEPARATE, HEADLINE_GENES, headline_macro


def out_dir_for(config, tag: str = None) -> Path:
    name = config.get("biology", "paper_v1") + (f"_{tag}" if tag else "")
    return Path("ai/checkpoints/reseeded") / name

QUICK_CONFIG = {**DEFAULT_CONFIG, "scenarios": ["pakistan_crisis"], "seeds_per_scenario": 1,
                "steps_per_run": 30, "epochs": 2, "patience": 2, "min_epochs": 0}


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
    pos_per_gene = np.zeros(N_GENES, dtype=int)
    for g0, _ in pairs:
        for k in ("node_features", "edge_index", "edge_features", "gene_labels"):
            a = g0[k]
            h.update(k.encode()); h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        n_edges += int(g0["edge_index"].shape[1])
        lab = np.asarray(g0["gene_labels"])
        n_pos += int(lab.sum())
        if lab.size:
            pos_per_gene += lab.sum(0).astype(int)
    return {"sha256": h.hexdigest(), "n_graph_pairs": len(pairs),
            "n_edges": n_edges, "n_edge_gene_positives": n_pos,
            "positives_per_gene": dict(zip(GENE_INDEX, pos_per_gene.tolist()))}


def add_headline(m: dict) -> dict:
    """Attach headline macros (ai/eval_genes.py policy) to a per-model metrics dict."""
    for metric in ("auroc", "auprc"):
        pg = m.get(f"per_gene_{metric}")
        if pg is not None:
            m[f"headline_{metric}"], m["headline_n_genes"] = headline_macro(pg)
    return m


def evaluate_gnn(model, te_ds, config, device) -> dict:
    from torch_geometric.loader import DataLoader
    loader = DataLoader(te_ds, batch_size=config["batch_size"], shuffle=False)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config["pos_weight"]] * N_GENES).to(device))
    model.to(device).eval()
    _, labels, probs = run_epoch(model, loader, None, crit, device, is_train=False)
    m = compute_metrics(labels, probs)
    f1 = calibrated_macro_f1(labels, probs)
    cal = compute_calibration(labels, probs)
    return add_headline({
        "auroc_macro": m["auroc_macro"], "auprc_macro": m["auprc_macro"],
        "f1_macro": f1["f1_macro"], "precision_macro": f1["precision_macro"],
        "recall_macro": f1["recall_macro"],
        "ece": cal["macro_ece"],
        "per_gene_auroc": {g: m.get(f"auroc_{g}", float("nan")) for g in GENE_INDEX},
        "per_gene_auprc": {g: m.get(f"auprc_{g}", float("nan")) for g in GENE_INDEX},
        "n_test_edges": int(labels.shape[0]),
    })


def _train(tr, val, te, cfg, ck, device, seed):
    model, _, hist = _train_core(tr, val, te, cfg, ck, device=device, verbose=False,
                                 print_steps=False, torch_seed=1000 + seed)
    best = max(hist, key=lambda h: h["val_auroc"]) if hist else {"epoch": None}
    return model, {"epochs_trained": len(hist), "best_epoch": best["epoch"]}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, model_seeds, ablation_seeds, out_dir: Path = None,
        gnn_hparams: dict = None, rf_hparams: dict = None, no_edge: bool = False,
        graph_free: bool = False, tag: str = None) -> dict:
    gcfg = {**config, **(gnn_hparams or {})}
    out_dir = out_dir or out_dir_for(config, tag)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    tmp = out_dir / "_tmp"
    tmp.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

    log(f"collecting data: {scenarios_for(config)} x {config['seeds_per_scenario']} data seeds, "
        f"{config['steps_per_run']} steps, dosing {dosing_for(config)}")
    grouped = config.get("split_by") == "run"
    if grouped:
        pairs, run_ids = collect_all_data(config, with_run_ids=True)
    else:
        pairs, run_ids = collect_all_data(config), None
    fp = dataset_fingerprint(pairs)
    log(f"dataset: {fp}")
    tr, val, te = split_dataset(pairs, config, run_ids=run_ids)
    log(f"split: train {len(tr)} / val {len(val)} / test {len(te)} graphs "
        f"(split_by={'run' if grouped else 'pair'}, split_seed "
        f"{config.get('split_seed', 42)})")
    composition = getattr(split_dataset, "last_composition", None) if grouped else None
    if composition:
        for scenario in sorted(composition):
            c = composition[scenario]
            log(f"    {scenario}: {len(c['train'])} train / {len(c['val'])} val / "
                f"{len(c['test'])} test runs")

    # Baseline subsample, scaled to hold the sampling FRACTION constant.
    # The reference run drew 100,000 of 1,210,040 training edges (8.264%). With a
    # larger dataset a fixed 100k would show the per-gene baselines a smaller
    # proportion of the positives, so rare genes would fall below the 5-positive
    # cutoff for a reason unrelated to the split being tested. Owner sign-off
    # 2026-10-01 (this changes baseline behaviour).
    REFERENCE_SUBSAMPLE_FRACTION = 100_000 / 1_210_040
    n_train_edges = sum(int(d.edge_index.shape[1]) for d in tr)
    max_train_samples = max(100_000, round(REFERENCE_SUBSAMPLE_FRACTION * n_train_edges))
    log(f"baseline subsample: {max_train_samples:,} of {n_train_edges:,} training edges "
        f"({100 * max_train_samples / max(1, n_train_edges):.3f}%; reference run was "
        f"100,000 of 1,210,040 = 8.264%)")
    edge_mask = EDGE_GROUP_SLICE if no_edge else None
    gtr, gval, gte = (mask_dataset(d, None, edge_mask) for d in (tr, val, te)) if no_edge else (tr, val, te)

    per_seed = []
    for s in model_seeds:
        log(f"--- model seed {s}: GNN (torch_seed {1000+s}, hparams {gnn_hparams or 'default'}, no_edge={no_edge})")
        model, info = _train(gtr, gval, gte, gcfg, str(out_dir / "models" / f"gnn_seed{s}.pt"), device, s)
        gnn = {**evaluate_gnn(model, gte, gcfg, device), **info}
        log(f"    GNN headline AUROC {gnn['headline_auroc']:.4f} (all-gene {gnn['auroc_macro']:.4f}) "
            f"| epochs {info['epochs_trained']} best@{info['best_epoch']}")
        rec = {"seed": s, "gnn": gnn}
        if graph_free:
            gf, gfinfo = _train(gtr, gval, gte, {**gcfg, "n_layers": 0}, str(tmp / "gf.pt"), device, s)
            rec["graph_free"] = {**evaluate_gnn(gf, gte, {**gcfg, "n_layers": 0}, device), **gfinfo}
            log(f"    graph-free headline AUROC {rec['graph_free']['headline_auroc']:.4f}")
        log(f"--- model seed {s}: baselines (RF/LR random_state {s}, subsample_seed {s}, RF {rf_hparams or 'default'})")
        bl = run_baselines(tr, te, results_path=str(tmp / "bl.json"),
                           subsample_seed=s, model_seed=s, include_gnn=False,
                           rf_params=rf_hparams, max_train_samples=max_train_samples)
        for k, v in bl.items():
            rec[k] = add_headline(v)
        per_seed.append(rec)
        log(f"    RF {rec['random_forest']['headline_auroc']:.4f} | "
            f"LR {rec['logistic_regression']['headline_auroc']:.4f} | "
            f"freq {rec['frequency_baseline']['headline_auroc']:.4f}  (headline AUROC)")

    models = ["gnn"] + (["graph_free"] if graph_free else []) + \
             ["random_forest", "logistic_regression", "frequency_baseline"]
    metrics = ["headline_auroc", "headline_auprc", "auroc_macro", "auprc_macro",
               "f1_macro", "precision_macro", "recall_macro"]
    summary = {m: {k: agg([r[m].get(k) for r in per_seed]) for k in metrics} for m in models}
    summary["gnn"]["ece"] = agg([r["gnn"]["ece"] for r in per_seed])
    per_gene = {m: {g: agg([r[m]["per_gene_auroc"].get(g) for r in per_seed]) for g in GENE_INDEX}
                for m in models}
    per_gene_auprc = {m: {g: agg([r[m].get("per_gene_auprc", {}).get(g) for r in per_seed])
                          for g in GENE_INDEX} for m in models}

    def paired(a, b):
        d = [r[a]["headline_auroc"] - r[b]["headline_auroc"] for r in per_seed]
        return {"diff": agg(d), "seeds_first_better": int(sum(x > 0 for x in d)), "n_seeds": len(d)}
    comparisons = {"gnn_minus_rf": paired("gnn", "random_forest")}
    if graph_free:
        comparisons["gnn_minus_graph_free"] = paired("gnn", "graph_free")

    # ── Ablation (same hparams and edge policy as the main GNN) ─────────────
    conditions = [(n, sl) for n, sl in NODE_GROUPS.items()]
    abl_edge = [("No edge features", EDGE_GROUP_SLICE)] if not no_edge else []
    abl_raw = {n: [] for n, _ in conditions + abl_edge}
    for s in ablation_seeds:
        for name, ns in conditions:
            log(f"--- ablation seed {s}: {name}")
            mtr, mval, mte = (mask_dataset(d, ns, edge_mask) for d in (tr, val, te))
            model, _ = _train(mtr, mval, mte, gcfg, str(tmp / "abl.pt"), device, s)
            abl_raw[name].append(evaluate_gnn(model, mte, gcfg, device)["headline_auroc"])
        for name, es in abl_edge:
            log(f"--- ablation seed {s}: {name}")
            mtr, mval, mte = (mask_dataset(d, None, es) for d in (tr, val, te))
            model, _ = _train(mtr, mval, mte, gcfg, str(tmp / "abl.pt"), device, s)
            abl_raw[name].append(evaluate_gnn(model, mte, gcfg, device)["headline_auroc"])
    base = abl_raw["All features (full model)"]
    ablation = {n: {"auroc": agg(v), "delta_vs_all": agg([a - b for a, b in zip(v, base)])}
                for n, v in abl_raw.items()}

    main_by_seed = {r["seed"]: r["gnn"]["headline_auroc"] for r in per_seed}
    repro = [abs(a - main_by_seed[s]) for s, a in zip(ablation_seeds, base) if s in main_by_seed]

    shutil.rmtree(tmp, ignore_errors=True)
    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "device": str(device), "torch": torch.__version__,
        "feature_dims": {"node": NODE_FEATURE_DIM, "edge": EDGE_FEATURE_DIM},
        "config": {k: v for k, v in config.items()},
        "biology": config.get("biology", "paper_v1"),
        "scenarios": scenarios_for(config),
        "dosing": dosing_for(config),
        "gnn_hparams": gnn_hparams, "rf_hparams": rf_hparams,
        "no_edge_features": no_edge, "min_epochs": gcfg.get("min_epochs", 0),
        "gene_policy": {"headline_genes": HEADLINE_GENES, "excluded": EXCLUDED, "separate": SEPARATE},
        "model_seeds": list(model_seeds), "ablation_seeds": list(ablation_seeds),
        "dataset": fp,
        "split": {"split_by": config.get("split_by", "pair"),
                  "split_seed": config.get("split_seed", 42),
                  "train_frac": config["train_frac"], "val_frac": config["val_frac"],
                  "n_graphs": {"train": len(tr), "val": len(val), "test": len(te)},
                  "runs_by_scenario": composition},
        "baseline_subsample": {"max_train_samples": max_train_samples,
                               "n_train_edges": n_train_edges,
                               "fraction": max_train_samples / max(1, n_train_edges),
                               "reference_fraction": REFERENCE_SUBSAMPLE_FRACTION},
        "summary": summary, "per_gene_auroc": per_gene, "per_gene_auprc": per_gene_auprc,
        "comparisons": comparisons,
        "ablation": ablation,
        "reproducibility_check": {
            "what": "|headline AUROC(main run) - (ablation 'All features' rerun)|, same data and torch_seed",
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
    ds = r["dataset"]
    L = [f"# Reseeded results ({r['generated']})", "",
         f"Biology {r['biology']}; scenarios {r['scenarios']}; dosing {r['dosing']}.",
         f"GNN hparams {r['gnn_hparams'] or 'default'}; edge features "
         f"{'ZEROED' if r['no_edge_features'] else 'used'}; min_epochs {r['min_epochs']}; "
         f"RF hparams {r['rf_hparams'] or 'default'}.",
         f"Model seeds {r['model_seeds']}; ablation seeds {r['ablation_seeds']}; device {r['device']}.",
         f"Dataset sha256 `{ds['sha256'][:16]}…`: {ds['n_graph_pairs']} graph pairs, "
         f"{ds['n_edges']:,} edges, {ds['n_edge_gene_positives']:,} positives; per gene "
         f"{ds.get('positives_per_gene')}.", "",
         f"**Headline macro** = mean over {r['gene_policy']['headline_genes']} with test positives. "
         f"Excluded: {list(r['gene_policy']['excluded'])}. Reported separately: "
         f"{list(r['gene_policy']['separate'])}.", "",
         "## Model comparison (test set, mean ± SD over seeds)", "",
         "| Model | Headline AUROC | Headline AUPRC | All-gene AUROC | F1 (calibrated) |",
         "|---|---|---|---|---|"]
    names = {"gnn": "AMRResistanceGNN", "graph_free": "GNN without message passing",
             "random_forest": "Random Forest", "logistic_regression": "Logistic Regression",
             "frequency_baseline": "Frequency baseline"}
    for m, label in names.items():
        if m not in r["summary"]:
            continue
        s = r["summary"][m]
        L.append(f"| {label} | {_ms(s['headline_auroc'])} | {_ms(s['headline_auprc'])} | "
                 f"{_ms(s['auroc_macro'])} | {_ms(s['f1_macro'])} |")
    L.append("")
    for k, c in r["comparisons"].items():
        L.append(f"{k} (headline AUROC, paired by seed): {_ms(c['diff'])}; first better on "
                 f"{c['seeds_first_better']}/{c['n_seeds']} seeds.")
    L += [f"GNN ECE: {_ms(r['summary']['gnn']['ece'])}.", "",
          "## Per-gene AUROC", "", "| Gene | GNN | RF | note |", "|---|---|---|---|"]
    for g in r["per_gene_auroc"]["gnn"]:
        note = ("EXCLUDED" if g in r["gene_policy"]["excluded"] else
                "SEPARATE - simplified mechanism" if g in r["gene_policy"]["separate"] else "")
        L.append(f"| {g} | {_ms(r['per_gene_auroc']['gnn'][g])} | "
                 f"{_ms(r['per_gene_auroc']['random_forest'][g])} | {note} |")
    for g, why in r["gene_policy"]["separate"].items():
        L.append(f"\n**{g} (reported separately):** {why}")
    L += ["", "## GNN feature-group ablation (retrained per condition, headline AUROC)", "",
          "| Condition | AUROC | Δ vs all features (paired) |", "|---|---|---|"]
    for n, a in r["ablation"].items():
        L.append(f"| {n} | {_ms(a['auroc'])} | {_ms(a['delta_vs_all'])} |")
    rc = r["reproducibility_check"]
    L += ["", f"Reproducibility check ({rc['what']}): max |Δ| = {rc['max_abs_diff']}.",
          "", "Per-seed training (GNN): " + "; ".join(
              f"seed {p['seed']}: {p['gnn']['epochs_trained']} epochs, best @ {p['gnn']['best_epoch']}"
              for p in r["per_seed"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--biology", default=None, help="override DEFAULT_CONFIG['biology']")
    ap.add_argument("--gnn-hparams", default=None, help="JSON dict, e.g. from ai/hparam_sweep")
    ap.add_argument("--rf-hparams", default=None, help="JSON dict")
    ap.add_argument("--no-edge", action="store_true", help="zero edge features in every GNN run")
    ap.add_argument("--graph-free", action="store_true", help="add the n_layers=0 arm")
    ap.add_argument("--tag", default=None, help="output subfolder suffix")
    ap.add_argument("--split-by", choices=["pair", "run"], default=None,
                    help="'run' holds out whole simulation runs (claim U13), "
                         "stratified by scenario; 'pair' is the original behaviour")
    ap.add_argument("--data-seeds", type=int, default=None,
                    help="data seeds per scenario (DEFAULT_CONFIG: 3)")
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    a = ap.parse_args()
    if a.quick:
        run(QUICK_CONFIG, range(2), range(1), out_dir=Path("ai/checkpoints/_reseeded_quick"),
            no_edge=a.no_edge, graph_free=a.graph_free)
    else:
        cfg = dict(DEFAULT_CONFIG)
        if a.biology:
            cfg["biology"] = a.biology
        if a.split_by:
            cfg["split_by"] = a.split_by
        if a.data_seeds:
            cfg["seeds_per_scenario"] = a.data_seeds
        if a.train_frac is not None:
            cfg["train_frac"] = a.train_frac
        if a.val_frac is not None:
            cfg["val_frac"] = a.val_frac
        run(cfg, range(a.seeds), range(a.ablation_seeds),
            gnn_hparams=json.loads(a.gnn_hparams) if a.gnn_hparams else None,
            rf_hparams=json.loads(a.rf_hparams) if a.rf_hparams else None,
            no_edge=a.no_edge, graph_free=a.graph_free, tag=a.tag)
