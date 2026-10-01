"""Three-way comparison: committed reference, 50-run pair split, 50-run run split.

ref      = ai/checkpoints/reseeded/lab_v2_tuned_noedge_mrsa   (15 runs, pair split)
pair50   = lab_v2_grouped50_pairsplit                         (50 runs, pair split)
run50    = lab_v2_grouped50_runsplit                          (50 runs, run split)

ref -> pair50 isolates the DATA-SIZE effect (method held constant).
pair50 -> run50 isolates the SPLIT effect (data held constant).
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath("."))
from ai.feature_engineering import GENE_INDEX  # noqa: E402

BASE = "ai/checkpoints/reseeded"
ARMS = {"ref": "lab_v2_tuned_noedge_mrsa",
        "pair50": "lab_v2_grouped50_pairsplit",
        "run50": "lab_v2_grouped50_runsplit"}
D = {k: json.load(open(f"{BASE}/{v}/results.json", encoding="utf-8"))
     for k, v in ARMS.items()}

HEAD4 = ["blaCTX-M-15", "blaNDM-1", "mcr-1", "tetM"]


def ms(d, path):
    cur = d
    for p in path:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return None, None
    return cur.get("mean"), cur.get("sd")


def fmt(m, s):
    return "n/a" if m is None else f"{m:.4f} ± {s:.4f}"


def subset(d, model, genes):
    vals = []
    for seed in d["per_seed"]:
        pg = seed[model]["per_gene_auroc"]
        v = [pg[g] for g in genes if pg.get(g) is not None]
        if len(v) == len(genes):
            vals.append(statistics.mean(v))
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def trainable(d, gene, model="random_forest"):
    return sum(1 for s in d["per_seed"]
               if s[model]["per_gene_auroc"].get(gene) not in (None, 0.5))


print("=" * 100)
print(f"{'':44s} {'ref (15, pair)':>18s} {'pair50 (50, pair)':>18s} {'run50 (50, run)':>18s}")
print("=" * 100)

rows = [
    ("S1  GNN headline AUROC", ("summary", "gnn", "headline_auroc")),
    ("S3  LR headline AUROC", ("summary", "logistic_regression", "headline_auroc")),
    ("    RF headline AUROC", ("summary", "random_forest", "headline_auroc")),
    ("    graph-free headline AUROC", ("summary", "graph_free", "headline_auroc")),
    ("S6  GNN ECE", ("summary", "gnn", "ece")),
    ("S7  GNN headline AUPRC", ("summary", "gnn", "headline_auprc")),
    ("S4  message passing gain", ("comparisons", "gnn_minus_graph_free", "diff")),
    ("S5  ablation: no genomic genes", ("ablation", "No genomic genes", "delta_vs_all")),
    ("U4  ablation: no antibiotic exposure",
     ("ablation", "No antibiotic exposure", "delta_vs_all")),
]
for label, path in rows:
    cells = [fmt(*ms(D[a], path)) for a in ("ref", "pair50", "run50")]
    print(f"{label:44s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")

print(f"\n{'S2  GNN on the 4 always-trainable genes':44s}", end="")
for a in ("ref", "pair50", "run50"):
    m, s = subset(D[a], "gnn", HEAD4)
    print(f" {fmt(m, s):>18s}", end="")
print(f"\n{'S2  RF  on the same 4 genes':44s}", end="")
for a in ("ref", "pair50", "run50"):
    m, s = subset(D[a], "random_forest", HEAD4)
    print(f" {fmt(m, s):>18s}", end="")
print(f"\n{'S2  gap (GNN - RF)':44s}", end="")
for a in ("ref", "pair50", "run50"):
    g, _ = subset(D[a], "gnn", HEAD4)
    r, _ = subset(D[a], "random_forest", HEAD4)
    print(f" {g - r:18.4f}", end="")
print(f"\n{'S2  seeds GNN ahead of RF':44s}", end="")
for a in ("ref", "pair50", "run50"):
    d = D[a]
    w = sum(1 for s in d["per_seed"]
            if statistics.mean(s["gnn"]["per_gene_auroc"][g] for g in HEAD4)
            > statistics.mean(s["random_forest"]["per_gene_auroc"][g] for g in HEAD4))
    print(f" {f'{w}/5':>18s}", end="")

print("\n\n" + "=" * 100)
print("S9 / S2b  per-gene GNN AUROC, and how many of 5 seeds RF could be fitted")
print("=" * 100)
print(f"{'gene':14s} {'ref GNN':>16s} {'pair50 GNN':>16s} {'run50 GNN':>16s} "
      f"{'RF fit ref':>11s} {'pair50':>7s} {'run50':>7s}")
for g in GENE_INDEX:
    cells = []
    for a in ("ref", "pair50", "run50"):
        row = D[a]["per_gene_auroc"]["gnn"].get(g, {})
        cells.append(fmt(row.get("mean"), row.get("sd")) if row.get("mean") is not None
                     else "not evaluable")
    tr = [f"{trainable(D[a], g)}/5" for a in ("ref", "pair50", "run50")]
    print(f"{g:14s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s} "
          f"{tr[0]:>11s} {tr[1]:>7s} {tr[2]:>7s}")

print("\n" + "=" * 100)
print("dataset / split provenance")
print("=" * 100)
for a in ("ref", "pair50", "run50"):
    d = D[a]
    ds = d["dataset"]
    sp = d.get("split", {})
    bs = d.get("baseline_subsample", {})
    print(f"{a:8s} pairs {ds['n_graph_pairs']:5d} edges {ds['n_edges']:>10,d} "
          f"positives {ds['n_edge_gene_positives']:5d} | split_by "
          f"{sp.get('split_by', 'pair'):4s} | subsample "
          f"{bs.get('max_train_samples', 100000):>7,d} "
          f"({100 * bs.get('fraction', 100000 / 1210040):.3f}%)")

out = {}
for a in ("ref", "pair50", "run50"):
    g, gs = subset(D[a], "gnn", HEAD4)
    r, rs = subset(D[a], "random_forest", HEAD4)
    out[a] = {"s2_gnn": [g, gs], "s2_rf": [r, rs],
              "headline": list(ms(D[a], ("summary", "gnn", "headline_auroc"))),
              "msg_passing": list(ms(D[a], ("comparisons", "gnn_minus_graph_free", "diff"))),
              "genomic": list(ms(D[a], ("ablation", "No genomic genes", "delta_vs_all"))),
              "rf_trainable": {g2: trainable(D[a], g2) for g2 in GENE_INDEX}}
with open(os.path.join(os.environ.get("SCRATCH", "."), "arm_comparison.json"),
          "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote arm_comparison.json")
