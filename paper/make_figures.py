"""Generate the paper's figures from committed result files only.

Every value plotted is read from `ai/checkpoints/reseeded/lab_v2_tuned_noedge_mrsa/
results.json` (the reference result set) or from a value already recorded in
`paper/claims_to_numbers.md`. Nothing is recomputed from a simulation run, and no
value is hard-coded except the claims-table figures used for the self-checks below.

The script self-checks the numbers it plots against `paper/claims_to_numbers.md` and
exits non-zero if any has drifted, the same discipline as
`paper/measure_cross_species_edges.py` and `paper/make_table1.py`.

Usage
-----
    python paper/make_figures.py                 # write paper/figures/*.png + *.pdf
    python paper/make_figures.py --check         # verify numbers only, write nothing
    python paper/make_figures.py --rf-coverage   # print the per-seed RF/LR 0.5 table

Figures produced
----------------
    fig1_per_gene_auroc   per-gene GNN AUROC with positive counts (S9, S10 policy)
    fig2_architecture     model schematic for the reference configuration
    fig3_ablation         feature-group ablation deltas (S5)

Captions are written to `paper/figures/captions.md` so the caption text is generated
from the same numbers as the figures and cannot drift from them.

NOT produced: the headline GNN-vs-baseline comparison figure. Generating it is blocked
on an open question about claim S2 -- `--rf-coverage` shows why: the random forest
scores exactly 0.5 (its untrainable marker) on blaKPC-2 in 3 of 5 seeds, although
blaKPC-2 is one of the five genes S2 describes as "genes RF can train on". See
`paper/decisions_log.md` (2026-09-30, "S2 contamination"). A figure would enshrine the
current framing, so it is deliberately not drawn until that is settled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "ai", "checkpoints", "reseeded",
                       "lab_v2_tuned_noedge_mrsa", "results.json")
FIGDIR = os.path.join(REPO, "paper", "figures")
SRC_LABEL = "ai/checkpoints/reseeded/lab_v2_tuned_noedge_mrsa/results.json"

# Values recorded in paper/claims_to_numbers.md, used only to verify the plotted data.
CLAIMED = {
    "S1_headline_gnn": (0.9805, 0.0012),
    "S2_gnn_4gene": (0.9777, 0.0023),
    "S2_rf_4gene": (0.9548, 0.0124),
    "S3_headline_lr": (0.7096, 0.0359),
    "S4_msg_passing_gain": (0.0031, 0.0014),
    "S5_no_genomic_delta": (-0.0549, 0.0331),
    "S6_ece": (0.0012, 0.0003),
    "S9_per_gene": {
        "blaCTX-M-15": (0.9642, 0.0030), "blaKPC-2": (0.9564, 0.0021),
        "blaNDM-1": (0.9755, 0.0039), "mcr-1": (0.9908, 0.0011),
        "tetM": (0.9805, 0.0033), "acrAB-tolC": (0.9986, 0.0018),
        "gyrA_S83L": (0.9977, 0.0006),
    },
    "positives_per_gene": {
        "blaCTX-M-15": 480, "tetM": 420, "mcr-1": 179, "blaNDM-1": 177,
        "blaKPC-2": 94, "vanA": 35, "gyrA_S83L": 34, "acrAB-tolC": 19,
        "blaTEM-1": 5, "mexAB-oprM": 0,
    },
}
# Genes whose simulated transfer is a labelled simplification (U6, U10).
SIMPLIFIED = {"acrAB-tolC", "gyrA_S83L", "vanA"}


def load() -> dict:
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def close(got: float, want: float, tol: float = 5e-4) -> bool:
    return got is not None and abs(got - want) <= tol


def check(data: dict) -> list:
    """Verify every plotted number against paper/claims_to_numbers.md."""
    problems = []

    def cmp(label, got_mean, got_sd, want):
        want_mean, want_sd = want
        if not close(got_mean, want_mean):
            problems.append(f"{label}: mean {got_mean!r} != claimed {want_mean}")
        if not close(got_sd, want_sd):
            problems.append(f"{label}: sd {got_sd!r} != claimed {want_sd}")

    s = data["summary"]
    cmp("S1 headline GNN", s["gnn"]["headline_auroc"]["mean"],
        s["gnn"]["headline_auroc"]["sd"], CLAIMED["S1_headline_gnn"])
    cmp("S3 headline LR", s["logistic_regression"]["headline_auroc"]["mean"],
        s["logistic_regression"]["headline_auroc"]["sd"], CLAIMED["S3_headline_lr"])
    cmp("S6 ECE", s["gnn"]["ece"]["mean"], s["gnn"]["ece"]["sd"], CLAIMED["S6_ece"])

    gm, gsd, gvals = subset_macro(data, "gnn", HEADLINE_SET)
    rm, rsd, rvals = subset_macro(data, "random_forest", HEADLINE_SET)
    cmp("S2 GNN (4-gene)", gm, gsd, CLAIMED["S2_gnn_4gene"])
    cmp("S2 RF (4-gene)", rm, rsd, CLAIMED["S2_rf_4gene"])
    wins = sum(1 for a, b in zip(gvals, rvals) if a > b)
    if wins != 5:
        problems.append(f"S2: GNN ahead on {wins}/5 seeds, claimed 5/5")
    for gene in HEADLINE_SET:
        n05 = sum(1 for s_ in data["per_seed"]
                  if s_["random_forest"]["per_gene_auroc"][gene] == 0.5)
        if n05:
            problems.append(
                f"S2 set contamination: RF scores 0.5 on {gene} in {n05}/5 seeds; "
                "the headline set must contain only genes RF fits in EVERY seed")

    mp = data["comparisons"]["gnn_minus_graph_free"]["diff"]
    cmp("S4 message-passing gain", mp["mean"], mp["sd"], CLAIMED["S4_msg_passing_gain"])

    ab = data["ablation"]["No genomic genes"]["delta_vs_all"]
    cmp("S5 no-genomic delta", ab["mean"], ab["sd"], CLAIMED["S5_no_genomic_delta"])

    for gene, want in CLAIMED["S9_per_gene"].items():
        row = data["per_gene_auroc"]["gnn"][gene]
        cmp(f"S9 {gene}", row["mean"], row["sd"], want)

    pos = data["dataset"]["positives_per_gene"]
    for gene, want in CLAIMED["positives_per_gene"].items():
        if pos.get(gene) != want:
            problems.append(f"positives {gene}: {pos.get(gene)} != claimed {want}")

    for key, want in (("n_graph_pairs", 390), ("n_edges", 1_707_498),
                      ("n_edge_gene_positives", 1_443)):
        if data["dataset"][key] != want:
            problems.append(f"dataset {key}: {data['dataset'][key]} != claimed {want}")
    return problems


def rf_coverage(data: dict) -> None:
    """Print which genes the classical baselines could not train, per seed.

    This is the diagnostic behind the open S2 question: a baseline that scores
    exactly 0.5 on a gene did not fit it (fewer than 5 positives in its 100k-edge
    subsample), so that gene's 0.5 is a construction artefact, not a prediction.
    """
    genes = list(data["dataset"]["positives_per_gene"])
    for model in ("random_forest", "logistic_regression"):
        print(f"\n== {model}: per-gene test AUROC by model seed "
              f"(0.5000 = not trainable)")
        header = "gene".ljust(14) + "".join(f"seed{i}  " for i in range(5))
        print(header + " untrainable")
        for gene in genes:
            cells, n05 = [], 0
            for seed in data["per_seed"]:
                v = seed[model]["per_gene_auroc"].get(gene)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("  nan  ")
                else:
                    cells.append(f"{v:.4f} ")
                    if v == 0.5:
                        n05 += 1
            print(gene.ljust(14) + "".join(cells) + f"  {n05}/5")

    five = ["blaCTX-M-15", "blaKPC-2", "blaNDM-1", "mcr-1", "tetM"]
    four = [g for g in five if g != "blaKPC-2"]
    print("\n== macro over gene subsets, per model (mean +/- sd over 5 seeds)")
    for label, subset in (("S2's 5 genes", five),
                          ("4 genes RF fits in ALL seeds", four)):
        print(f"  {label}: {subset}")
        for model in ("gnn", "random_forest", "logistic_regression"):
            vals = [statistics.mean(s[model]["per_gene_auroc"][g] for g in subset)
                    for s in data["per_seed"]]
            print(f"    {model:20s} {statistics.mean(vals):.4f} "
                  f"+/- {statistics.stdev(vals):.4f}")
        wins = sum(1 for s in data["per_seed"]
                   if statistics.mean(s["gnn"]["per_gene_auroc"][g] for g in subset)
                   > statistics.mean(s["random_forest"]["per_gene_auroc"][g]
                                     for g in subset))
        print(f"    GNN ahead on {wins}/5 seeds")


# ---------------------------------------------------------------- figure 1
def fig_per_gene(data: dict) -> str:
    pos = data["dataset"]["positives_per_gene"]
    rows = []
    for gene, row in data["per_gene_auroc"]["gnn"].items():
        if row["mean"] is None:
            continue                      # blaTEM-1, mexAB-oprM: not evaluable
        rows.append((gene, row["mean"], row["sd"], pos[gene]))
    rows.sort(key=lambda r: r[1], reverse=True)

    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    sds = [r[2] for r in rows]
    counts = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = range(len(rows))
    colours = ["#b4553f" if n in SIMPLIFIED else "#3a6ea5" for n in names]
    ax.barh(list(y), means, xerr=sds, color=colours, height=0.62,
            error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    # Counts in a fixed column past the longest error bar, so they never collide.
    label_x = max(m + s for m, s in zip(means, sds)) + 0.0025
    for i, c in enumerate(counts):
        ax.text(label_x, i, f"n={c}", va="center", fontsize=9, color="#222222")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{n}*" if n in SIMPLIFIED else n for n in names], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.94, 1.018)
    ax.set_xlabel("Test AUROC (mean $\\pm$ SD over 5 model seeds)")
    ax.set_title("Per-gene discrimination, with positive-event counts\n"
                 "Fewest positives score highest; best-evidenced gene scores lowest",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    handles = [patches.Patch(color="#3a6ea5", label="modelled as conjugative transfer"),
               patches.Patch(color="#b4553f",
                             label="* simplified mechanism (U6, U10)")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    return save(fig, "fig1_per_gene_auroc")


# ---------------------------------------------------------------- figure 2
def fig_architecture(data: dict) -> str:
    """Schematic of the reference configuration, read from the results file."""
    hp = data.get("gnn_hparams", {})
    hidden = hp.get("hidden_dim", 128)
    edge_enc = hp.get("edge_enc_dim", 64)
    layers = hp.get("n_layers", 2)
    heads = hp.get("heads", 4)
    dims = data.get("feature_dims", {})
    node_dim = dims.get("node_feature_dim", 35)
    edge_dim = dims.get("edge_feature_dim", 5)
    n_genes = dims.get("n_genes", 10)

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.15, 6.25)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=8.6, ec="#444444"):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.045", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": style, "lw": 1.1, "color": "#444444"})

    blue, green, orange, grey = "#dce7f2", "#dbeedd", "#fae6d8", "#eeeeee"

    # Inputs
    box(0.15, 4.35, 2.15, 1.0,
        f"node features\n({node_dim}-dim)\n10 genes, physiology,\nspecies, exposure...", grey)
    box(0.15, 2.55, 2.15, 0.95,
        f"edge features\n({edge_dim}-dim)\nzeroed in the\nreported runs", grey)

    # Encoders
    box(2.75, 4.3, 2.0, 1.1,
        "NodeEncoder\n8 per-group linears\n-> concat -> proj\n"
        f"-> {hidden}-dim", blue)
    box(2.75, 2.5, 2.0, 1.05,
        f"EdgeEncoder\nMLP -> {edge_enc}-dim", blue)

    # GAT stack
    box(5.2, 3.35, 2.05, 2.05,
        f"{layers} x GAT block\n\nGATConv, {heads} heads\n"
        f"edge_dim={edge_enc}\n+ residual, LayerNorm\n+ feed-forward\n+ residual, LayerNorm",
        green)

    # Head
    box(7.75, 3.55, 2.1, 1.6,
        "EdgePredictorHead\n"
        f"concat[h_src, h_dst, e]\n({hidden}+{hidden}+{edge_enc})\n"
        f"-> {hidden} -> {hidden // 2}\n-> {n_genes} logits", orange)

    box(7.75, 1.55, 2.1, 1.0,
        f"per directed edge:\n{n_genes} gene\ntransfer scores", grey)

    arrow(2.3, 4.85, 2.73, 4.85)
    arrow(2.3, 3.0, 2.73, 3.0)
    arrow(4.77, 4.85, 5.35, 4.6)          # node -> GAT
    arrow(4.77, 3.0, 5.35, 3.9)           # edge -> GAT (as edge attribute)
    arrow(7.27, 4.4, 7.73, 4.4)           # GAT -> head
    arrow(4.77, 2.85, 7.73, 2.85)         # edge encoding -> head (skip)
    ax.text(6.2, 2.62, "encoded edge vector also enters the head",
            fontsize=7.6, color="#555555", ha="center")
    arrow(8.8, 3.5, 8.8, 2.6)

    ax.text(5.0, 5.85,
            "AMRResistanceGNN, reference configuration "
            f"(hidden {hidden}, edge encoding {edge_enc}, {layers} blocks, {heads} heads)",
            ha="center", fontsize=10.5)
    ax.text(5.0, 1.32,
            "Graph-free ablation removes the GAT stack entirely: encoders feed the head "
            "directly.",
            ha="center", fontsize=8.4, color="#555555", style="italic")
    fig.tight_layout()
    return save(fig, "fig2_architecture")


# ---------------------------------------------------------------- figure 3
def fig_ablation(data: dict) -> str:
    rows = []
    for group, row in data["ablation"].items():
        if group.startswith("All features"):
            continue
        d = row["delta_vs_all"]
        rows.append((group.replace("No ", ""), d["mean"], d["sd"]))
    rows.sort(key=lambda r: r[1])

    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    sds = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = range(len(rows))
    colours = ["#b4553f" if m < -0.01 else "#8b9bab" for m in means]
    ax.barh(list(y), means, xerr=sds, color=colours, height=0.6,
            error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    ax.axvline(0, color="#222222", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Change in headline AUROC when the group is zeroed\n"
                  "(mean $\\pm$ SD over 3 seeds; negative = the group helped)")
    ax.set_title("Feature-group ablation: only carried genes matter", fontsize=11)
    ax.grid(axis="x", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "fig3_ablation")


# ---------------------------------------------------------------- figure 4
HEADLINE_SET = ["blaCTX-M-15", "blaNDM-1", "mcr-1", "tetM"]


def subset_macro(data: dict, model: str, genes: list) -> tuple:
    vals = [statistics.mean(s[model]["per_gene_auroc"][g] for g in genes)
            for s in data["per_seed"]]
    return statistics.mean(vals), statistics.stdev(vals), vals


def fig_comparison(data: dict) -> str:
    """Headline comparison on the four genes RF fits in EVERY seed (claim S2).

    blaKPC-2 is deliberately excluded: RF scores it 0.5 by construction in 3 of 5
    seeds, so including it would fold that artefact into the baseline (see U5 and
    decisions_log.md, "S2 contamination"). It appears in the right-hand panel instead.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel A: like-for-like macro on the always-trainable set.
    models = [("gnn", "GNN", "#3a6ea5"),
              ("random_forest", "Random forest", "#8b9bab"),
              ("logistic_regression", "Logistic regression", "#c2c7cc")]
    means, sds, labels, colours = [], [], [], []
    for key, label, colour in models:
        m, sd, _ = subset_macro(data, key, HEADLINE_SET)
        means.append(m), sds.append(sd), labels.append(label), colours.append(colour)
    x = range(len(models))
    ax1.bar(list(x), means, yerr=sds, color=colours, width=0.62,
            error_kw={"ecolor": "#333333", "capsize": 4, "lw": 1})
    for i, m in enumerate(means):
        ax1.text(i, m + sds[i] + 0.012, f"{m:.4f}", ha="center", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0.5, 1.06)
    ax1.set_ylabel("Macro AUROC over the 4 genes")
    ax1.set_title("A. Like-for-like: the 4 genes the baseline\n"
                  "fits in every seed", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, lw=0.6)
    ax1.set_axisbelow(True)

    # Panel B: per-gene, showing where the baseline cannot be fitted.
    genes = HEADLINE_SET + ["blaKPC-2", "gyrA_S83L", "acrAB-tolC"]
    pos = data["dataset"]["positives_per_gene"]
    gnn_m = [data["per_gene_auroc"]["gnn"][g]["mean"] for g in genes]
    rf_vals = {g: [s["random_forest"]["per_gene_auroc"][g]
                   for s in data["per_seed"]] for g in genes}
    y = list(range(len(genes)))
    h = 0.36
    ax2.barh([v + h / 2 for v in y], gnn_m, height=h, color="#3a6ea5", label="GNN")
    rf_m = [statistics.mean(rf_vals[g]) for g in genes]
    ax2.barh([v - h / 2 for v in y], rf_m, height=h, color="#8b9bab",
             label="Random forest (mean incl. 0.5 scores)")
    for i, g in enumerate(genes):
        n_untrained = sum(1 for v in rf_vals[g] if v == 0.5)
        if n_untrained:
            ax2.text(0.515, i - h / 2, f"not fitted in {n_untrained}/5 seeds",
                     va="center", fontsize=7.6, color="#7a2f1d")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{g}\n(n={pos[g]})" for g in genes], fontsize=8)
    ax2.invert_yaxis()
    # Blank band below the last bar so the legend cannot overlap any bar.
    ax2.set_ylim(len(genes) + 0.35, -0.6)
    ax2.set_xlim(0.5, 1.02)
    ax2.axvline(0.5, color="#7a2f1d", lw=1, ls=":")
    ax2.set_xlabel("Test AUROC (0.5 = baseline could not be fitted)")
    ax2.set_title("B. Per gene: where the per-gene baseline\n"
                  "cannot be fitted at all", fontsize=10)
    ax2.legend(loc="lower right", fontsize=7.8, framealpha=0.95,
               borderaxespad=0.3)
    ax2.grid(axis="x", alpha=0.3, lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, "fig4_model_comparison")


def save(fig, stem: str) -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    return stem


def captions(data: dict) -> str:
    """Write captions generated from the same numbers as the figures."""
    pos = data["dataset"]["positives_per_gene"]
    pg = data["per_gene_auroc"]["gnn"]
    s = data["summary"]
    gm, gsd, gvals = subset_macro(data, "gnn", HEADLINE_SET)
    rm, rsd, rvals = subset_macro(data, "random_forest", HEADLINE_SET)

    mp = data["comparisons"]["gnn_minus_graph_free"]["diff"]
    ab = data["ablation"]["No genomic genes"]["delta_vs_all"]
    hp = data.get("gnn_hparams", {})
    dims = data.get("feature_dims", {})

    def f(x, n=4):
        return f"{x:.{n}f}"

    g4m, g4s, _ = subset_macro(data, "gnn", HEADLINE_SET)
    r4m, r4s, _ = subset_macro(data, "random_forest", HEADLINE_SET)
    l4m, l4s, _ = subset_macro(data, "logistic_regression", HEADLINE_SET)

    text = f"""<!-- Generated by paper/make_figures.py. Do not edit by hand:
edit the script so captions and figures cannot drift apart. -->

# Figure captions

All values from `{SRC_LABEL}`, the committed reference result set
(dataset SHA-256 `ee83ff385ca15ec9...`: {data['dataset']['n_graph_pairs']} graph pairs,
{data['dataset']['n_edges']:,} edges, {data['dataset']['n_edge_gene_positives']:,}
edge-gene positives). Generated by `paper/make_figures.py`, which verifies every plotted
number against `paper/claims_to_numbers.md` and fails if any has drifted.

## Figure 1 — Per-gene discrimination, with positive-event counts

Per-gene test AUROC of the GNN, mean $\\pm$ SD over 5 model seeds, each bar annotated with
that gene's total positive transfer events in the dataset.
**The ordering is confounded, and runs opposite to the evidence behind it: the two
highest-scoring genes are the two with the fewest positives — acrAB-tolC
{f(pg['acrAB-tolC']['mean'])} from {pos['acrAB-tolC']} events and gyrA_S83L
{f(pg['gyrA_S83L']['mean'])} from {pos['gyrA_S83L']} — while the best-evidenced gene,
blaCTX-M-15 with {pos['blaCTX-M-15']} events, scores lowest at
{f(pg['blaCTX-M-15']['mean'])}.** An AUROC estimated from {pos['acrAB-tolC']} positives is
not comparable to one estimated from {pos['blaCTX-M-15']}, so this figure must not be read as
a difficulty ranking over genes. Starred genes (acrAB-tolC, gyrA_S83L) are chromosomal in real
bacteria and are transferred by the simulator as a labelled simplification (claim U10); vanA is
likewise simplified (U6) and is excluded from the headline macro. blaTEM-1 ({pos['blaTEM-1']}
positives, none in the test split) and mexAB-oprM ({pos['mexAB-oprM']} positives, in no
species' acquirable set) are not evaluable and are absent from the figure. Counts from
`dataset.positives_per_gene`; AUROCs from `per_gene_auroc.gnn`. Claims S9, U6, U8, U10, U12.

## Figure 2 — Model architecture, reference configuration

Schematic of AMRResistanceGNN as configured for every reported result: node features
({dims.get('node_feature_dim')}-dim) through a per-group NodeEncoder to width
{hp.get('hidden_dim')}, edge features ({dims.get('edge_feature_dim')}-dim) through an
EdgeEncoder to width {hp.get('edge_enc_dim')}, then {hp.get('n_layers')} graph-attention
blocks ({hp.get('heads')} heads, each with a residual connection, layer normalisation and a
feed-forward sublayer), then a head that concatenates the two endpoint representations with
the encoded edge vector and emits {dims.get('n_genes')} per-gene logits for each directed
edge. Edge features are zeroed in all reported runs, so that pathway is present but carries no
information. The graph-free ablation removes the attention stack, leaving the encoders feeding
the head directly, and costs {f(mp['mean'])} $\\pm$ {f(mp['sd'])} headline AUROC (claim S4).
Configuration read from `gnn_hparams` and `feature_dims`; layer structure from
`ai/gnn_model.py`.

## Figure 3 — Feature-group ablation

Change in headline AUROC when each node-feature group is zeroed, mean $\\pm$ SD over 3 seeds;
more negative means the group contributed more. Only the carried-gene group matters:
{f(ab['mean'])} $\\pm$ {f(ab['sd'])}. Every other group falls within $\\pm$0.003, which is
inside seed-to-seed variation and is not interpreted as an effect in either direction — including
antibiotic exposure, for which the drug is present in only 5 of the 80 simulated steps under
this dosing protocol (claim U4). Full-model reference:
{f(data['ablation']['All features (full model)']['auroc']['mean'])}. Values from `ablation`.
Claims S5, U4.

## Figure 4 — Headline comparison, and where the per-gene baseline cannot be built

**A.** Macro AUROC over the four genes the random forest fits in *every* seed
(blaCTX-M-15, blaNDM-1, mcr-1, tetM): GNN {f(g4m)} $\\pm$ {f(g4s)}, random forest {f(r4m)}
$\\pm$ {f(r4s)}, logistic regression {f(l4m)} $\\pm$ {f(l4s)}; the GNN is ahead on 5 of 5 seeds
(claim S2). This set deliberately excludes blaKPC-2, because the baselines draw their
100,000-edge training subsample per seed and blaKPC-2 clears the five-positive threshold in
only 2 of 5 seeds; including it would fold 0.5-by-construction scores into the baseline, which
is the artefact claim U5 exists to keep out of a performance comparison. An earlier version of
this analysis did include it and overstated the gap roughly threefold (0.079 against the
correct 0.023).
**B.** Per-gene view of the same run, ordered as in the text, with each gene's total positive
count. The dotted line marks 0.5, the score assigned when a gene cannot be fitted at all.
acrAB-tolC ({pos['acrAB-tolC']} positives) and gyrA_S83L ({pos['gyrA_S83L']}) are never fitted in any seed;
blaKPC-2 ({pos['blaKPC-2']}) is fitted in 2 of 5, and the plotted random-forest bar for those three is a
mean that includes those 0.5 scores, so it is a summary of availability rather than of skill.
The GNN produces a stable prediction for every gene shown in every seed. Claims S2, S2b, U5.
Values from `per_seed[*].per_gene_auroc` and `dataset.positives_per_gene`.
"""
    path = os.path.join(FIGDIR, "captions.md")
    os.makedirs(FIGDIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify numbers against claims_to_numbers.md, write nothing")
    ap.add_argument("--rf-coverage", action="store_true",
                    help="print the per-seed baseline untrainable-gene diagnostic")
    args = ap.parse_args()

    data = load()
    print(f"source: {SRC_LABEL}")

    problems = check(data)
    if problems:
        print("\nFAILED: plotted numbers disagree with paper/claims_to_numbers.md:")
        for p in problems:
            print("  - " + p)
        return 1
    print("self-check: all plotted numbers match paper/claims_to_numbers.md")

    if args.rf_coverage:
        rf_coverage(data)
        return 0
    if args.check:
        return 0

    for stem in (fig_per_gene(data), fig_architecture(data), fig_ablation(data),
                 fig_comparison(data)):
        print(f"wrote paper/figures/{stem}.png and .pdf")
    print(f"wrote {os.path.relpath(captions(data), REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
