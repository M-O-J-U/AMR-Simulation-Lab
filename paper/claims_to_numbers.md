# Claim → number → source (for the new paper)

Written 2026-09-30. Every number below is copied from a committed results file; re-read the
file before citing. **No paper text has been drafted** (user instruction). The retired IEEE JBHI
draft (`Downloads/amr_gnn.tex`, AUROC 0.9934, 36-dim) is invalid and is not a target.

**Reference result set** (current pipeline): `ai/checkpoints/reseeded/lab_v2_tuned_noedge_mrsa/results.json`
(+ `summary.md`), with hyperparameters selected in `ai/checkpoints/sweep/lab_v2_mrsa_noedge/`.
Protocol: lab_v2 biology; 5 scenarios (ecoli_cipro, klebsiella_carbapenem, pakistan_crisis,
xdr_acinetobacter, mrsa_hospital) × data seeds 100–102, 80 steps; dosing 0.25 µg/mL cleared
after 5 steps (invented protocol parameters); GNN hidden 128 / lr 1e-3 / 2 GAT layers,
**edge features zeroed**, early-stopping warm-up 20 epochs; RF 300 trees depth 8; model
seeds 0–4 (ablation seeds 0–2); split_seed 42. Dataset SHA-256 `ee83ff385ca15ec9…`:
390 graph pairs, 1,707,498 edges, 1,443 positives (tetM 420, vanA 35, blaTEM-1 5, acrAB-tolC 19,
gyrA_S83L 34, mexAB-oprM 0). "Headline" = macro over genes in `ai/eval_genes.py`
(mexAB-oprM excluded; vanA separate). Values are mean ± SD over seeds.

## Claims the numbers support

| # | Claim | Number(s) | Source |
|---|---|---|---|
| S1 | A tuned GNN predicts per-edge HGT events with high discrimination, stable across seeds | Headline AUROC 0.9805 ± 0.0012 (n=5) | reseeded/lab_v2_tuned_noedge_mrsa/results.json → `summary.gnn.headline_auroc` |
| S2 | **HEADLINE (abstract, main text):** the GNN outperforms Random Forest on the genes RF can train on | GNN **0.9735 ± 0.0017** vs RF **0.8950 ± 0.0386** on {blaCTX-M-15, blaKPC-2, blaNDM-1, mcr-1, tetM}; GNN better on 5/5 seeds | computed from `per_seed[*].{gnn,random_forest}.per_gene_auroc` (decision 2026-09-30, `paper/decisions_log.md`) |
| S2b | **Stated separately, as a real advantage of the approach:** the per-gene RF/LR baselines as implemented (one-vs-rest, ≥5 positives in the 100k-edge training subsample; cutoff NOT lowered) can't be trained for rare genes, while the GNN, trained jointly on all genes, gives evaluable predictions for them | RF/LR untrainable for acrAB-tolC and gyrA_S83L (scored 0.5 by construction). GNN evaluable on 7 of 8 headline genes (blaTEM-1: no test positives, so no model can be scored on it) + vanA separately = 8. GNN on the RF-untrainable genes: acrAB-tolC 0.9986 ± 0.0018, gyrA_S83L 0.9977 ± 0.0006 — **quote with their total positive counts (19 and 34)** | same file → `per_gene_auroc`; `dataset.positives_per_gene` |
| S3 | …and Logistic Regression | Headline LR 0.7096 ± 0.0359 | same file → `summary.logistic_regression.headline_auroc` |
| S4 | Message passing adds a small but consistent gain | GNN − no-message-passing: +0.0031 ± 0.0014 headline AUROC, 5/5 seeds (0.9805 vs 0.9774 ± 0.0006) | same file → `comparisons.gnn_minus_graph_free`; also sweep/lab_v2_mrsa_noedge/summary.md |
| S5 | Carried resistance genes are the dominant feature group | Ablation "No genomic genes" Δ = −0.0549 ± 0.0331 (n=3); all other groups within ±0.003 | same file → `ablation` |
| S6 | Predicted probabilities are well calibrated | GNN ECE 0.0012 ± 0.0003 | same file → `summary.gnn.ece` |
| S7 | Headline AUPRC far above base rate (extreme imbalance) | GNN headline AUPRC 0.0316 ± 0.0293 (high variance); base rate ≈ 8.5e-5 (1,443 / (1,707,498 × 10)) | same file → `summary.gnn.headline_auprc`; `dataset` |
| S8 | Results are reproducible: deterministic data; seeded training | Simulation + training pairs byte-identical under different PYTHONHASHSEED (tests/test_paper_v1_frozen.py for paper_v1); GPU retrain same seed max |Δ| 2.1e-4 | tests/test_paper_v1_frozen.py; results.json → `reproducibility_check` |
| S9 | Per-gene discrimination (GNN) | blaCTX-M-15 0.9642 ± 0.0030; blaKPC-2 0.9564 ± 0.0021; blaNDM-1 0.9755 ± 0.0039; mcr-1 0.9908 ± 0.0011; tetM 0.9805 ± 0.0033; acrAB-tolC 0.9986 ± 0.0018 and gyrA_S83L 0.9977 ± 0.0006 (**few positives: 19 / 34 total**) | same file → `per_gene_auroc.gnn` |

## Claims the numbers do NOT support (do not make them)

| # | Claim (from the retired draft or earlier reasoning) | Why not |
|---|---|---|
| U1 | AUROC 0.9934 / RF 0.9896 / LR 0.9746 | Pre-leakage-fix run (2026-06-07); 36-dim nodes incl. sos_active, 8-dim edges; confirmed by exact parameter count (CLAUDE.md "Leakage-fix provenance") |
| U2 | "Graph structure drives prediction" / large graph advantage | Message passing adds only +0.003 (S4); edge features add nothing (below) |
| U3 | Edge features carry signal | Tuned: GNN with vs without edge features ≈ equal on paper_v1 (0.9665 ± 0.0048 vs 0.9673 ± 0.0035). They are zeroed in the reference set. **Not evidence either way:** the lab_v2 (4-scenario) seed that scored 0.739 with edge features was an early-stopping artifact (a lucky epoch-1 validation peak ended training at epoch 13; reproducible; fixed by the 20-epoch warm-up, after which that seed reaches 0.969 — see `paper/decisions_log.md`), **not** an edge-feature effect, so it must not be cited as a cost of edge features |
| U4 | Antibiotic exposure is the strongest signal (retired: ΔAUROC −0.0284) | Ablation Δ = +0.0001 ± 0.0001 (S5 file). Caveat: under the lab_v2 protocol drug is present for only 5 of 80 steps |
| U5 | GNN beats RF by ~0.20 | The 8-gene headline gap (0.1984) mixes in two genes RF/LR can't train on (scored 0.5 by construction). The GNN-vs-RF number is S2; the coverage difference is stated separately as S2b |
| U6 | vanA prediction is a biological result | vanA AUROC 0.9988 reflects a SIMPLIFIED mechanism (de novo in MRSA, MRSA→MRSA spread; documented route is Tn1546 from *E. faecalis* — Weigel 2003 doi:10.1126/science.1090956; Clark 2005 doi:10.1128/AAC.49.1.470-472.2005) and species confinement; report separately, labelled |
| U7 | External validation against real genomes | Current external validation is non-functional (phenotype tables, no gene calls; Spearman NaN). ResFinder re-annotation = FUTURE WORK |
| U8 | blaTEM-1 performance, or "the GNN evaluates all 8 headline genes" | Not evaluable: 5 positives total, none in test — the GNN is evaluated on 7 of 8 headline genes (+ vanA separately) |
| U9 | Default-hyperparameter numbers (e.g. lab_v2 GNN 0.68) | Superseded; defaults (lr 3e-4) were mistuned |

## Earlier result sets (context for a before/after table; not the reference)

| Setting | GNN | RF | Source |
|---|---|---|---|
| Retired draft (invalid) | 0.9934 | 0.9896 | Downloads/amr_gnn.tex; logs/gnn_training_20260607_013207.log |
| paper_v1, default hparams | 0.9261 ± 0.0211 | 0.9226 ± 0.0263 | reseeded/paper_v1/summary.md |
| paper_v1, tuned (edge features on) | 0.9665 ± 0.0048 | 0.9306 ± 0.0317 | sweep/paper_v1/summary.md |
| lab_v2 4-scenario, default | 0.6802 ± 0.0381 | 0.7739 ± 0.0446 | reseeded/lab_v2/summary.md |
| lab_v2 4-scenario, tuned (edge features on; pre-warm-up) | 0.9272 ± 0.1051 (bimodal, see decisions log) | 0.7862 ± 0.0534 | sweep/lab_v2/summary.md |
| **lab_v2 5-scenario, tuned, no edge features, warm-up (reference)** | **0.9805 ± 0.0012 (headline)** | **0.7822 ± 0.0276 (headline; see U5)** | reseeded/lab_v2_tuned_noedge_mrsa/summary.md |

Note: earlier rows use the all-gene macro; the reference row uses the headline macro
(all-gene macro there: GNN 0.9828 ± 0.0011, RF 0.7594 ± 0.0270).
