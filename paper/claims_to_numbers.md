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
390 graph pairs, 1,707,498 edges, 1,443 positives. Positives per gene (all 10, read from
`results.json` → `dataset.positives_per_gene` on 2026-09-30; they sum to 1,443):
blaCTX-M-15 480, tetM 420, mcr-1 179, blaNDM-1 177, blaKPC-2 94, vanA 35, gyrA_S83L 34,
acrAB-tolC 19, blaTEM-1 5, mexAB-oprM 0. "Headline" = macro over genes in `ai/eval_genes.py`
(mexAB-oprM excluded; vanA separate). Values are mean ± SD over seeds.

## Claims the numbers support

| # | Claim | Number(s) | Source |
|---|---|---|---|
| S1 | A tuned GNN predicts per-edge HGT events with high discrimination, stable across seeds | Headline AUROC 0.9805 ± 0.0012 (n=5) | reseeded/lab_v2_tuned_noedge_mrsa/results.json → `summary.gnn.headline_auroc` |
| S2 | **HEADLINE (abstract, main text):** the GNN outperforms Random Forest on the genes RF can train on | GNN **0.9735 ± 0.0017** vs RF **0.8950 ± 0.0386** on {blaCTX-M-15, blaKPC-2, blaNDM-1, mcr-1, tetM}; GNN better on 5/5 seeds | computed from `per_seed[*].{gnn,random_forest}.per_gene_auroc` (decision 2026-09-30, `paper/decisions_log.md`) |
| S2b | **Stated separately, as a real advantage of the approach:** the per-gene RF/LR baselines as implemented (one-vs-rest, ≥5 positives in the 100k-edge training subsample; cutoff NOT lowered) can't be trained for rare genes, while the GNN, trained jointly on all genes, gives evaluable predictions for them | RF/LR untrainable for acrAB-tolC and gyrA_S83L (scored 0.5 by construction). GNN evaluable on 7 of 8 headline genes (blaTEM-1: no test positives, so no model can be scored on it) + vanA separately = 8. GNN on the RF-untrainable genes: acrAB-tolC 0.9986 ± 0.0018, gyrA_S83L 0.9977 ± 0.0006 — **quote with their total positive counts (19 and 34)**, and **label both as SIMPLIFIED MECHANISMS** (see U10): the coverage advantage is real, but these two genes' "transfer events" are not documented conjugative transfer | same file → `per_gene_auroc`; `dataset.positives_per_gene` |
| S3 | …and Logistic Regression | Headline LR 0.7096 ± 0.0359 | same file → `summary.logistic_regression.headline_auroc` |
| S4 | Message passing adds a small but consistent gain | GNN − no-message-passing: +0.0031 ± 0.0014 headline AUROC, 5/5 seeds (0.9805 vs 0.9774 ± 0.0006) | same file → `comparisons.gnn_minus_graph_free`; also sweep/lab_v2_mrsa_noedge/summary.md |
| S5 | Carried resistance genes are the dominant feature group | Ablation "No genomic genes" Δ = −0.0549 ± 0.0331 (n=3); all other groups within ±0.003 | same file → `ablation` |
| S6 | Predicted probabilities are well calibrated | GNN ECE 0.0012 ± 0.0003 | same file → `summary.gnn.ece` |
| S7 | Headline AUPRC far above base rate (extreme imbalance) | GNN headline AUPRC 0.0316 ± 0.0293 (high variance); base rate ≈ 8.5e-5 (1,443 / (1,707,498 × 10)) | same file → `summary.gnn.headline_auprc`; `dataset` |
| S8 | Results are reproducible: deterministic data; seeded training | Simulation + training pairs byte-identical under different PYTHONHASHSEED (tests/test_paper_v1_frozen.py for paper_v1); GPU retrain same seed max |Δ| 2.1e-4 | tests/test_paper_v1_frozen.py; results.json → `reproducibility_check` |
| S10 | The evaluation is almost entirely within-species: cross-species contacts are rare and never carry a transfer | Cross-species edges **4,696 of 1,707,498 = 0.275%**; they occur in `pakistan_crisis` only (4,696 of its 192,984 edges); the other four scenarios are single-species and have 0. **Positives on cross-species edges: 0**; all 1,443 positives are same-species (as `_attempt_hgt` implies). Cross-species edge-gene pairs are 0.275% of the negative class | Measured 2026-09-30 by deterministically regenerating the reference dataset with the committed config (`ai/gnn_trainer.DEFAULT_CONFIG` + `scenarios_for`/`dosing_for`) and counting per edge. The regeneration reproduced the committed totals exactly (390 pairs / 1,707,498 edges / 1,443 positives), which independently re-confirms S8. Script kept in the session scratchpad, not the repo |
| S9 | Per-gene discrimination (GNN) — **discrimination only; see U12, the counts are set by the seeding code, not by gene biology** | blaCTX-M-15 0.9642 ± 0.0030; blaKPC-2 0.9564 ± 0.0021; blaNDM-1 0.9755 ± 0.0039; mcr-1 0.9908 ± 0.0011; tetM 0.9805 ± 0.0033; acrAB-tolC 0.9986 ± 0.0018 and gyrA_S83L 0.9977 ± 0.0006 (**few positives: 19 / 34 total**) | same file → `per_gene_auroc.gnn` |

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
| U14 | The intraspecies-only restriction makes the task easier by supplying many trivially separable (cross-species) negatives | **Measured and refuted 2026-09-30** — see S10: cross-species contacts are 0.275% of the dataset, so they cannot be padding the negative class. This argument was drafted into the Abstract, Discussion §6.1 and Limitations §5.2 before being checked, and was removed from all three. The correct statement of the U11 consequence is *absence*, not class balance: the interspecies case is effectively missing from the evaluation, so the model is assessed almost entirely on within-species pairs. Any future draft must not reintroduce the class-balance version |
| U13 | The held-out test set measures generalisation to unseen simulations | `split_dataset` shuffles **graph pairs**, not runs. Each run yields ~26 pairs three steps apart, so snapshots from one run (same founders, same seed, largely the same cells) land in train, val and test. The test set measures generalisation to new time windows of seen runs, not to unseen runs/scenarios/species. Found 2026-09-30 while reading for Methods; disclosed in Methods §3.3.3 and to be repeated in Limitations. A grouped (by-run) split would be the fix and would change every number, so it is future work |
| U11 | The simulated transfers model HGT between bacteria in general | `_attempt_hgt` transfers **only between cells of the same species**, so every cross-species contact is a negative by construction. Real transfer of these genes crosses species and genera (mcr-1 plasmid maintained in E. coli, K. pneumoniae and P. aeruginosa — Liu 2016; Tn916 tet(M) *Lactococcus*→*Enterococcus* — Boguslawska 2009; NDM-1 plasmid *Citrobacter*→E. coli — Dolejska 2012). Disclose in Methods and Limitations; see `paper/gene_mechanism_audit.md` §3.1 |
| U12 | The per-gene positive counts, or S9's per-gene ordering, reflect gene epidemiology | No acquired gene exists at t=0; each is seeded by `_attempt_mutation`'s donor-free branch, which draws uniformly from the species' acquirable pool, then spreads. The counts are therefore an outcome of the seeding code plus stochastic amplification. **Do not claim a tidier law than that** — checked against the full counts 2026-09-30: availability and extra seeding routes clearly matter (vanA 35 from a 2-gene MRSA pool vs blaTEM-1 5 from a 5-gene E. coli pool; acrAB-tolC 19 has an extra SOS seeding route, blaTEM-1 has none), **but the counts are not monotonic in `acquisition_prob`** — blaKPC-2 (94), blaNDM-1 (177) and mcr-1 (179) have identical species availability and probabilities 0.02/0.015/0.02, so a ~2× spread remains unexplained by parameters and is attributable to when each gene happened to seed. Report S9 as per-gene discrimination only, never as biology; see `paper/gene_mechanism_audit.md` §3.2 |
| U10 | gyrA_S83L / acrAB-tolC "transfer events" are conjugative transfer | Both are chromosomal in real biology (gyrA_S83L a point mutation; AcrAB-TolC a chromosomal RND efflux system), but `core/bacterium_agent.py::_attempt_hgt` moves them between same-species neighbours like any other gene. Their events are a **SIMPLIFIED MECHANISM**, same category as vanA (U6). Decided 2026-09-30: label, do not rerun. The S2b coverage advantage still holds (it is a property of the learner, not of the genes), but must be stated with this label. Sourced 2026-09-30 (Hooper & Jacoby 2015: target mutations vs the distinct plasmid-borne qnr/efflux genes; Li 2015: AcrAB-TolC chromosomal and ubiquitous) — with the caveat that plasmid-borne RND pump genes do exist, so the objection is to transferring E. coli's *resident* AcrAB-TolC, not to RND transfer in general. Full detail: `paper/gene_mechanism_audit.md` |

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
