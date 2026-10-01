# Claim → number → source (for the new paper)

Updated **2026-10-01**: the reference set was replaced after the run-grouped split was built
(claim U13). Every number below is copied from a committed results file; re-read the file
before citing. The retired IEEE JBHI draft (`Downloads/amr_gnn.tex`, AUROC 0.9934, 36-dim) is
invalid and is not a target.

**Reference result set:** `ai/checkpoints/reseeded/lab_v2_grouped50_runsplit/results.json`
(+ `summary.md`), hyperparameters selected in `ai/checkpoints/sweep/lab_v2_mrsa_noedge/`.
Protocol: lab_v2 biology; 5 scenarios (ecoli_cipro, klebsiella_carbapenem, pakistan_crisis,
xdr_acinetobacter, mrsa_hospital) × data seeds 100–109 (**10 per scenario, 50 runs**), 80 steps;
dosing 0.25 µg/mL cleared after 5 steps (invented protocol parameters); GNN hidden 128 /
lr 1e-3 / 2 GAT layers, **edge features zeroed**, early-stopping warm-up 20 epochs;
RF 300 trees depth 8; model seeds 0–4 (ablation seeds 0–2); split_seed 42.
**Split: by RUN, stratified by scenario** (6 train / 2 val / 2 test runs per scenario =
30/10/10 runs, 780/260/260 graphs). Baseline subsample 303,449 edges, holding the earlier
run's 8.264% sampling fraction. Dataset SHA-256 `7492f6e14601ed8f…`:
1,300 graph pairs, 6,029,316 edges, 4,763 positives. Positives per gene (sum 4,763):
blaCTX-M-15 1,446, tetM 1,230, mcr-1 766, blaNDM-1 554, blaKPC-2 536, gyrA_S83L 109,
vanA 75, blaTEM-1 24, acrAB-tolC 23, mexAB-oprM 0. Test-split positives:
tetM 311, mcr-1 217, blaCTX-M-15 212, blaKPC-2 61, gyrA_S83L 24, blaNDM-1 16,
acrAB-tolC 5, vanA 5, blaTEM-1 0, mexAB-oprM 0.
"Headline" = macro over genes in `ai/eval_genes.py` (mexAB-oprM excluded; vanA separate).
Values are mean ± SD over model seeds.

**Superseded sets, retained ONLY as the historical demonstration of split inflation (U13).
Never quote them as current results:**
- `lab_v2_tuned_noedge_mrsa` — 15 runs, pair-level split. The former reference.
- `lab_v2_grouped50_pairsplit` — 50 runs, pair-level split. Same data as the current
  reference, differing only in the split, so the pair of them isolates the split effect from
  the data-size effect. Reproduce the three-way comparison with
  `python paper/compare_split_arms.py`.

## Claims the numbers support

All values from the run-grouped 50-run reference set unless stated. Where a claim changed on
2026-10-01, the previous value is given so the change is visible rather than silent.

| # | Claim | Number(s) | Source / change |
|---|---|---|---|
| S1 | A tuned GNN predicts per-edge HGT events with high discrimination | Headline AUROC **0.9676 ± 0.0076** (n=5); all-gene macro 0.9551 ± 0.0120 | `summary.gnn.headline_auroc`. **Was 0.9805 ± 0.0012** on the pair-split 15-run set. Of the −0.0129 change, +0.0047 is more data and **−0.0176 is the split correction** (U13) |
| S2 | **HEADLINE:** the GNN outperforms Random Forest on the genes RF can be fitted to in every seed | **Five genes** at this reference set {blaCTX-M-15, **blaKPC-2**, blaNDM-1, mcr-1, tetM}: GNN **0.9805 ± 0.0021** vs RF **0.9647 ± 0.0031**; gap **0.0158**; GNN ahead **5/5 seeds** | computed from `per_seed[*].per_gene_auroc`; reproduce with `python paper/compare_split_arms.py`. The set follows the stated criterion applied to the current data — blaKPC-2 is now fitted in 5/5 seeds (it was 2/5 at 15 runs, which is why it was excluded then). **Continuity figure** on the earlier 4-gene set: GNN 0.9816 ± 0.0020 vs RF 0.9605 ± 0.0037, gap 0.0211, 5/5. **Robust either way:** the 4-gene gap was 0.0229 (15 runs, pair) and 0.0204 (50 runs, pair) — stable to within 0.003 across a 3.3× data increase and the split fix |
| S2b | ⚠ **DEMOTED 2026-10-01 — no longer a standalone contribution; report as an inconclusive finding.** A per-gene baseline cannot be fitted at all for a gene with too few positives, while a jointly trained model still emits a prediction | At 50 runs this holds for **acrAB-tolC alone** (RF fitted 0/5 seeds; GNN 0.8890 ± 0.0506 on **5 test positives**). The other two genes it used to rest on are now fitted: gyrA_S83L 3/5, vanA 5/5; blaKPC-2 5/5. **The effect shrinks as data grows**, so it describes a small-data regime, not the method | `per_seed[*].random_forest.per_gene_auroc` (0.5 = not fitted; NaN = no test positives). Was: acrAB-tolC + gyrA_S83L never fitted, blaKPC-2 2/5 |
| S3 | …and Logistic Regression | Headline LR **0.7911 ± 0.0336**; RF headline 0.8711 ± 0.0366 | `summary`. Was LR 0.7096 ± 0.0359, RF 0.7822 ± 0.0276 |
| S4 | ⚠ **NO LONGER SUPPORTED as a consistent gain.** Message passing shows **no reliable benefit** under the corrected evaluation | GNN − no-message-passing = **+0.0065 ± 0.0086**, positive in only **3 of 5 seeds** (values +0.0188, −0.0005, +0.0052, +0.0112, −0.0022). The SD exceeds the mean and two seeds are negative | `comparisons.gnn_minus_graph_free`. **Was +0.0031 ± 0.0014, 5/5 seeds** under the pair split — the consistency was an artefact of that split. Do not write "small but consistent" |
| S5 | Carried resistance genes are the dominant feature group | "No genomic genes" Δ = **−0.0442 ± 0.0071** (n=3). **No other group is distinguishable from zero** — every one has an SD at least as large as its own mean (largest: species −0.0046 ± 0.0102, spatial +0.0052 ± 0.0083, physiological +0.0051 ± 0.0125) | `ablation`. Was −0.0549 ± 0.0331 with others "within ±0.003"; the new estimate is far tighter |
| S6 | Predicted probabilities are well calibrated | GNN ECE **0.0012 ± 0.0003** | `summary.gnn.ece`. Unchanged |
| S7 | Headline AUPRC far above base rate | **0.0145 ± 0.0057**; base rate ≈ 7.9e-5 (4,763 / (6,029,316 × 10)) | `summary.gnn.headline_auprc`. Was 0.0316 ± 0.0293 — still unstable but less so |
| S8 | Results are reproducible: deterministic data; seeded training | Simulation + training pairs byte-identical under different PYTHONHASHSEED (`tests/test_paper_v1_frozen.py` for paper_v1); GPU retrain same seed max \|Δ\| **1.6e-3** | `reproducibility_check`. **Was 2.1e-4** — the larger spread comes with the harder grouped task; quote the new figure |
| S9 | Per-gene discrimination (GNN) | blaNDM-1 0.9860 ± 0.0054 (16 test pos); tetM 0.9844 ± 0.0010 (311); gyrA_S83L 0.9822 ± 0.0049 (24); blaCTX-M-15 0.9821 ± 0.0024 (212); blaKPC-2 0.9761 ± 0.0045 (61); mcr-1 0.9738 ± 0.0018 (217); **acrAB-tolC 0.8890 ± 0.0506 (5)**; **vanA 0.8669 ± 0.0504 (5)** | `per_gene_auroc.gnn`. **Quote with test-positive counts.** The former near-perfect rare-gene values (acrAB-tolC 0.9986, gyrA_S83L 0.9977, vanA 0.9988) were **same-run leakage** under the pair split — see U13 |
| S10 | The evaluation is almost entirely within-species | Re-measured on the 50-run dataset; see `paper/measure_cross_species_edges.py --data-seeds 10`. Cross-species contacts remain a small minority and carry no transfers | measured; invariant pinned by `tests/test_cross_species.py` |

## Claims the numbers do NOT support (do not make them)

| # | Claim (from the retired draft or earlier reasoning) | Why not |
|---|---|---|
| U1 | AUROC 0.9934 / RF 0.9896 / LR 0.9746 | Pre-leakage-fix run (2026-06-07); 36-dim nodes incl. sos_active, 8-dim edges; confirmed by exact parameter count (CLAUDE.md "Leakage-fix provenance") |
| U2 | "Graph structure drives prediction" / large graph advantage | Message passing adds only +0.003 (S4); edge features add nothing (below) |
| U3 | Edge features carry signal | Tuned: GNN with vs without edge features ≈ equal on paper_v1 (0.9665 ± 0.0048 vs 0.9673 ± 0.0035). They are zeroed in the reference set. **Not evidence either way:** the lab_v2 (4-scenario) seed that scored 0.739 with edge features was an early-stopping artifact (a lucky epoch-1 validation peak ended training at epoch 13; reproducible; fixed by the 20-epoch warm-up, after which that seed reaches 0.969 — see `paper/decisions_log.md`), **not** an edge-feature effect, so it must not be cited as a cost of edge features |
| U4 | Antibiotic exposure is the strongest signal (retired draft: ΔAUROC −0.0284) | Ablation Δ = **+0.0006 ± 0.0011** (removing it does not hurt). Caveat unchanged: under the lab_v2 protocol drug is present for only 5 of 80 steps, so this holds for this protocol, not for selection pressure in general. Was +0.0001 ± 0.0001 |
| U5 | GNN beats RF by the headline-macro difference | At 50 runs the 8-gene headline gap is 0.0965 (0.9676 vs 0.8711), but it still mixes in acrAB-tolC, which RF cannot fit in any seed and which is scored 0.5 by construction. The GNN-vs-RF number is S2 (0.0211 on the always-fitted genes); coverage is S2b. **Do not quote the headline-macro difference as a performance gap** (was 0.1984 on the old set) |
| U6 | vanA prediction is a biological result | vanA AUROC 0.8669 ± 0.0504 (5 test positives) reflects a SIMPLIFIED mechanism (de novo in MRSA, MRSA→MRSA spread; documented route is Tn1546 from *E. faecalis* — Weigel 2003 doi:10.1126/science.1090956; Clark 2005 doi:10.1128/AAC.49.1.470-472.2005) and species confinement; report separately, labelled |
| U7 | External validation against real genomes | Current external validation is non-functional (phenotype tables, no gene calls; Spearman NaN). ResFinder re-annotation = FUTURE WORK |
| U8 | blaTEM-1 performance, or "the GNN evaluates all 8 headline genes" | Still not evaluable at 50 runs: 24 positives dataset-wide but **0 in the test split**, so no model can be scored on it. The GNN is evaluated on 7 of 8 headline genes (+ vanA separately). (Under the superseded 50-run *pair* split it was evaluable at 0.9985 — another instance of U13, not a result.) |
| U9 | Default-hyperparameter numbers (e.g. lab_v2 GNN 0.68) | Superseded; defaults (lr 3e-4) were mistuned |
| U14 | The intraspecies-only restriction makes the task easier by supplying many trivially separable (cross-species) negatives | **Measured and refuted 2026-09-30** — see S10: cross-species contacts are 0.275% of the dataset, so they cannot be padding the negative class. This argument was drafted into the Abstract, Discussion §6.1 and Limitations §5.2 before being checked, and was removed from all three. The correct statement of the U11 consequence is *absence*, not class balance: the interspecies case is effectively missing from the evaluation, so the model is assessed almost entirely on within-species pairs. Any future draft must not reintroduce the class-balance version |
| U13 | ✅ **RESOLVED 2026-10-01.** (Was: the held-out test set measures generalisation to unseen simulations.) The reference set now splits by RUN, stratified by scenario, so test runs are never seen in training | **The pair-level split was inflating the headline by 0.0176 AUROC** (0.9852 pair vs 0.9676 run, same 50-run data). It inflated the rare genes far more: acrAB-tolC 0.9986→0.8890, vanA 0.9988→0.8669. Residual limitation: the test runs are unseen *runs*, but still the same five scenarios and five species — generalisation to unseen scenarios or species is still untested |
| U11 | The simulated transfers model HGT between bacteria in general | `_attempt_hgt` transfers **only between cells of the same species**, so every cross-species contact is a negative by construction. Real transfer of these genes crosses species and genera (mcr-1 plasmid maintained in E. coli, K. pneumoniae and P. aeruginosa — Liu 2016; Tn916 tet(M) *Lactococcus*→*Enterococcus* — Boguslawska 2009; NDM-1 plasmid *Citrobacter*→E. coli — Dolejska 2012). Disclose in Methods and Limitations; see `paper/gene_mechanism_audit.md` §3.1 |
| U12 | The per-gene positive counts, or S9's per-gene ordering, reflect gene epidemiology | **Independently reconfirmed 2026-10-01 by tripling the data.** If counts were a gene property every gene would scale with the dataset (3.53×). They scale from **1.21× (acrAB-tolC 19→23) to 5.70× (blaKPC-2 94→536)** — a 4.7-fold spread. No acquired gene exists at t=0; each is seeded by `_attempt_mutation`'s donor-free branch, then amplified by chance. Report S9 as discrimination only, never as biology; see `paper/gene_mechanism_audit.md` §3.2 |
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
