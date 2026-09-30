# Decisions log (new-paper / lab work)

Dated record of decisions and their evidence. **OPEN** = awaiting the project owner.
See also `CLAUDE.md` (rules, paper-phase notes, RESUME HERE) and `paper/claims_to_numbers.md`.

## 2026-09-29
- The IEEE JBHI manuscript was rejected on 2026-06-29 and not resubmitted. It's retired
  (`Downloads/amr_gnn.tex`, AUROC 0.9934); a new paper will be written from the current
  pipeline. No paper text until the owner says so.
- **Two-version biology:** `paper_v1` is frozen (byte-identical guard); `lab_v2` is corrected
  (mecA; EUCAST rule 1.7 for Klebsiella; mass-conserving diffusion). The pipeline default is `lab_v2`.
- Seeding bugs fixed (sorted gene iteration; seeded graph subsampling; seeded LR).
- All 10 CARD IDs corrected; Table 1 is generated from code (`paper/make_table1.py`).

## 2026-09-30
- **lab_v2 dosing (approved):** 0.25 µg/mL, cleared after 5 steps (invented protocol
  parameters; `TRAINING_PROTOCOLS` in `ai/gnn_trainer.py`). paper_v1 keeps 1.5 µg/mL, never
  cleared. Follow-up: a real step duration plus sourced half-lives.
- **Future work, not blockers (approved):** ResFinder/ARIBA re-annotation of BV-BRC data
  (~1–2 weeks); the software-paper track. Both are to be named in the new paper's
  Limitations/Future Work.
- **Rare genes (approved):**
  - Add the existing `mrsa_hospital` scenario to lab_v2 training. This makes tetM
    evaluable (420 positives).
  - **mexAB-oprM: excluded from evaluation.** It's intrinsic and never acquirable in the
    model, so it has zero transfer events by construction. CARD ARO:3000386 describes it as
    a *P. aeruginosa* system. Separate flag, unfixed: the model gives it to *A. baumannii*
    as intrinsic.
  - **vanA: evaluated but reported separately**, labelled as a SIMPLIFIED MECHANISM (the
    documented route is Tn1546 from *E. faecalis*: Weigel 2003, Clark 2005; *Enterococcus*
    isn't modelled). It is never in the headline macro.
  - Policy code: `ai/eval_genes.py`.
- **blaTEM-1 (observation, no decision taken):** in the reference dataset it has 5 positives
  in total and none in the test split, so it's reported as "not evaluable". Options if
  wanted: more data seeds, or a scenario with TEM-1 donors. **OPEN**, not proposed.
- **Bimodal seed: diagnosed and fixed.**
  - Symptom: in the lab_v2 (4-scenario) tuned GNN with edge features, seed 0 scored 0.7392
    while the other seeds scored 0.97–0.98.
  - Diagnosis: a reproducible early-stopping artifact (identical on retraining).
    Validation AUROC peaked at epoch 1 (0.787, an untrained model), dipped to 0.59, and was
    recovering when patience-12 stopped training at epoch 13. That kept the epoch-1 model.
    It was worse on every gene.
  - Fix: early-stopping warm-up, `min_epochs = 20` (DEFAULT_CONFIG). Seed 0 goes to 0.969;
    a normal seed is unchanged (0.9706).
  - Applied to all new runs. In the reference set, all 5 seeds trained 27–38 epochs
    (best epoch 15–32).
  - Evidence: `ai/gnn_trainer.py` comment; commit 2e8fcd1. The investigation scripts were in
    the session scratchpad (not kept).
- **Edge features zeroed in the reference set (approved as part of the combined rerun).**
- **Fair RF comparison policy: DECIDED — report both.**
  - **Headline (abstract, main text):** GNN **0.9735 ± 0.0017** vs RF **0.8950 ± 0.0386**,
    on the 5 genes RF can train on (blaCTX-M-15, blaKPC-2, blaNDM-1, mcr-1, tetM).
    GNN better on 5/5 seeds.
  - **Separately:** the RF/LR baselines can't be trained below the 5-positive cutoff, while
    the GNN is trained jointly on all genes and gives evaluable predictions for them.
    Framed as a real advantage of the approach, not an artifact.
  - **RF's cutoff is NOT lowered.** The existing baseline stays as is.
  - **Accuracy notes for the framing** (added when recording the decision):
    - Scope the claim to the baselines *as implemented*: per-gene one-vs-rest, with at least
      5 positives needed in the 100k-edge training subsample. The cutoff is a property of
      this baseline design, not of Random Forests in general.
    - The GNN is *evaluated* on 7 of the 8 headline genes: blaTEM-1 has no test positives
      (5 in total), so no model can be scored on it. Adding vanA (reported separately)
      gives 8 evaluated genes. It is *trained* on all 10 outputs jointly.
    - The two genes only the GNN covers have few positives in total (acrAB-tolC 19,
      gyrA_S83L 34); their GNN AUROCs (0.9986, 0.9977) should be quoted with those counts.
  - The headline gap on all 8 genes (0.1984) is **not** to be quoted as the GNN-vs-RF
    difference.
  - Previously OPEN (options offered: compare on common genes, or let RF train on rare
    genes); decided by the owner 2026-09-30.
- **Next steps: OPEN.** Lab Phases 3–4 aren't started. No paper text.

## 2026-09-30 (later, found during the Related Work citation audit)
- **DECIDED 2026-09-30 (owner): label as simplified mechanisms, do NOT rerun.**
  gyrA_S83L and acrAB-tolC are handled like vanA (U6): the coverage advantage in S2b stays,
  stated with the label. Recorded as `U10` in `paper/claims_to_numbers.md`; Introduction
  contribution 3 discloses it. Methods and Limitations must repeat the label.
  **Future work** (alongside the mecA/SCCmec note in the lab-phase section of CLAUDE.md):
  the simulator has exactly one transfer mechanism, conjugation-style transfer between
  same-species neighbours, and it is applied to every gene regardless of that gene's real
  mobility. A version that models per-gene mobility (conjugative plasmid vs transposon vs
  chromosomal point mutation vs SCCmec) would need new biology and a full rerun.
- The finding, as originally logged:
  - `core/bacterium_agent.py` `_attempt_hgt` transfers every carried gene that's in the
    recipient's `acquired_resistance_pool` and not in `_no_transfer`.
  - `gyrA_S83L` is a chromosomal point mutation. The code arises it by mutation
    (`_attempt_mutation`); `data/card_loader.py` comments it "point mutation, not HGT". It is
    *also* conjugated between same-species neighbours.
  - `acrAB-tolC` is a chromosomal efflux system. It is gained via SOS upregulation *and*
    conjugated.
  - These are exactly the two genes in claim S2b (GNN-only coverage, 34 and 19 positives).
    Their "transfer events" are a simplified mechanism, like vanA (U6), not documented
    conjugative transfer.
  - Options offered: (1) label, (2) add to `_no_transfer` in lab_v2 and rerun, (3) both.
    Option 1 was chosen. No code changed; no rerun.
- **Also noted: invented per-step transfer probabilities.** `acquisition_prob` values in
  `data/card_loader.py` (0.01–0.05) have no cited source. They must be labelled as invented
  parameters in Methods (CLAUDE.md rule 7).

- **Full transfer-mechanism audit of all 11 genes** (owner-requested before Methods;
  `paper/gene_mechanism_audit.md`, 11 sources verified against their own text). No code
  changed, no number changed. Outcome:
  - 6 genes are mobile in the right kind of way: blaTEM-1, blaCTX-M-15, blaKPC-2,
    blaNDM-1, mcr-1 (plasmids) and tetM (Tn916 conjugative transposon).
  - 3 are simplified: gyrA_S83L and acrAB-tolC (U10, now sourced) and vanA (U6).
  - 2 are correctly never transferred: mexAB-oprM (in no acquirable pool → 0 events) and
    mecA (`non_transferable`).
  - **New, U11:** transfer is same-species only, so every cross-species contact is a
    negative by construction, while real transfer of these genes crosses species and
    genera. Must be disclosed in Methods and Limitations.
  - **New, U12:** no acquired gene exists at t=0; each is seeded by a donor-free branch of
    `_attempt_mutation` that draws uniformly from the species' acquirable pool. The per-gene
    positive counts therefore track pool size and `acquisition_prob`, i.e. the seeding code,
    not gene epidemiology. This is the mechanical explanation of U8 (blaTEM-1, 5 positives)
    and it constrains how S9 and S2b may be worded.
  - **OPEN (§3.4):** `gyrA_S83L` is CARD's *E. coli* gyrA entry but sits in Klebsiella's and
    P. aeruginosa's pools; no source found establishing S83L as the dominant variant in all
    three. Recommendation: describe it generically in Methods ("a gyrA target-site
    mutation") or restrict the claim to E. coli. **Owner decision on wording.**
  - **OPEN, IMPORTANT — "S2 contamination" (found 2026-09-30 while preparing figures).
    The headline comparison is contaminated by the very artefact U5 warns about.**
    - S2 states: GNN 0.9735 ± 0.0017 vs RF 0.8950 ± 0.0386 on
      {blaCTX-M-15, **blaKPC-2**, blaNDM-1, mcr-1, tetM}, "the genes RF can train on".
      Those two numbers reproduce exactly from `per_seed`, so the arithmetic is right.
    - **But RF scores exactly 0.5 — its untrainable marker — on blaKPC-2 in 3 of 5 seeds**
      (seeds 0, 2, 4); LR likewise. blaKPC-2 has 94 positives dataset-wide, but the
      100k-edge subsample is drawn per seed, so in 3 seeds it fell below the 5-positive
      cutoff. blaKPC-2 therefore is NOT a gene "RF can train on"; it is trainable in 2 of
      5 seeds.
    - Consequence: RF's 0.8950 is depressed by 0.5-by-construction entries, which is
      exactly what U5 says must not be folded into a performance comparison. RF's large SD
      (0.0386) is the tell — it is bimodal between seeds where blaKPC-2 trained (0.9399,
      0.9342) and seeds where it did not (0.8654, 0.8627, 0.8730).
    - On the **4 genes RF fits in all 5 seeds** {blaCTX-M-15, blaNDM-1, mcr-1, tetM}:
      **GNN 0.9777 ± 0.0023 vs RF 0.9548 ± 0.0124, GNN ahead on 5/5 seeds.**
      The gap falls from 0.0785 to 0.0229 — roughly a third of the size.
    - So the direction of S2 holds (GNN ahead, every seed, on either subset) but **the
      current headline overstates the margin by about 3×.**
    - Options (not chosen):
      1. Restate the headline on the 4-gene all-seeds-trainable set (0.9777 vs 0.9548) and
         report blaKPC-2 with the other partially-trainable genes under the coverage claim.
      2. Keep 5 genes but define trainability per seed and average only over seeds where a
         gene trained — changes what the mean means, and needs care.
      3. Keep S2 as-is and disclose the blaKPC-2 3/5 issue in the text.
      Recommendation: **option 1.** It is the only one that makes the headline mean what it
      says, and it strengthens rather than weakens the paper's credibility; the coverage
      claim absorbs blaKPC-2 naturally, since partial trainability is the same phenomenon
      as no trainability.
    - Touches the headline number: **owner decision.** No claim changed, no figure drawn.
      Reproduce with `python paper/make_figures.py --rf-coverage`.
    - The headline comparison FIGURE is deliberately not generated until this is settled.
  - **OPEN — code-comment claims found while reading for Methods (2026-09-30). No code
    changed.** These are claims in docstrings, i.e. CLAUDE.md rules 2 and 7 territory:
    1. **`ai/gnn_model.py` cites "Orenstein et al. 2021 — GNNs for microbial ecology".
       I could not find this paper** on PubMed (no Orenstein hit for graph neural networks
       or microbial ecology) or on the web. It looks like a fabricated citation of exactly
       the kind the SLURP audits found. **Recommendation: delete the line.** Not done
       unilaterally because it is a claim change.
    2. Same file claims the model is "validated against CARD transfer rates". No such
       validation exists anywhere in the repo. **Recommendation: delete.**
    3. Same file claims "First GNN applied directly to agent-based AMR simulation state" —
       an unqualified novelty claim. The Related Work draft states the weaker, defensible
       version ("we did not find prior work that…"). **Recommendation: delete or soften.**
    4. Stale docstrings that contradict the code: `gnn_model.py` says 3 GAT layers and
       8-dim edges (reference config is 2 layers, 5-dim); `gnn_trainer.py` says
       "patience=10" (12) and "stratified by scenario" (a plain shuffle);
       `feature_engineering.py`'s return comment says edge features are `(E, 8)` (5).
       `gnn_trainer.py` also asserts "blaTEM-1 transfers very frequently", which the
       dataset contradicts (5 positives — see U12).
    5. `collect_all_data` computes its progress total from 4 scenarios while iterating 5,
       so run counts in logs read "n/12" for 15 runs. Cosmetic.
    Methods was written from the code, not from these docstrings.
  - Disclosures with no identified effect on the numbers: §3.3 invented probabilities,
    §3.5 blaTEM-1 confined to E. coli though common in Klebsiella (Cuzon 2010), §3.6
    A. baumannii carries mexAB-oprM and acrAB-tolC although its RND systems are
    AdeABC/AdeIJK/AdeFGH (Coyne 2011).
