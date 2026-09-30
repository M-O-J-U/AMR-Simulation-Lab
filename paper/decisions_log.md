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
- **Fair RF comparison policy: OPEN (not decided).**
  - Fact: `run_baselines` doesn't train RF/LR on a gene with fewer than 5 positives in its
    100k-edge training subsample; that gene scores 0.5 by construction. In the reference
    set this affects acrAB-tolC and gyrA_S83L, which inflates the headline gap
    (0.198 vs 0.079 on the genes RF trained on).
  - Options put to the owner: (a) compare only on genes both models trained on (as in
    `claims_to_numbers.md` S2); (b) let RF train on rare genes (lower the 5-positive cutoff,
    or subsample stratified by gene) and rerun.
  - Until decided, cite S2, not the headline gap.
- **Next steps: OPEN.** Lab Phases 3–4 aren't started. No paper text.
