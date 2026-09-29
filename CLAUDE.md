# AMR Simulation Lab — Interactive Upgrade (project first, paper second)

Repo: amr-simulation-lab (Mesa ABM + FastAPI + single-page canvas frontend).
Paper: AMRResistanceGNN, IEEE JBHI, under review — DO NOT touch yet. This phase is the
lab only. Scope: bacteria only, no viral pathogens.

## Non-negotiable rules
1. **Simulation biology and existing validated numbers do not change without approval.**
   The GNN paper's results (AUROC 0.9934, the four benchmark validations, etc.) depend on
   the simulation producing the same distributions it always has. Frontend and
   infrastructure changes must not alter simulation logic, RNG behavior, or step semantics
   unless explicitly asked. If a visual fix requires a simulation change, STOP and flag it
   as a decision point before touching `simulation/` or `core/`.
2. **No claim in the README or UI outruns the code.** The current README says
   "WebSocket" when the frontend actually polls every 2.5s — fix the mismatch by making one
   match the other (see Phase 1), not by leaving the false claim.
3. **Don't add real disease/pathogen data (viral, fungal, parasitic) without approval.**
   Scope is bacteria only. If the ABM design would make later extension easier, note it,
   don't build it now.
4. **Verify biological/pharmacological claims against sources before adding or changing
   them**, same standard as the papers: CARD ARO IDs, EUCAST breakpoints, fitness costs,
   Hill-equation parameters — check the primary source, don't assume the existing numbers
   are right just because they're already in the code.
5. **Never rewrite or delete existing simulation output data/checkpoints
   (`ai/checkpoints/*.json`) without approval** — these back the paper's numbers.
6. **Privacy/publishing**: same as other projects. Repo push only by me. No
   "Co-Authored-By" trailers (confirm the attribution setting is active in this folder
   too). Nothing public without explicit approval.
7. **Never fabricate a citation or a "based on real biology" claim** for any new mechanism
   added to the sim. If something is a simplification or an invented parameter (not from a
   cited source), label it as such in code comments and any UI copy — don't imply it's
   empirically grounded when it isn't.

## STOP-AND-REPORT (standing rule, updated 2026-09-29 — replaces the Decision Points list)
Proceed on own judgment for architecture, file structure, and how much to rebuild vs.
patch — report after, don't ask before. Still STOP (report findings/options/
recommendation and wait for approval) before:
- (a) any biological, pharmacological, or clinical-advice claim, new or changed. Verify
  against a real primary source first (as done for data/expected_resistance.py) and flag
  anything a source doesn't clearly settle rather than presenting it as fact.
- (b) deleting or overwriting anything that isn't recoverable from git history.
- (c) anything that touches the GNN paper's numbers, or logic in `simulation/` / `core/`.
Rule 6 (nothing public, no push) still applies.

## Current state (found during read, verify before relying on this)
- Frontend: one `frontend/index.html`, inline CSS/JS, `<canvas>` 2D rendering. Originally
  polled `GET /state` every 2.5 s (the README's WebSocket claim was false); since Phase 1
  it consumes `WS /ws` and only polls as a fallback. Still single-file (Phase 2 decides).
- Backend (corrected 2026-09-29, Phase 0): the original `api/server.py` had been
  overwritten with a byte-for-byte copy of `ai/gnn_inference.py`, so there was no server
  at all and `python main.py server` failed. It was rebuilt as a REST-only FastAPI server
  matching the frontend's calls (13 routes incl. GNN + analytics), with
  `tests/test_api.py` covering every route. Server defaults to `127.0.0.1`.
- Phase 2 done (2026-09-29): frontend split into `frontend/{index.html,css/,js/,vendor/}`
  (no build step; plain scripts on `window.AMR`; also served at `/ui/`); rAF canvas renderer
  with interpolated motion and event animations; uPlot charts (vendored, MIT); live
  inspector; responsive layout. See commit fed7280 for details and measurements.
- Phase 1 done (2026-09-29): `WS /ws` live stream (`api/stream.py`: snapshot + per-step
  field-level diffs, seq/resync, per-client bounded queues, events incl. birth with
  parent_id, age derived client-side, detail fields only for inspected cells, 8-bit
  heatmaps). Play is server-side (`/resume`, `/pause`, `/speed`). Frontend consumes the
  stream, falls back to polling. `parent_id` is read from the agent by the server;
  `core/` and `simulation/` are unchanged (`git diff a098ff9 -- core simulation` empty).
- `core/antibiotic_agent.py` (0-byte, unreferenced) was deleted as dead code
  (2026-09-29, approved). Antibiotics are concentration grids on the model, diffused and
  decayed in `AMRSimulationModel._diffuse_antibiotics()`; profiles live in
  `data/card_loader.py` (`AntibioticProfile`).
- Root-level `test_gnn.py` (pre-leakage-fix edge layout) archived to
  `archive/pre_leakage_fix/`; `pytest.ini` restricts collection to `tests/`.
- Repo is under git as of 2026-09-29 (local only, never pushed). Raw BV-BRC CSVs
  (`ecoli_amr.csv`, `klebsiella_amr.csv`, ~830 MB) are gitignored.
- Simulation: Mesa 3.x, 80x60 MultiGrid, bacteria agents with SOS response, persister
  switching, biofilm, HGT conjugation, Hill-equation PK/PD killing. 5 pathogen species, 6
  antibiotics, 10 resistance genes from CARD, 7 preset scenarios.
- Tests exist (`tests/test_simulation.py`, `tests/test_gnn.py`) — run them before and
  after any change to confirm nothing broke.
- `requirements.txt` was wrong (found 2026-09-29, Phase 0): it pinned
  `mesa>=2.1.0,<3.0.0`, but the code targets Mesa 3.x and the installed, passing version
  is Mesa 3.5.1. A fresh install from requirements.txt would have produced an environment
  the paper's numbers were never generated on. Now pinned to `mesa==3.5.1`.

## Phase 0 — orientation (do first, no code changes)
1. Run the existing test suite and report pass/fail as a baseline.
2. Run the app locally (`python main.py server`), open the frontend, and describe what
   actually renders and what the real update cadence is (confirm the 2.5s poll).
3. Map every REST endpoint the frontend calls, and note any dead code (starting with
   `core/antibiotic_agent.py`).
4. Report all of this before proposing an architecture. STOP.

## Phase 1 — real-time transport (backend)
- Replace or supplement the polling loop with a WebSocket stream from FastAPI, pushing
  state diffs (not full state) on each simulation step, so the frontend gets push updates
  instead of polling.
- Keep the REST endpoints for actions (step, apply_antibiotic, spawn, reset) — WebSocket
  is for the live state stream, not for commands, unless there's a clear reason to change
  that (propose it, don't just do it).
- Fix the README to match reality once this lands.
- STOP with a design proposal before implementing: message format, diff strategy, what
  happens on reconnect/dropped frames.

## Phase 2 — frontend rebuild (visual + interaction upgrade)
Goal: modern, smooth, informative, still fast at 4800 grid cells with hundreds of agents.
- Propose 2-3 concrete architecture options (e.g., keep canvas2D but redesign the render
  loop and add interpolation/easing between frames; move to WebGL via a lightweight
  library for large-agent-count performance; a hybrid where the petri-dish view stays
  canvas/WebGL and charts move to a proper charting library) with tradeoffs on
  performance, complexity, and how much of the existing 59KB file survives. STOP for a
  choice before building.
- Concrete upgrades to include once an approach is chosen:
  - Smooth interpolated motion between simulation steps (agents shouldn't jump/teleport
    visually even if the sim itself steps discretely) — this is the "motion" the user
    wants, delivered via client-side interpolation/tweening, not an external animation
    tool.
  - Visual encoding upgrades: current dot-color/outline/halo system, made clearer —
    propose specific improvements (size, glow intensity tied to real values, smoother
    color transitions) rather than a full redesign for its own sake.
  - Event visuals: HGT transfer events, death, division, biofilm formation, SOS
    activation — animate these as discrete visual events tied to the WebSocket stream,
    not just static state snapshots.
  - Live charts (population, resistance prevalence, MIC distributions) on a proper
    charting library instead of hand-rolled canvas chart code, still real-time.
  - An inspector/detail panel for a clicked agent, richer than "click a bacterium to
    inspect it" — show its actual state (genes, fitness, stress, biofilm status) live as
    it changes.
  - Responsive layout check — current CSS uses fixed assumptions; verify it works at
    different window sizes.
- Split the single 59KB inline file into organized modules (HTML/CSS/JS separated, JS
  split by concern: rendering, networking, UI controls, charts) unless there's a reason to
  keep it single-file (e.g., if this needs to stay a simple static artifact with no build
  step — ask if unsure).
- Every visual change gets a before/after description (or screenshot if the environment
  allows generating one) — I can't currently see screenshots, so describe changes
  precisely enough that I can picture them, and flag anything genuinely hard to describe
  in text.

## Phase 3 — UX / control upgrades
- Review the 7 scenario presets, antibiotic dosing controls, and spawn controls: propose
  concrete UX improvements (clearer controls, in-app explanations of what's happening
  biologically, maybe a guided "experiment" mode) rather than just re-skinning existing
  controls.
- Consider (propose, don't build silently) a way to compare two runs side by side
  (e.g., resistant strain vs. no treatment), since that's close to what a "virtual lab
  experiment" implies.

## Phase 4 — validation
- Re-run the full test suite.
- Confirm simulation output is byte-identical (or statistically identical across seeds)
  to before the frontend/backend changes, to protect the paper's numbers.
- Report a summary of what changed, what didn't, and any open issues.

## Decision points (SUPERSEDED 2026-09-29 by the STOP-AND-REPORT standing rule above; kept for history)
1. Whether `core/antibiotic_agent.py` is dead code to remove or an unfinished module.
2. WebSocket message design (Phase 1).
3. Frontend architecture choice (Phase 2).
4. Any change that touches `simulation/` or `core/bacterium_agent.py` logic.
5. Any new biological/pharmacological claim or parameter added to the sim or UI.
6. Whether to keep the frontend single-file or split into a build setup.
7. Anything that would make the repo or any deployment public.

## Explicitly out of scope for this phase
- The AMRResistanceGNN paper itself — separate phase, after the lab is done, own
  CLAUDE.md addendum.
- Any non-bacterial pathogen.
- Changing simulation biology/parameters beyond what's needed to fix real bugs found in
  Phase 0 (report bugs found, don't silently "improve" biology while upgrading visuals).

- Two-version biology (2026-09-29, approved): `data/biology.py`. `paper_v1` = the exact
  card_loader tables, FROZEN, and `AMRSimulationModel`'s default (paper pipeline unchanged);
  `tests/test_paper_v1_frozen.py` proves byte-identical states + GNN training pairs against
  `tests/golden/paper_v1.json` (generated from pre-change code, PYTHONHASHSEED=0).
  `lab_v2` = the API server's default: MRSA carries mecA (CARD ARO:3000617) instead of
  tetM; acrAB-tolC removed from MRSA's acquirable pool; mecA non-transferring. mecA's
  fitness cost lives in `data/lab_v2_config.json`.
  lab_v2 Klebsiella: intrinsic acrAB-tolC removed; EUCAST ERP v1.2 rule 1.7 applied via
  `Biology.intrinsic_resistance` (ampicillin, 0.90 gene convention; ticarcillin not simulated).
  lab_v2 diffusion conserves total drug (2026-09-29); decay_rate values unchanged and
  UNVALIDATED against real PK/PD. mecA fitness cost 0.275 (Ender 2004, RA120 vs BB255).
- **Known limitation (lab_v2): SCCmec-style transfer of mecA is not modelled.** The sim's
  only HGT mechanism is conjugation-style transfer between neighbours, which is the wrong
  mechanism for SCCmec, so mecA is marked non-transferring rather than moved by it.

## Paper-phase TODO (do NOT act on these during the lab phase)
**Status 2026-09-29:** the IEEE JBHI manuscript (JBHI-03955-2026) was rejected on
2026-06-29 and not resubmitted. It is retired, along with `Downloads/amr_gnn.tex` and
its numbers (AUROC 0.9934, 36-dim). A NEW paper will be written from the current
pipeline; nothing below is to be reconciled with the retired draft.
- **Leakage-fix provenance (confirmed 2026-09-29, for the new paper's methods note):**
  the retired 0.9934 came from the 2026-06-07 01:32 run (`logs/gnn_training_20260607_013207.log`:
  test AUROC 0.9934, AUPRC 0.0602, early stop epoch 42; 349,370 params). Rebuilding the
  current architecture with the pre-fix inputs (behavioural node group 3-dim incl.
  sos_active -> 36-dim nodes; 8-dim edges incl. raw distance, same_species,
  transferable_genes_norm, either_sos) gives exactly 349,370 params; the post-fix 35/5
  layout gives exactly 349,162, the count logged by every run from 2026-07-04 on (test
  AUROC 0.9292-0.9599). Four features were removed, not three: sos_active too.
  `ai/checkpoints/feature_leakage_remediation.md`, cited in feature_engineering.py as the
  full audit, does not exist in the repo.
- **NaN AUROC for mexAB-oprM** (noted 2026-09-29, not investigated): the checkpoint's
  stored validation metrics (`ai/checkpoints/best_model.pt`, `val_metrics`) and
  `ai/checkpoints/training_results.json` contain `"auroc_mexAB-oprM": NaN`. Found only
  because `/gnn/status` failed to JSON-encode it (the API now reports it as null; the
  files are untouched). Before the paper phase, determine how macro AUROC handles this
  gene and whether any reported figure depends on it.
- **8 of 10 CARD ARO IDs were wrong** — FIXED IN CODE 2026-09-29 (`data/card_loader.py`,
  pinned by `tests/test_biology.py::test_card_ids_verified`). Corrected Table 1 for the new
  paper is generated from code: `paper/make_table1.py` -> `paper/table1_genes.tex`. Original audit
  (audited 2026-09-29 against card.mcmaster.ca). Correct: blaNDM-1 ARO:3000589, tetM
  ARO:3000186. Wrong (what the ID actually is): blaTEM-1 and blaCTX-M-15 both ARO:3000237
  (TolC); blaKPC-2 ARO:3000159 (generic efflux term); mexAB-oprM ARO:3000157 (rifamycin drug
  class); acrAB-tolC ARO:3000055 (SME beta-lactamase); gyrA_S83L ARO:3000181 (tet(V));
  mcr-1 ARO:3000745 (dihydrofolate reductase); vanA ARO:3000089 (AER beta-lactamase).
  `card_id` is metadata only (no code reads it), so no simulated number depends on it — but
  check whether the paper cites these IDs.
- **Klebsiella / MRSA profile biology** (see lab-phase proposal, 2026-09-29): every Klebsiella
  starts with `acrAB-tolC`, which the sim treats as 0.90 protection vs ciprofloxacin,
  tetracycline and ampicillin; EUCAST ERP v1.2 rule 1.7 lists K. pneumoniae as expected
  resistant only to ampicillin/amoxicillin and ticarcillin. Klebsiella is in 2 of the 4
  DEFAULT_CONFIG training scenarios, so this shapes the paper's training data.
- **Seeding bugs — FIXED 2026-09-29 (approved):** (1) seeded runs depended on
  PYTHONHASHSEED because gene *sets* were iterated around RNG draws and float sums
  (`_attempt_hgt`, `get_resistance_to`, `_recalculate_fitness`; plus output order in
  `to_dict`, gene distribution, HGT-burst message) — all now iterate `sorted(...)`;
  (2) `build_graph_from_state` subsampled with the global unseeded `random.sample` — now a
  seeded local RNG (per-run in `collect_training_snapshots`, step-seeded otherwise).
  Verified: identical fingerprints under PYTHONHASHSEED 11/22/33 with no global seeding;
  `tests/test_paper_v1_frozen.py` checks two different hash seeds. This intentionally
  changed paper_v1 trajectories; golden regenerated (previous baseline: commit 89cb849).
- **Antibiotic "diffusion" removes drug instead of spreading it** (found 2026-09-29, not
  fixed): `_diffuse_antibiotics` convolves with a kernel summing to 1.0 and then multiplies
  by `diffusion_rate`, so total drug is scaled by diffusion_rate*(1-decay_rate) every step
  (half-lives 0.57-6 steps vs 69-231 from decay_rate alone; lower "diffusion_rate" = faster
  loss). Contradicts the manuscript's "Diffusion follows a discrete Laplacian (Fick's second
  law)", which conserves mass. Neither rate cites a source, and no step duration is defined
  anywhere, so decay rates cannot be checked against real PK yet. Training runs dose once at
  step 15, so this shapes the training data.
- **Gene-less cells get no biofilm or persister protection** (found 2026-09-29, not fixed —
  `core/`, both versions): `BacteriumAgent.get_resistance_to` returns 0.0 when a cell has no
  resistance genes *before* applying biofilm (60%) and persister (95%) protection. The
  manuscript says persisters are tolerant "regardless of genotype". Affects E. coli (no
  intrinsic genes) in both versions, and lab_v2 Klebsiella for non-ampicillin drugs.
- Lab-phase note: the analytics expected-resistance table (`data/expected_resistance.py`)
  includes EUCAST header rules (e.g. glycopeptides for all Gram-negatives); the lab_v2
  *simulation* applies only rule 1.7 for Klebsiella, so the sim still kills Gram-negatives
  with vancomycin while analytics reports them resistant.
- **Follow-up (not done): real time scale for dosing.** lab_v2 uses a time-limited course
  measured in steps (approved option (b), 2026-09-29). Replacing it with a defined step
  duration and SOURCED per-drug half-lives (option (c)) is real follow-up work; nothing in
  the code or manuscript currently defines how long a step is.
- **External validation is currently non-functional** (found 2026-09-29): the BV-BRC files
  (`ecoli_amr.csv`, `klebsiella_amr.csv`) are AMR *phenotype* tables with no gene calls, so
  `run_patric_validation.py` yields degenerate prevalences (1.0/0.0) and Spearman rho = NaN
  (`ai/checkpoints/patric_validation.json`); the `--synthetic` path crashes on a console
  encoding error (`external_validation_rerun.txt`). Real validation needs gene re-annotation
  of BV-BRC assemblies (scoped, not started).

## RESUME HERE (saved 2026-09-29; last commit fffd163; tree clean; nothing pushed)
**Waiting on the user for two decisions; do not proceed on either without an answer.**
1. **lab_v2 dosing protocol.** Approved: option (b), time-limited course. Time-limiting alone
   fails at the current dose 1.5 ug/mL (pakistan_crisis extinct at every N). PROPOSED, not
   approved: dose **0.25 ug/mL, cleared after N=5 steps** -> 0/12 runs extinct, 312 pairs,
   1,011 positives (paper_v1: 301 / 1,222); pakistan_crisis min pop 119 (real pressure).
   Survey scripts were in the session scratchpad (lost); rerun via
   `collect_training_snapshots(..., biology="lab_v2", dose=..., dose_duration=...)`.
   On approval: set `DEFAULT_CONFIG["dose_duration_steps"]=5` and add a config key for the
   dose (pipeline currently passes only dose_duration; `dose` defaults to 1.5 — thread
   `config["dose"]` through the 3 callers like dose_duration), label both as invented
   protocol parameters, then run:
   `python -m ai.reseeded_results --biology lab_v2` and
   `python -m ai.hparam_sweep --part gnn --biology lab_v2` / `--part rf` / `--summarize`,
   then report the three-column summary: retired draft -> paper_v1 corrected -> lab_v2
   corrected (every headline number). NO paper text until the user says so.
2. **External validation:** user is deciding whether the paper waits for a ResFinder
   re-annotation of BV-BRC assemblies (~1-2 weeks; gene->phenotype concordance is the
   feasible, useful check; per-edge HGT cannot be validated from genomes) or ships with it as
   a labelled limitation.

**Numbers so far (paper_v1, corrected; test AUROC mean ± SD over seeds 0-4):**
- Default hparams (`reseeded/paper_v1`): GNN 0.9261±0.0211, RF 0.9226±0.0263, LR 0.8466±0.0406.
- Tuned (`sweep/paper_v1`, validation-selected): GNN full 0.9665±0.0048 (hidden 128, lr 1e-3,
  2 layers); edge features zeroed 0.9673±0.0035; graph-free (no message passing)
  0.9546±0.0084; RF 0.9306±0.0317 (300 trees, depth 8). Supports: GNN > RF (5/5 seeds);
  message passing +0.012 (5/5); edge feature vectors add nothing. Default lr 3e-4 was the cause
  of the earlier "GNN ≈ RF, edges hurt" result.
- Retired draft (invalid, pre-leakage-fix, 36-dim): GNN 0.9934, RF 0.9896, LR 0.9746.

**Lab (Phases 1-2) is done** (WebSocket stream, frontend rebuild). Phase 3 (UX/experiment
mode, run comparison) and Phase 4 (final validation) not started.
