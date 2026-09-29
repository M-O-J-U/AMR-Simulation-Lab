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

## Paper-phase TODO (do NOT act on these during the lab phase)
- **NaN AUROC for mexAB-oprM** (noted 2026-09-29, not investigated): the checkpoint's
  stored validation metrics (`ai/checkpoints/best_model.pt`, `val_metrics`) and
  `ai/checkpoints/training_results.json` contain `"auroc_mexAB-oprM": NaN`. Found only
  because `/gnn/status` failed to JSON-encode it (the API now reports it as null; the
  files are untouched). Before the paper phase, determine how macro AUROC handles this
  gene and whether any reported figure depends on it.
- **8 of 10 CARD ARO IDs in `data/card_loader.py` point at the wrong CARD entry** (audited
  2026-09-29 against card.mcmaster.ca; not fixed). Correct: blaNDM-1 ARO:3000589, tetM
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
