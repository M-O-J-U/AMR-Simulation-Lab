# RESUME HERE — paper phase (paused 2026-09-30)

Read this first, then `paper/STATUS.md` (complete-vs-open inventory),
`paper/claims_to_numbers.md` (claim → number → source) and `paper/decisions_log.md`.

**This version of the paper will never be submitted to a venue** (owner, 2026-09-30). It is
being finished as a standalone document. Nothing is pushed; the repo has **no git remote
configured**, and the owner pushes, always.

## The six-step plan, and where it stopped

| Step | What | State |
|---|---|---|
| 1 | Abstract, written last | **done** — commit `8a9e20d` |
| 2 | Future Work promoted to its own section | **done** — `72602ba` |
| 3 | Figures, from committed result files only | **done** — `5787985`, `4b46b7e` |
| 4 | Assemble into one formatted document | **done** — `79218fa` |
| 5 | Portfolio pack `portfolio/paper4_portfolio.md` | **NOT STARTED** |
| 6 | Push to the private repo (owner runs it) | **NOT STARTED** |

## Step 5 — portfolio pack

Match the SLURP format. Two worked examples, both self-contained so a website AI can render
them without reading the paper:

- `C:\Users\mojua\Desktop\SLURP\portfolio\paper1_portfolio.md` (465 lines)
- `C:\Users\mojua\Desktop\SLURP\portfolio\paper2_portfolio.md` (583 lines)

Their structure, in order:

1. YAML front matter: `title`, `pitch`, `status`, `dataset`, `models`, `tags`, `role`,
   `contact`, `code`.
2. H1 title.
3. **Instructions for the website builder** — suggested layout top to bottom, tone rules
   ("plain and honest… no hype words, no superlatives, no claims of priority"), how to draw
   charts, and "use only the numbers in this file".
4. **Story** — Problem / Approach / Findings / What surprised me / Limitations / Technical
   summary.
5. **Key numbers** — bullet "cards".
6. **Charts** — one H3 each, with a fenced ` ```json ` block carrying `id`, `title`,
   `chart_type`, `x_axis`, `y_axis`, `series`, plus `caption`, `takeaway` and
   `fallback_image`.
7. **Method diagram**, **Glossary**, **Data provenance**, **Do not claim**.

Owner's required specifics for this one:

- front matter `status`: **"Draft, not submitted"**
- the do-not-claim list must include: no venue claim, no "first", no external-validation
  claim, no per-gene biological interpretation.

Add these to the do-not-claim list as well — each is a finding this project actually made,
and all are binding:

- Do not quote the retired 0.9934 or any pre-leakage-fix number (U1).
- Do not quote the superseded 5-gene headline 0.9735 vs 0.8950 — it was contaminated by
  blaKPC-2 (see `decisions_log.md`, "S2 contamination").
- Do not quote an 8-gene GNN-vs-RF gap (U5).
- Do not say the model was tested on interspecies transfer (U11) or on unseen simulations
  (U13).
- Do not describe the simulated transfer of gyrA_S83L, acrAB-tolC or vanA as real
  conjugation (U6, U10).
- Do not say the intraspecies restriction supplies many easy negatives — measured and
  refuted, 0.275% (U14).

Numbers for the cards and charts come from `paper/claims_to_numbers.md` only (S1–S10).
For chart series, take values from the committed results file rather than retyping them;
`python paper/make_figures.py --check` verifies the plotted set still matches the claims
table.

Suggested charts (all data already gathered for `paper/figures/`):

- headline comparison on the 4 always-trainable genes (S2) — bar;
- per-gene AUROC with positive counts (S9 + U12) — bar, with the backwards-confound as the
  `takeaway`;
- feature-group ablation (S5) — bar;
- baseline trainability by gene and seed (S2b) — the 0/5, 2/5, 5/5 tiers.

## Step 6 — push

- Artifacts already live in `paper/`. The owner mentioned `paper/` or `docs/`; `paper/` is
  where everything is, so no move is needed unless they want one.
- **`README.md` does not yet mention the paper.** Updating it is part of this step.
- `git remote -v` is empty, so the push command needs the private repo URL from the owner.
- Prepare the commit, then hand over the exact command. Never run a push.

## State of the work

- Document: `paper/amr_hgt_gnn.{md,html,pdf}` — 31 pages, 15,424 words, 4 figures,
  29 references. Rebuild with `python paper/assemble.py`; `--check` validates citations and
  figure anchors without writing.
- Drafts: `paper/draft/00_abstract.md` … `08_conclusion.md`, all awaiting owner review.
- 280 tests pass, including the `paper_v1` byte-identity guard. Run `python -m pytest -q`
  before and after any change.
- Reproduce the measurements: `python paper/make_figures.py`,
  `python paper/measure_cross_species_edges.py`.
- Open decisions: **none.** The last one (S2 contamination) was decided and applied.
- Biggest known weakness, already disclosed in the document: the train/test split is over
  snapshot pairs rather than simulation runs (U13), so nothing here measures generalisation
  to unseen runs. `Future Work` §7.1 ranks fixing it first.
