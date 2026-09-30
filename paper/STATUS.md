# Paper status — as of 2026-09-30

Local only; nothing pushed. No venue chosen. **No submission target**, so submission prep
(figures, LaTeX build, remaining citation checks, repo/portfolio work) is deliberately NOT
started.

Read first: `paper/claims_to_numbers.md` (claim → number → source), then
`paper/decisions_log.md` (dated decisions; OPEN items marked), then this file.

## Drafted (all awaiting owner review; none approved yet)

| File | Section | Notes |
|---|---|---|
| `paper/draft/01_introduction.md` | 1 Introduction | Citations verified; Scope paragraph covers U11 + U13; contribution 2 qualified |
| `paper/draft/02_related_work.md` | 2 Related Work | Audited citations only |
| `paper/draft/03_methods.md` | 3 Methods | Parameters read from code, not docstrings |
| `paper/draft/04_results.md` | 4 Results | All 10 per-gene counts present |
| `paper/draft/05_limitations.md` | 5 Limitations | Full caveat queue assembled |
| `paper/draft/06_discussion.md` | 6 Discussion | Thesis: joint training, not graph structure |
| `paper/draft/07_future_work.md` | 7 Future Work | 6 prioritised items, each with effort + claims affected |
| `paper/draft/08_conclusion.md` | 8 Conclusion | Short; upgrades no earlier claim |

| `paper/draft/00_abstract.md` | Abstract | Written last; scope weighted as in Discussion |

## Assembled document

`paper/amr_hgt_gnn.{md,html,pdf}` — the whole paper as one standalone document
(31 pages), built by `paper/assemble.py` from `paper/draft/`. Citations are resolved to
numbered references from `paper/refs.bib`; figures are inserted with inline captions and
numbered in document order; Appendix A records how far each source was verified.
Rebuild with `python paper/assemble.py` (`--check` validates citations and figure
anchors without writing). Build-time extras: the `markdown` package, and Chrome or Edge
for the PDF step.

## Supporting files complete

- `paper/citations.md` — per-reference audit: existence, venue, year, authors, DOI, the text
  that supports the claim, and a verification level per source.
- `paper/refs.bib` — 32 entries, all with a matching audit entry.
- `paper/gene_mechanism_audit.md` — all 11 genes' real mobility vs what the simulator does,
  11 sources verified against their own text.
- `paper/claims_to_numbers.md` — S1–S9 supported, U1–U13 not-supported, earlier-results table.
- `paper/make_table1.py` → `paper/table1_genes.tex` — Table 1 generated from code.

## Open — needs an owner decision

**None.** The last one (whether to mention the retired 2016 review's 10-million projection) was
decided on 2026-09-30: excluded entirely; the AMR burden is cited from the GBD papers only.

**This version will not be submitted to any venue** (owner, 2026-09-30). It is being completed
as a standalone document, so "submission prep" items below are scoped to that, not to a venue's
requirements.

## Open — work not started

**Submission prep (blocked by design: no venue).**
- ~~Figures~~ done: four, from committed result files, by `paper/make_figures.py`.
- ~~Assembly~~ done: `paper/assemble.py` resolves citations and emits Markdown/HTML/PDF.
  No LaTeX or IEEE template, by instruction — there is no venue.
- ~~Abstract~~ done.
- Full-text re-verification of the 23 abstract-only citations (of 29 cited), listed in
  Appendix A of the assembled document and in `paper/citations.md`. Each is currently cited
  only for what its abstract states, which is defensible for a draft and is disclosed in
  the document itself.

**Methodological work that would change the numbers** (from Discussion §6.5, in priority order):
1. Split by run instead of by snapshot pair (U13). Cheapest; changes how every number reads.
2. External validation with gene-level calls — needs ResFinder/ARIBA re-annotation of BV-BRC
   assemblies (U7). Scoped earlier at ~1–2 weeks.
3. Per-gene transfer mobility, so one mechanism is not applied to every gene (U10, U11).
4. A defined step duration plus sourced half-lives, without which no rate can be validated.

**Repo hygiene, unresolved:**
- `ai/checkpoints/feature_leakage_remediation.md` is referenced in source comments as the full
  leakage audit but does not exist. Methods §3.4 is currently the record. Either write the file
  or drop the reference.
- Stale docstrings outside `ai/gnn_model.py` (already cleaned): `ai/gnn_trainer.py` says
  patience 10 (is 12) and "stratified by scenario" (is a plain shuffle), and asserts
  "blaTEM-1 transfers very frequently" (5 positives); `ai/feature_engineering.py` has a stale
  `(E, 8)` comment. Logged 2026-09-30 in `decisions_log.md`, not fixed.
- `collect_all_data` computes its progress total from 4 scenarios while iterating 5. Cosmetic.

**Lab phases 3–4 (separate track, not started):** UX/control upgrades and the
validation/byte-identity re-check.

## Things that must not regress

- 280 tests pass, including `tests/test_paper_v1_frozen.py`, which guards `paper_v1`
  byte-for-byte. Run `python -m pytest -q` before and after any change.
- The reference result set is `ai/checkpoints/reseeded/lab_v2_tuned_noedge_mrsa/`
  (+ `sweep/lab_v2_mrsa_noedge/`). Dataset SHA-256 `ee83ff385ca15ec9…`.
- Do not quote the 8-gene GNN-vs-RF gap (U5), the retired 0.9934 (U1), or any per-gene
  ordering as biology (U12).
