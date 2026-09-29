# ai/checkpoints — which results are current

**Current (for the new paper):** `reseeded/<biology>/` — produced by
`python -m ai.reseeded_results [--biology paper_v1|lab_v2]` after the 2026-09-29 seeding
fixes. Every headline number is mean ± SD over model seeds on a fixed, fingerprinted dataset
(`results.json` → `dataset.sha256`), with a human-readable `summary.md`.
`reseeded/paper_v1/` was generated before the pipeline default switched to lab_v2.
Hyperparameter sweeps: `sweep/<biology>/` (`python -m ai.hparam_sweep`).

**Superseded (kept for history; do not cite):** the single-run files in this directory —
`training_results.json`, `baseline_results.json`, `calibration_results.json`,
`full_comparison.json`, `gnn_ablation_results.json`, `multiseed_comparison.json`,
`external_validation.json`, `patric_validation.json`, and `best_model.pt` / `_ablation/` /
`_multiseed/` models. They were produced before the seeding fixes, when:

- seeded simulation runs depended on `PYTHONHASHSEED` (different trajectories per process),
- GNN graph subsampling used the global, unseeded `random.sample`,
- logistic regression's `saga` solver was unseeded, and the ablation's GNN retrains had no
  `torch_seed`,

so none of them can be regenerated exactly, and several are single observations of a quantity
with large run-to-run spread.

**Retired:** the IEEE JBHI manuscript's numbers (AUROC 0.9934, 36-dim node features) came
from a 2026-06-07 run that used the pre-leakage-fix feature layout (see CLAUDE.md). No file
here reproduces them, and none should.

`best_model.pt` is still what the lab UI's GNN panel loads; it is a 35/5-dim post-leakage-fix
model trained 2026-07-31 (single run, pre-seeding-fix).
