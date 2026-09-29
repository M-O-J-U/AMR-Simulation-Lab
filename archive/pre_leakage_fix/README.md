# Archived: pre-leakage-fix GNN tests

`test_gnn.py` is an older copy of `tests/test_gnn.py` (last modified 2026-06-07),
kept as a record of the GNN edge-feature layout *before* the label-leakage
remediation in `ai/feature_engineering.py`.

It tests the old 8-dim edge layout, including `distance_norm`, `same_species`
and `transferable_genes_norm`, which were removed because they directly
encoded the HGT label-generating rule. Against the current code, 8 of its 42
tests fail by design. Do not run it or port its tests back.

This folder is excluded from test collection (`pytest.ini`: `testpaths = tests`,
`norecursedirs` includes `archive`).
