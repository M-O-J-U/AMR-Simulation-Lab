"""
paper_v1 must stay byte-identical to the code that produced the paper.

tests/golden/paper_v1.json was generated at commit b31ebae (before the
two-version biology change) by `python tests/golden/fingerprint.py --write`.
It fingerprints, for the paper's DEFAULT_CONFIG training runs (4 scenarios x
seeds 100-102, 80 steps): every step's full simulation state, and the GNN
training pairs produced by ai.feature_engineering.collect_training_snapshots.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "golden"))
sys.path.insert(0, os.path.dirname(HERE))

import fingerprint as fp

GOLD = json.load(open(fp.GOLDEN))


@pytest.mark.parametrize("run", sorted(GOLD))
def test_paper_v1_run_is_byte_identical(run):
    scenario, seed = run.split("/")
    seed = int(seed)
    ph, n = fp.pairs_fingerprint(scenario, seed)
    assert fp.state_fingerprint(scenario, seed) == GOLD[run]["state"], "simulation state changed"
    assert n == GOLD[run]["n_pairs"]
    assert ph == GOLD[run]["pairs"], "GNN training pairs changed"
