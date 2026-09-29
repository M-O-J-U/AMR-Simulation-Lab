"""
The paper_v1 pipeline must be deterministic across processes and must not
drift silently.

tests/golden/paper_v1.json fingerprints, for the DEFAULT_CONFIG training runs
(4 scenarios x seeds 100-102, 80 steps): every step's full simulation state,
and the GNN training pairs from ai.feature_engineering.collect_training_snapshots.

Baseline history:
  89cb849  pre-seeding-fix code (only reproducible under PYTHONHASHSEED=0)
  current  after the 2026-09-29 seeding fixes (sorted gene iteration; seeded
           graph subsampling) - regenerated deliberately, approved change.

The fingerprint runs in subprocesses under DIFFERENT PYTHONHASHSEED values;
both must match the golden exactly.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "golden", "fingerprint.py")


@pytest.mark.parametrize("hashseed", ["1", "987654"])
def test_paper_v1_matches_golden_under_any_hashseed(hashseed):
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONWARNINGS": "ignore"}
    r = subprocess.run([sys.executable, SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "MATCH" in r.stdout, r.stdout + r.stderr[-2000:]


def test_graph_subsampling_ignores_global_random():
    """Fix #2: training pairs must not depend on the global `random` state
    (xdr_acinetobacter seed 100 subsamples ~20 times)."""
    import random
    import numpy as np
    sys.path.insert(0, ROOT)
    from ai.feature_engineering import collect_training_snapshots

    def run(global_seed):
        random.seed(global_seed)
        pairs = collect_training_snapshots(n_steps=40, scenario="xdr_acinetobacter",
                                           seed=100, snapshot_interval=3)
        return [(g0["node_ids"], g0["node_features"].tobytes(), g0["edge_index"].tobytes())
                for g0, _ in pairs]
    assert run(1) == run(2)
