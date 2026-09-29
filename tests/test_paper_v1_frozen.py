"""
paper_v1 must stay byte-identical to the code that produced the paper.

tests/golden/paper_v1.json fingerprints, for the paper's DEFAULT_CONFIG
training runs (4 scenarios x seeds 100-102, 80 steps): every step's full
simulation state, and the GNN training pairs produced by
ai.feature_engineering.collect_training_snapshots. It was generated with
PYTHONHASHSEED=0 from simulation code unchanged since commit b31ebae.

The check runs in a subprocess because the trajectory depends on
PYTHONHASHSEED (set iteration order feeds RNG draw order), and pytest's own
process has a random hash seed.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "golden", "fingerprint.py")


def test_paper_v1_byte_identical_to_golden():
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONWARNINGS": "ignore"}
    r = subprocess.run([sys.executable, SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "MATCH" in r.stdout, r.stdout + r.stderr[-2000:]
