"""
Fingerprints of the paper_v1 simulation, used to prove that the frozen paper
biology stays byte-identical as the lab code evolves.

  PYTHONHASHSEED=0 python tests/golden/fingerprint.py --write   # (re)generate
  PYTHONHASHSEED=0 python tests/golden/fingerprint.py           # compare

PYTHONHASHSEED MUST be fixed: the simulation's trajectory depends on Python's
per-process string-hash randomisation (set iteration order feeds RNG draw
order; found 2026-09-29, see CLAUDE.md paper-phase TODO). The script refuses
to run without it.

Covers exactly the paper's training data path (ai/gnn_trainer.py
DEFAULT_CONFIG: 4 scenarios x seeds 100-102, 80 steps, snapshot_interval 3,
via ai.feature_engineering.collect_training_snapshots) plus the per-step
full simulation state of the same runs. cell_id (uuid4, never seeded, display
only) is excluded from state hashes.
"""
import hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_v1.json")

SCENARIOS = ["ecoli_cipro", "klebsiella_carbapenem", "pakistan_crisis", "xdr_acinetobacter"]
SEEDS = [100, 101, 102]
STEPS = 80
INTERVAL = 3


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def state_fingerprint(scenario, seed):
    """Replays collect_training_snapshots' schedule (15 warm-up steps, then
    each scenario antibiotic at 1.5 uniform) and hashes every step's state."""
    from simulation.amr_model import AMRSimulationModel
    m = AMRSimulationModel(scenario=scenario, initial_bacteria=120, seed=seed,
                           enable_logging=False)
    h = hashlib.sha256()
    def add():
        s = m.get_full_state()
        for b in s["bacteria"]:
            b.pop("cell_id", None)
        h.update(_h(s).encode())
    for _ in range(15):
        m.step(); add()
    if scenario != "validation":
        for ab in m.active_antibiotic_keys:
            m.apply_antibiotic(ab, concentration=1.5, mode="uniform")
    for _ in range(STEPS):
        m.step(); add()
    return h.hexdigest()


# build_graph_from_state() subsamples populations > max_nodes with Python's
# GLOBAL, unseeded `random.sample`, so the paper pipeline's training pairs are
# not reproducible run-to-run (found 2026-09-29; 7 of these 12 runs are
# affected; logged in CLAUDE.md as a paper-phase item, NOT fixed). This harness
# seeds the global RNG before each run only so the fingerprint is stable; the
# paper pipeline itself is untouched.
HARNESS_GLOBAL_SEED = 20260929


def pairs_fingerprint(scenario, seed):
    import random
    import numpy as np
    from ai.feature_engineering import collect_training_snapshots
    saved = random.getstate()
    random.seed(HARNESS_GLOBAL_SEED)
    try:
        pairs = collect_training_snapshots(n_steps=STEPS, scenario=scenario, seed=seed,
                                           snapshot_interval=INTERVAL)
    finally:
        random.setstate(saved)
    h = hashlib.sha256()
    for g0, g1 in pairs:
        for g in (g0, g1):
            for k in sorted(g):
                v = g[k]
                if hasattr(v, "tobytes"):
                    h.update(k.encode()); h.update(str(v.dtype).encode())
                    h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
                elif k == "bacteria":
                    h.update(k.encode())
                    h.update(_h([{kk: vv for kk, vv in b.items() if kk != "cell_id"}
                                 for b in v]).encode())
                else:
                    h.update(k.encode()); h.update(_h(v).encode())
    return h.hexdigest(), len(pairs)


def compute():
    out = {}
    for sc in SCENARIOS:
        for sd in SEEDS:
            t = time.time()
            ph, n = pairs_fingerprint(sc, sd)
            out[f"{sc}/{sd}"] = {"state": state_fingerprint(sc, sd), "pairs": ph, "n_pairs": n}
            print(f"  {sc}/{sd}: {n} pairs ({time.time()-t:.1f}s)", file=sys.stderr)
    return out


REQUIRED_HASHSEED = "0"

if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != REQUIRED_HASHSEED:
        sys.exit(f"set PYTHONHASHSEED={REQUIRED_HASHSEED} (trajectories depend on it)")
    fp = compute()
    if "--write" in sys.argv:
        json.dump(fp, open(GOLDEN, "w"), indent=1)
        print("wrote", GOLDEN)
    else:
        gold = json.load(open(GOLDEN))
        bad = [k for k in gold if gold[k] != fp.get(k)]
        print("MATCH" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
