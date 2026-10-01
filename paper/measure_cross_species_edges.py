"""Measure the cross-species share of contacts in the reference dataset.

This script produced claim **S10** in `paper/claims_to_numbers.md`, and the numbers
quoted in the Abstract, Discussion §6.1 and Limitations §5.2:

    cross-species edges        6,428 of 6,029,316  (0.1066%)   [50-run reference set]
    where                      pakistan_crisis only (6,428 of its 682,678 edges);
                               the other four scenarios are single-species -> 0
    positives on those edges   0  (all 4,763 positives are same-species)

    On the superseded 15-run set the figures were 4,696 of 1,707,498 (0.275%),
    likewise with zero positives. The conclusion is unchanged and strengthened.

It was written to CHECK a claim that turned out to be false — that intraspecies-only
transfer floods the negative class with easily separable cross-species pairs. It does
not: they are 0.275% of the negative class. That refuted argument is recorded as U14 so
it is not reintroduced.

The run also reproduces the committed dataset totals exactly (390 graph pairs,
1,707,498 edges, 1,443 edge-gene positives) from a fresh process, which independently
re-confirms the determinism claim S8.

Read-only: it regenerates the dataset in memory and writes nothing unless --json is
given. It does not touch `ai/checkpoints/`.

Usage
-----
From the repository root:

    python paper/measure_cross_species_edges.py

    # also write the full breakdown to a file
    python paper/measure_cross_species_edges.py --json out.json

    # spot-check a single scenario/seed instead of all 15 runs (fast)
    python paper/measure_cross_species_edges.py --scenario pakistan_crisis --seed 100

Takes a few minutes for the full dataset, most of it in the per-edge Python loop.
The invariant it pins (no positive ever lies on a cross-species edge) is also covered
quickly by `tests/test_cross_species.py`.

Configuration comes from `ai.gnn_trainer.DEFAULT_CONFIG` plus `scenarios_for` and
`dosing_for`, i.e. the same committed reference protocol the results use — it is not
duplicated here, so it cannot drift from the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import GENE_INDEX, collect_training_snapshots
from ai.gnn_trainer import DEFAULT_CONFIG, dosing_for, scenarios_for

# Committed reference totals, for the self-check (paper/claims_to_numbers.md header).
# The run50 reference set (50 runs). The earlier 15-run set was
# 390 / 1,707,498 / 1,443.
EXPECTED = {"graph_pairs": 1300, "edges": 6_029_316, "edge_gene_positives": 4_763}


def tally(pairs: list) -> dict:
    """Count edges and positives, split by whether the endpoints share a species."""
    out = {"graph_pairs": 0, "edges": 0, "edge_gene_positives": 0,
           "same_species_edges": 0, "cross_species_edges": 0,
           "positives_same_species": 0, "positives_cross_species": 0}
    for g0, _g1 in pairs:
        if g0.get("gene_labels") is None:
            continue
        out["graph_pairs"] += 1
        species = [b.get("species", "") for b in g0["bacteria"]]
        edge_index, labels = g0["edge_index"], g0["gene_labels"]
        n_edges = edge_index.shape[1]
        out["edges"] += n_edges
        out["edge_gene_positives"] += int(labels.sum())
        for e in range(n_edges):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            n_pos = int(labels[e].sum())
            if species[i] != species[j]:
                out["cross_species_edges"] += 1
                out["positives_cross_species"] += n_pos
            else:
                out["same_species_edges"] += 1
                out["positives_same_species"] += n_pos
    return out


def merge(into: dict, more: dict) -> dict:
    for key, value in more.items():
        into[key] = into.get(key, 0) + value
    return into


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", help="measure one scenario only (default: all five)")
    ap.add_argument("--seed", type=int, help="measure one data seed only, e.g. 100")
    ap.add_argument("--json", help="write the full breakdown to this path")
    ap.add_argument("--data-seeds", type=int, default=None,
                    help="data seeds per scenario (DEFAULT_CONFIG: 3; the run50 "
                         "reference set uses 10)")
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.data_seeds:
        config["seeds_per_scenario"] = args.data_seeds
    dosing = dosing_for(config)
    scenarios = [args.scenario] if args.scenario else scenarios_for(config)
    seeds = ([args.seed] if args.seed is not None
             else [s + 100 for s in range(config["seeds_per_scenario"])])
    partial = bool(args.scenario or args.seed is not None)

    print(f"biology   : {config['biology']}")
    print(f"scenarios : {scenarios}")
    print(f"seeds     : {seeds}")
    print(f"dosing    : {dosing}")
    print(f"steps/run : {config['steps_per_run']}, "
          f"snapshot interval {config['snapshot_interval']}\n")

    totals: dict = {}
    per_scenario: dict = {}
    for scenario in scenarios:
        scenario_totals: dict = {}
        for seed in seeds:
            pairs = collect_training_snapshots(
                n_steps=config["steps_per_run"], scenario=scenario, seed=seed,
                snapshot_interval=config["snapshot_interval"],
                biology=config["biology"], **dosing)
            merge(scenario_totals, tally(pairs))
        print(f"  {scenario:24s} {scenario_totals['edges']:>9,} edges, "
              f"{scenario_totals['cross_species_edges']:>6,} cross-species, "
              f"{scenario_totals['positives_cross_species']} cross-species positives")
        per_scenario[scenario] = scenario_totals
        merge(totals, scenario_totals)

    edges = totals["edges"]
    n_genes = len(GENE_INDEX)
    negatives = edges * n_genes - totals["edge_gene_positives"]
    cross_pct = 100.0 * totals["cross_species_edges"] / edges if edges else 0.0
    cross_share_neg = (100.0 * totals["cross_species_edges"] * n_genes / negatives
                       if negatives else 0.0)

    print(f"\ngraph pairs         : {totals['graph_pairs']:,}")
    print(f"edges               : {edges:,}")
    print(f"edge-gene positives : {totals['edge_gene_positives']:,}")
    print(f"cross-species edges : {totals['cross_species_edges']:,} ({cross_pct:.4f}%)")
    print(f"  ...as a share of the negative class: {cross_share_neg:.4f}%")
    print(f"positives on cross-species edges: {totals['positives_cross_species']} "
          f"(must be 0; transfer is intraspecies only)")

    assert totals["positives_cross_species"] == 0, (
        "INVARIANT VIOLATED: a recorded transfer lies on a cross-species edge, but "
        "BacteriumAgent._attempt_hgt skips neighbours of a different species.")

    if partial:
        print("\n(partial run: totals are not compared against the committed dataset)")
    else:
        mismatched = {k: (totals[k], v) for k, v in EXPECTED.items() if totals[k] != v}
        if mismatched:
            print("\nWARNING: totals differ from the committed reference dataset:")
            for key, (got, want) in mismatched.items():
                print(f"  {key}: got {got:,}, expected {want:,}")
            print("The reference numbers in paper/ may no longer describe this code.")
            return 1
        print("\nTotals match the committed reference dataset "
              "(1,300 / 6,029,316 / 4,763): S8 determinism re-confirmed.")

    if args.json:
        payload = {"totals": totals, "per_scenario": per_scenario,
                   "cross_species_edge_pct": cross_pct,
                   "cross_species_share_of_negatives_pct": cross_share_neg,
                   "config": {"biology": config["biology"], "scenarios": scenarios,
                              "seeds": seeds, "dosing": dosing,
                              "steps_per_run": config["steps_per_run"],
                              "snapshot_interval": config["snapshot_interval"]}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
