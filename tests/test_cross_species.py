"""Pin the intraspecies-transfer invariant behind claim S10.

`BacteriumAgent._attempt_hgt` skips any neighbour of a different species, so no
recorded transfer can ever lie on a cross-species edge. Claim S10 in
`paper/claims_to_numbers.md` reports 0 such positives across the whole reference
dataset; the paper's Abstract, Discussion and Limitations all rest on that.

Regenerating the full dataset takes minutes, so these tests exercise the invariant
on the one multi-species training scenario (`pakistan_crisis`) plus a short
single-species run. Use `paper/measure_cross_species_edges.py` for the full count.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.feature_engineering import collect_training_snapshots  # noqa: E402
from ai.gnn_trainer import DEFAULT_CONFIG, dosing_for  # noqa: E402
from paper.measure_cross_species_edges import tally  # noqa: E402


def _collect(scenario: str, seed: int, n_steps: int) -> list:
    config = dict(DEFAULT_CONFIG)
    return collect_training_snapshots(
        n_steps=n_steps, scenario=scenario, seed=seed,
        snapshot_interval=config["snapshot_interval"],
        biology=config["biology"], **dosing_for(config))


@pytest.fixture(scope="module")
def mixed_scenario_counts() -> dict:
    """pakistan_crisis, data seed 100: the reference protocol's mixed-species run."""
    return tally(_collect("pakistan_crisis", 100, DEFAULT_CONFIG["steps_per_run"]))


def test_mixed_scenario_actually_has_cross_species_contacts(mixed_scenario_counts):
    """Guard against the invariant test passing vacuously."""
    assert mixed_scenario_counts["cross_species_edges"] > 0, (
        "pakistan_crisis produced no cross-species edges, so the invariant test "
        "below would pass without exercising anything. Has the scenario's species "
        "composition or the edge radius changed?")


def test_no_recorded_transfer_lies_on_a_cross_species_edge(mixed_scenario_counts):
    """The invariant behind S10, and behind the paper's intraspecies-only claims."""
    assert mixed_scenario_counts["positives_cross_species"] == 0
    # ...and every positive is therefore accounted for as same-species.
    assert (mixed_scenario_counts["positives_same_species"]
            == mixed_scenario_counts["edge_gene_positives"])


def test_cross_species_contacts_are_a_small_minority(mixed_scenario_counts):
    """S10 reports 0.275% dataset-wide; this scenario is the only source of them.

    Kept deliberately loose: it pins the order of magnitude, not the exact value,
    so it fails on a structural change (e.g. species seeded as one cluster) rather
    than on incidental drift.
    """
    counts = mixed_scenario_counts
    share = counts["cross_species_edges"] / counts["edges"]
    assert 0.0 < share < 0.10, (
        f"cross-species share of contacts in pakistan_crisis is {share:.4%}; "
        "S10 and the paper describe it as a small minority (0.275% dataset-wide, "
        "a few percent within this scenario)")


def test_single_species_scenario_has_no_cross_species_edges():
    """A short run is enough: the count must be exactly zero at any length."""
    counts = tally(_collect("ecoli_cipro", 100, 24))
    assert counts["edges"] > 0, "no edges collected; cannot check anything"
    assert counts["cross_species_edges"] == 0
    assert counts["positives_cross_species"] == 0
