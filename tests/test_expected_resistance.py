"""
Tests for species-level expected resistance (data/expected_resistance.py) in the
analytics layer, and for the treatment advisory's "safe" logic.

The simulation must NOT consume this table; that is checked here too.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from data.card_loader import ANTIBIOTIC_PROFILES, GERM_PROFILES, get_germ
from data.expected_resistance import (
    EXPECTED_RESISTANCE, is_expected_resistant, germ_key_for,
)
from ai.resistance_analytics import estimate_population_mic, recommend_treatment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class B:
    """Minimal bacterium: what the analytics functions read."""
    def __init__(self, germ_key, genes=()):
        self.species = get_germ(germ_key).species
        self.resistance_genes = set(genes)
        self.fitness = 0.8


def pop(germ_key, n=20, genes=()):
    return [B(germ_key, genes) for _ in range(n)]


def rec_by_key(recs):
    return {r["key"]: r for r in recs}


# ─────────────────────────────────────────────────────────────────────────────
# TABLE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestTable:
    def test_keys_are_real_germs_and_antibiotics(self):
        for germ, row in EXPECTED_RESISTANCE.items():
            assert germ in GERM_PROFILES
            for ab, entry in row.items():
                assert ab in ANTIBIOTIC_PROFILES
                assert entry.basis in ("expected_phenotype", "defining_phenotype")
                assert entry.source.startswith(("[ERP]", "[BP14]"))

    def test_exact_table_contents(self):
        """Pin the verified table so any edit is a deliberate, reviewed change."""
        got = {g: sorted(r) for g, r in EXPECTED_RESISTANCE.items()}
        assert got == {
            "e_coli":                  ["vancomycin"],
            "klebsiella_pneumoniae":   ["ampicillin", "vancomycin"],
            "acinetobacter_baumannii": ["ampicillin", "tetracycline", "vancomycin"],
            "pseudomonas_aeruginosa":  ["ampicillin", "tetracycline", "vancomycin"],
            "mrsa":                    ["ampicillin", "colistin", "meropenem"],
        }

    def test_species_name_and_key_both_resolve(self):
        assert germ_key_for("Escherichia coli") == "e_coli"
        assert germ_key_for("e_coli") == "e_coli"
        assert is_expected_resistant("Escherichia coli", "vancomycin")
        assert not is_expected_resistant("Escherichia coli", "ciprofloxacin")
        assert not is_expected_resistant(None, "vancomycin")
        assert not is_expected_resistant("Not a species", "vancomycin")

    def test_simulation_does_not_import_table(self):
        """Biology in core/ and simulation/ must be untouched by this table."""
        for d in ("core", "simulation"):
            for f in os.listdir(os.path.join(ROOT, d)):
                if f.endswith(".py"):
                    src = open(os.path.join(ROOT, d, f), encoding="utf-8").read()
                    assert "expected_resistance" not in src, f"{d}/{f}"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_ecoli_not_susceptible_to_vancomycin(self):
        r = rec_by_key(recommend_treatment(pop("e_coli"), ["vancomycin"]))["vancomycin"]
        assert r["pct_susceptible"] == 0.0
        assert r["pct_expected_resistant"] == 100.0
        assert "EUCAST" in r["rationale"]

    def test_ecoli_still_susceptible_to_ciprofloxacin(self):
        r = rec_by_key(recommend_treatment(pop("e_coli"), ["ciprofloxacin"]))["ciprofloxacin"]
        assert r["pct_susceptible"] == 100.0
        assert r["pct_expected_resistant"] == 0.0

    @pytest.mark.parametrize("ab", ["ampicillin", "meropenem", "colistin"])
    def test_mrsa_expected_resistant(self, ab):
        assert estimate_population_mic(pop("mrsa"), ab)["resistant_pct"] == 100.0

    def test_mrsa_vancomycin_not_marked(self):
        assert estimate_population_mic(pop("mrsa"), "vancomycin")["susceptible_pct"] == 100.0

    def test_mixed_population_fraction(self):
        """Half E. coli, half MRSA: only MRSA is expected-resistant to colistin."""
        mixed = pop("e_coli", 10) + pop("mrsa", 10)
        d = estimate_population_mic(mixed, "colistin")
        assert d["resistant_pct"] == 50.0 and d["susceptible_pct"] == 50.0

    def test_objects_without_species_unchanged(self):
        """Callers that pass objects with no species get the old gene-only result."""
        class Bare:
            resistance_genes = set(); fitness = 0.8
        r = recommend_treatment([Bare() for _ in range(5)], ["vancomycin"])[0]
        assert r["pct_susceptible"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# ADVISORY "SAFE" LOGIC
# ─────────────────────────────────────────────────────────────────────────────

ALL_ABS = ["ciprofloxacin", "meropenem", "colistin",
           "vancomycin", "ampicillin", "tetracycline"]


def bacterium_dict(germ_key, genes=()):
    g = get_germ(germ_key)
    return {"species": g.species, "resistance_genes": list(genes), "fitness": 0.8}


@pytest.fixture(scope="module")
def engine():
    from ai.gnn_inference import GNNInferenceEngine
    return GNNInferenceEngine.load_untrained()


def advise(engine, monkeypatch, bacteria, gene_probs=None):
    """Run treatment_advisory with a controlled GNN output."""
    monkeypatch.setattr(engine, "predict", lambda state, **kw: {
        "model_ready": True, "gene_transfer_probs": gene_probs or {"tetM": 0.01},
        "high_risk_cells": []})
    return engine.treatment_advisory({"bacteria": bacteria}, ALL_ABS)


class TestAdvisory:
    def test_present_gene_resistance_excluded_from_safe(self, engine, monkeypatch):
        """All E. coli carry gyrA_S83L: ciprofloxacin must not be 'safe' even
        though the GNN predicts nothing spreading (the reported bug)."""
        d = advise(engine, monkeypatch,
                   [bacterium_dict("e_coli", ["gyrA_S83L"]) for _ in range(20)])
        assert d["risk_level"] == "LOW"
        assert "ciprofloxacin" not in d["safe_antibiotics"]
        assert "ciprofloxacin" in d["already_resistant"]
        assert "ciprofloxacin" in d["advisory_text"]
        assert "remain effective" not in d["advisory_text"]

    def test_expected_resistance_excluded_from_safe(self, engine, monkeypatch):
        d = advise(engine, monkeypatch, [bacterium_dict("e_coli") for _ in range(20)])
        assert "vancomycin" not in d["safe_antibiotics"]
        assert set(d["safe_antibiotics"]) >= {"ciprofloxacin", "meropenem", "colistin"}

    def test_threshold_boundary(self, engine, monkeypatch):
        """Exactly 80% susceptible stays safe; below it does not."""
        from ai.gnn_inference import SAFE_MIN_SUSCEPTIBLE_PCT
        assert SAFE_MIN_SUSCEPTIBLE_PCT == 80.0
        at = [bacterium_dict("e_coli", ["gyrA_S83L"])] * 2 + [bacterium_dict("e_coli")] * 8
        below = [bacterium_dict("e_coli", ["gyrA_S83L"])] * 3 + [bacterium_dict("e_coli")] * 7
        assert "ciprofloxacin" in advise(engine, monkeypatch, at)["safe_antibiotics"]
        assert "ciprofloxacin" not in advise(engine, monkeypatch, below)["safe_antibiotics"]

    def test_gnn_threat_still_excludes(self, engine, monkeypatch):
        """Original behaviour kept: a predicted-imminent gene removes its drug class."""
        d = advise(engine, monkeypatch, [bacterium_dict("e_coli") for _ in range(20)],
                   gene_probs={"mcr-1": 0.9})
        assert "colistin" not in d["safe_antibiotics"]

    def test_no_safe_options_text(self, engine, monkeypatch):
        """MRSA carrying vanA + gyrA + tetM: nothing available works."""
        d = advise(engine, monkeypatch,
                   [bacterium_dict("mrsa", ["vanA", "gyrA_S83L", "tetM"]) for _ in range(10)])
        assert d["safe_antibiotics"] == []
        assert "no available antibiotic is effective" in d["advisory_text"]
