"""
Tests for versioned biology (data/biology.py).

paper_v1 byte-identity is proven separately by tests/test_paper_v1_frozen.py;
here: paper_v1 is literally card_loader's tables and the model default, and
lab_v2 contains exactly the approved, sourced changes and nothing else.
"""
import dataclasses
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from data import card_loader
from data.biology import PAPER_V1, LAB_V2, BIOLOGIES, Biology, build_lab_v2, load_lab_v2_config
from simulation.amr_model import AMRSimulationModel
from core.bacterium_agent import BacteriumAgent


def living(m):
    return [a for a in m.agents if isinstance(a, BacteriumAgent)]


class TestPaperV1:
    def test_is_card_loader_objects(self):
        assert PAPER_V1.genes is card_loader.RESISTANCE_GENES
        assert PAPER_V1.germs is card_loader.GERM_PROFILES
        assert PAPER_V1.antibiotics is card_loader.ANTIBIOTIC_PROFILES
        assert PAPER_V1.non_transferable == frozenset() and PAPER_V1.warnings == []

    def test_model_default_is_paper_v1(self):
        m = AMRSimulationModel(scenario="validation", initial_bacteria=5, seed=1,
                               enable_logging=False)
        assert m.biology is PAPER_V1

    def test_card_loader_untouched_by_lab_v2(self):
        assert "mecA" not in card_loader.RESISTANCE_GENES
        assert card_loader.GERM_PROFILES["mrsa"].natural_resistances == ["tetM"]
        assert "acrAB-tolC" in card_loader.GERM_PROFILES["mrsa"].acquired_resistance_pool


class TestLabV2Contents:
    def test_only_the_approved_differences(self):
        """Everything except MRSA and the added mecA is shared with paper_v1."""
        assert set(LAB_V2.genes) - set(PAPER_V1.genes) == {"mecA"}
        for k, g in PAPER_V1.genes.items():
            assert LAB_V2.genes[k] is g
        for k, g in PAPER_V1.germs.items():
            if k != "mrsa":
                assert LAB_V2.germs[k] is g          # incl. Klebsiella: unchanged for now
        assert LAB_V2.antibiotics is PAPER_V1.antibiotics   # diffusion/decay unchanged for now
        a, b = dataclasses.asdict(PAPER_V1.germs["mrsa"]), dataclasses.asdict(LAB_V2.germs["mrsa"])
        assert {k for k in a if a[k] != b[k]} == {"natural_resistances", "acquired_resistance_pool"}

    def test_mrsa_profile(self):
        mrsa = LAB_V2.germs["mrsa"]
        assert mrsa.natural_resistances == ["mecA"]
        assert "tetM" not in mrsa.natural_resistances
        assert "acrAB-tolC" not in mrsa.acquired_resistance_pool
        assert set(mrsa.acquired_resistance_pool) == {"vanA", "tetM"}

    def test_meca_gene(self):
        g = LAB_V2.genes["mecA"]
        assert g.card_id == "ARO:3000617"
        assert g.acquisition_prob == 0.0
        assert "mecA" in LAB_V2.non_transferable
        assert {"penicillin", "carbapenem"} <= set(g.drug_classes)
        for germ in LAB_V2.germs.values():
            assert "mecA" not in germ.acquired_resistance_pool

    def test_meca_fitness_cost_from_ender_2004(self):
        """0.275 = 1 - 29/40 (BB255 vs RA120 doubling times, Ender et al. 2004)."""
        assert load_lab_v2_config()["mecA_fitness_cost"] == 0.275
        assert LAB_V2.genes["mecA"].fitness_cost == 0.275
        assert LAB_V2.warnings == []

    def test_meca_fitness_cost_unset_is_explicit(self):
        bio = build_lab_v2(mecA_fitness_cost=None)
        assert bio.genes["mecA"].fitness_cost == 0.0
        assert any("UNSET" in w for w in bio.warnings)

    def test_meca_fitness_cost_configurable(self):
        bio = build_lab_v2(mecA_fitness_cost=0.07)
        assert bio.genes["mecA"].fitness_cost == 0.07 and bio.warnings == []
        m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=10, seed=1,
                               enable_logging=False, biology=bio)
        base = bio.germs["mrsa"].baseline_fitness
        assert all(abs(a.fitness - (base - 0.07)) < 1e-9 for a in living(m))
        with pytest.raises(ValueError):
            build_lab_v2(mecA_fitness_cost=1.5)


class TestLabV2Behaviour:
    def test_mrsa_cells_carry_meca_and_resist_beta_lactams(self):
        m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=20, seed=2,
                               enable_logging=False, biology=LAB_V2)
        cells = living(m)
        assert cells and all(a.resistance_genes == {"mecA"} for a in cells)
        for ab in ("ampicillin", "meropenem"):
            assert cells[0].get_resistance_to(LAB_V2.antibiotics[ab]) > 0.8
        assert cells[0].get_resistance_to(LAB_V2.antibiotics["vancomycin"]) == 0.0

    def test_paper_v1_mrsa_unchanged(self):
        m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=20, seed=2,
                               enable_logging=False)
        assert all(a.resistance_genes == {"tetM"} for a in living(m))

    def test_meca_never_transfers_even_if_it_otherwise_could(self):
        """Force every other HGT condition to favour mecA; it must still not move."""
        forced_gene = dataclasses.replace(LAB_V2.genes["mecA"], acquisition_prob=1.0)
        mrsa = dataclasses.replace(LAB_V2.germs["mrsa"],
                                   natural_resistances=[],
                                   acquired_resistance_pool=["mecA", "vanA", "tetM"])
        bio = Biology("forced", {**LAB_V2.genes, "mecA": forced_gene},
                      {**LAB_V2.germs, "mrsa": mrsa}, LAB_V2.antibiotics,
                      non_transferable=frozenset({"mecA"}))
        m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=60, seed=4,
                               enable_logging=False, biology=bio)
        cells = living(m)
        for a in cells[:30]:
            a.resistance_genes.add("mecA")
        for _ in range(10):
            m.step()
        assert not [e for e in m.hgt_events if e.gene == "mecA"]

    def test_non_transferable_is_what_blocks_it(self):
        """Control for the test above: without the flag, the forced setup does transfer."""
        forced_gene = dataclasses.replace(LAB_V2.genes["mecA"], acquisition_prob=1.0)
        mrsa = dataclasses.replace(LAB_V2.germs["mrsa"], natural_resistances=[],
                                   acquired_resistance_pool=["mecA", "vanA", "tetM"])
        bio = Biology("forced-control", {**LAB_V2.genes, "mecA": forced_gene},
                      {**LAB_V2.germs, "mrsa": mrsa}, LAB_V2.antibiotics,
                      non_transferable=frozenset())
        m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=60, seed=4,
                               enable_logging=False, biology=bio)
        for a in living(m)[:30]:
            a.resistance_genes.add("mecA")
        for _ in range(10):
            m.step()
        assert [e for e in m.hgt_events if e.gene == "mecA"]

    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_lab_v2_mrsa_survives_meropenem(self, seed):
        """paper_v1 'MRSA' is killed by a carbapenem (it has no mecA); lab_v2 is not.
        (Meropenem, MBC 0.25, is used rather than ampicillin, MBC 4.0: a 3.0
        ampicillin dose drains below killing levels in both versions before it
        can act — see the diffusion finding in CLAUDE.md.)"""
        def survivors(bio):
            m = AMRSimulationModel(scenario="mrsa_hospital", initial_bacteria=80, seed=seed,
                                   enable_logging=False, biology=bio)
            for _ in range(5): m.step()
            m.apply_antibiotic("meropenem", concentration=2.0, mode="uniform")
            for _ in range(6): m.step()
            return m.count_living_bacteria()
        v1, v2 = survivors(PAPER_V1), survivors(LAB_V2)
        assert v1 < 40 and v2 >= 80


class TestServerBiology:
    def test_server_defaults_to_lab_v2_and_can_select_paper_v1(self):
        from api.server import create_app
        with TestClient(create_app(enable_logging=False)) as c:
            s = c.get("/state").json()
            assert s["biology"] == {"name": "lab_v2", "warnings": []}
            r = c.post("/reset", json={"scenario": "mrsa_hospital", "initial_bacteria": 10,
                                       "biology": "paper_v1"})
            assert r.status_code == 200
            s = c.get("/state").json()
            assert s["biology"] == {"name": "paper_v1", "warnings": []}
            assert all(b["resistance_genes"] == ["tetM"] for b in s["bacteria"])
            assert c.post("/reset", json={"biology": "v3"}).status_code == 422

    def test_spawn_uses_session_biology(self):
        from api.server import create_app
        with TestClient(create_app(enable_logging=False)) as c:
            c.post("/reset", json={"scenario": "validation", "initial_bacteria": 5})
            c.post("/spawn_bacteria", json={"germ_key": "mrsa", "count": 5})
            mrsa = [b for b in c.get("/state").json()["bacteria"] if b["species_key"] == "mrsa"]
            assert len(mrsa) == 5 and all(b["resistance_genes"] == ["mecA"] for b in mrsa)
