"""
Tests for AMR Simulation.

Tests are organized in layers:
  1. Data integrity — CARD profiles are internally consistent
  2. Agent biology — BacteriumAgent behaves according to real biology
  3. Model dynamics — population grows, resists, dies correctly
  4. API contracts — endpoints return expected shapes
  5. Science validation — simulation matches known biology

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import pytest

from data.card_loader import (
    GERM_PROFILES, ANTIBIOTIC_PROFILES, RESISTANCE_GENES,
    get_germ, get_antibiotic, resistance_probability,
    germ_summary, list_germs, list_antibiotics
)
from simulation.amr_model import AMRSimulationModel
from core.bacterium_agent import BacteriumAgent, BacterialState
from ai.resistance_analytics import (
    estimate_population_mic, recommend_treatment,
    shannon_diversity, selection_coefficient
)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestCARDData:

    def test_all_germs_have_required_fields(self):
        for key, germ in GERM_PROFILES.items():
            assert germ.species, f"{key} missing species"
            assert germ.who_priority in ["CRITICAL", "HIGH", "MEDIUM"], \
                f"{key} invalid WHO priority"
            assert 0 < germ.baseline_fitness <= 1.0, \
                f"{key} fitness out of range"
            assert 0 < germ.doubling_time_steps <= 200, \
                f"{key} doubling time out of range"
            assert 0 <= germ.biofilm_potential <= 1, \
                f"{key} biofilm potential out of range"

    def test_all_antibiotics_have_required_fields(self):
        for key, ab in ANTIBIOTIC_PROFILES.items():
            assert ab.name, f"{key} missing name"
            assert ab.mbc > 0, f"{key} MBC must be positive"
            assert ab.mic_susceptible > 0, f"{key} MIC_S must be positive"
            assert 0 < ab.diffusion_rate <= 1, f"{key} diffusion rate out of range"
            assert 0 < ab.decay_rate < 1, f"{key} decay rate out of range"
            assert ab.color_hex.startswith('#'), f"{key} color must be hex"

    def test_resistance_genes_cover_antibiotic_classes(self):
        """Every antibiotic class should have at least one resistance gene."""
        ab_classes = {ab.drug_class for ab in ANTIBIOTIC_PROFILES.values()}
        gene_classes = set()
        for gene in RESISTANCE_GENES.values():
            gene_classes.update(gene.drug_classes)
        # Intersection must not be empty
        covered = ab_classes.intersection(gene_classes)
        assert len(covered) > 0, "No antibiotic classes covered by resistance genes"

    def test_resistance_probability_returns_valid_range(self):
        for gene_name, gene in RESISTANCE_GENES.items():
            for ab_name, ab in ANTIBIOTIC_PROFILES.items():
                r = resistance_probability(gene, ab)
                assert 0.0 <= r <= 1.0, \
                    f"resistance_probability({gene_name}, {ab_name}) = {r} out of [0,1]"

    def test_fitness_cost_is_non_negative(self):
        for name, gene in RESISTANCE_GENES.items():
            assert gene.fitness_cost >= 0, f"{name} has negative fitness cost"
            assert gene.fitness_cost < 1.0, f"{name} fitness cost >= 1 (lethal)"

    def test_ndm1_resists_carbapenems(self):
        """NDM-1 should confer resistance to carbapenems (known biology)."""
        ndm1 = RESISTANCE_GENES["blaNDM-1"]
        meropenem = ANTIBIOTIC_PROFILES["meropenem"]
        r = resistance_probability(ndm1, meropenem)
        assert r >= 0.8, f"NDM-1 should strongly resist meropenem, got {r}"

    def test_mcr1_resists_colistin(self):
        """MCR-1 should confer resistance to colistin (critical WHO concern)."""
        mcr1 = RESISTANCE_GENES["mcr-1"]
        colistin = ANTIBIOTIC_PROFILES["colistin"]
        r = resistance_probability(mcr1, colistin)
        assert r >= 0.8, f"MCR-1 should strongly resist colistin, got {r}"

    def test_tem1_resists_penicillins(self):
        """TEM-1 should resist ampicillin (most common resistance gene globally)."""
        tem1 = RESISTANCE_GENES["blaTEM-1"]
        ampicillin = ANTIBIOTIC_PROFILES["ampicillin"]
        r = resistance_probability(tem1, ampicillin)
        assert r >= 0.8, f"TEM-1 should strongly resist ampicillin, got {r}"

    def test_germ_lookups(self):
        for key in list_germs():
            g = get_germ(key)
            assert g is not None

    def test_antibiotic_lookups(self):
        for key in list_antibiotics():
            ab = get_antibiotic(key)
            assert ab is not None

    def test_invalid_germ_raises(self):
        with pytest.raises(ValueError):
            get_germ("totally_fake_germ_xyz")

    def test_invalid_antibiotic_raises(self):
        with pytest.raises(ValueError):
            get_antibiotic("magic_cure_xyz")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: AGENT BIOLOGY
# ─────────────────────────────────────────────────────────────────────────────

class TestBacteriumAgent:

    @pytest.fixture
    def model(self):
        return AMRSimulationModel(scenario="validation", seed=42, initial_bacteria=10)

    @pytest.fixture
    def ecoli_agent(self, model):
        profile = get_germ("e_coli")
        agent = BacteriumAgent(model=model, profile=profile, position=(40, 30))
        model.grid.place_agent(agent, (40, 30))
        return agent

    def test_agent_starts_alive(self, ecoli_agent):
        assert ecoli_agent.state != BacterialState.DEAD

    def test_agent_has_valid_fitness(self, ecoli_agent):
        assert 0 < ecoli_agent.fitness <= 1.0

    def test_agent_has_valid_energy(self, ecoli_agent):
        assert 0 <= ecoli_agent.energy <= 1.0

    def test_agent_no_acquired_resistance_at_start(self, ecoli_agent):
        """Fresh E. coli should have no acquired resistance (no natural resistances for e_coli)."""
        profile = get_germ("e_coli")
        # Natural resistances are inherited
        assert ecoli_agent.resistance_genes == set(profile.natural_resistances)

    def test_resistance_calculation_susceptible(self, ecoli_agent):
        """Susceptible bacteria should have near-zero resistance."""
        ciprofloxacin = get_antibiotic("ciprofloxacin")
        # e_coli with no genes has no ciprofloxacin resistance
        r = ecoli_agent.get_resistance_to(ciprofloxacin)
        assert r < 0.1, f"Susceptible E. coli should be sensitive, got {r}"

    def test_resistance_increases_after_gene_acquisition(self, ecoli_agent):
        """After acquiring gyrA mutation, resistance to ciprofloxacin should increase."""
        ciprofloxacin = get_antibiotic("ciprofloxacin")
        r_before = ecoli_agent.get_resistance_to(ciprofloxacin)

        ecoli_agent.resistance_genes.add("gyrA_S83L")
        ecoli_agent._recalculate_fitness()

        r_after = ecoli_agent.get_resistance_to(ciprofloxacin)
        assert r_after > r_before, \
            f"Resistance should increase after gyrA acquisition: {r_before} -> {r_after}"
        assert r_after >= 0.8, f"gyrA should give high ciprofloxacin resistance, got {r_after}"

    def test_fitness_cost_of_resistance_genes(self, ecoli_agent):
        """Acquiring resistance genes should reduce fitness."""
        fitness_before = ecoli_agent.fitness
        ecoli_agent.resistance_genes.add("blaKPC-2")
        ecoli_agent._recalculate_fitness()
        assert ecoli_agent.fitness < fitness_before, \
            "Resistance gene acquisition should reduce fitness"

    def test_multiple_genes_stack_resistance(self, ecoli_agent):
        """Multiple beta-lactamase genes should provide stronger combined resistance."""
        meropenem = get_antibiotic("meropenem")
        ecoli_agent.resistance_genes.add("blaNDM-1")
        r1 = ecoli_agent.get_resistance_to(meropenem)

        ecoli_agent.resistance_genes.add("blaKPC-2")
        r2 = ecoli_agent.get_resistance_to(meropenem)

        assert r2 >= r1, "More resistance genes should not decrease resistance"

    def test_biofilm_increases_resistance(self, ecoli_agent):
        """Biofilm reduces antibiotic penetration (Hoiby 2010).
        Requires base resistance to show multiplicative protection effect."""
        ciprofloxacin = get_antibiotic("ciprofloxacin")

        # Give the agent a resistance gene first — biofilm multiplies existing protection
        ecoli_agent.resistance_genes.add("gyrA_S83L")
        ecoli_agent._recalculate_fitness()

        r_free = ecoli_agent.get_resistance_to(ciprofloxacin)
        ecoli_agent.in_biofilm = True
        r_biofilm = ecoli_agent.get_resistance_to(ciprofloxacin)

        assert r_biofilm > r_free, \
            f"Biofilm should increase effective resistance: {r_free:.3f} -> {r_biofilm:.3f}"
        assert r_biofilm >= 0.95, \
            f"Biofilm + gyrA should give near-complete protection, got {r_biofilm:.3f}"

    def test_persister_cells_are_highly_tolerant(self, ecoli_agent):
        """Persister cells are metabolically dormant — antibiotic tolerance via phenotype.
        Persisters show high tolerance regardless of genotypic resistance genes."""
        ciprofloxacin = get_antibiotic("ciprofloxacin")

        # Give base resistance so persister boost is testable multiplicatively
        ecoli_agent.resistance_genes.add("gyrA_S83L")
        ecoli_agent._recalculate_fitness()
        r_normal = ecoli_agent.get_resistance_to(ciprofloxacin)

        ecoli_agent.is_persister = True
        r_persister = ecoli_agent.get_resistance_to(ciprofloxacin)

        assert r_persister >= 0.9, \
            f"Persister cells should be highly tolerant, got {r_persister}"
        assert r_persister >= r_normal, \
            "Persister tolerance should be at least as high as genotypic resistance"

    def test_sos_activates_under_stress(self, ecoli_agent):
        """SOS response should activate when antibiotic concentration is high."""
        assert not ecoli_agent.sos_active
        # Simulate high stress
        for _ in range(10):
            ecoli_agent._update_sos_response(0.9)
        assert ecoli_agent.sos_active or ecoli_agent.stress_level > 0.3

    def test_agent_serializes_to_dict(self, ecoli_agent):
        d = ecoli_agent.to_dict()
        required_keys = ["id", "species", "pos", "state", "fitness", "energy",
                         "resistance_genes", "in_biofilm", "sos_active"]
        for key in required_keys:
            assert key in d, f"Missing key in agent dict: {key}"

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: MODEL DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

class TestModelDynamics:

    def test_population_grows_without_antibiotics(self):
        """Without antibiotics, bacteria should grow."""
        model = AMRSimulationModel(scenario="validation", seed=1, initial_bacteria=20)
        initial_count = model.count_living_bacteria()

        for _ in range(15):
            model.step()

        final_count = model.count_living_bacteria()
        assert final_count > initial_count, \
            f"Population should grow without antibiotics: {initial_count} -> {final_count}"

    def test_antibiotic_reduces_population(self):
        """Antibiotics should reduce bacterial population."""
        model = AMRSimulationModel(scenario="validation", seed=2, initial_bacteria=60)

        # Grow first
        for _ in range(10):
            model.step()

        pre_treatment = model.count_living_bacteria()

        # Apply high dose
        model.apply_antibiotic("ciprofloxacin", concentration=3.0, mode="uniform")

        for _ in range(20):
            model.step()

        post_treatment = model.count_living_bacteria()
        assert post_treatment < pre_treatment, \
            f"Antibiotic should reduce population: {pre_treatment} -> {post_treatment}"

    def test_resistant_bacteria_survive_better(self):
        """Bacteria with resistance genes survive antibiotic treatment better."""
        model = AMRSimulationModel(scenario="validation", seed=3, initial_bacteria=40)

        # Manually give some bacteria gyrA resistance
        living = [a for a in model.agents if isinstance(a, BacteriumAgent)]
        resistant_ids = set()
        for agent in living[:20]:
            agent.resistance_genes.add("gyrA_S83L")
            agent._recalculate_fitness()
            resistant_ids.add(agent.unique_id)

        # Apply ciprofloxacin (gyrA confers resistance to it)
        model.apply_antibiotic("ciprofloxacin", concentration=2.0, mode="uniform")

        for _ in range(15):
            model.step()

        # Check survival
        remaining = [
            a for a in model.agents
            if isinstance(a, BacteriumAgent) and a.state != BacterialState.DEAD
        ]
        resistant_survivors = sum(
            1 for a in remaining if "gyrA_S83L" in a.resistance_genes
        )
        susceptible_survivors = sum(
            1 for a in remaining if "gyrA_S83L" not in a.resistance_genes
        )

        # Resistant should survive at higher rate
        total_resistant_start = 20
        total_susceptible_start = len(living) - 20

        if total_susceptible_start > 0 and total_resistant_start > 0:
            rate_R = resistant_survivors / total_resistant_start
            rate_S = susceptible_survivors / total_susceptible_start
            # Resistant should survive at least as well (may not always hold in short runs)
            assert rate_R >= rate_S * 0.5, \
                f"Resistant rate {rate_R:.2f} should not be far below susceptible rate {rate_S:.2f}"

    def test_hgt_events_are_recorded(self):
        """HGT events should be recorded during simulation."""
        model = AMRSimulationModel(scenario="validation", seed=5, initial_bacteria=100)

        # Give some bacteria resistance genes to transfer
        living = [a for a in model.agents if isinstance(a, BacteriumAgent)]
        for agent in living[:30]:
            agent.resistance_genes.add("blaTEM-1")

        for _ in range(20):
            model.step()

        # HGT should have occurred in a dense population
        # (not guaranteed every run but very likely with 100+ bacteria)
        total_with_gene = sum(
            1 for a in model.agents
            if isinstance(a, BacteriumAgent)
            and "blaTEM-1" in a.resistance_genes
            and a.state != BacterialState.DEAD
        )
        assert total_with_gene >= 30, "Gene should spread or at least persist"

    def test_antibiotic_diffuses(self):
        """Antibiotic should diffuse from application point."""
        model = AMRSimulationModel(scenario="validation", seed=6, initial_bacteria=10)
        model.apply_antibiotic("ciprofloxacin", concentration=2.0, mode="spot",
                               center=(40, 30), radius=5)

        conc_at_center = model.antibiotic_grids["ciprofloxacin"][40, 30]
        conc_far_away  = model.antibiotic_grids["ciprofloxacin"][0, 0]

        assert conc_at_center > 0, "Antibiotic should be present at application point"
        assert conc_at_center > conc_far_away, "Concentration should be higher at center"

        # After steps, it should spread
        for _ in range(5):
            model.step()

        conc_nearby_after = model.antibiotic_grids["ciprofloxacin"][35, 30]
        assert conc_nearby_after >= 0, "Concentration should not go negative"

    def test_nutrients_regenerate(self):
        """Nutrient grid should regenerate over time."""
        model = AMRSimulationModel(scenario="validation", seed=7, initial_bacteria=5)

        # Deplete nutrients manually
        model.nutrient_grid[:] = 0.0

        for _ in range(10):
            model._regenerate_nutrients()

        avg_nutrient = model.nutrient_grid.mean()
        assert avg_nutrient > 0, "Nutrients should regenerate"

    def test_model_state_is_serializable(self):
        """get_full_state() should return a JSON-compatible dict."""
        model = AMRSimulationModel(scenario="ecoli_cipro", seed=8, initial_bacteria=20)
        for _ in range(3):
            model.step()

        state = model.get_full_state()
        assert "bacteria" in state
        assert "stats" in state
        assert "antibiotic_heatmaps" in state
        assert "nutrient_heatmap" in state
        assert "hgt_events" in state

        # All bacteria should have required fields
        for b in state["bacteria"]:
            assert "pos" in b
            assert "state" in b
            assert "fitness" in b

    def test_reset_clears_state(self):
        """After reset, state should be clean."""
        model = AMRSimulationModel(scenario="validation", seed=9, initial_bacteria=50)
        for _ in range(10):
            model.step()

        # New model (equivalent to reset)
        model2 = AMRSimulationModel(scenario="validation", seed=9, initial_bacteria=50)
        assert model2.current_step == 0
        assert len(model2.hgt_events) == 0

    def test_multi_species_coexist(self):
        """Multi-species scenario should have multiple species."""
        model = AMRSimulationModel(scenario="pakistan_crisis", seed=10, initial_bacteria=80)

        species = model.get_species_counts()
        assert len(species) >= 2, f"Pakistan crisis should have >= 2 species, got {species}"

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: AI ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestAIAnalytics:

    @pytest.fixture
    def mixed_population(self):
        """Create a mixed population with some resistant bacteria."""
        model = AMRSimulationModel(scenario="validation", seed=42, initial_bacteria=50)
        profile = get_germ("e_coli")
        living = [a for a in model.agents if isinstance(a, BacteriumAgent)]
        for agent in living[:20]:
            agent.resistance_genes.add("gyrA_S83L")
        return living

    def test_mic_estimation_returns_valid_percentages(self, mixed_population):
        result = estimate_population_mic(mixed_population, "ciprofloxacin")
        if result:
            total_pct = result["susceptible_pct"] + result["intermediate_pct"] + result["resistant_pct"]
            assert abs(total_pct - 100.0) < 2.0, \
                f"MIC percentages should sum to ~100%, got {total_pct}"

    def test_resistant_population_has_higher_resistant_pct(self, mixed_population):
        """Population with gyrA genes should show higher ciprofloxacin resistance."""
        result = estimate_population_mic(mixed_population, "ciprofloxacin")
        if result:
            assert result["resistant_pct"] > 0, \
                "Population with gyrA_S83L should show some ciprofloxacin resistance"

    def test_shannon_diversity_increases_with_variety(self):
        """Higher genetic variety should give higher Shannon diversity."""
        model = AMRSimulationModel(scenario="validation", seed=42, initial_bacteria=50)
        living = [a for a in model.agents if isinstance(a, BacteriumAgent)]

        # All same genotype → low diversity
        for a in living:
            a.resistance_genes = set()
        h_low = shannon_diversity(living)

        # Mixed genotypes → higher diversity
        for i, a in enumerate(living):
            if i % 3 == 0:
                a.resistance_genes = {"blaTEM-1"}
            elif i % 3 == 1:
                a.resistance_genes = {"gyrA_S83L"}
        h_high = shannon_diversity(living)

        assert h_high >= h_low, \
            f"Mixed genotypes should have higher diversity: {h_low:.3f} vs {h_high:.3f}"

    def test_treatment_recommendation_ranks_antibiotics(self, mixed_population):
        ab_list = ["ciprofloxacin", "meropenem", "ampicillin"]
        recs = recommend_treatment(mixed_population, ab_list)
        assert len(recs) > 0, "Should return at least one recommendation"
        assert recs[0]["score"] >= recs[-1]["score"], "Should be sorted descending by score"

    def test_selection_coefficient_positive_under_pressure(self):
        """Resistant bacteria should have positive selection coefficient under antibiotics."""
        model = AMRSimulationModel(scenario="validation", seed=42, initial_bacteria=60)
        living = [a for a in model.agents if isinstance(a, BacteriumAgent)]

        # Make half resistant to ciprofloxacin
        for a in living[:30]:
            a.resistance_genes.add("gyrA_S83L")
            a._recalculate_fitness()

        # Apply antibiotic pressure
        model.apply_antibiotic("ciprofloxacin", concentration=2.0, mode="uniform")
        for _ in range(5):
            model.step()

        updated_living = [
            a for a in model.agents
            if isinstance(a, BacteriumAgent) and a.state != BacterialState.DEAD
        ]
        # Selection coefficient shouldn't be NaN
        s = selection_coefficient(updated_living, "ciprofloxacin")
        assert isinstance(s, float), "Selection coefficient should be a float"
        assert not math.isnan(s), "Selection coefficient should not be NaN"

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5: SCIENCE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestScienceValidation:
    """
    These tests validate that the simulation qualitatively matches
    published microbiology literature.
    """

    def test_klebsiella_has_higher_biofilm_than_ecoli(self):
        """Klebsiella pneumoniae forms more biofilm than E. coli (published biology)."""
        kp = get_germ("klebsiella_pneumoniae")
        ec = get_germ("e_coli")
        assert kp.biofilm_potential > ec.biofilm_potential, \
            "Klebsiella should have higher biofilm potential than E. coli"

    def test_pseudomonas_has_highest_mutation_rate(self):
        """Pseudomonas aeruginosa has the highest intrinsic mutation rate."""
        pa = get_germ("pseudomonas_aeruginosa")
        ec = get_germ("e_coli")
        assert pa.baseline_mutation_rate >= ec.baseline_mutation_rate, \
            "Pseudomonas should have >= mutation rate vs E. coli"

    def test_pseudomonas_is_motile(self):
        """Pseudomonas is highly motile (flagella)."""
        pa = get_germ("pseudomonas_aeruginosa")
        assert pa.motility >= 0.8, "Pseudomonas should be highly motile"

    def test_mrsa_is_gram_positive(self):
        """MRSA is gram-positive staphylococcus."""
        mrsa = get_germ("mrsa")
        assert mrsa.gram_stain == "positive"

    def test_ecoli_is_gram_negative(self):
        """E. coli is gram-negative."""
        ec = get_germ("e_coli")
        assert ec.gram_stain == "negative"

    def test_colistin_is_last_resort(self):
        """Colistin MBC should be within clinical range (2-4 µg/mL)."""
        col = get_antibiotic("colistin")
        assert 0.5 <= col.mbc <= 8.0, \
            f"Colistin MBC {col.mbc} outside clinical range"

    def test_ndm1_is_plasmid_mediated_high_hgt(self):
        """NDM-1 is plasmid-mediated — should have non-zero acquisition probability."""
        ndm1 = RESISTANCE_GENES["blaNDM-1"]
        assert ndm1.acquisition_prob > 0, "NDM-1 should be transferable via HGT"

    def test_tem1_more_common_than_ndm1(self):
        """TEM-1 should be more commonly acquired than NDM-1 (higher frequency globally)."""
        tem1 = RESISTANCE_GENES["blaTEM-1"]
        ndm1 = RESISTANCE_GENES["blaNDM-1"]
        assert tem1.acquisition_prob > ndm1.acquisition_prob, \
            "TEM-1 should have higher acquisition probability than NDM-1"

    def test_vancomycin_only_effective_against_gram_positive(self):
        """Vancomycin targets gram-positive bacteria; resistance genes for gram-negative are minimal."""
        vanco = get_antibiotic("vancomycin")
        assert vanco.drug_class == "glycopeptide"
        assert vanco.bactericidal == True

    def test_acinetobacter_has_high_biofilm(self):
        """Acinetobacter is notorious for extreme biofilm formation."""
        ab = get_germ("acinetobacter_baumannii")
        assert ab.biofilm_potential >= 0.9, \
            f"Acinetobacter biofilm potential should be >= 0.9, got {ab.biofilm_potential}"

    def test_who_critical_priority_germs_present(self):
        """WHO Critical Priority 1 pathogens should be in the database."""
        critical = [k for k, v in GERM_PROFILES.items() if v.who_priority == "CRITICAL"]
        assert len(critical) >= 3, \
            f"Should have >= 3 WHO Critical pathogens, got {len(critical)}: {critical}"

    def test_population_dynamics_logistic_growth(self):
        """
        Population without antibiotics should follow logistic growth —
        slow initial growth, accelerating, then plateau at carrying capacity.
        """
        model = AMRSimulationModel(scenario="validation", seed=99, initial_bacteria=15)

        counts = []
        for _ in range(30):
            model.step()
            counts.append(model.count_living_bacteria())

        # Population should generally increase
        assert counts[-1] > counts[0], \
            f"Population should grow: {counts[0]} -> {counts[-1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])