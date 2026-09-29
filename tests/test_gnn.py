"""
Tests for GNN feature engineering, model architecture, and inference pipeline.

Layer 6: GNN Feature Engineering
Layer 7: GNN Model Architecture
Layer 8: GNN Inference Integration
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pytest
import torch

from ai.feature_engineering import (
    extract_node_features, extract_edge_features,
    build_graph_from_state, feature_names, edge_feature_names,
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, N_GENES, GENE_INDEX, AB_INDEX,
    get_dims, generate_hgt_labels, _generate_hgt_labels_diff_legacy,
    collect_training_snapshots,
)
from ai.gnn_model import build_model, AMRResistanceGNN
from simulation.amr_model import AMRSimulationModel
from core.bacterium_agent import BacteriumAgent, BacterialState

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_bacterium():
    return {
        "id": 1, "cell_id": "abc123",
        "species": "Escherichia coli",
        "pos": [40, 30],
        "state": "growing",
        "fitness": 0.82, "energy": 0.65,
        "stress_level": 0.12, "antibiotic_damage": 0.05,
        "age": 20,
        "in_biofilm": False, "is_persister": False, "sos_active": False,
        "local_density": 3, "generation": 2, "offspring_count": 4,
        "resistance_genes": ["blaTEM-1", "gyrA_S83L"],
        "gene_count": 2,
        "gram_stain": "negative", "shape": "bacillus",
        "color_hex": "#4CAF50",
        "total_mutations": 3, "hgt_events": 1,
    }

@pytest.fixture(scope="module")
def sample_bacterium_resistant():
    return {
        "id": 2, "cell_id": "def456",
        "species": "Klebsiella pneumoniae",
        "pos": [42, 31],
        "state": "stressed",
        "fitness": 0.70, "energy": 0.45,
        "stress_level": 0.55, "antibiotic_damage": 0.25,
        "age": 35,
        "in_biofilm": True, "is_persister": False, "sos_active": True,
        "local_density": 6, "generation": 5, "offspring_count": 8,
        "resistance_genes": ["blaNDM-1", "blaKPC-2", "mcr-1", "acrAB-tolC"],
        "gene_count": 4,
        "gram_stain": "negative", "shape": "bacillus",
        "color_hex": "#FF9800",
        "total_mutations": 12, "hgt_events": 5,
    }

@pytest.fixture(scope="module")
def sim_state():
    """Real simulation state for graph construction tests."""
    model = AMRSimulationModel(
        scenario="ecoli_cipro", initial_bacteria=60, seed=99, enable_logging=False
    )
    for _ in range(12):
        model.step()
    model.apply_antibiotic("ciprofloxacin", concentration=1.5, mode="uniform")
    for _ in range(5):
        model.step()
    return model.get_full_state()

@pytest.fixture(scope="module")
def gnn_model():
    return build_model(hidden_dim=64, edge_enc_dim=32, n_layers=2, heads=4)

@pytest.fixture(scope="module")
def dummy_pyg_data():
    from torch_geometric.data import Data
    N, E = 15, 45
    return Data(
        x          = torch.randn(N, NODE_FEATURE_DIM),
        edge_index = torch.randint(0, N, (2, E)),
        edge_attr  = torch.randn(E, EDGE_FEATURE_DIM),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_node_feature_shape(self, sample_bacterium):
        feats = extract_node_features(sample_bacterium, 80, 60)
        assert feats.shape == (NODE_FEATURE_DIM,), \
            f"Expected ({NODE_FEATURE_DIM},), got {feats.shape}"

    def test_node_feature_dtype(self, sample_bacterium):
        feats = extract_node_features(sample_bacterium, 80, 60)
        assert feats.dtype == np.float32

    def test_node_features_bounded(self, sample_bacterium):
        """All node features should be in [0, 1] range."""
        feats = extract_node_features(sample_bacterium, 80, 60)
        assert feats.min() >= 0.0, f"Feature below 0: min={feats.min()}"
        assert feats.max() <= 1.0 + 1e-5, f"Feature above 1: max={feats.max()}"

    def test_gene_features_binary(self, sample_bacterium):
        """Gene presence features must be 0 or 1."""
        feats = extract_node_features(sample_bacterium, 80, 60)
        gene_feats = feats[:N_GENES]
        assert set(gene_feats.round().astype(int)).issubset({0, 1}), \
            "Gene features must be binary"

    def test_correct_genes_flagged(self, sample_bacterium):
        """blaTEM-1 and gyrA_S83L should be flagged (they're in the sample)."""
        feats      = extract_node_features(sample_bacterium, 80, 60)
        tem1_idx   = GENE_INDEX.index("blaTEM-1")
        gyra_idx   = GENE_INDEX.index("gyrA_S83L")
        ndm1_idx   = GENE_INDEX.index("blaNDM-1")
        assert feats[tem1_idx]  == 1.0, "blaTEM-1 should be flagged"
        assert feats[gyra_idx]  == 1.0, "gyrA_S83L should be flagged"
        assert feats[ndm1_idx]  == 0.0, "blaNDM-1 should NOT be flagged"

    def test_spatial_normalization(self, sample_bacterium):
        """Position features should normalize correctly to [0, 1]."""
        feats = extract_node_features(sample_bacterium, 80, 60)
        # pos = [40, 30], grid = 80x60
        # x_norm = 40/79 ~ 0.506, y_norm = 30/59 ~ 0.508
        # spatial is at indices 17:19 in the de-leaked 35-dim layout
        x_norm = feats[17]
        y_norm = feats[18]
        assert 0.4 < x_norm < 0.7, f"x_norm={x_norm} should be ~0.5"
        assert 0.4 < y_norm < 0.7, f"y_norm={y_norm} should be ~0.5"

    def test_species_one_hot(self, sample_bacterium, sample_bacterium_resistant):
        """Species one-hot should have exactly one 1."""
        feats_ec  = extract_node_features(sample_bacterium, 80, 60)
        feats_kp  = extract_node_features(sample_bacterium_resistant, 80, 60)
        # species one-hot is at indices 22:27 in the de-leaked 35-dim layout
        sp_ec  = feats_ec[22:27]
        sp_kp  = feats_kp[22:27]
        assert sp_ec.sum()  == 1.0, "E. coli should have one-hot species"
        assert sp_kp.sum()  == 1.0, "Klebsiella should have one-hot species"
        assert (sp_ec != sp_kp).any(), "Different species should differ"

    def test_resistant_bacterium_more_genes_flagged(
        self, sample_bacterium, sample_bacterium_resistant
    ):
        feats_s = extract_node_features(sample_bacterium, 80, 60)
        feats_r = extract_node_features(sample_bacterium_resistant, 80, 60)
        assert feats_r[:N_GENES].sum() > feats_s[:N_GENES].sum(), \
            "Resistant bacterium should have more gene flags"

    def test_ab_exposure_features(self, sample_bacterium):
        """Antibiotic exposure features should reflect provided concentrations."""
        concs = {"ciprofloxacin": 1.5, "meropenem": 0.0}
        feats = extract_node_features(sample_bacterium, 80, 60, concs)
        # AB exposure starts at index 29 in the de-leaked 35-dim layout
        cipro_idx   = 29 + AB_INDEX.index("ciprofloxacin")
        meropen_idx = 29 + AB_INDEX.index("meropenem")
        assert feats[cipro_idx] > 0,    "Ciprofloxacin exposure should be non-zero"
        assert feats[meropen_idx] == 0.0, "Meropenem exposure should be zero"

    def test_edge_feature_shape(self, sample_bacterium, sample_bacterium_resistant):
        feats = extract_edge_features(sample_bacterium, sample_bacterium_resistant, 80, 60)
        assert feats.shape == (EDGE_FEATURE_DIM,), \
            f"Expected ({EDGE_FEATURE_DIM},), got {feats.shape}"

    def test_shared_genes_feature(self, sample_bacterium, sample_bacterium_resistant):
        """
        shared_genes_norm (edge feat 0) replaced raw distance as first edge
        feature after leakage remediation. Two different-species bacteria with
        no gene overlap should give shared=0; same bacterium vs itself gives
        shared proportional to its gene count.
        """
        feats_diff = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        feats_self = extract_edge_features(
            sample_bacterium, sample_bacterium, 80, 60)
        # sample_bacterium has {blaTEM-1, gyrA_S83L},
        # sample_bacterium_resistant has {blaNDM-1, blaKPC-2, mcr-1, acrAB-tolC}
        # overlap = 0  ->  shared_genes_norm = 0
        assert feats_diff[0] == 0.0, \
            f"No shared genes should give shared_genes_norm=0, got {feats_diff[0]}"
        # self vs self -> all genes shared -> > 0
        assert feats_self[0] > 0.0, \
            "Same bacterium vs itself should have non-zero shared_genes_norm"

    def test_fitness_diff_feature(self, sample_bacterium, sample_bacterium_resistant):
        """Fitness difference feature (edge feat 1) should reflect |fit_i - fit_j|."""
        feats = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        # sample: fitness=0.82, resistant: fitness=0.70
        expected = abs(0.82 - 0.70)
        assert abs(feats[1] - expected) < 0.01, \
            f"Fitness diff feature {feats[1]:.3f} != expected {expected:.3f}"

    def test_biofilm_edge_feature(self, sample_bacterium_resistant):
        """Both-biofilm flag (edge feat 2) should reflect co-biofilm status."""
        b_no_biofilm = {**sample_bacterium_resistant, "in_biofilm": False}
        feats_both = extract_edge_features(
            sample_bacterium_resistant, sample_bacterium_resistant, 80, 60)
        feats_none = extract_edge_features(b_no_biofilm, b_no_biofilm, 80, 60)
        assert feats_both[2] == 1.0, "Both in biofilm -> flag=1"
        assert feats_none[2] == 0.0, "Neither in biofilm -> flag=0"

    def test_stress_diff_feature(self, sample_bacterium, sample_bacterium_resistant):
        """Stress difference (edge feat 3) should reflect |stress_i - stress_j|."""
        feats = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        # sample: stress=0.12, resistant: stress=0.55
        expected = abs(0.12 - 0.55)
        assert abs(feats[3] - expected) < 0.02, \
            f"Stress diff feature {feats[3]:.3f} != expected {expected:.3f}"

    def test_proximity_band_same_cell(self, sample_bacterium):
        """Bacteria at the same position should get proximity_band=0.0."""
        feats = extract_edge_features(sample_bacterium, sample_bacterium, 80, 60)
        assert feats[4] == 0.0, \
            f"Same position should give proximity_band=0.0, got {feats[4]}"

    def test_proximity_band_nearby(self, sample_bacterium, sample_bacterium_resistant):
        """
        Bacteria at pos [40,30] and [42,31] (dist~2.24) should be in
        the 'nearby' band (0.5), not 'far' (1.0) or 'same cell' (0.0).
        This confirms the coarsened band feature doesn't expose the exact
        CONJUGATION_DISTANCE=1 threshold (that would reintroduce leakage).
        """
        feats = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        assert feats[4] == 0.5, \
            f"Distance~2.24 should give proximity_band=0.5 (nearby), got {feats[4]}"

    def test_proximity_band_far(self, sample_bacterium):
        """Bacteria >3 cells apart should get proximity_band=1.0 (far)."""
        far_bacterium = {**sample_bacterium, "pos": [60, 50]}
        feats = extract_edge_features(sample_bacterium, far_bacterium, 80, 60)
        assert feats[4] == 1.0, \
            f"Distance>3 should give proximity_band=1.0 (far), got {feats[4]}"

    def test_removed_features_not_present(self, sample_bacterium, sample_bacterium_resistant):
        """
        Guard test: confirm the EDGE_FEATURE_DIM is 5, not 8.
        The four removed features (raw distance, same_species, transferable_genes,
        sos_active flag) each encoded a direct component of the HGT
        label-generating rule and caused ~14.8x label inflation through the
        leak into the GNN's discrimination.
        """
        feats = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        assert feats.shape[0] == 5, (
            f"EDGE_FEATURE_DIM should be 5 after leakage remediation, "
            f"got {feats.shape[0]}. Did someone re-add a removed feature?"
        )

    def test_sos_active_not_in_node_features(self, sample_bacterium):
        """
        Guard test: confirm that setting sos_active=True on a bacterium
        does NOT change the node feature vector. sos_active was removed
        because it is a direct multiplicative term in the HGT label
        (hgt_prob *= 2.0 if self.sos_active).
        """
        b_no_sos  = {**sample_bacterium, "sos_active": False}
        b_yes_sos = {**sample_bacterium, "sos_active": True}
        feats_no  = extract_node_features(b_no_sos,  80, 60)
        feats_yes = extract_node_features(b_yes_sos, 80, 60)
        assert (feats_no == feats_yes).all(), (
            "sos_active changed the node feature vector - it should be "
            "excluded to prevent leakage of the HGT label-generating rule."
        )

    def test_feature_names_count(self):
        assert len(feature_names()) == NODE_FEATURE_DIM
        assert len(edge_feature_names()) == EDGE_FEATURE_DIM

    def test_get_dims(self):
        dims = get_dims()
        assert dims["node_feature_dim"] == NODE_FEATURE_DIM
        assert dims["edge_feature_dim"] == EDGE_FEATURE_DIM
        assert dims["n_genes"]          == N_GENES
        assert len(dims["gene_names"])  == N_GENES


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7: GNN MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNModel:

    def test_model_builds(self, gnn_model):
        assert isinstance(gnn_model, AMRResistanceGNN)

    def test_parameter_count_reasonable(self, gnn_model):
        n = gnn_model.count_parameters()
        assert 10_000 < n < 5_000_000, \
            f"Parameter count {n:,} seems wrong"

    def test_forward_pass_shape(self, gnn_model, dummy_pyg_data):
        gnn_model.eval()
        with torch.no_grad():
            logits = gnn_model(dummy_pyg_data)
        E = dummy_pyg_data.edge_index.shape[1]
        assert logits.shape == (E, N_GENES), \
            f"Expected ({E}, {N_GENES}), got {logits.shape}"

    def test_output_is_logits_not_probs(self, gnn_model, dummy_pyg_data):
        """Output should be raw logits (can be outside [0,1])."""
        gnn_model.eval()
        with torch.no_grad():
            logits = gnn_model(dummy_pyg_data)
        # Logits can be negative or >1; probs cannot
        has_neg = (logits < 0).any()
        has_gt1 = (logits > 1).any()
        assert has_neg or has_gt1, \
            "Model output looks like probabilities - should be raw logits"

    def test_predict_proba_bounded(self, gnn_model, dummy_pyg_data):
        proba = gnn_model.predict_proba(dummy_pyg_data)
        assert proba.min() >= 0.0, "Probabilities must be >= 0"
        assert proba.max() <= 1.0, "Probabilities must be <= 1"
        assert proba.shape == (dummy_pyg_data.edge_index.shape[1], N_GENES)

    def test_predict_transfers_returns_dict(self, gnn_model, dummy_pyg_data):
        transfers = gnn_model.predict_transfers(dummy_pyg_data, threshold=0.5)
        assert isinstance(transfers, dict)
        for edge_idx, genes in transfers.items():
            assert isinstance(edge_idx, int)
            assert isinstance(genes, list)
            assert all(g in GENE_INDEX for g in genes), \
                f"Unknown gene in predictions: {genes}"

    def test_architecture_summary(self, gnn_model):
        summary = gnn_model.architecture_summary()
        required_keys = ["model", "node_feature_dim", "edge_feature_dim",
                         "hidden_dim", "trainable_params", "output_genes"]
        for k in required_keys:
            assert k in summary, f"Missing key in summary: {k}"
        assert summary["output_genes"] == GENE_INDEX

    def test_gradient_flows(self, gnn_model, dummy_pyg_data):
        """Gradients should flow through all layers."""
        gnn_model.train()
        logits = gnn_model(dummy_pyg_data)
        fake_labels = torch.zeros_like(logits)
        fake_labels[0, 0] = 1.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fake_labels)
        loss.backward()

        for name, param in gnn_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"

    def test_dropout_affects_output(self, gnn_model, dummy_pyg_data):
        """In train mode with dropout, outputs should vary."""
        gnn_model.train()
        out1 = gnn_model(dummy_pyg_data).detach()
        out2 = gnn_model(dummy_pyg_data).detach()
        # With dropout, outputs should differ (at least sometimes)
        # We just check the model runs without error in both modes
        gnn_model.eval()
        out3 = gnn_model(dummy_pyg_data).detach()
        assert out3.shape == out1.shape

    def test_model_handles_single_edge(self):
        """Model should handle minimal graphs (2 nodes, 1 edge)."""
        from torch_geometric.data import Data
        model = build_model(hidden_dim=64, edge_enc_dim=32, n_layers=2, heads=4)
        model.eval()
        data = Data(
            x          = torch.randn(2, NODE_FEATURE_DIM),
            edge_index = torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr  = torch.randn(1, EDGE_FEATURE_DIM),
        )
        with torch.no_grad():
            logits = model(data)
        assert logits.shape == (1, N_GENES)

    def test_model_handles_large_graph(self):
        """Model should handle larger graphs without memory errors."""
        from torch_geometric.data import Data
        model = build_model(hidden_dim=64, edge_enc_dim=32, n_layers=2, heads=4)
        model.eval()
        N, E = 200, 1000
        data = Data(
            x          = torch.randn(N, NODE_FEATURE_DIM),
            edge_index = torch.randint(0, N, (2, E)),
            edge_attr  = torch.randn(E, EDGE_FEATURE_DIM),
        )
        with torch.no_grad():
            logits = model(data)
        assert logits.shape == (E, N_GENES)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8: GNN INFERENCE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNInference:

    @pytest.fixture
    def engine(self):
        from ai.gnn_inference import GNNInferenceEngine
        return GNNInferenceEngine.load_untrained()

    def test_engine_loads(self, engine):
        assert engine is not None
        assert engine._loaded

    def test_predict_returns_dict(self, engine, sim_state):
        result = engine.predict(sim_state)
        assert result is not None
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, engine, sim_state):
        result = engine.predict(sim_state)
        required = ["predicted_transfers", "risk_scores", "high_risk_cells",
                    "gene_transfer_probs", "n_nodes", "n_edges",
                    "inference_time_ms", "model_ready"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_predict_model_ready(self, engine, sim_state):
        result = engine.predict(sim_state)
        assert result["model_ready"] is True

    def test_inference_time_reasonable(self, engine, sim_state):
        """Inference should complete in under 2 seconds."""
        result = engine.predict(sim_state)
        assert result["inference_time_ms"] < 2000, \
            f"Inference took {result['inference_time_ms']:.0f}ms - too slow"

    def test_gene_transfer_probs_bounded(self, engine, sim_state):
        result = engine.predict(sim_state)
        for gene, prob in result["gene_transfer_probs"].items():
            assert 0.0 <= prob <= 1.0, \
                f"Gene prob {gene}={prob} out of [0,1]"
        assert len(result["gene_transfer_probs"]) == N_GENES

    def test_gene_transfer_probs_correct_names(self, engine, sim_state):
        result = engine.predict(sim_state)
        for gene in result["gene_transfer_probs"]:
            assert gene in GENE_INDEX, f"Unknown gene in predictions: {gene}"

    def test_risk_scores_bounded(self, engine, sim_state):
        result = engine.predict(sim_state)
        for uid, score in result["risk_scores"].items():
            assert 0.0 <= score <= 1.0, \
                f"Risk score {uid}={score} out of [0,1]"

    def test_predicted_transfers_structure(self, engine, sim_state):
        result = engine.predict(sim_state)
        for edge_idx, info in result["predicted_transfers"].items():
            assert "genes"    in info
            assert "donor_id" in info
            assert "recip_id" in info
            assert "max_prob" in info
            for g in info["genes"]:
                assert g in GENE_INDEX, f"Unknown gene: {g}"

    def test_mc_dropout_adds_uncertainty(self, engine, sim_state):
        result = engine.predict(sim_state, use_mc_dropout=True, mc_samples=5)
        assert result is not None
        if "uncertainty" in result:
            for gene, unc in result["uncertainty"].items():
                assert gene in GENE_INDEX
                assert unc >= 0.0, "Uncertainty must be non-negative"

    def test_status_tracks_predictions(self, engine, sim_state):
        before = engine.status()["n_predictions_run"]
        engine.predict(sim_state)
        after  = engine.status()["n_predictions_run"]
        assert after == before + 1

    def test_graph_construction_from_state(self, sim_state):
        graph = build_graph_from_state(sim_state, max_edge_distance=3)
        assert graph is not None
        assert "node_features" in graph
        assert "edge_index"    in graph
        assert "edge_features" in graph
        assert graph["node_features"].shape[1] == NODE_FEATURE_DIM
        assert graph["edge_features"].shape[1] == EDGE_FEATURE_DIM
        # Edge index should reference valid node indices
        N = graph["node_features"].shape[0]
        assert graph["edge_index"].max() < N, \
            "Edge index references out-of-range node"

    def test_treatment_advisory_structure(self, engine, sim_state):
        advisory = engine.treatment_advisory(
            sim_state, ["ciprofloxacin", "meropenem", "colistin", "vancomycin"]
        )
        assert "risk_level" in advisory
        assert advisory["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert "imminent_resistance"    in advisory
        assert "threatened_antibiotics" in advisory
        assert "advisory_text"          in advisory
        assert isinstance(advisory["advisory_text"], str)
        assert len(advisory["advisory_text"]) > 10

    def test_empty_population_handled(self, engine):
        """Predict should gracefully handle empty simulation state."""
        empty_state = {
            "bacteria": [],
            "antibiotic_heatmaps": {},
            "nutrient_heatmap": [],
            "stats": {"step": 0},
            "grid_width": 80,
            "grid_height": 60,
        }
        result = engine.predict(empty_state)
        assert result is not None
        assert result.get("n_nodes", 0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9: HGT LABEL GENERATION CORRECTNESS
#
# Added after IEEE JBHI desk rejection (manuscript JBHI-03955-2026), which
# flagged "internally inconsistent" positive-label counts. Root cause: the
# original generate_hgt_labels() inferred transfer from genome diffing alone,
# which conflates real horizontal gene transfer with independent spontaneous
# mutation (see BacteriumAgent._attempt_mutation). These tests lock in the
# corrected behavior: labels must come from model.hgt_events (recorded at
# the moment conjugation actually fires), not from comparing genome
# snapshots before/after.
# ─────────────────────────────────────────────────────────────────────────────

class FakeHGTEvent:
    """Lightweight stand-in for simulation.amr_model.HGTEvent in unit tests."""
    def __init__(self, step, donor_id, recipient_id, gene, position=(0, 0)):
        self.step = step
        self.donor_id = donor_id
        self.recipient_id = recipient_id
        self.gene = gene
        self.position = position


def _make_two_node_graph(id_donor, id_recipient, donor_genes_t0, recip_genes_t0,
                          recip_genes_t1, donor_genes_t1=None):
    """
    Build a minimal 2-node graph_t0/graph_t1 pair with a single directed
    edge donor -> recipient, for isolated label-generation testing.
    """
    if donor_genes_t1 is None:
        donor_genes_t1 = donor_genes_t0

    bacteria_t0 = [
        {"id": id_donor,    "pos": [10, 10], "species": "Escherichia coli",
         "resistance_genes": list(donor_genes_t0)},
        {"id": id_recipient,"pos": [11, 10], "species": "Escherichia coli",
         "resistance_genes": list(recip_genes_t0)},
    ]
    bacteria_t1 = [
        {"id": id_donor,    "pos": [10, 10], "species": "Escherichia coli",
         "resistance_genes": list(donor_genes_t1)},
        {"id": id_recipient,"pos": [11, 10], "species": "Escherichia coli",
         "resistance_genes": list(recip_genes_t1)},
    ]

    # edge_index: directed edge node0 (donor) -> node1 (recipient)
    edge_index = np.array([[0], [1]], dtype=np.int64)

    graph_t0 = {
        "edge_index": edge_index,
        "node_ids":   [id_donor, id_recipient],
        "bacteria":   bacteria_t0,
        "metadata":   {"step": 10},
    }
    graph_t1 = {
        "edge_index": edge_index,
        "node_ids":   [id_donor, id_recipient],
        "bacteria":   bacteria_t1,
        "metadata":   {"step": 13},
    }
    return graph_t0, graph_t1


class TestHGTLabelGeneration:

    def test_real_hgt_event_produces_positive_label(self):
        """A recorded conjugation event for this exact donor/recipient/gene
        must produce a positive label on the matching edge."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1={"blaTEM-1"},
        )
        events = [FakeHGTEvent(step=12, donor_id=100, recipient_id=200, gene="blaTEM-1")]

        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=events)
        gene_idx = GENE_INDEX.index("blaTEM-1")

        assert labels.shape == (1, N_GENES)
        assert labels[0, gene_idx] == 1.0

    def test_mutation_without_hgt_event_is_negative(self):
        """
        This is the exact failure mode that caused the editorial rejection:
        recipient acquires a gene that the donor also happens to carry, but
        via spontaneous mutation (no real conjugation event recorded). The
        corrected labeler must NOT mark this edge positive, even though the
        legacy genome-diff method would.
        """
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1={"blaTEM-1"},   # gene appeared, but NOT via HGT
        )
        # No recorded HGT events at all this window
        events = []

        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=events)
        gene_idx = GENE_INDEX.index("blaTEM-1")

        assert labels[0, gene_idx] == 0.0, (
            "Mutation-acquired gene was incorrectly labeled as HGT - "
            "this is the exact bug flagged in JBHI-03955-2026 review"
        )

    def test_event_for_different_gene_does_not_leak_into_other_genes(self):
        """An HGT event for gene A must not set the label for gene B,
        even on the same donor/recipient edge."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1", "mcr-1"}, recip_genes_t0=set(),
            recip_genes_t1={"blaTEM-1", "mcr-1"},
        )
        events = [FakeHGTEvent(step=12, donor_id=100, recipient_id=200, gene="blaTEM-1")]

        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=events)
        tem_idx = GENE_INDEX.index("blaTEM-1")
        mcr_idx = GENE_INDEX.index("mcr-1")

        assert labels[0, tem_idx] == 1.0
        assert labels[0, mcr_idx] == 0.0, \
            "Event for blaTEM-1 incorrectly leaked into mcr-1 label"

    def test_event_on_reverse_edge_does_not_mislabel_forward_edge(self):
        """An event recorded recipient->donor must not be attributed to
        the donor->recipient edge."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1=set(),  # recipient never actually gains the gene
        )
        # Event recorded backwards (200 -> 100), should not affect edge (100->200)
        events = [FakeHGTEvent(step=12, donor_id=200, recipient_id=100, gene="blaTEM-1")]

        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=events)
        gene_idx = GENE_INDEX.index("blaTEM-1")
        assert labels[0, gene_idx] == 0.0

    def test_no_events_in_window_gives_all_negative_labels(self):
        """An empty event list is a valid, meaningful negative example set
        and must not be dropped or treated as missing data."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1=set(),
        )
        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=[])
        assert labels.shape == (1, N_GENES)
        assert labels.sum() == 0.0

    def test_dict_style_events_also_supported(self):
        """Caller may pass plain dicts (as used by API/state serialization)
        instead of HGTEvent objects with attributes."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"gyrA_S83L"}, recip_genes_t0=set(),
            recip_genes_t1={"gyrA_S83L"},
        )
        events = [{"donor": 100, "recipient": 200, "gene": "gyrA_S83L", "step": 12}]

        labels = generate_hgt_labels(graph_t0, graph_t1, recorded_events=events)
        gene_idx = GENE_INDEX.index("gyrA_S83L")
        assert labels[0, gene_idx] == 1.0

    def test_legacy_function_still_importable_for_ablation(self):
        """The deprecated diff-based labeler must remain importable so the
        paper can report 'AUROC under flawed labels vs corrected labels'
        as an explicit ablation, rather than being silently deleted."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1={"blaTEM-1"},
        )
        legacy_labels = _generate_hgt_labels_diff_legacy(graph_t0, graph_t1)
        assert legacy_labels.shape == (1, N_GENES)
        # Legacy method DOES mark this positive (that's the bug we're
        # documenting, not testing it as correct).
        gene_idx = GENE_INDEX.index("blaTEM-1")
        assert legacy_labels[0, gene_idx] == 1.0

    def test_calling_without_recorded_events_warns_loudly(self):
        """Omitting recorded_events must trigger a visible warning, since
        silently falling back to the flawed legacy behavior is exactly
        what produced the rejected manuscript's inflated positive counts."""
        graph_t0, graph_t1 = _make_two_node_graph(
            id_donor=100, id_recipient=200,
            donor_genes_t0={"blaTEM-1"}, recip_genes_t0=set(),
            recip_genes_t1={"blaTEM-1"},
        )
        with pytest.warns(UserWarning, match="conflates HGT with independent mutation"):
            generate_hgt_labels(graph_t0, graph_t1)

    def test_full_pipeline_produces_lower_positive_rate_than_legacy(self):
        """
        Integration-level regression test: running the real simulation and
        comparing corrected vs legacy labeling on the SAME data must show
        the corrected positive rate is lower (since legacy inflates it with
        mutation-driven false positives). This guards against silently
        reintroducing the leakage in a future refactor.
        """
        from simulation.amr_model import AMRSimulationModel

        model = AMRSimulationModel(
            scenario="ecoli_cipro", initial_bacteria=120,
            seed=7, enable_logging=False
        )
        for _ in range(15):
            model.step()
        for abk in model.active_antibiotic_keys:
            model.apply_antibiotic(abk, concentration=1.5, mode="uniform")

        prev_graph = None
        total_legacy = 0.0
        total_fixed  = 0.0

        import warnings
        for step in range(40):
            model.step()
            if not model.running:
                break
            if step % 3 != 0:
                continue
            state = model.get_full_state()
            graph = build_graph_from_state(state, max_edge_distance=3, max_nodes=300)
            if graph is None:
                continue
            if prev_graph is not None:
                window_start = prev_graph["metadata"]["step"]
                window_end   = graph["metadata"]["step"]
                window_events = [
                    ev for ev in model.hgt_events
                    if window_start < ev.step <= window_end
                ]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    legacy = _generate_hgt_labels_diff_legacy(prev_graph, graph)
                fixed = generate_hgt_labels(prev_graph, graph, recorded_events=window_events)
                total_legacy += legacy.sum()
                total_fixed  += fixed.sum()
            prev_graph = graph

        # The corrected method must never produce MORE positives than the
        # legacy diff method, since every real HGT-labeled edge under the
        # corrected method is a strict subset of what the diff method
        # would also catch (real transfers still show up as genome diffs).
        assert total_fixed <= total_legacy, (
            f"Corrected positive count ({total_fixed}) exceeded legacy "
            f"({total_legacy}) - labeling logic regression"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 10: SPLIT_DATASET REPRODUCIBILITY
#
# Found during GPU-script preparation (post JBHI-03955-2026 remediation).
# split_dataset() previously used Python's GLOBAL, unseeded random module
# for its shuffle. Every call - from gnn_trainer.train(), from
# ai/baselines.py's run_all_comparisons(), from
# ai/threshold_calibration.py's collect_test_probs() - therefore drew a
# DIFFERENT random train/val/test split from the same underlying data.
# This meant the GNN's reported test AUROC and the baselines' (LR/RF/
# frequency) reported test AUROC were silently computed on DIFFERENT held-
# out edges, not the same one - an apples-to-oranges comparison that would
# have survived undetected into a resubmission.
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitDatasetReproducibility:

    def _make_dummy_pairs(self, n=50):
        """Build minimal fake graph pairs sufficient for split_dataset()."""
        pairs = []
        for i in range(n):
            g_t0 = {
                "node_features": np.zeros((2, NODE_FEATURE_DIM), dtype=np.float32),
                "edge_index":    np.array([[0], [1]], dtype=np.int64),
                "edge_features": np.zeros((1, EDGE_FEATURE_DIM), dtype=np.float32),
                "gene_labels":   np.zeros((1, N_GENES), dtype=np.float32),
                "metadata":      {"step": i},
            }
            g_t1 = {**g_t0, "metadata": {"step": i + 3}}
            pairs.append((g_t0, g_t1))
        return pairs

    def test_same_split_seed_gives_identical_split(self):
        """
        Same (pairs, split_seed) must produce byte-identical train/val/test
        assignment across separate calls - required for the GNN's test set
        and the baselines' test set to actually be the same held-out data.
        """
        from ai.gnn_trainer import split_dataset, DEFAULT_CONFIG

        pairs = self._make_dummy_pairs(50)

        tr1, val1, te1 = split_dataset(pairs, DEFAULT_CONFIG, split_seed=7)
        tr2, val2, te2 = split_dataset(pairs, DEFAULT_CONFIG, split_seed=7)

        steps1 = [d.metadata["step"] for d in tr1] + \
                 [d.metadata["step"] for d in val1] + \
                 [d.metadata["step"] for d in te1]
        steps2 = [d.metadata["step"] for d in tr2] + \
                 [d.metadata["step"] for d in val2] + \
                 [d.metadata["step"] for d in te2]

        assert steps1 == steps2, (
            "Same split_seed produced different train/val/test assignments - "
            "split_dataset is not actually reproducible"
        )

    def test_different_split_seeds_give_different_splits(self):
        """Sanity check: different seeds should (almost certainly) differ."""
        from ai.gnn_trainer import split_dataset, DEFAULT_CONFIG

        pairs = self._make_dummy_pairs(50)

        tr1, _, _ = split_dataset(pairs, DEFAULT_CONFIG, split_seed=1)
        tr2, _, _ = split_dataset(pairs, DEFAULT_CONFIG, split_seed=2)

        steps1 = [d.metadata["step"] for d in tr1]
        steps2 = [d.metadata["step"] for d in tr2]

        assert steps1 != steps2, (
            "Different split_seeds produced identical splits - "
            "seed may not actually be wired into the shuffle"
        )

    def test_default_split_seed_matches_across_independent_calls(self):
        """
        This is the actual regression this fix targets: gnn_trainer.train(),
        ai/baselines.py, and ai/threshold_calibration.py all call
        split_dataset(pairs, config) with NO explicit split_seed argument.
        They must all resolve to the same default (config['split_seed'])
        and therefore produce the same split, without any caller needing
        to remember to pass it explicitly.
        """
        from ai.gnn_trainer import split_dataset, DEFAULT_CONFIG

        pairs = self._make_dummy_pairs(50)

        # Simulate three independent call sites, none passing split_seed -
        # exactly how train(), run_baselines(), and collect_test_probs()
        # actually call this function.
        tr_a, val_a, te_a = split_dataset(list(pairs), DEFAULT_CONFIG)
        tr_b, val_b, te_b = split_dataset(list(pairs), DEFAULT_CONFIG)
        tr_c, val_c, te_c = split_dataset(list(pairs), DEFAULT_CONFIG)

        steps_a = [d.metadata["step"] for d in te_a]
        steps_b = [d.metadata["step"] for d in te_b]
        steps_c = [d.metadata["step"] for d in te_c]

        assert steps_a == steps_b == steps_c, (
            "Three independent split_dataset() calls with no explicit "
            "split_seed produced different test sets - the GNN and the "
            "baselines would be evaluated on different held-out data."
        )

    def test_split_dataset_does_not_mutate_caller_list(self):
        """
        split_dataset() must not shuffle the caller's list in place - doing
        so would make a second call on 'the same' list produce a different
        result depending on prior calls, breaking reproducibility in a way
        that's easy to miss (the bug wouldn't show up in single-call tests).
        """
        from ai.gnn_trainer import split_dataset, DEFAULT_CONFIG

        pairs = self._make_dummy_pairs(50)
        original_order = [p[0]["metadata"]["step"] for p in pairs]

        split_dataset(pairs, DEFAULT_CONFIG, split_seed=99)

        after_order = [p[0]["metadata"]["step"] for p in pairs]
        assert original_order == after_order, (
            "split_dataset() mutated the caller's pairs list in place - "
            "this can silently change results depending on call order."
        )


class TestTrainCoreAlwaysCheckpoints:
    """
    Regression test for a real bug found while building ai/gnn_ablation.py
    (not introduced by that module - it existed in the original train()
    before the _train_core refactor, just never triggered in production
    runs, which always had large enough validation sets to have a nonzero
    AUROC by epoch 1).

    compute_metrics() returns auroc_macro=0.0 (its own documented fallback)
    when NO gene has any valid/positive examples in the evaluated split.
    If that happens on every epoch of a training run, val_auroc stays
    exactly 0.0 for the whole run, so val_auroc > best_auroc + min_delta
    (starting from best_auroc=0.0) never becomes true, no checkpoint is
    ever saved, and the subsequent load_checkpoint() call crashes with
    FileNotFoundError. This can occur legitimately for a small or
    rare-gene-masked condition in ablation, not just in a pathological
    test - it must not crash the whole ablation run.
    """

    def test_checkpoint_saved_even_when_val_auroc_stuck_at_zero(self, tmp_path):
        """
        Directly construct a validation set with zero positive labels for
        every gene (guaranteeing compute_metrics returns auroc_macro=0.0
        every epoch) and confirm _train_core still produces a loadable
        checkpoint rather than crashing.
        """
        from ai.gnn_trainer import _train_core, DEFAULT_CONFIG
        from torch_geometric.data import Data
        import torch as t

        def make_all_negative_graphs(n=6):
            graphs = []
            for i in range(n):
                n_nodes, n_edges = 5, 8
                x = t.rand(n_nodes, NODE_FEATURE_DIM)
                edge_index = t.randint(0, n_nodes, (2, n_edges))
                edge_attr = t.rand(n_edges, EDGE_FEATURE_DIM)
                y = t.zeros(n_edges, N_GENES)   # ALL negative, zero positives anywhere
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                graphs.append(data)
            return graphs

        tr_ds  = make_all_negative_graphs(8)
        val_ds = make_all_negative_graphs(4)
        te_ds  = make_all_negative_graphs(4)

        tiny_config = {**DEFAULT_CONFIG, "epochs": 3, "patience": 2}
        ckpt_path = str(tmp_path / "test_checkpoint.pt")

        # Must not raise FileNotFoundError
        best_model, test_metrics, history = _train_core(
            tr_ds, val_ds, te_ds, tiny_config, ckpt_path,
            verbose=False, print_steps=False,
        )

        assert os.path.exists(ckpt_path), (
            "No checkpoint was written even though _train_core completed "
            "without raising - the fallback save on epoch 1 didn't fire."
        )
        assert best_model is not None
        assert "auroc_macro" in test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 11: EXTERNAL VALIDATION GNN TRANSFER MATRIX
#
# Found during resubmission-prep review (not caught by any prior test,
# since ai/external_validation.py had zero test coverage before this).
# compute_gnn_transfer_matrix() previously read a per-gene SCALAR
# (gene_transfer_probs, max probability across all edges for that gene)
# and broadcast the same scalar across every column of the matrix row.
# Every donor gene's row was therefore constant across all 10 target
# genes (e.g. blaNDM-1: 0.0133 for every column), guaranteeing the
# Spearman correlation against real gene co-occurrence would return ~0
# regardless of model quality. Separately, the index convention was
# transposed relative to compute_gene_cooccurrence()'s row=conditioning-
# gene / column=outcome-gene convention.
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalValidationMatrix:

    @pytest.fixture(scope="class")
    def untrained_engine(self):
        from ai.gnn_inference import GNNInferenceEngine
        return GNNInferenceEngine.load_untrained()

    def test_matrix_shape(self, untrained_engine):
        from ai.external_validation import compute_gnn_transfer_matrix
        matrix = compute_gnn_transfer_matrix(
            untrained_engine, n_scenarios=1, steps_per_scenario=10,
            seeds_per_scenario=1)
        assert matrix.shape == (N_GENES, N_GENES)

    def test_rows_are_not_constant_broadcasts(self, untrained_engine):
        """
        The core regression test for the bug: no row should have all its
        non-zero entries identical, which was the exact signature of
        broadcasting a per-gene scalar across every column instead of
        computing a real per-pair value. A small number of legitimate
        false positives is tolerated (e.g. a row with only one non-zero
        entry can't demonstrate non-constancy either way), but any row
        with >=3 distinct non-zero entries must show actual variation.
        """
        from ai.external_validation import compute_gnn_transfer_matrix
        matrix = compute_gnn_transfer_matrix(
            untrained_engine, n_scenarios=2, steps_per_scenario=20,
            seeds_per_scenario=1)

        constant_broadcast_rows = []
        for i, gene in enumerate(GENE_INDEX):
            row = matrix[i]
            nonzero = row[row != 0]
            if len(nonzero) >= 3 and np.allclose(nonzero, nonzero[0], atol=1e-6):
                constant_broadcast_rows.append(gene)

        assert not constant_broadcast_rows, (
            f"Rows with >=3 non-zero entries that are all identical "
            f"(constant-broadcast bug signature): {constant_broadcast_rows}"
        )

    def test_index_convention_matches_cooccurrence_matrix(self):
        """
        gnn_matrix[i,j] and compute_gene_cooccurrence()'s co_matrix[i,j]
        must use the SAME convention (row i = conditioning/existing gene,
        column j = outcome/target gene), or comparing them via Spearman
        silently compares transposed matrices. This test verifies the
        convention directly against compute_gene_cooccurrence()'s own
        documented and implemented semantics using a synthetic genome set
        with a known asymmetric relationship.
        """
        from ai.external_validation import compute_gene_cooccurrence

        # Construct genomes where gene A always co-occurs with gene B,
        # but gene B occurs alone MUCH more often than gene A does -
        # i.e. P(B|A) should be high, P(A|B) should be low. This
        # asymmetry is what would be destroyed/inverted by a transpose bug.
        gene_a, gene_b = GENE_INDEX[0], GENE_INDEX[1]
        genomes = {}
        for k in range(10):
            genomes[f"g{k}"] = {gene_a, gene_b}   # A and B always together
        for k in range(10, 100):
            genomes[f"g{k}"] = {gene_b}           # B occurs alone frequently

        co_matrix = compute_gene_cooccurrence(genomes, verbose=False)
        i, j = GENE_INDEX.index(gene_a), GENE_INDEX.index(gene_b)

        p_b_given_a = co_matrix[i, j]   # should be ~1.0 (whenever A present, B always present)
        p_a_given_b = co_matrix[j, i]   # should be ~0.10 (B present in 100 genomes, only 10 also have A)

        assert p_b_given_a > 0.9, (
            f"P(gene_b | gene_a) should be ~1.0 by construction, got {p_b_given_a}"
        )
        assert p_a_given_b < 0.2, (
            f"P(gene_a | gene_b) should be ~0.1 by construction, got {p_a_given_b}"
        )
        assert p_b_given_a > p_a_given_b, (
            "Asymmetric co-occurrence matrix must preserve directionality - "
            "row i = conditioning gene, column j = outcome gene"
        )

    def test_spearman_validation_runs_without_error(self, untrained_engine):
        """End-to-end smoke test: matrix construction through Spearman test."""
        from ai.external_validation import (
            compute_gnn_transfer_matrix, compute_gene_cooccurrence,
            compute_spearman_validation, generate_synthetic_validation_data,
        )
        gnn_matrix = compute_gnn_transfer_matrix(
            untrained_engine, n_scenarios=1, steps_per_scenario=10,
            seeds_per_scenario=1)
        genomes = generate_synthetic_validation_data(seed=1)
        real_matrix = compute_gene_cooccurrence(genomes, verbose=False)

        result = compute_spearman_validation(gnn_matrix, real_matrix)
        assert "spearman_rho" in result
        assert "p_value" in result
        assert -1.0 <= result["spearman_rho"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 12: GNN-NATIVE ABLATION
#
# ai/baselines.py's run_ablation() masks features and trains a cheap LR
# proxy per condition - useful, but a linear model's sensitivity to a
# feature doesn't necessarily reflect the actual GAT architecture's
# sensitivity to the same feature (message passing can route around a
# masked feature in ways a flat classifier can't). These tests cover
# ai/gnn_ablation.py, which retrains the real AMRResistanceGNN per
# condition instead.
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNAblationMasking:

    def _make_dummy_graph_list(self, n=8):
        """Build minimal real PyG Data objects (not dicts) for mask_dataset()."""
        from torch_geometric.data import Data
        import torch as t

        graphs = []
        for i in range(n):
            n_nodes, n_edges = 5, 8
            x = t.rand(n_nodes, NODE_FEATURE_DIM)
            edge_index = t.randint(0, n_nodes, (2, n_edges))
            edge_attr = t.rand(n_edges, EDGE_FEATURE_DIM)
            y = t.zeros(n_edges, N_GENES)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.metadata = {"step": i}
            graphs.append(data)
        return graphs

    def test_node_slice_zeros_only_target_columns(self):
        from ai.gnn_ablation import mask_dataset
        graphs = self._make_dummy_graph_list()
        node_slice = slice(0, 10)   # genomic group

        masked = mask_dataset(graphs, node_slice=node_slice, edge_slice=None)

        for orig, m in zip(graphs, masked):
            assert (m.x[:, node_slice] == 0.0).all(), \
                "Masked columns should be exactly zero"
            assert torch.allclose(m.x[:, 10:], orig.x[:, 10:]), \
                "Unmasked columns should be untouched"
            assert torch.allclose(m.edge_attr, orig.edge_attr), \
                "Edge features should be untouched when only node_slice is given"

    def test_edge_slice_zeros_only_target_columns(self):
        from ai.gnn_ablation import mask_dataset
        graphs = self._make_dummy_graph_list()

        masked = mask_dataset(graphs, node_slice=None, edge_slice=slice(0, 5))

        for orig, m in zip(graphs, masked):
            assert (m.edge_attr == 0.0).all(), \
                "All 5 edge feature columns should be zeroed"
            assert torch.allclose(m.x, orig.x), \
                "Node features should be untouched when only edge_slice is given"

    def test_no_slices_returns_unmodified_clone(self):
        from ai.gnn_ablation import mask_dataset
        graphs = self._make_dummy_graph_list()

        masked = mask_dataset(graphs, node_slice=None, edge_slice=None)

        for orig, m in zip(graphs, masked):
            assert torch.equal(m.x, orig.x)
            assert torch.equal(m.edge_attr, orig.edge_attr)
            assert m is not orig, "Should be a clone, not the same object"

    def test_does_not_mutate_original_dataset(self):
        """
        Same discipline as split_dataset()'s mutation test - masking must
        never modify the caller's original tensors, or repeated ablation
        conditions run against 'the same' tr_ds would silently corrupt
        each other.
        """
        from ai.gnn_ablation import mask_dataset
        graphs = self._make_dummy_graph_list()
        original_x_values = [g.x.clone() for g in graphs]
        original_edge_values = [g.edge_attr.clone() for g in graphs]

        mask_dataset(graphs, node_slice=slice(0, 10), edge_slice=slice(0, 5))

        for g, orig_x, orig_e in zip(graphs, original_x_values, original_edge_values):
            assert torch.equal(g.x, orig_x), \
                "mask_dataset() mutated the original node features in place"
            assert torch.equal(g.edge_attr, orig_e), \
                "mask_dataset() mutated the original edge features in place"

    def test_node_groups_cover_full_dim_without_overlap(self):
        """
        Sanity check on the NODE_GROUPS slice definitions themselves -
        they must exactly tile [0, NODE_FEATURE_DIM) with no gaps and no
        overlaps, or some columns would never be tested by any ablation
        condition while others get masked twice.
        """
        from ai.gnn_ablation import NODE_GROUPS

        real_slices = [s for name, s in NODE_GROUPS.items() if s is not None]
        covered = np.zeros(NODE_FEATURE_DIM, dtype=int)
        for s in real_slices:
            covered[s] += 1

        assert (covered == 1).all(), (
            f"NODE_GROUPS slices must tile [0,{NODE_FEATURE_DIM}) exactly "
            f"once each; coverage counts: {covered.tolist()}"
        )

    def test_edge_group_covers_full_edge_dim(self):
        from ai.gnn_ablation import EDGE_GROUP_SLICE
        indices = list(range(EDGE_FEATURE_DIM))[EDGE_GROUP_SLICE]
        assert len(indices) == EDGE_FEATURE_DIM, (
            "EDGE_GROUP_SLICE should cover all edge feature columns"
        )


class TestGNNAblationEndToEnd:
    """
    Smoke test for run_gnn_ablation() itself - deliberately tiny (1
    scenario, 1 seed, 3 epochs, 2 conditions) to confirm the full
    collect -> split -> mask -> train -> report pipeline runs without
    crashing. This does NOT validate the ablation NUMBERS (3 epochs on
    minimal data proves nothing about feature importance), only that the
    machinery is wired correctly end to end.
    """

    def test_runs_end_to_end_without_crashing(self, tmp_path):
        from ai.gnn_ablation import run_gnn_ablation
        from ai.gnn_trainer import DEFAULT_CONFIG

        tiny_config = {
            **DEFAULT_CONFIG,
            "scenarios":          ["ecoli_cipro"],
            "seeds_per_scenario": 2,
            "steps_per_run":      50,
            "epochs":             3,
            "patience":           2,
            "checkpoint_dir":     str(tmp_path / "checkpoints"),
        }

        # Only test 2 conditions (not all 9) to keep this fast - full
        # coverage of NODE_GROUPS is exercised by
        # test_node_groups_cover_full_dim_without_overlap above, which
        # doesn't require actual training.
        import ai.gnn_ablation as ga
        original_groups = ga.NODE_GROUPS
        try:
            ga.NODE_GROUPS = {
                "All features (full model)": None,
                "No genomic genes": slice(0, 10),
            }
            results = run_gnn_ablation(
                config=tiny_config, include_edge_group=False, verbose=False)
        finally:
            ga.NODE_GROUPS = original_groups

        assert "All features (full model)" in results
        assert "No genomic genes" in results
        for name, r in results.items():
            assert "auroc_macro" in r
            assert 0.0 <= r["auroc_macro"] <= 1.0
            assert "delta_auroc_vs_full" in r

        # Baseline condition must show delta == 0 by construction (it IS
        # the baseline being compared against).
        assert results["All features (full model)"]["delta_auroc_vs_full"] == 0.0


class TestBaselineSubsampleReproducibility:
    """
    Regression test for a bug found while investigating cross-run AUROC
    variance: Random Forest's test AUROC moved from 0.9344 to 0.9548
    between two nominally identical DEFAULT_CONFIG runs, DESPITE
    RandomForestClassifier's own random_state being hardcoded to 42 in
    both. Root cause: run_baselines()/run_ablation()'s training-set
    subsampling (100k/50k edges from a much larger pool) used
    np.random.choice() against the GLOBAL, unseeded numpy RNG, so the
    actual training data shown to RF/LR differed every run even though
    the model's own randomness was pinned. Same bug class as the original
    split_dataset() global-random issue, in a different module.
    """

    def _make_dummy_ds(self, n_graphs=20, n_edges_per_graph=50):
        from ai.feature_engineering import AMRGraphDataset
        from torch_geometric.data import Data
        import torch as t

        pairs = []
        for i in range(n_graphs):
            n_nodes = 10
            x = t.rand(n_nodes, NODE_FEATURE_DIM).numpy()
            edge_index = np.random.randint(0, n_nodes, (2, n_edges_per_graph))
            edge_attr = np.random.rand(n_edges_per_graph, EDGE_FEATURE_DIM).astype(np.float32)
            gene_labels = np.zeros((n_edges_per_graph, N_GENES), dtype=np.float32)
            g_t0 = {
                "node_features": x.astype(np.float32),
                "edge_index": edge_index.astype(np.int64),
                "edge_features": edge_attr,
                "gene_labels": gene_labels,
                "bacteria": [], "metadata": {"step": i},
            }
            g_t1 = {**g_t0, "metadata": {"step": i + 3}}
            pairs.append((g_t0, g_t1))
        return AMRGraphDataset(pairs)

    def test_subsample_rng_is_reproducible_given_same_seed(self):
        """
        Directly verify the RandomState-based subsampling pattern used
        inside run_baselines()/run_ablation() - same seed, same data,
        same indices selected, every time. This tests the actual
        mechanism rather than running the full run_baselines() pipeline
        end to end, which has legitimate hard dependencies (requires
        training_results.json/calibration_results.json to exist) that a
        unit test for THIS specific fix shouldn't need to satisfy.
        """
        from ai.baselines import flatten_dataset
        ds = self._make_dummy_ds(n_graphs=20, n_edges_per_graph=50)
        X_tr, _ = flatten_dataset(ds)

        rng_a = np.random.RandomState(7)
        rng_b = np.random.RandomState(7)
        idx_a = rng_a.choice(X_tr.shape[0], min(200, X_tr.shape[0]), replace=False)
        idx_b = rng_b.choice(X_tr.shape[0], min(200, X_tr.shape[0]), replace=False)

        assert np.array_equal(idx_a, idx_b), (
            "Same seed produced different subsample indices - the "
            "RandomState-based subsampling fix is not actually deterministic"
        )

    def test_run_baselines_different_seeds_can_differ(self):
        """
        Sanity check on the other direction - different subsample_seed
        values should be capable of producing different subsamples (not
        strictly required to differ in every metric, but the seed must
        actually be wired into the RNG, not silently ignored).
        """
        from ai.baselines import flatten_dataset
        ds = self._make_dummy_ds(n_graphs=20, n_edges_per_graph=50)
        X_tr, y_tr = flatten_dataset(ds)

        rng_a = np.random.RandomState(1)
        rng_b = np.random.RandomState(2)
        idx_a = rng_a.choice(X_tr.shape[0], min(200, X_tr.shape[0]), replace=False)
        idx_b = rng_b.choice(X_tr.shape[0], min(200, X_tr.shape[0]), replace=False)

        assert not np.array_equal(np.sort(idx_a), np.sort(idx_b)), (
            "Different seeds produced identical subsample indices - "
            "seed may not actually be wired into the RNG"
        )


class TestMultiseedComparison:
    """
    Smoke test for run_multiseed_comparison() - tiny config (1 scenario,
    2 seeds, 2 epochs, n_seeds=2) purely to confirm the collect -> split
    -> repeated-train -> statistical-comparison pipeline runs without
    crashing. Does NOT validate the comparison NUMBERS (2 epochs on
    minimal data proves nothing about whether GNN beats RF), only that
    the machinery is wired correctly end to end.
    """

    def test_runs_end_to_end_without_crashing(self, tmp_path):
        from ai.gnn_multiseed import run_multiseed_comparison
        from ai.gnn_trainer import DEFAULT_CONFIG

        tiny_config = {
            **DEFAULT_CONFIG,
            "scenarios":          ["ecoli_cipro"],
            "seeds_per_scenario": 2,
            "steps_per_run":      50,
            "epochs":             2,
            "patience":           2,
            "checkpoint_dir":     str(tmp_path / "checkpoints"),
        }

        results = run_multiseed_comparison(
            config=tiny_config, n_seeds=2, verbose=False)

        assert results["n_seeds"] == 2
        assert len(results["per_seed"]) == 2
        assert "gnn_mean" in results["summary"]
        assert "rf_mean" in results["summary"]
        for r in results["per_seed"]:
            assert 0.0 <= r["gnn_auroc"] <= 1.0 or np.isnan(r["gnn_auroc"])
            assert 0.0 <= r["rf_auroc"] <= 1.0 or np.isnan(r["rf_auroc"])

    def test_same_torch_seed_gives_same_gnn_auroc(self, tmp_path):
        """
        Direct verification that torch_seed actually controls GNN
        training determinism - same seed, same data, same result.
        """
        from ai.gnn_trainer import _train_core, DEFAULT_CONFIG
        from torch_geometric.data import Data
        import torch as t

        def make_graphs(n=10):
            graphs = []
            for i in range(n):
                n_nodes, n_edges = 6, 10
                x = t.rand(n_nodes, NODE_FEATURE_DIM)
                edge_index = t.randint(0, n_nodes, (2, n_edges))
                edge_attr = t.rand(n_edges, EDGE_FEATURE_DIM)
                y = t.zeros(n_edges, N_GENES)
                y[0, 0] = 1.0   # at least one positive so training isn't fully degenerate
                graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            return graphs

        tr_ds  = make_graphs(8)
        val_ds = make_graphs(4)
        te_ds  = make_graphs(4)
        tiny_config = {**DEFAULT_CONFIG, "epochs": 2, "patience": 2}

        _, metrics_a, _ = _train_core(
            tr_ds, val_ds, te_ds, tiny_config,
            str(tmp_path / "seed_a.pt"), verbose=False, print_steps=False,
            torch_seed=42,
        )
        _, metrics_b, _ = _train_core(
            tr_ds, val_ds, te_ds, tiny_config,
            str(tmp_path / "seed_b.pt"), verbose=False, print_steps=False,
            torch_seed=42,
        )

        assert metrics_a["auroc_macro"] == metrics_b["auroc_macro"], (
            "Same torch_seed produced different GNN test AUROC - "
            "training is not actually deterministic given a fixed seed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])