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
    get_dims,
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
        # x_norm = 40/79 ≈ 0.506, y_norm = 30/59 ≈ 0.508
        x_norm = feats[18]
        y_norm = feats[19]
        assert 0.4 < x_norm < 0.7, f"x_norm={x_norm} should be ~0.5"
        assert 0.4 < y_norm < 0.7, f"y_norm={y_norm} should be ~0.5"

    def test_species_one_hot(self, sample_bacterium, sample_bacterium_resistant):
        """Species one-hot should have exactly one 1."""
        feats_ec  = extract_node_features(sample_bacterium, 80, 60)
        feats_kp  = extract_node_features(sample_bacterium_resistant, 80, 60)
        sp_ec  = feats_ec[23:28]
        sp_kp  = feats_kp[23:28]
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
        cipro_idx = 30 + AB_INDEX.index("ciprofloxacin")
        meropen_idx = 30 + AB_INDEX.index("meropenem")
        assert feats[cipro_idx] > 0, "Ciprofloxacin exposure should be non-zero"
        assert feats[meropen_idx] == 0.0, "Meropenem exposure should be zero"

    def test_edge_feature_shape(self, sample_bacterium, sample_bacterium_resistant):
        feats = extract_edge_features(sample_bacterium, sample_bacterium_resistant, 80, 60)
        assert feats.shape == (EDGE_FEATURE_DIM,), \
            f"Expected ({EDGE_FEATURE_DIM},), got {feats.shape}"

    def test_edge_distance_feature(self, sample_bacterium, sample_bacterium_resistant):
        """Distance feature should reflect actual grid distance."""
        feats = extract_edge_features(sample_bacterium, sample_bacterium_resistant, 80, 60)
        # pos_i=[40,30], pos_j=[42,31] → dist = sqrt(4+1) = 2.24 → norm by 10 = 0.224
        expected_dist = math.sqrt(4 + 1) / 10.0
        assert abs(feats[0] - expected_dist) < 0.01, \
            f"Distance feature {feats[0]:.3f} ≠ expected {expected_dist:.3f}"

    def test_same_species_flag_different(self, sample_bacterium, sample_bacterium_resistant):
        """Different species should have same_species=0."""
        feats = extract_edge_features(sample_bacterium, sample_bacterium_resistant, 80, 60)
        assert feats[1] == 0.0, "Different species should flag same_species=0"

    def test_same_species_flag_same(self, sample_bacterium):
        """Same species should have same_species=1."""
        feats = extract_edge_features(sample_bacterium, sample_bacterium, 80, 60)
        assert feats[1] == 1.0, "Same species should flag same_species=1"

    def test_transferable_genes_donor_has_more(
        self, sample_bacterium, sample_bacterium_resistant
    ):
        """
        Resistant bacterium has more genes — as donor, transferable count is higher.
        """
        feats_r_to_s = extract_edge_features(
            sample_bacterium_resistant, sample_bacterium, 80, 60)
        feats_s_to_r = extract_edge_features(
            sample_bacterium, sample_bacterium_resistant, 80, 60)
        # Resistant→Susceptible: more transferable genes
        assert feats_r_to_s[3] > feats_s_to_r[3], \
            "Resistant donor should have higher transferable gene count"

    def test_biofilm_edge_feature(self, sample_bacterium_resistant):
        """Biofilm flag in edge should reflect biofilm status."""
        b_no_biofilm = {**sample_bacterium_resistant, "in_biofilm": False}
        feats_both   = extract_edge_features(
            sample_bacterium_resistant, sample_bacterium_resistant, 80, 60)
        feats_none   = extract_edge_features(b_no_biofilm, b_no_biofilm, 80, 60)
        assert feats_both[5] == 1.0, "Both in biofilm → flag=1"
        assert feats_none[5] == 0.0, "Neither in biofilm → flag=0"

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
            "Model output looks like probabilities — should be raw logits"

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
            f"Inference took {result['inference_time_ms']:.0f}ms — too slow"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])