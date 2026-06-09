"""
GNN Inference Engine — loads a trained model and runs predictions
against a live simulation state, feeding results back to the API.

Provides:
  - Live prediction: given current state, predict next HGT transfers
  - Risk scoring: which bacteria are at highest risk of acquiring resistance
  - Treatment advisory: combine GNN predictions with analytics engine
  - Uncertainty estimates: Monte Carlo dropout for prediction confidence
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import (
    build_graph_from_state, GENE_INDEX, N_GENES, NODE_FEATURE_DIM
)
from ai.gnn_model import AMRResistanceGNN, build_model

DEFAULT_CHECKPOINT = "ai/checkpoints/best_model.pt"
TRANSFER_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class GNNInferenceEngine:
    """
    Wraps a trained AMRResistanceGNN for live inference against
    the running simulation.

    Usage:
        engine = GNNInferenceEngine.load("ai/checkpoints/best_model.pt")
        result = engine.predict(state_dict)
    """

    def __init__(self, model: AMRResistanceGNN, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        self._loaded = True
        self._n_predictions = 0
        self._prediction_times: List[float] = []

    @classmethod
    def load(cls, checkpoint_path: str = DEFAULT_CHECKPOINT,
             device: str = "cpu") -> Optional["GNNInferenceEngine"]:
        """Load from checkpoint. Returns None if checkpoint doesn't exist yet."""
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        try:
            ckpt  = torch.load(str(path), map_location=device)
            model = build_model(**ckpt.get("model_config", {}))
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            engine = cls(model, device)
            engine._checkpoint_path = str(path)
            engine._checkpoint_metrics = ckpt.get("metrics", {})
            return engine
        except Exception as e:
            print(f"[GNNInferenceEngine] Failed to load checkpoint: {e}")
            return None

    @classmethod
    def load_untrained(cls, device: str = "cpu") -> "GNNInferenceEngine":
        """
        Create an untrained model for structural testing.
        Predictions will be random but architecture is correct.
        """
        model = build_model()
        return cls(model, device)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        state: dict,
        threshold:        float = TRANSFER_THRESHOLD,
        max_nodes:        int   = 300,
        max_edge_distance:int   = 3,
        use_mc_dropout:   bool  = False,
        mc_samples:       int   = 10,
    ) -> Optional[dict]:
        """
        Run GNN inference on a simulation state snapshot.

        Args:
          state:             Full simulation state dict (from model.get_full_state())
          threshold:         Probability cutoff for positive prediction
          max_nodes:         Max bacteria to include in graph (for performance)
          max_edge_distance: Edge connection radius in grid cells
          use_mc_dropout:    Use MC Dropout for uncertainty estimation
          mc_samples:        Number of MC samples (if use_mc_dropout=True)

        Returns dict with:
          predicted_transfers: {edge_idx: [gene_names]}  — likely next transfers
          risk_scores:         {bacterium_id: float}     — resistance acquisition risk
          high_risk_cells:     list of bacterium dicts at highest risk
          gene_transfer_probs: {gene_name: float}        — aggregate transfer prob
          n_nodes, n_edges, inference_time_ms
          uncertainty:         per-edge std dev (if MC dropout used)
          model_ready:         bool
        """
        t0 = time.time()

        try:
            from torch_geometric.data import Data
        except ImportError:
            return {"error": "torch_geometric not installed", "model_ready": False}

        graph = build_graph_from_state(
            state, max_edge_distance=max_edge_distance, max_nodes=max_nodes
        )
        if graph is None or graph["node_features"].shape[0] < 2:
            return {
                "predicted_transfers": {},
                "risk_scores": {},
                "high_risk_cells": [],
                "gene_transfer_probs": {g: 0.0 for g in GENE_INDEX},
                "n_nodes": 0, "n_edges": 0,
                "inference_time_ms": 0,
                "model_ready": True,
            }

        # Build PyG Data
        data = Data(
            x          = torch.tensor(graph["node_features"], dtype=torch.float),
            edge_index = torch.tensor(graph["edge_index"],    dtype=torch.long),
            edge_attr  = torch.tensor(graph["edge_features"], dtype=torch.float),
        ).to(self.device)

        # Run inference
        if use_mc_dropout:
            probs, uncertainty = self._mc_dropout_predict(data, mc_samples)
        else:
            with torch.no_grad():
                logits = self.model(data)
                probs  = torch.sigmoid(logits).cpu().numpy()   # (E, N_GENES)
            uncertainty = None

        # Predicted transfers: edges + genes above threshold
        predicted_transfers = {}
        edge_src = graph["edge_index"][0]
        edge_dst = graph["edge_index"][1]
        node_ids = graph["node_ids"]
        bacteria = graph["bacteria"]

        for e_idx in range(probs.shape[0]):
            genes_predicted = [
                GENE_INDEX[g_idx]
                for g_idx in range(N_GENES)
                if probs[e_idx, g_idx] >= threshold
            ]
            if genes_predicted:
                src_id = node_ids[edge_src[e_idx]] if edge_src[e_idx] < len(node_ids) else -1
                dst_id = node_ids[edge_dst[e_idx]] if edge_dst[e_idx] < len(node_ids) else -1
                predicted_transfers[e_idx] = {
                    "genes":     genes_predicted,
                    "donor_id":  int(src_id),
                    "recip_id":  int(dst_id),
                    "max_prob":  float(probs[e_idx].max()),
                }

        # Risk scores: per-node probability of acquiring any new gene
        #   For node j, sum over all incoming edges the max transfer prob
        n_nodes   = len(node_ids)
        risk_raw  = np.zeros(n_nodes, dtype=np.float32)
        for e_idx in range(probs.shape[0]):
            j = edge_dst[e_idx]
            if j < n_nodes:
                risk_raw[j] = max(risk_raw[j], float(probs[e_idx].max()))

        # Normalize risk to [0, 1]
        if risk_raw.max() > 0:
            risk_norm = risk_raw / risk_raw.max()
        else:
            risk_norm = risk_raw

        risk_scores = {
            int(node_ids[i]): float(risk_norm[i])
            for i in range(n_nodes)
        }

        # High-risk cells: top 10 bacteria by risk score
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        high_risk_ids = {uid for uid, _ in sorted_risks[:10]}
        high_risk_cells = [
            {**b, "risk_score": risk_scores.get(b["id"], 0.0)}
            for b in bacteria if b["id"] in high_risk_ids
        ]

        # Aggregate gene transfer probabilities across all edges
        gene_transfer_probs = {
            GENE_INDEX[g]: float(probs[:, g].max()) if probs.shape[0] > 0 else 0.0
            for g in range(N_GENES)
        }

        elapsed_ms = (time.time() - t0) * 1000
        self._n_predictions += 1
        self._prediction_times.append(elapsed_ms)

        result = {
            "predicted_transfers":  predicted_transfers,
            "risk_scores":          risk_scores,
            "high_risk_cells":      high_risk_cells[:5],
            "gene_transfer_probs":  gene_transfer_probs,
            "n_nodes":              n_nodes,
            "n_edges":              probs.shape[0],
            "n_predicted_transfers":len(predicted_transfers),
            "inference_time_ms":    round(elapsed_ms, 2),
            "model_ready":          True,
            "threshold_used":       threshold,
        }

        if uncertainty is not None:
            result["uncertainty"] = {
                GENE_INDEX[g]: float(uncertainty[:, g].mean())
                for g in range(N_GENES)
            }

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # MONTE CARLO DROPOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _mc_dropout_predict(
        self, data, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enable dropout at inference time for uncertainty estimation.
        Returns (mean_probs, std_probs) each of shape (E, N_GENES).

        MC Dropout reference: Gal & Ghahramani 2016
        """
        self.model.train()   # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(data)
                probs  = torch.sigmoid(logits).cpu().numpy()
                samples.append(probs)
        self.model.eval()

        samples_arr = np.stack(samples, axis=0)   # (n_samples, E, N_GENES)
        mean_probs  = samples_arr.mean(axis=0)    # (E, N_GENES)
        std_probs   = samples_arr.std(axis=0)     # (E, N_GENES)
        return mean_probs, std_probs

    # ─────────────────────────────────────────────────────────────────────────
    # TREATMENT ADVISORY (GNN + analytics combined)
    # ─────────────────────────────────────────────────────────────────────────

    def treatment_advisory(
        self,
        state:                dict,
        available_antibiotics: List[str],
    ) -> dict:
        """
        Combine GNN predictions with treatment recommendation engine
        to produce a clinically-framed advisory.

        Returns structured advisory with:
          - GNN-predicted imminent resistance genes
          - Which antibiotics those genes will resist
          - Recommended antibiotics that won't be undermined by predicted transfers
          - Risk level (LOW/MEDIUM/HIGH/CRITICAL)
        """
        from ai.resistance_analytics import recommend_treatment
        from data.card_loader import RESISTANCE_GENES, ANTIBIOTIC_PROFILES

        gnn_result = self.predict(state)
        if not gnn_result or not gnn_result.get("model_ready"):
            return {"error": "Model not ready", "advisory": None}

        # Which genes are about to spread?
        imminent_genes = set()
        gene_probs     = gnn_result.get("gene_transfer_probs", {})
        for gene, prob in gene_probs.items():
            if prob >= 0.3:
                imminent_genes.add(gene)

        # Which antibiotics do these genes resist?
        threatened_antibiotics = set()
        for gene_name in imminent_genes:
            if gene_name in RESISTANCE_GENES:
                gene = RESISTANCE_GENES[gene_name]
                threatened_antibiotics.update(gene.drug_classes)

        # Safe antibiotics: available ones NOT threatened by imminent genes
        safe_antibiotics = [
            ab for ab in available_antibiotics
            if ab in ANTIBIOTIC_PROFILES and
            ANTIBIOTIC_PROFILES[ab].drug_class not in threatened_antibiotics
        ]

        # Risk level based on how many genes are spreading
        n_imminent = len(imminent_genes)
        risk_level = (
            "CRITICAL" if n_imminent >= 4 else
            "HIGH"     if n_imminent >= 2 else
            "MEDIUM"   if n_imminent >= 1 else
            "LOW"
        )

        # Get current bacteria for treatment recommendation
        bacteria = state.get("bacteria", [])
        recs     = recommend_treatment(
            [type("B", (), {"resistance_genes": set(b["resistance_genes"]),
                             "fitness": b["fitness"]})()
             for b in bacteria],
            safe_antibiotics or available_antibiotics
        )

        return {
            "risk_level":            risk_level,
            "imminent_resistance":   list(imminent_genes),
            "threatened_antibiotics":list(threatened_antibiotics),
            "safe_antibiotics":      safe_antibiotics,
            "recommendations":       recs[:3],
            "n_high_risk_cells":     len(gnn_result.get("high_risk_cells", [])),
            "gnn_confidence":        float(np.mean(list(gene_probs.values()))),
            "advisory_text":         self._build_advisory_text(
                risk_level, imminent_genes, threatened_antibiotics, safe_antibiotics
            ),
        }

    def _build_advisory_text(
        self,
        risk_level:    str,
        imminent:      set,
        threatened:    set,
        safe_options:  list,
    ) -> str:
        if risk_level == "LOW":
            return ("No imminent resistance gene transfers predicted. "
                    "Current treatment options remain effective.")
        elif risk_level == "MEDIUM":
            genes_str = ", ".join(imminent) or "unknown"
            return (f"GNN predicts {genes_str} may spread within the next "
                    f"few simulation steps. Monitor resistance emergence.")
        elif risk_level == "HIGH":
            genes_str = ", ".join(imminent)
            ab_str    = ", ".join(threatened) or "none"
            return (f"WARNING: {genes_str} predicted to spread imminently. "
                    f"Drug classes at risk: {ab_str}. "
                    f"Recommend switching to: {', '.join(safe_options[:2]) or 'consult specialist'}.")
        else:  # CRITICAL
            return (f"CRITICAL: {len(imminent)} resistance genes spreading rapidly. "
                    f"Multiple drug classes threatened. Combination therapy required. "
                    f"Consider colistin + carbapenem or seek specialist guidance.")

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS / DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        avg_ms = (
            sum(self._prediction_times) / len(self._prediction_times)
            if self._prediction_times else 0.0
        )
        return {
            "model_ready":       self._loaded,
            "n_predictions_run": self._n_predictions,
            "avg_inference_ms":  round(avg_ms, 2),
            "architecture":      self.model.architecture_summary(),
            "checkpoint":        getattr(self, "_checkpoint_path", "untrained"),
            "val_metrics":       getattr(self, "_checkpoint_metrics", {}),
        }