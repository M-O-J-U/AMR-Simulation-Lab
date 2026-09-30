"""
GNN Resistance Predictor — Graph Attention Network (GAT).

Architecture:
  Input:  node features (NODE_FEATURE_DIM = 35) + edge features (EDGE_FEATURE_DIM = 5)
  Layers: n_layers GAT conv blocks (constructor default 3; the configuration used
          for the reported results uses 2 — see paper/draft/03_methods.md §3.5)
  Output: per-edge, per-gene transfer probability (E × 10)

Why GAT rather than a non-attentional convolution: attention weights neighbours
rather than averaging them uniformly, and GATConv accepts an edge attribute, so
the encoded edge vector participates in attention. Note that the reported runs
zero the edge features, so this capacity is available but unused there.

What this model does: it predicts, per directed edge and per gene, whether that
gene is transferred along that contact in the next snapshot window, rather than
predicting only whether resistance emerges in the population.

CAUTION on interpreting attention: we have NOT run any analysis of what
individual heads attend to, so no claim is made here about heads specialising to
particular biological factors. (An earlier version of this docstring asserted
that heads learn "distance" and "SOS" — besides being unverified, both of those
features were REMOVED as label leakage; see ai/feature_engineering.py.)

CAUTION on edge features: they encode transfer CORRELATES (shared genes, fitness
difference, both-in-biofilm, stress difference, coarsened proximity), not the
conjugation preconditions. The features that were the actual preconditions were
removed as leakage.

No external validation of this model exists in this repository. Earlier text here
claimed validation "against CARD transfer rates"; there is no such experiment.
See paper/claims_to_numbers.md U7.

References (both verified 2026-09-30):
  - Veličković et al. 2018 — Graph Attention Networks (ICLR 2018, arXiv:1710.10903)
  - Gilmer et al. 2017 — Neural Message Passing for Quantum Chemistry
    (Gilmer, Schoenholz, Riley, Vinyals, Dahl; arXiv:1704.01212)
  (A third reference, "Orenstein et al. 2021 — GNNs for microbial ecology", was
  removed on 2026-09-30: no such paper could be found on PubMed or the open web,
  so it is treated as fabricated. CLAUDE.md rule 7.)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from ai.feature_engineering import (
    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, N_GENES, GENE_INDEX
)

# ─────────────────────────────────────────────────────────────────────────────
# EDGE FEATURE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class EdgeEncoder(nn.Module):
    """
    Encodes the edge feature vector (edge_dim, = EDGE_FEATURE_DIM = 5) into a
    hidden representation of width hidden_dim. The result is passed to each GAT
    layer as GATConv's edge attribute, and is also concatenated with the two
    endpoint representations in the prediction head.
    """
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


# ─────────────────────────────────────────────────────────────────────────────
# NODE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """
    Projects raw 35-dim node features into hidden_dim space.
    Separate sub-encoders for each feature group (genomic, physiological etc.)
    then concatenates and projects. The rationale is that the groups have very
    different scales and semantics; we have NOT compared this against a single
    linear projection, so no claim is made that it performs better.

    NOTE: behavioral group is 2-dim (in_biofilm, is_persister), not 3 —
    sos_active was removed as direct leakage of the HGT label-generating
    rule (see ai/feature_engineering.py module docstring for the full
    audit). Slice boundaries below were updated accordingly; this class
    must be kept in sync with NODE_FEATURE_DIM and the feature order in
    extract_node_features().
    """

    # Feature group slice indices (must match extract_node_features() order)
    GENE_SLICE    = slice(0,  10)
    PHYSIO_SLICE  = slice(10, 15)
    BEHAV_SLICE   = slice(15, 17)   # was slice(15,18) — now 2 dims, not 3
    SPATIAL_SLICE = slice(17, 19)
    POP_SLICE     = slice(19, 22)
    SPECIES_SLICE = slice(22, 27)
    GRAM_SLICE    = slice(27, 29)
    AB_SLICE      = slice(29, 35)

    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        # Separate encoders per feature group
        self.enc_genes   = nn.Linear(10, hidden_dim // 4)
        self.enc_physio  = nn.Linear(5,  hidden_dim // 8)
        self.enc_behav   = nn.Linear(2,  hidden_dim // 8)   # was 3
        self.enc_spatial = nn.Linear(2,  hidden_dim // 8)
        self.enc_pop     = nn.Linear(3,  hidden_dim // 8)
        self.enc_species = nn.Linear(5,  hidden_dim // 8)
        self.enc_ab      = nn.Linear(6,  hidden_dim // 8)
        self.enc_gram    = nn.Linear(2,  hidden_dim // 8)

        # Compute concat dim
        concat_dim = (hidden_dim//4 + hidden_dim//8 * 7)
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            F.gelu(self.enc_genes(   x[:, self.GENE_SLICE])),
            F.gelu(self.enc_physio(  x[:, self.PHYSIO_SLICE])),
            F.gelu(self.enc_behav(   x[:, self.BEHAV_SLICE])),
            F.gelu(self.enc_spatial( x[:, self.SPATIAL_SLICE])),
            F.gelu(self.enc_pop(     x[:, self.POP_SLICE])),
            F.gelu(self.enc_species( x[:, self.SPECIES_SLICE])),
            F.gelu(self.enc_gram(    x[:, self.GRAM_SLICE])),
            F.gelu(self.enc_ab(      x[:, self.AB_SLICE])),
        ]
        return self.proj(torch.cat(parts, dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# GAT LAYER WITH EDGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class GATLayerWithEdge(nn.Module):
    """
    GAT block. Given node states and the already-encoded edge vectors, it:
    1. runs GATConv with the encoded edge vector as its edge attribute
       (edge_dim = edge_enc_dim, which need not equal hidden_dim),
    2. adds a residual connection and normalises,
    3. applies a position-wise feed-forward network,
    4. adds a second residual connection and normalises.

    Note: the edge encoding is consumed by GATConv's own edge_dim pathway. It is
    NOT added to the source node features before attention (an earlier version of
    this docstring said it was; the code never did that).
    """
    def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.gat  = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> torch.Tensor:
        # GAT with residual
        x2 = self.gat(x, edge_index, edge_attr=edge_attr)
        x  = self.norm(x + x2)
        # Feed-forward with residual
        x  = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# EDGE PREDICTOR HEAD
# ─────────────────────────────────────────────────────────────────────────────

class EdgePredictorHead(nn.Module):
    """
    Predicts per-gene transfer probability for each directed edge (i→j).

    Takes:
      - h_i : source node hidden state (donor)
      - h_j : target node hidden state (recipient)
      - e_ij: edge feature encoding

    Outputs: (E, N_GENES) probabilities via sigmoid

    Biological interpretation: for each gene g, P(g transfers from i to j)
    """
    def __init__(self, hidden_dim: int, edge_enc_dim: int, n_genes: int, dropout: float):
        super().__init__()
        in_dim = hidden_dim * 2 + edge_enc_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_genes),
        )

    def forward(self, h_src: torch.Tensor,
                h_dst: torch.Tensor,
                e_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_src, h_dst, e_enc], dim=-1)
        return self.net(x)   # raw logits — apply sigmoid outside for loss


# ─────────────────────────────────────────────────────────────────────────────
# FULL GNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AMRResistanceGNN(nn.Module):
    """
    Full Graph Attention Network for AMR resistance gene transfer prediction.

    Forward pass:
      1. Encode node features (NODE_FEATURE_DIM=35 → hidden_dim)
      2. Encode edge features (EDGE_FEATURE_DIM=5 → edge_enc_dim)
      3. n_layers rounds of GAT message passing (0 = no message passing, the
         graph-free ablation)
      4. For each directed edge (i→j): predict which genes will transfer

    Args:
      hidden_dim  : node hidden dimension (default 128)
      edge_enc_dim: edge encoding dimension (default 64)
      n_layers    : number of GAT layers (default 3)
      heads       : attention heads per layer (default 4)
      dropout     : dropout rate (default 0.15)
    """

    def __init__(
        self,
        node_dim:    int = NODE_FEATURE_DIM,
        edge_dim:    int = EDGE_FEATURE_DIM,
        hidden_dim:  int = 128,
        edge_enc_dim:int = 64,
        n_layers:    int = 3,
        heads:       int = 4,
        n_genes:     int = N_GENES,
        dropout:     float = 0.15,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.edge_enc_dim = edge_enc_dim
        self.n_genes      = n_genes
        self.n_layers     = n_layers

        # Encoders
        self.node_encoder = NodeEncoder(node_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_dim, edge_enc_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayerWithEdge(hidden_dim, heads, edge_enc_dim, dropout)
            for _ in range(n_layers)
        ])

        # Edge prediction head
        self.edge_head = EdgePredictorHead(hidden_dim, edge_enc_dim, n_genes, dropout)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Args:
          data: PyG Data object with x, edge_index, edge_attr

        Returns:
          logits: (E, N_GENES) — raw logits per edge per gene
                  Apply sigmoid for probabilities
        """
        x          = data.x           # (N, NODE_FEATURE_DIM) = (N, 35)
        edge_index = data.edge_index  # (2, E)
        edge_attr  = data.edge_attr   # (E, EDGE_FEATURE_DIM) = (E, 5)

        # Encode nodes and edges
        h = self.node_encoder(x)           # (N, hidden_dim)
        e = self.edge_encoder(edge_attr)   # (E, edge_enc_dim)

        # GAT message passing
        for layer in self.gat_layers:
            h = layer(h, edge_index, e)

        # Edge prediction: for each directed edge (i→j)
        src = edge_index[0]   # (E,)
        dst = edge_index[1]   # (E,)

        h_src = h[src]        # (E, hidden_dim)
        h_dst = h[dst]        # (E, hidden_dim)

        logits = self.edge_head(h_src, h_dst, e)  # (E, N_GENES)
        return logits

    def predict_proba(self, data) -> torch.Tensor:
        """Returns sigmoid probabilities (E, N_GENES)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(data))

    def predict_transfers(
        self,
        data,
        threshold: float = 0.35,
    ) -> Dict[int, List[str]]:
        """
        High-level inference: given a graph snapshot, return
        predicted gene transfers as {edge_idx: [gene_names]}.

        threshold: probability above which we call a transfer likely
        """
        proba = self.predict_proba(data)   # (E, N_GENES)
        preds = (proba >= threshold).cpu().numpy()

        result = {}
        for e_idx in range(preds.shape[0]):
            genes = [GENE_INDEX[g] for g in range(N_GENES) if preds[e_idx, g]]
            if genes:
                result[e_idx] = genes
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> dict:
        return {
            "model": "AMRResistanceGNN",
            "node_feature_dim":  NODE_FEATURE_DIM,
            "edge_feature_dim":  EDGE_FEATURE_DIM,
            "hidden_dim":        self.hidden_dim,
            "edge_enc_dim":      self.edge_enc_dim,
            "n_gat_layers":      self.n_layers,
            "attention_heads":   self.gat_layers[0].gat.heads if self.n_layers else None,
            "output_dim":        self.n_genes,
            "output_genes":      GENE_INDEX,
            "trainable_params":  self.count_parameters(),
            "architecture":      (f"NodeEncoder → EdgeEncoder → {self.n_layers}×GAT+EdgeAttn → EdgePredictorHead"
                                  if self.n_layers else
                                  "NodeEncoder → EdgeEncoder → EdgePredictorHead (no message passing)"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    hidden_dim:   int   = 128,
    edge_enc_dim: int   = 64,
    n_layers:     int   = 3,
    heads:        int   = 4,
    dropout:      float = 0.15,
) -> "AMRResistanceGNN":
    if not PYG_AVAILABLE:
        raise ImportError(
            "torch_geometric not installed. Run: "
            "pip install torch-geometric --break-system-packages"
        )
    return AMRResistanceGNN(
        hidden_dim=hidden_dim,
        edge_enc_dim=edge_enc_dim,
        n_layers=n_layers,
        heads=heads,
        dropout=dropout,
    )


def load_model(path: str, device: str = "cpu") -> "AMRResistanceGNN":
    """Load a saved model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(**checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model