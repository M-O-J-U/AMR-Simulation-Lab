"""
Feature Engineering Pipeline for AMR GNN.

Converts raw bacterial agent state + CARD gene profiles into numeric
feature tensors suitable for graph neural network training.

Every feature is grounded in real biology — nothing is fabricated:

Node features (per bacterium, 32-dim vector):
  - Genomic: which resistance genes it carries (10 binary flags)
  - Physiological: fitness, energy, stress, damage, age (5 floats)
  - Behavioral: in_biofilm, is_persister, sos_active (3 binary)
  - Spatial: normalized x/y position (2 floats)
  - Population: local density, generation, offspring count (3 floats)
  - Species: gram stain, shape one-hot (4 + 3 = 7 floats)
  - Antibiotic exposure: per-drug concentration at position (2 floats)

Edge features (per bacterium pair, 8-dim vector):
  - Euclidean distance (normalized)
  - Same species flag
  - Shared resistance genes count
  - Donor has gene recipient doesn't (transfer potential per gene)
  - Relative fitness difference
  - Both in biofilm flag

Labels (what the GNN predicts):
  - Binary per gene: will this gene transfer from donor to recipient
    in the next N steps? (supervised from simulation ground truth)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from data.card_loader import (
    GERM_PROFILES, RESISTANCE_GENES, ANTIBIOTIC_PROFILES,
    GermProfile, resistance_probability
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Canonical ordered list of resistance genes — index = position in feature vector
GENE_INDEX: List[str] = [
    "blaTEM-1", "blaCTX-M-15", "blaKPC-2", "blaNDM-1",
    "mexAB-oprM", "acrAB-tolC", "gyrA_S83L", "mcr-1", "tetM", "vanA",
]
N_GENES = len(GENE_INDEX)

# Canonical antibiotic list for exposure features
AB_INDEX: List[str] = [
    "ciprofloxacin", "meropenem", "colistin",
    "vancomycin", "ampicillin", "tetracycline",
]
N_ABS = len(AB_INDEX)

# Species one-hot
SPECIES_INDEX: List[str] = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Staphylococcus aureus (MRSA)",
]
N_SPECIES = len(SPECIES_INDEX)

# Feature dimensions
DIM_GENOMIC     = N_GENES       # 10  binary resistance gene flags
DIM_PHYSIO      = 5             # fitness, energy, stress, ab_damage, age_norm
DIM_BEHAVIORAL  = 3             # in_biofilm, is_persister, sos_active
DIM_SPATIAL     = 2             # x_norm, y_norm
DIM_POPULATION  = 3             # local_density_norm, generation_norm, offspring_norm
DIM_SPECIES     = N_SPECIES     # 5   one-hot species
DIM_GRAM        = 2             # gram_positive, gram_negative
DIM_AB_EXPOSURE = N_ABS         # 6   antibiotic concentrations at position

NODE_FEATURE_DIM = (DIM_GENOMIC + DIM_PHYSIO + DIM_BEHAVIORAL +
                    DIM_SPATIAL  + DIM_POPULATION + DIM_SPECIES +
                    DIM_GRAM     + DIM_AB_EXPOSURE)  # = 36

EDGE_FEATURE_DIM = 8

# ─────────────────────────────────────────────────────────────────────────────
# NODE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_node_features(
    bacterium_dict: dict,
    grid_width: int,
    grid_height: int,
    antibiotic_concs: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a bacterium agent dict
    (as returned by BacteriumAgent.to_dict()).

    Returns: np.ndarray of shape (NODE_FEATURE_DIM,), dtype float32
    """
    feats = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    idx   = 0

    # ── 1. Genomic features (binary gene presence) ──────────────────────────
    genes = set(bacterium_dict.get("resistance_genes", []))
    for gene in GENE_INDEX:
        feats[idx] = 1.0 if gene in genes else 0.0
        idx += 1
    # idx = 10

    # ── 2. Physiological features ───────────────────────────────────────────
    feats[idx]   = float(bacterium_dict.get("fitness",            0.8))
    feats[idx+1] = float(bacterium_dict.get("energy",             0.5))
    feats[idx+2] = float(bacterium_dict.get("stress_level",       0.0))
    feats[idx+3] = float(bacterium_dict.get("antibiotic_damage",  0.0))
    feats[idx+4] = min(1.0, float(bacterium_dict.get("age", 0)) / 200.0)
    idx += 5
    # idx = 15

    # ── 3. Behavioral features (binary) ─────────────────────────────────────
    feats[idx]   = 1.0 if bacterium_dict.get("in_biofilm",  False) else 0.0
    feats[idx+1] = 1.0 if bacterium_dict.get("is_persister",False) else 0.0
    feats[idx+2] = 1.0 if bacterium_dict.get("sos_active",  False) else 0.0
    idx += 3
    # idx = 18

    # ── 4. Spatial features (normalized 0–1) ────────────────────────────────
    pos = bacterium_dict.get("pos", [0, 0])
    feats[idx]   = pos[0] / max(1, grid_width  - 1)
    feats[idx+1] = pos[1] / max(1, grid_height - 1)
    idx += 2
    # idx = 20

    # ── 5. Population features ───────────────────────────────────────────────
    feats[idx]   = min(1.0, float(bacterium_dict.get("local_density",   0)) / 10.0)
    feats[idx+1] = min(1.0, float(bacterium_dict.get("generation",      0)) / 50.0)
    feats[idx+2] = min(1.0, float(bacterium_dict.get("offspring_count", 0)) / 20.0)
    idx += 3
    # idx = 23

    # ── 6. Species one-hot ───────────────────────────────────────────────────
    species = bacterium_dict.get("species", "")
    for sp in SPECIES_INDEX:
        feats[idx] = 1.0 if species == sp else 0.0
        idx += 1
    # idx = 28

    # ── 7. Gram stain ────────────────────────────────────────────────────────
    gram = bacterium_dict.get("gram_stain", "negative")
    feats[idx]   = 1.0 if gram == "positive" else 0.0
    feats[idx+1] = 1.0 if gram == "negative" else 0.0
    idx += 2
    # idx = 30

    # ── 8. Antibiotic exposure at position ───────────────────────────────────
    ab_concs = antibiotic_concs or {}
    for ab_name in AB_INDEX:
        feats[idx] = min(1.0, ab_concs.get(ab_name, 0.0) / 5.0)  # normalize to [0,1]
        idx += 1
    # idx = 36

    assert idx == NODE_FEATURE_DIM, f"Feature dim mismatch: {idx} != {NODE_FEATURE_DIM}"
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# EDGE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_edge_features(
    b_i: dict,
    b_j: dict,
    grid_width:  int,
    grid_height: int,
    max_dist:    float = 10.0,
) -> np.ndarray:
    """
    Extract edge feature vector between two bacteria.

    Features encode biological transfer potential:
      - How close are they? (distance — HGT requires physical proximity)
      - Same species? (same-species HGT is much more efficient)
      - Shared gene count (co-carriers can transfer to each other)
      - Transfer potential per gene class
      - Fitness difference (fitter donor = stronger selection pressure)
      - Both in biofilm (biofilm enhances gene transfer)
      - Both SOS active (stress-induced conjugation is higher)
      - Relative antibiotic pressure

    Returns: np.ndarray of shape (EDGE_FEATURE_DIM,), dtype float32
    """
    feats = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)

    # 1. Euclidean distance (normalized by max_dist)
    pos_i = b_i.get("pos", [0, 0])
    pos_j = b_j.get("pos", [0, 0])
    dist  = math.sqrt((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2)
    feats[0] = min(1.0, dist / max_dist)

    # 2. Same species flag
    feats[1] = 1.0 if b_i.get("species") == b_j.get("species") else 0.0

    # 3. Shared gene count (normalized)
    genes_i = set(b_i.get("resistance_genes", []))
    genes_j = set(b_j.get("resistance_genes", []))
    shared  = len(genes_i & genes_j)
    feats[2] = min(1.0, shared / max(1, N_GENES))

    # 4. Transfer potential: i has gene that j doesn't
    #    (measures how much i can "teach" j)
    transferable = len(genes_i - genes_j)
    feats[3] = min(1.0, transferable / max(1, N_GENES))

    # 5. Fitness difference |fit_i - fit_j| (higher = stronger selection)
    fit_i = float(b_i.get("fitness", 0.8))
    fit_j = float(b_j.get("fitness", 0.8))
    feats[4] = abs(fit_i - fit_j)

    # 6. Both in biofilm (biofilm enhances plasmid transfer 10–1000x)
    feats[5] = 1.0 if (b_i.get("in_biofilm") and b_j.get("in_biofilm")) else 0.0

    # 7. Either has SOS active (SOS upregulates conjugation)
    feats[6] = 1.0 if (b_i.get("sos_active") or b_j.get("sos_active")) else 0.0

    # 8. Antibiotic stress difference (|stress_i - stress_j|)
    stress_i = float(b_i.get("stress_level", 0.0))
    stress_j = float(b_j.get("stress_level", 0.0))
    feats[7] = abs(stress_i - stress_j)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION FROM SIMULATION STATE
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_state(
    state: dict,
    max_edge_distance: int = 3,
    max_nodes: int = 500,
) -> Optional[dict]:
    """
    Build a graph from a full simulation state snapshot.

    Nodes  = bacteria (up to max_nodes, sampled if more)
    Edges  = pairs of bacteria within max_edge_distance grid cells
    Labels = per-edge, per-gene transfer label (used during training)

    Returns a dict with:
      node_features  : (N, NODE_FEATURE_DIM) float32
      edge_index     : (2, E) int64
      edge_features  : (E, EDGE_FEATURE_DIM) float32
      node_ids       : list of unique_ids (for label matching)
      gene_labels    : (E, N_GENES) binary  — set during training only
      metadata       : dict with step, scenario etc.
    """
    bacteria = state.get("bacteria", [])
    if not bacteria:
        return None

    # Sample if too many
    if len(bacteria) > max_nodes:
        import random
        bacteria = random.sample(bacteria, max_nodes)

    gw = state.get("grid_width",  80)
    gh = state.get("grid_height", 60)

    # Pre-compute antibiotic concentrations at each position from heatmaps
    ab_heatmaps = state.get("antibiotic_heatmaps", {})

    def get_ab_concs(pos):
        concs = {}
        x, y = pos[0]//2, pos[1]//2  # heatmap is downsampled 2x
        for key, hm in ab_heatmaps.items():
            data = hm.get("data", [])
            if data and y < len(data) and x < len(data[0]):
                concs[key] = float(data[y][x]) if isinstance(data[y], list) else 0.0
        return concs

    # Build node features
    node_features = []
    node_ids      = []
    pos_lookup    = {}

    for b in bacteria:
        ab_concs = get_ab_concs(b["pos"])
        feats    = extract_node_features(b, gw, gh, ab_concs)
        node_features.append(feats)
        node_ids.append(b["id"])
        pos_lookup[b["id"]] = (b["pos"][0], b["pos"][1])

    N = len(node_features)

    # Build edges: connect bacteria within max_edge_distance
    edge_src, edge_dst, edge_feats = [], [], []

    for i in range(N):
        xi, yi = pos_lookup[node_ids[i]]
        for j in range(i + 1, N):
            xj, yj = pos_lookup[node_ids[j]]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist <= max_edge_distance:
                # Undirected: add both directions
                b_i, b_j = bacteria[i], bacteria[j]
                ef_ij = extract_edge_features(b_i, b_j, gw, gh)
                ef_ji = extract_edge_features(b_j, b_i, gw, gh)

                edge_src.append(i); edge_dst.append(j); edge_feats.append(ef_ij)
                edge_src.append(j); edge_dst.append(i); edge_feats.append(ef_ji)

    if not edge_src:
        return None

    return {
        "node_features": np.array(node_features, dtype=np.float32),  # (N, 36)
        "edge_index":    np.array([edge_src, edge_dst], dtype=np.int64),  # (2, E)
        "edge_features": np.array(edge_feats, dtype=np.float32),      # (E, 8)
        "node_ids":      node_ids,
        "bacteria":      bacteria,
        "gene_labels":   None,   # filled during training
        "metadata": {
            "step":     state.get("stats", {}).get("step", 0),
            "scenario": state.get("stats", {}).get("scenario", ""),
            "n_nodes":  N,
            "n_edges":  len(edge_src),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION (for training data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_hgt_labels(
    graph_t0: dict,
    graph_t1: dict,
) -> np.ndarray:
    """
    Given two consecutive graph snapshots (t0 and t1), generate
    binary labels: for each edge (i→j) and each gene,
    did gene transfer from i to j between t0 and t1?

    Returns: (E, N_GENES) binary float32 array
    """
    # Map node_id → gene set at t0 and t1
    genes_t0 = {b["id"]: set(b["resistance_genes"]) for b in graph_t0["bacteria"]}
    genes_t1 = {b["id"]: set(b["resistance_genes"]) for b in graph_t1.get("bacteria", [])}

    n_edges  = graph_t0["edge_index"].shape[1]
    labels   = np.zeros((n_edges, N_GENES), dtype=np.float32)

    edge_src = graph_t0["edge_index"][0]
    edge_dst = graph_t0["edge_index"][1]
    node_ids = graph_t0["node_ids"]

    for e_idx in range(n_edges):
        i = edge_src[e_idx]
        j = edge_dst[e_idx]

        id_i = node_ids[i]
        id_j = node_ids[j]

        genes_i_t0 = genes_t0.get(id_i, set())
        genes_j_t0 = genes_t0.get(id_j, set())
        genes_j_t1 = genes_t1.get(id_j, set())

        # Gene transferred from i to j: j didn't have it at t0, has it at t1,
        # AND i had it at t0 (plausible donor)
        for g_idx, gene in enumerate(GENE_INDEX):
            if (gene in genes_i_t0 and
                gene not in genes_j_t0 and
                gene in genes_j_t1):
                labels[e_idx, g_idx] = 1.0

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# DATASET COLLECTION (run simulation, collect snapshots)
# ─────────────────────────────────────────────────────────────────────────────

def collect_training_snapshots(
    n_steps:    int  = 60,
    scenario:   str  = "ecoli_cipro",
    seed:       int  = 42,
    snapshot_interval: int = 3,
) -> List[Tuple[dict, dict]]:
    """
    Run a simulation headlessly and collect (graph_t0, graph_t1) pairs
    for supervised training.

    Each pair: state at step t and state at step t + snapshot_interval.
    Labels: which HGT transfers occurred between t and t+interval.

    Returns list of (graph_t0, graph_t1) tuples with labels filled in graph_t0.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from simulation.amr_model import AMRSimulationModel

    model = AMRSimulationModel(
        scenario=scenario, initial_bacteria=120,
        seed=seed, enable_logging=False
    )

    # Apply antibiotic after warmup so we get resistance selection data
    for _ in range(15):
        model.step()
    if scenario != "validation":
        for ab_key in model.active_antibiotic_keys:
            model.apply_antibiotic(ab_key, concentration=1.5, mode="uniform")

    pairs      = []
    prev_state = None
    prev_graph = None

    for step in range(n_steps):
        model.step()
        if not model.running:
            break

        if step % snapshot_interval == 0:
            state = model.get_full_state()
            graph = build_graph_from_state(state, max_edge_distance=3, max_nodes=300)
            if graph is None:
                continue

            if prev_graph is not None and prev_state is not None:
                labels = generate_hgt_labels(prev_graph, graph)
                prev_graph["gene_labels"] = labels
                # Keep all pairs — GNN needs negative examples (no-transfer)
                # to learn the boundary. Positive (HGT) events are rare but
                # class weighting in the loss handles the imbalance.
                pairs.append((prev_graph, graph))

            prev_graph = graph
            prev_state = state

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH GEOMETRIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

class AMRGraphDataset:
    """
    Plain Python dataset wrapping AMR simulation graph snapshots.
    Does NOT inherit InMemoryDataset — avoids PyG version compatibility issues.

    Each item is a PyG Data object:
      x          : node features  (N, NODE_FEATURE_DIM)
      edge_index : graph edges    (2, E)
      edge_attr  : edge features  (E, EDGE_FEATURE_DIM)
      y          : transfer labels (E, N_GENES) — binary multi-label
    """

    def __init__(self, graph_pairs: list):
        from torch_geometric.data import Data

        self._data_list: List = []
        for g_t0, _g_t1 in graph_pairs:
            if g_t0.get("gene_labels") is None:
                continue
            if g_t0["node_features"].shape[0] < 2:
                continue

            x          = torch.tensor(g_t0["node_features"], dtype=torch.float)
            edge_index = torch.tensor(g_t0["edge_index"],    dtype=torch.long)
            edge_attr  = torch.tensor(g_t0["edge_features"], dtype=torch.float)
            y          = torch.tensor(g_t0["gene_labels"],   dtype=torch.float)

            # Sanity-check shapes before adding
            E = edge_index.shape[1]
            if edge_attr.shape[0] != E or y.shape[0] != E:
                continue

            data          = Data(x=x, edge_index=edge_index,
                                 edge_attr=edge_attr, y=y)
            data.metadata = g_t0.get("metadata", {})
            self._data_list.append(data)

    def __len__(self):          return len(self._data_list)
    def __getitem__(self, idx): return self._data_list[idx]
    def __iter__(self):         return iter(self._data_list)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def feature_names() -> List[str]:
    """Return human-readable names for each node feature dimension."""
    names = []
    names += [f"gene_{g}" for g in GENE_INDEX]
    names += ["fitness", "energy", "stress_level", "ab_damage", "age_norm"]
    names += ["in_biofilm", "is_persister", "sos_active"]
    names += ["x_norm", "y_norm"]
    names += ["local_density_norm", "generation_norm", "offspring_norm"]
    names += [f"species_{sp.split()[0]}" for sp in SPECIES_INDEX]
    names += ["gram_positive", "gram_negative"]
    names += [f"ab_conc_{ab}" for ab in AB_INDEX]
    return names

def edge_feature_names() -> List[str]:
    return [
        "distance_norm", "same_species", "shared_genes_norm",
        "transferable_genes_norm", "fitness_diff",
        "both_biofilm", "either_sos", "stress_diff",
    ]

def get_dims() -> dict:
    return {
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "n_genes":          N_GENES,
        "gene_names":       GENE_INDEX,
    }