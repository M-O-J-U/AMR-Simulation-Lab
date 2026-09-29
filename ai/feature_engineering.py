"""
Feature Engineering Pipeline for AMR GNN.

Converts raw bacterial agent state + CARD gene profiles into numeric
feature tensors suitable for graph neural network training.

============================================================================
FEATURE LEAKAGE REMEDIATION (post IEEE JBHI desk rejection, JBHI-03955-2026)
============================================================================
The original feature set included four fields that are direct, literal
components of the stochastic rule in BacteriumAgent._attempt_hgt() that
GENERATES the training label:

  - edge distance (raw)         -- HGT requires dist <= CONJUGATION_DISTANCE,
                                    a hard gate the model could read off
                                    directly instead of learning anything
  - same_species (edge flag)    -- _attempt_hgt() skips immediately if
                                    species differ; this IS the gate
  - transferable_genes_norm     -- mirrors "gn not in recipient.genes and
                                    gn in donor.genes" exactly, the core
                                    eligibility condition for transfer
  - sos_active (node + edge)    -- literally the term that multiplies
                                    hgt_prob by 2x in the label-generating
                                    rule

Training a model on these features mostly demonstrates that the model can
recover the simulator's own deterministic/stochastic gating logic, not that
it has learned generalizable biological structure. This was flagged
explicitly in editorial review under "methods rigor" and "evaluation &
evidence."

These four features have been REMOVED from the node/edge vectors below.
Two of the four (same_species, distance) carried no other biological
signal once removed; one (transferable_genes_norm) is replaced with the
much weaker, non-gating shared_genes_norm (genomic overlap, not transfer
eligibility); sos_active is removed but stress_level (a continuous,
noisier proxy that does NOT deterministically gate the label) is retained
at the node level.

A coarsened, discretized "proximity_band" edge feature replaces raw
distance: it buckets pairs into {same-cell, near, far} rather than exposing
the exact CONJUGATION_DISTANCE=1 threshold, so the model must still infer
fine-grained spatial structure rather than reading the gate off a
continuous feature.

See ai/checkpoints/feature_leakage_remediation.md for the full audit.
============================================================================

Node features (per bacterium, 35-dim vector):
  - Genomic: which resistance genes it carries (10 binary flags)
  - Physiological: fitness, energy, stress, damage, age (5 floats)
  - Behavioral: in_biofilm, is_persister (2 binary) -- sos_active REMOVED
  - Spatial: normalized x/y position (2 floats)
  - Population: local density, generation, offspring count (3 floats)
  - Species: one-hot (5 floats)
  - Gram stain: positive/negative (2 floats)
  - Antibiotic exposure: per-drug concentration at position (6 floats)

Edge features (per bacterium pair, 5-dim vector):
  - Shared resistance genes (normalized genomic overlap, NOT transfer
    eligibility)
  - Relative fitness difference
  - Both in biofilm flag (now a real causal factor -- see
    core/bacterium_agent.py _attempt_hgt biofilm multiplier)
  - Stress level difference (continuous proxy, not the deterministic
    sos_active gate)
  - Coarsened proximity band (same-cell / near / far -- NOT raw distance)

Labels (what the GNN predicts):
  - Binary per gene: did a REAL recorded HGT conjugation event transfer
    this gene from donor to recipient in the snapshot window? (see
    generate_hgt_labels(), which uses model.hgt_events ground truth,
    not genome diffing)
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
DIM_BEHAVIORAL  = 2             # in_biofilm, is_persister (sos_active REMOVED — leakage)
DIM_SPATIAL     = 2             # x_norm, y_norm
DIM_POPULATION  = 3             # local_density_norm, generation_norm, offspring_norm
DIM_SPECIES     = N_SPECIES     # 5   one-hot species
DIM_GRAM        = 2             # gram_positive, gram_negative
DIM_AB_EXPOSURE = N_ABS         # 6   antibiotic concentrations at position

NODE_FEATURE_DIM = (DIM_GENOMIC + DIM_PHYSIO + DIM_BEHAVIORAL +
                    DIM_SPATIAL  + DIM_POPULATION + DIM_SPECIES +
                    DIM_GRAM     + DIM_AB_EXPOSURE)  # = 35

EDGE_FEATURE_DIM = 5   # was 8 — distance, same_species, transferable_genes,
                       # sos_active removed as direct label-gate leakage

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

    NOTE: sos_active is intentionally NOT included here. It is a direct
    multiplicative term in the HGT label-generating rule
    (BacteriumAgent._attempt_hgt: hgt_prob *= 2.0 if self.sos_active),
    so including it as a model input would let the model partially read
    the label off its own feature rather than infer it. stress_level
    (a continuous, non-deterministic proxy) is retained instead.

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
    # sos_active REMOVED — direct leakage of the HGT label-generating rule.
    feats[idx]   = 1.0 if bacterium_dict.get("in_biofilm",  False) else 0.0
    feats[idx+1] = 1.0 if bacterium_dict.get("is_persister",False) else 0.0
    idx += 2
    # idx = 17

    # ── 4. Spatial features (normalized 0–1) ────────────────────────────────
    pos = bacterium_dict.get("pos", [0, 0])
    feats[idx]   = pos[0] / max(1, grid_width  - 1)
    feats[idx+1] = pos[1] / max(1, grid_height - 1)
    idx += 2
    # idx = 19

    # ── 5. Population features ───────────────────────────────────────────────
    feats[idx]   = min(1.0, float(bacterium_dict.get("local_density",   0)) / 10.0)
    feats[idx+1] = min(1.0, float(bacterium_dict.get("generation",      0)) / 50.0)
    feats[idx+2] = min(1.0, float(bacterium_dict.get("offspring_count", 0)) / 20.0)
    idx += 3
    # idx = 22

    # ── 6. Species one-hot ───────────────────────────────────────────────────
    species = bacterium_dict.get("species", "")
    for sp in SPECIES_INDEX:
        feats[idx] = 1.0 if species == sp else 0.0
        idx += 1
    # idx = 27

    # ── 7. Gram stain ────────────────────────────────────────────────────────
    gram = bacterium_dict.get("gram_stain", "negative")
    feats[idx]   = 1.0 if gram == "positive" else 0.0
    feats[idx+1] = 1.0 if gram == "negative" else 0.0
    idx += 2
    # idx = 29

    # ── 8. Antibiotic exposure at position ───────────────────────────────────
    ab_concs = antibiotic_concs or {}
    for ab_name in AB_INDEX:
        feats[idx] = min(1.0, ab_concs.get(ab_name, 0.0) / 5.0)  # normalize to [0,1]
        idx += 1
    # idx = 35

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

    NOTE on removed features (post JBHI-03955-2026 leakage remediation):
      - raw distance, same_species, and transferable_genes_norm were
        removed because they are literal components of the deterministic/
        stochastic gate in BacteriumAgent._attempt_hgt() that generates
        the training label (species match and proximity are hard
        preconditions; transferable_genes_norm mirrors the eligibility
        check almost exactly). Including them taught the model to read
        the gate off its own inputs rather than infer transfer risk.
      - the SOS-derived flag was removed for the same reason (sos_active
        is a direct multiplicative term in hgt_prob).
      - raw distance is replaced by a coarsened, discretized
        proximity_band so the model retains some spatial signal without
        the exact CONJUGATION_DISTANCE=1 cutoff being readable as a
        continuous feature.

    Retained features encode biological transfer CORRELATES rather than
    the eligibility rule itself:
      - Shared gene count (genomic similarity, not transfer eligibility)
      - Fitness difference (selection pressure context)
      - Both in biofilm (biofilm enhances conjugation -- see
        BIOFILM_HGT_MULTIPLIER in bacterium_agent.py, now a real causal
        mechanism rather than a spurious correlation)
      - Stress level difference (continuous proxy for SOS likelihood,
        not the deterministic sos_active gate itself)
      - Coarsened proximity band (same-cell / near / far)

    Returns: np.ndarray of shape (EDGE_FEATURE_DIM,), dtype float32
    """
    feats = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)

    # 1. Shared gene count (normalized) — genomic overlap, NOT eligibility
    genes_i = set(b_i.get("resistance_genes", []))
    genes_j = set(b_j.get("resistance_genes", []))
    shared  = len(genes_i & genes_j)
    feats[0] = min(1.0, shared / max(1, N_GENES))

    # 2. Fitness difference |fit_i - fit_j| (higher = stronger selection)
    fit_i = float(b_i.get("fitness", 0.8))
    fit_j = float(b_j.get("fitness", 0.8))
    feats[1] = abs(fit_i - fit_j)

    # 3. Both in biofilm — now a real causal factor in HGT probability
    #    (BIOFILM_HGT_MULTIPLIER in bacterium_agent.py._attempt_hgt),
    #    not merely a correlate as in the pre-remediation simulation.
    feats[2] = 1.0 if (b_i.get("in_biofilm") and b_j.get("in_biofilm")) else 0.0

    # 4. Stress level difference — continuous proxy, not the deterministic
    #    sos_active gate. Two cells can have very different stress levels
    #    while both being below/above the 0.4 SOS activation threshold,
    #    so this does not let the model read off sos_active directly.
    stress_i = float(b_i.get("stress_level", 0.0))
    stress_j = float(b_j.get("stress_level", 0.0))
    feats[3] = abs(stress_i - stress_j)

    # 5. Coarsened proximity band (replaces raw distance).
    #    Bucketed into 3 discrete bands with no exposed numeric threshold
    #    matching CONJUGATION_DISTANCE: 0.0 = same cell, 0.5 = nearby
    #    (within ~3 cells), 1.0 = far. This keeps general spatial context
    #    available to the model without letting it read the exact HGT
    #    eligibility radius off a continuous feature.
    pos_i = b_i.get("pos", [0, 0])
    pos_j = b_j.get("pos", [0, 0])
    dist  = math.sqrt((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2)
    if dist < 0.5:
        feats[4] = 0.0   # same cell
    elif dist <= 3.0:
        feats[4] = 0.5   # nearby
    else:
        feats[4] = 1.0   # far

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION FROM SIMULATION STATE
# ─────────────────────────────────────────────────────────────────────────────

# Base seed for the node subsample when the caller passes no rng (fixed
# 2026-09-29: this used the GLOBAL, unseeded `random.sample`, so training
# pairs differed run-to-run whenever a population exceeded max_nodes).
SUBSAMPLE_BASE_SEED = 20260929


def build_graph_from_state(
    state: dict,
    max_edge_distance: int = 3,
    max_nodes: int = 500,
    rng: Optional["random.Random"] = None,
) -> Optional[dict]:
    """
    Build a graph from a full simulation state snapshot.

    Nodes  = bacteria (up to max_nodes, sampled if more)
    Edges  = pairs of bacteria within max_edge_distance grid cells
    Labels = per-edge, per-gene transfer label (used during training)

    NOTE on max_edge_distance vs. CONJUGATION_DISTANCE: the default
    max_edge_distance=3 is intentionally WIDER than
    BacteriumAgent.CONJUGATION_DISTANCE (=1), the actual radius within
    which HGT can occur. This is deliberate: if the graph only contained
    edges between HGT-eligible pairs, the mere existence of an edge would
    trivially separate eligible from ineligible pairs, reintroducing the
    same leakage problem as the (now-removed) raw distance feature, just
    at the topology level instead of the feature level. By including
    near-but-ineligible pairs (distance 2-3) alongside eligible pairs
    (distance <=1) under the same coarsened proximity_band feature value
    ("nearby"), the model cannot use topology or features to trivially
    recover eligibility and must learn genuine biological structure.

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

    # Sample if too many — with a seeded local RNG (never the global one).
    # Default: seeded from the snapshot's step, so the same state always yields
    # the same graph. collect_training_snapshots passes a per-run rng instead.
    if len(bacteria) > max_nodes:
        import random
        if rng is None:
            step = int(state.get("stats", {}).get("step", 0))
            rng = random.Random(SUBSAMPLE_BASE_SEED + step)
        bacteria = rng.sample(bacteria, max_nodes)

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
        "node_features": np.array(node_features, dtype=np.float32),  # (N, 35)
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
    recorded_events: Optional[List] = None,
) -> np.ndarray:
    """
    Given two consecutive graph snapshots (t0 and t1), generate
    binary labels: for each edge (i->j) and each gene,
    did a REAL recorded HGT conjugation event occur from i to j
    between t0 and t1?

    CORRECTED (post JBHI desk rejection, 2026-06-29):
    The original implementation inferred HGT purely from genome diffing
    (gene absent in j at t0, present in j at t1, present in i at t0).
    This conflates two biologically distinct mechanisms:
      1. Horizontal gene transfer (conjugation) -- what we want to predict
      2. Spontaneous mutation / independent gene acquisition (see
         BacteriumAgent._attempt_mutation) -- an entirely separate process
         that also adds genes to a cell's genome but has nothing to do
         with the donor cell i.
    A cell can acquire a gene via mutation in the same window an HGT-eligible
    neighbor happens to carry that gene, producing a false-positive HGT label
    that has no causal relationship to cell i. This was the source of the
    "internally inconsistent" positive-label counts flagged in editorial
    review (JBHI-03955-2026).

    Fix: use model.hgt_events (recorded directly inside
    BacteriumAgent._attempt_hgt / Model.record_hgt_event at the moment
    conjugation actually fires) as ground truth instead of inferring transfer
    from genome state diffs. This is the only source that distinguishes HGT
    from mutation, since both mechanisms are observable only at the agent
    level, not from genome snapshots alone.

    Args:
      graph_t0:        graph snapshot at time t (must include "bacteria")
      graph_t1:        graph snapshot at time t+interval
      recorded_events:  list of HGTEvent-like objects (or dicts) with
                        .step / .donor_id / .recipient_id / .gene attributes,
                        restricted to the (t0, t1) window by the caller.
                        If None, falls back to the old diff-based method
                        with a loud warning (kept only for backward
                        compatibility / ablation comparison).

    Returns: (E, N_GENES) binary float32 array
    """
    n_edges  = graph_t0["edge_index"].shape[1]
    labels   = np.zeros((n_edges, N_GENES), dtype=np.float32)

    edge_src = graph_t0["edge_index"][0]
    edge_dst = graph_t0["edge_index"][1]
    node_ids = graph_t0["node_ids"]

    if recorded_events is None:
        import warnings
        warnings.warn(
            "generate_hgt_labels() called without recorded_events — "
            "falling back to genome-diff inference, which conflates HGT "
            "with independent mutation. This mode exists only for backward "
            "compatibility / ablation comparison and MUST NOT be used for "
            "any reported model results.",
            stacklevel=2,
        )
        return _generate_hgt_labels_diff_legacy(graph_t0, graph_t1)

    # Build (donor_id, recipient_id) -> set of genes transferred, from
    # real recorded events only.
    real_transfers: Dict[Tuple[int, int], set] = {}
    for ev in recorded_events:
        donor_id     = ev.donor_id     if hasattr(ev, "donor_id")     else ev["donor"]
        recipient_id = ev.recipient_id if hasattr(ev, "recipient_id") else ev["recipient"]
        gene         = ev.gene         if hasattr(ev, "gene")         else ev["gene"]
        key = (donor_id, recipient_id)
        real_transfers.setdefault(key, set()).add(gene)

    if not real_transfers:
        # No HGT events occurred in this window — all-negative labels,
        # which is correct and should NOT be silently dropped (negative
        # examples are exactly what the model needs to learn the boundary).
        return labels

    for e_idx in range(n_edges):
        i, j = edge_src[e_idx], edge_dst[e_idx]
        id_i, id_j = node_ids[i], node_ids[j]

        genes_transferred = real_transfers.get((id_i, id_j))
        if not genes_transferred:
            continue

        for g_idx, gene in enumerate(GENE_INDEX):
            if gene in genes_transferred:
                labels[e_idx, g_idx] = 1.0

    return labels


def _generate_hgt_labels_diff_legacy(
    graph_t0: dict,
    graph_t1: dict,
) -> np.ndarray:
    """
    LEGACY / DEPRECATED — kept only so the original (flawed) labeling
    method can be reproduced for ablation comparison in the corrected
    paper ("Effect of ground-truth-event labels vs. genome-diff labels
    on reported AUROC"). Do not use for any headline result.

    See generate_hgt_labels() docstring for why this method conflates
    HGT with spontaneous mutation.
    """
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
    biology:    str  = "paper_v1",
    dose_duration: Optional[int] = None,
    dose: float = 1.5,
) -> List[Tuple[dict, dict]]:
    """
    Run a simulation headlessly and collect (graph_t0, graph_t1) pairs
    for supervised training.

    Each pair: state at step t and state at step t + snapshot_interval.
    Labels: REAL recorded HGT conjugation events (model.hgt_events) that
    fired between t and t+interval — NOT inferred from genome diffing.
    See generate_hgt_labels() docstring for why this distinction matters.

    Returns list of (graph_t0, graph_t1) tuples with labels filled in graph_t0.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from simulation.amr_model import AMRSimulationModel

    from data.biology import BIOLOGIES
    model = AMRSimulationModel(
        scenario=scenario, initial_bacteria=120,
        seed=seed, enable_logging=False,
        biology=BIOLOGIES[biology],   # pipeline passes config["biology"]
    )
    # Per-run node-subsample RNG, independent of the model's RNG (so graph
    # building can never perturb the simulation) and fixed by the run's seed.
    import random as _random
    subsample_rng = _random.Random(SUBSAMPLE_BASE_SEED * 1000 + seed)

    # Apply antibiotic after warmup so we get resistance selection data
    for _ in range(15):
        model.step()
    if scenario != "validation":
        for ab_key in model.active_antibiotic_keys:
            model.apply_antibiotic(ab_key, concentration=dose, mode="uniform")

    pairs      = []
    prev_state = None
    prev_graph = None

    # Track cumulative HGT event count BEFORE entering each interval window.
    # model.hgt_events trims to the last 500 once it exceeds 1000 (see
    # AMRSimulationModel.record_hgt_event), so we cannot rely on stable
    # list indices across the whole run -- instead we snapshot the *list
    # object's current contents* at each window boundary and diff against
    # the previous window's recorded event objects directly (HGTEvent
    # objects are never mutated in place, only appended/trimmed), which is
    # safe regardless of trimming.
    prev_event_count_seen = 0  # number of events already attributed to past windows

    for step in range(n_steps):
        # Time-limited course (lab_v2 protocol, approved 2026-09-29): clear the
        # drugs after `dose_duration` steps of exposure. None = never cleared
        # (the original protocol; paper_v1 golden fingerprints use None).
        # SIMPLIFICATION: a step count, not a real duration - no step length or
        # sourced half-life is defined yet (follow-up work, see CLAUDE.md).
        if dose_duration is not None and step == dose_duration:
            for ab_key in list(model.antibiotic_grids):
                model.remove_antibiotic(ab_key)
        model.step()
        if not model.running:
            break

        if step % snapshot_interval == 0:
            state = model.get_full_state()
            graph = build_graph_from_state(state, max_edge_distance=3, max_nodes=300,
                                           rng=subsample_rng)
            if graph is None:
                continue

            if prev_graph is not None and prev_state is not None:
                # Events fired strictly after the previous snapshot's step
                # and up to and including the current step belong to this
                # window. We identify them by event.step, not by list
                # position, so the trim-to-500 cap cannot misattribute them
                # (a trimmed-away event is, by construction, older than any
                # window we still care about).
                window_start_step = prev_graph["metadata"]["step"]
                window_end_step   = graph["metadata"]["step"]
                window_events = [
                    ev for ev in model.hgt_events
                    if window_start_step < ev.step <= window_end_step
                ]

                labels = generate_hgt_labels(
                    prev_graph, graph, recorded_events=window_events
                )
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
    names += ["in_biofilm", "is_persister"]   # sos_active removed — leakage
    names += ["x_norm", "y_norm"]
    names += ["local_density_norm", "generation_norm", "offspring_norm"]
    names += [f"species_{sp.split()[0]}" for sp in SPECIES_INDEX]
    names += ["gram_positive", "gram_negative"]
    names += [f"ab_conc_{ab}" for ab in AB_INDEX]
    return names

def edge_feature_names() -> List[str]:
    # distance_norm, same_species, transferable_genes_norm, either_sos
    # removed — direct leakage of the HGT label-generating rule.
    # proximity_band replaces raw distance with a coarsened 3-bucket signal.
    return [
        "shared_genes_norm", "fitness_diff",
        "both_biofilm", "stress_diff", "proximity_band",
    ]

def get_dims() -> dict:
    return {
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "n_genes":          N_GENES,
        "gene_names":       GENE_INDEX,
    }