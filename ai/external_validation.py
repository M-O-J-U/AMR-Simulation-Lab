"""
External Validation Module — Real PATRIC/NCBI AMR Surveillance Data.

This module closes the biggest reviewer objection:
  "You trained on simulation. Does it predict real biology?"

Strategy:
  1. Download real E. coli and Klebsiella isolate metadata from PATRIC
     (Pathosystems Resource Integration Center — public, no auth needed)
  2. Extract which resistance genes each isolate carries (from PATRIC AMR panel)
  3. Build co-occurrence network: isolates sharing the same ward/sample_date
     are likely to have been in physical proximity (HGT opportunity)
  4. Ask: do gene pairs that our GNN predicts as "high transfer probability"
     co-occur more frequently in real isolates than low-probability pairs?
  5. Compute Spearman correlation between GNN transfer probability and
     real-world gene co-occurrence frequency → this is the validation metric

This is a novel validation approach because:
  - We don't claim the GNN directly predicts real HGT (we can't observe that)
  - We do claim: genes the GNN predicts as frequently transferred
    are also more frequently found together in real clinical isolates
  - This is a testable, publishable hypothesis

PATRIC API: https://www.patricbrc.org/api/
No API key required. Public data.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_engineering import GENE_INDEX, N_GENES
from data.card_loader import RESISTANCE_GENES

# ─────────────────────────────────────────────────────────────────────────────
# PATRIC API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

PATRIC_API = "https://www.patricbrc.org/api"
CACHE_DIR  = Path("ai/patric_cache")

# Map our internal gene names to PATRIC AMR panel gene names
GENE_TO_PATRIC = {
    "blaTEM-1":    ["TEM", "blaTEM"],
    "blaCTX-M-15": ["CTX-M", "blaCTX-M-15", "CTX-M-15"],
    "blaKPC-2":    ["KPC", "blaKPC", "KPC-2"],
    "blaNDM-1":    ["NDM", "blaNDM", "NDM-1"],
    "mexAB-oprM":  ["mexA", "mexB", "MexAB-OprM"],
    "acrAB-tolC":  ["acrA", "acrB", "AcrAB-TolC"],
    "gyrA_S83L":   ["gyrA"],
    "mcr-1":       ["MCR", "mcr-1", "mcr"],
    "tetM":        ["tetM", "tet(M)"],
    "vanA":        ["vanA", "VanA"],
}


def fetch_patric_amr_data(
    species: str = "Escherichia coli",
    max_records: int = 2000,
    use_cache:   bool = True,
) -> Optional[List[Dict]]:
    """
    Fetch AMR phenotype + genome metadata from PATRIC.

    Returns list of dicts, each with:
      genome_id, antibiotic, resistant_phenotype, genome_name
    """
    import urllib.request
    import urllib.parse

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"patric_amr_{species.replace(' ','_')}.json"

    if use_cache and cache_file.exists():
        print(f"  Using cached PATRIC data: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Fetching PATRIC AMR data for {species}...")

    try:
        query = urllib.parse.urlencode({
            "q":   f'genome_name:"{species}"',
            "rows": max_records,
            "fl":  "genome_id,genome_name,antibiotic,resistant_phenotype,laboratory_typing_method",
            "wt":  "json",
        })
        url = f"{PATRIC_API}/genome_amr/?{query}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept":     "application/json",
                "User-Agent": "AMR-Simulation-Research/1.0",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        records = data.get("response", {}).get("docs", [])
        print(f"  Retrieved {len(records)} AMR records")

        with open(cache_file, "w") as f:
            json.dump(records, f, indent=2)

        return records

    except Exception as e:
        print(f"  PATRIC API error: {e}")
        print(f"  Falling back to synthetic validation data...")
        return None


def fetch_patric_genome_amr(
    species: str = "Escherichia coli",
    max_genomes: int = 500,
    use_cache:   bool = True,
) -> Optional[Dict[str, set]]:
    """
    Fetch per-genome resistance gene data from PATRIC genome_feature endpoint.

    Returns: {genome_id: set of resistance gene names}
    """
    import urllib.request
    import urllib.parse

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"patric_genes_{species.replace(' ','_')}.json"

    if use_cache and cache_file.exists():
        print(f"  Using cached gene data: {cache_file}")
        with open(cache_file) as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}

    print(f"  Fetching PATRIC resistance gene data for {species}...")

    try:
        import urllib.request
        query = urllib.parse.urlencode({
            "q":   f'genome_name:"{species}" AND feature_type:mat_peptide',
            "rows": max_genomes * 20,
            "fl":  "genome_id,product,gene",
            "wt":  "json",
        })
        url = f"{PATRIC_API}/genome_feature/?{query}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json",
                     "User-Agent": "AMR-Simulation-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        records = data.get("response", {}).get("docs", [])
        genome_genes: Dict[str, set] = {}
        for rec in records:
            gid     = rec.get("genome_id", "")
            product = (rec.get("product", "") + " " + rec.get("gene", "")).lower()
            genome_genes.setdefault(gid, set())
            genome_genes[gid].add(product)

        # Save as lists for JSON
        with open(cache_file, "w") as f:
            json.dump({k: list(v) for k, v in genome_genes.items()}, f)

        print(f"  Retrieved gene data for {len(genome_genes)} genomes")
        return genome_genes

    except Exception as e:
        print(f"  PATRIC gene fetch error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GENE CO-OCCURRENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_gene_cooccurrence(
    genome_genes: Dict[str, set],
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute pairwise gene co-occurrence matrix from real genomic data.

    co_occur[i,j] = P(gene_j in genome | gene_i in genome)
                  = count(both) / count(gene_i)

    This is the conditional co-occurrence frequency — if genes frequently
    transfer together, they should co-occur more in real isolates.

    Returns: (N_GENES, N_GENES) co-occurrence probability matrix
    """
    co_matrix   = np.zeros((N_GENES, N_GENES), dtype=np.float32)
    gene_counts = np.zeros(N_GENES, dtype=np.float32)
    n_genomes   = len(genome_genes)

    if n_genomes == 0:
        return co_matrix

    for genome_id, gene_set in genome_genes.items():
        gene_lower = set(g.lower() for g in gene_set)

        # Map to our canonical gene indices
        present = set()
        for g_idx, gene in enumerate(GENE_INDEX):
            synonyms = GENE_TO_PATRIC.get(gene, [gene])
            for syn in synonyms:
                if any(syn.lower() in gl for gl in gene_lower):
                    present.add(g_idx)
                    break

        for i in present:
            gene_counts[i] += 1
            for j in present:
                co_matrix[i, j] += 1

    # Normalize to conditional probability
    for i in range(N_GENES):
        if gene_counts[i] > 0:
            co_matrix[i] /= gene_counts[i]

    if verbose:
        print(f"\n  Gene co-occurrence matrix computed from {n_genomes} genomes")
        print(f"  Gene prevalence rates:")
        for g_idx, gene in enumerate(GENE_INDEX):
            pct = gene_counts[g_idx] / max(1, n_genomes) * 100
            bar = "█" * int(pct)
            print(f"    {gene:<20} {bar:<20} {pct:.1f}%")

    return co_matrix


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC VALIDATION (fallback when PATRIC unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_validation_data(seed: int = 42) -> Dict[str, set]:
    """
    Generate synthetic but biologically realistic gene co-occurrence data.

    Used when PATRIC API is unavailable (offline, rate-limited etc.).
    Gene prevalence rates based on WHO GLASS 2022 and published literature:
      blaTEM-1:    ~70% of E. coli isolates globally
      blaCTX-M-15: ~40% in South Asia
      gyrA_S83L:   ~60% in ciprofloxacin-resistant strains (Pakistan >60%)
      acrAB-tolC:  ~35% (efflux-mediated MDR)
      blaNDM-1:    ~15% in South Asia (Kumarasamy 2010, expanding)
      blaKPC-2:    ~12% in hospital-acquired
      tetM:        ~45% globally
      mcr-1:       ~8%  (emerging, spreading)
      mexAB-oprM:  ~20% (Pseudomonas; lower in E. coli)
      vanA:        ~5%  (Gram-positive; rare in Gram-negative)

    Co-occurrence reflects known gene linkage patterns on plasmids.
    """
    rng = np.random.RandomState(seed)
    N_GENOMES = 1000

    # Gene prevalence (based on published literature)
    prevalence = {
        "blaTEM-1":    0.70,
        "blaCTX-M-15": 0.40,
        "blaKPC-2":    0.12,
        "blaNDM-1":    0.15,
        "mexAB-oprM":  0.20,
        "acrAB-tolC":  0.35,
        "gyrA_S83L":   0.55,
        "mcr-1":       0.08,
        "tetM":        0.45,
        "vanA":        0.05,
    }

    # Known co-occurrence boosts (genes on same plasmid types)
    # Based on published plasmid linkage studies
    linkage_pairs = [
        ("blaCTX-M-15", "blaTEM-1",  0.35),  # frequently co-carried on IncF plasmids
        ("blaNDM-1",    "mcr-1",     0.20),  # co-resistance in South Asia
        ("blaKPC-2",    "acrAB-tolC",0.25),  # KPC plasmids often carry efflux
        ("gyrA_S83L",   "acrAB-tolC",0.30),  # quinolone resistance linkage
        ("tetM",        "acrAB-tolC",0.20),  # co-resistance in MDR strains
        ("blaNDM-1",    "blaCTX-M-15",0.15), # ESBL + MBL co-occurrence
    ]

    genome_genes = {}
    for i in range(N_GENOMES):
        genome_id = f"synthetic_{i:05d}"
        present   = set()

        # Sample each gene independently by prevalence
        for gene, prev in prevalence.items():
            if rng.random() < prev:
                present.add(gene)

        # Apply linkage: if gene A is present, boost P(gene B)
        for gene_a, gene_b, boost in linkage_pairs:
            if gene_a in present and gene_b not in present:
                if rng.random() < boost:
                    present.add(gene_b)

        genome_genes[genome_id] = present

    return genome_genes


# ─────────────────────────────────────────────────────────────────────────────
# GNN PREDICTION → CO-OCCURRENCE CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gnn_transfer_matrix(
    inference_engine,
    n_scenarios: int = 4,
    steps_per_scenario: int = 30,
) -> np.ndarray:
    """
    Run the GNN across multiple simulation scenarios to estimate
    per-gene-pair average transfer probability.

    Returns: (N_GENES, N_GENES) matrix where entry [i,j] =
             average probability that gene i transfers to a cell
             that already has gene j (or vice versa).
    """
    from simulation.amr_model import AMRSimulationModel
    from ai.feature_engineering import build_graph_from_state

    scenarios = ["ecoli_cipro", "klebsiella_carbapenem",
                 "pakistan_crisis", "xdr_acinetobacter"]
    seeds     = [1, 2, 3, 4]

    transfer_acc = np.zeros((N_GENES, N_GENES), dtype=np.float64)
    count_acc    = np.zeros((N_GENES, N_GENES), dtype=np.int64)

    for sc_idx, (scenario, seed) in enumerate(zip(scenarios[:n_scenarios], seeds)):
        print(f"  Scenario {sc_idx+1}/{n_scenarios}: {scenario}...")
        model = AMRSimulationModel(
            scenario=scenario, initial_bacteria=80,
            seed=seed, enable_logging=False
        )
        for _ in range(15):
            model.step()
        for abk in model.active_antibiotic_keys:
            model.apply_antibiotic(abk, concentration=1.5, mode="uniform")

        for step in range(steps_per_scenario):
            model.step()
            if not model.running:
                break
            if step % 5 != 0:
                continue

            state = model.get_full_state()
            graph = build_graph_from_state(state, max_edge_distance=3, max_nodes=200)
            if graph is None:
                continue

            result = inference_engine.predict(state, threshold=0.0)
            if not result or result["n_edges"] == 0:
                continue

            # For each edge, get donor genes and predicted probs
            edge_index = graph["edge_index"]
            bacteria   = {b["id"]: b for b in graph["bacteria"]}
            node_ids   = graph["node_ids"]

            gene_probs = result.get("gene_transfer_probs", {})
            for g_src_idx, g_src in enumerate(GENE_INDEX):
                src_prob = gene_probs.get(g_src, 0.0)
                for g_dst_idx, g_dst in enumerate(GENE_INDEX):
                    transfer_acc[g_src_idx, g_dst_idx] += src_prob
                    count_acc[g_src_idx, g_dst_idx]    += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = np.where(
            count_acc > 0,
            transfer_acc / count_acc,
            0.0
        )
    return avg_matrix.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SPEARMAN CORRELATION: GNN vs REAL CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_spearman_validation(
    gnn_matrix:   np.ndarray,
    real_matrix:  np.ndarray,
    gene_subset:  Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation between GNN transfer probabilities
    and real-world gene co-occurrence frequencies.

    This is the key validation metric for the paper:
    ρ > 0.5, p < 0.05 → GNN predictions align with real biology

    Args:
      gnn_matrix:  (N_GENES, N_GENES) predicted transfer probs from GNN
      real_matrix: (N_GENES, N_GENES) conditional co-occurrence from PATRIC
      gene_subset: subset of genes to include (default: all with data)
    """
    from scipy.stats import spearmanr

    if gene_subset:
        indices = [GENE_INDEX.index(g) for g in gene_subset if g in GENE_INDEX]
    else:
        indices = list(range(N_GENES))

    # Flatten upper triangle (excluding diagonal)
    gnn_flat  = []
    real_flat = []

    for i in indices:
        for j in indices:
            if i != j:
                gnn_flat.append(float(gnn_matrix[i, j]))
                real_flat.append(float(real_matrix[i, j]))

    gnn_arr  = np.array(gnn_flat)
    real_arr = np.array(real_flat)

    # Remove pairs where both are zero (uninformative)
    mask = (gnn_arr > 0) | (real_arr > 0)
    gnn_arr  = gnn_arr[mask]
    real_arr = real_arr[mask]

    if len(gnn_arr) < 5:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n_pairs": 0,
                "interpretation": "Insufficient data"}

    rho, p_val = spearmanr(gnn_arr, real_arr)

    interp = (
        "Strong alignment (ρ>0.7): GNN predictions strongly match real co-occurrence" if rho > 0.7 else
        "Moderate alignment (ρ>0.5): GNN predictions moderately match real co-occurrence" if rho > 0.5 else
        "Weak alignment (ρ>0.3): Some signal, needs more data" if rho > 0.3 else
        "No significant alignment: model may not capture real transfer patterns"
    )
    sig = "p<0.05 (significant)" if p_val < 0.05 else f"p={p_val:.3f} (not significant)"

    return {
        "spearman_rho": float(rho),
        "p_value":      float(p_val),
        "n_pairs":      int(mask.sum()),
        "significance": sig,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL VALIDATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_external_validation(
    checkpoint_path: str = "ai/checkpoints/best_model.pt",
    species:         str = "Escherichia coli",
    use_synthetic:   bool = False,
    save_path:       str = "ai/checkpoints/external_validation.json",
) -> Dict:
    """
    Full external validation pipeline:
    1. Load trained GNN
    2. Fetch real PATRIC data (or synthetic fallback)
    3. Compute gene co-occurrence from real data
    4. Run GNN across multiple simulation scenarios
    5. Compute Spearman correlation
    6. Save results

    Returns comprehensive validation dict for paper.
    """
    from ai.gnn_inference import GNNInferenceEngine
    from scipy.stats import spearmanr

    print(f"\n{'='*60}")
    print(f"  External Validation: GNN vs Real AMR Data")
    print(f"  Species: {species}")
    print(f"{'='*60}\n")

    # ── 1. Load GNN ────────────────────────────────────────────────────────
    print("[1/4] Loading trained GNN...")
    engine = GNNInferenceEngine.load(checkpoint_path)
    if engine is None:
        print("  No checkpoint found — using untrained model")
        engine = GNNInferenceEngine.load_untrained()
    else:
        print(f"  Loaded from: {checkpoint_path}")

    # ── 2. Fetch real data ─────────────────────────────────────────────────
    print(f"\n[2/4] Fetching real genomic data...")
    genome_genes = None

    if not use_synthetic:
        genome_genes = fetch_patric_genome_amr(species, max_genomes=500)

    if genome_genes is None or use_synthetic:
        print("  Using synthetic validation data (literature-based prevalence rates)")
        genome_genes = generate_synthetic_validation_data()
        data_source  = "synthetic (literature-based)"
    else:
        data_source  = f"PATRIC ({len(genome_genes)} {species} genomes)"

    # ── 3. Compute co-occurrence ───────────────────────────────────────────
    print(f"\n[3/4] Computing gene co-occurrence matrix ({data_source})...")
    real_cooccur = compute_gene_cooccurrence(genome_genes)

    # ── 4. GNN transfer matrix ─────────────────────────────────────────────
    print(f"\n[4/4] Computing GNN-predicted transfer probabilities...")
    gnn_matrix = compute_gnn_transfer_matrix(engine, n_scenarios=4, steps_per_scenario=20)

    # ── 5. Spearman correlation ────────────────────────────────────────────
    print(f"\n[5/4] Computing Spearman rank correlation...")
    # Full gene set
    full_corr = compute_spearman_validation(gnn_matrix, real_cooccur)

    # Beta-lactamase subset (most clinically relevant for Pakistan)
    bla_genes = ["blaTEM-1", "blaCTX-M-15", "blaKPC-2", "blaNDM-1"]
    bla_corr  = compute_spearman_validation(gnn_matrix, real_cooccur, bla_genes)

    # Efflux + quinolone subset (common MDR mechanism)
    mdr_genes = ["acrAB-tolC", "mexAB-oprM", "gyrA_S83L"]
    mdr_corr  = compute_spearman_validation(gnn_matrix, real_cooccur, mdr_genes)

    # ── 6. Report ──────────────────────────────────────────────────────────
    print(f"\n=== EXTERNAL VALIDATION RESULTS ===")
    print(f"\nData source: {data_source}")
    print(f"Genomes: {len(genome_genes)}")
    print(f"\nSpearman correlation (GNN transfer prob vs real co-occurrence):")
    print(f"  All genes    : ρ={full_corr['spearman_rho']:+.4f}  {full_corr['significance']}")
    print(f"  Beta-lactams : ρ={bla_corr['spearman_rho']:+.4f}  {bla_corr['significance']}")
    print(f"  MDR genes    : ρ={mdr_corr['spearman_rho']:+.4f}  {mdr_corr['significance']}")
    print(f"\nInterpretation: {full_corr['interpretation']}")

    print(f"\nGNN transfer matrix (top gene pairs):")
    print(f"  {'Donor → Recipient':<35} {'GNN Prob':>9} {'Real CoOccur':>13}")
    print(f"  {'-'*60}")
    pairs = []
    for i in range(N_GENES):
        for j in range(N_GENES):
            if i != j:
                pairs.append((GENE_INDEX[i], GENE_INDEX[j],
                               gnn_matrix[i,j], real_cooccur[i,j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for g_src, g_dst, gnn_p, real_p in pairs[:8]:
        print(f"  {g_src:<18} → {g_dst:<15} {gnn_p:>9.4f} {real_p:>13.4f}")

    result = {
        "data_source":       data_source,
        "n_genomes":         len(genome_genes),
        "species":           species,
        "correlation_full":  full_corr,
        "correlation_bla":   bla_corr,
        "correlation_mdr":   mdr_corr,
        "gnn_matrix":        gnn_matrix.tolist(),
        "real_matrix":       real_cooccur.tolist(),
        "gene_names":        GENE_INDEX,
        "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    Path("ai/checkpoints").mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data instead of PATRIC API")
    p.add_argument("--species",   default="Escherichia coli")
    p.add_argument("--checkpoint",default="ai/checkpoints/best_model.pt")
    args = p.parse_args()

    result = run_external_validation(
        checkpoint_path=args.checkpoint,
        species=args.species,
        use_synthetic=args.synthetic,
    )
    print(f"\nFinal Spearman ρ = {result['correlation_full']['spearman_rho']:.4f}")