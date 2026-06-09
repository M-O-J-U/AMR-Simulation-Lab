import csv
import json
import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.feature_engineering import GENE_INDEX, N_GENES

# Maps BV-BRC gene/product strings to our canonical GENE_INDEX names
GENE_SYNONYMS = {
    "blaTEM-1":    ["blaTEM", "TEM-1", "TEM"],
    "blaCTX-M-15": ["CTX-M-15", "CTX-M"],
    "blaKPC-2":    ["KPC-2", "KPC"],
    "blaNDM-1":    ["NDM-1", "NDM"],
    "mexAB-oprM":  ["mexA", "mexB", "OprM"],
    "acrAB-tolC":  ["acrA", "acrB", "TolC"],
    "gyrA_S83L":   ["gyrA"],
    "mcr-1":       ["mcr-1", "MCR"],
    "tetM":        ["tet(M)", "tetM"],
    "vanA":        ["vanA"],
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV READER
# ─────────────────────────────────────────────────────────────────────────────

def read_bvbrc_csv(filepath: str) -> dict:
    """
    Read a BV-BRC AMR Phenotypes CSV file.
    Returns {genome_id: set(canonical_gene_names)}.

    BV-BRC CSV columns include: Genome ID, Genome Name, Antibiotic,
    Resistant Phenotype, Laboratory Typing Method, Laboratory Typing Platform,
    Vendor, Laboratory Typing Method Version, Testing Standard,
    MIC Comparator, MIC Value, Testing Standard Year, Assertion, Gene, Evidence
    """
    genome_genes = {}
    path = Path(filepath)
    if not path.exists():
        return {}

    print(f"  Reading: {filepath} ({path.stat().st_size // 1024} KB)")

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        # Detect delimiter
        sample = f.read(2048)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)

        rows_read = 0
        for row in reader:
            rows_read += 1
            # Try multiple possible column names for genome_id
            gid = (row.get("Genome ID") or row.get("genome_id") or
                   row.get("genome ID") or "").strip()
            if not gid:
                continue

            # Combine all text columns for gene matching
            text = " ".join(str(v) for v in row.values()).lower()

            genome_genes.setdefault(gid, set())
            for canonical, synonyms in GENE_SYNONYMS.items():
                for syn in synonyms:
                    if syn.lower() in text:
                        genome_genes[gid].add(canonical)
                        break

    print(f"  Rows: {rows_read} | Genomes with matched genes: {len(genome_genes)}")
    return genome_genes


def read_bvbrc_txt(filepath: str) -> dict:
    """
    Read BV-BRC tab-delimited .txt file (same format as CSV, just .txt extension).
    """
    path = Path(filepath)
    if not path.exists():
        return {}
    # Same logic as CSV — just force tab delimiter
    genome_genes = {}
    print(f"  Reading: {filepath} ({path.stat().st_size // 1024} KB)")
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows_read = 0
        for row in reader:
            rows_read += 1
            gid = (row.get("Genome ID") or row.get("genome_id") or "").strip()
            if not gid:
                continue
            text = " ".join(str(v) for v in row.values()).lower()
            genome_genes.setdefault(gid, set())
            for canonical, synonyms in GENE_SYNONYMS.items():
                for syn in synonyms:
                    if syn.lower() in text:
                        genome_genes[gid].add(canonical)
                        break
        print(f"  Rows: {rows_read} | Genomes with matched genes: {len(genome_genes)}")
    return genome_genes


def load_local_files() -> dict:
    """
    Auto-detect all local BV-BRC data files and load them.
    Looks for: ecoli_amr.csv, klebsiella_amr.csv, *.csv, *.txt in project root.
    """
    all_genes = {}
    found = []

    # Priority: named files first, then any CSV/TXT
    candidates = [
        "ecoli_amr.csv", "ecoli_amr.txt",
        "klebsiella_amr.csv", "klebsiella_amr.txt",
        "bvbrc_amr.csv", "patric_amr.csv",
        "amr.csv", "amr.txt",
    ]
    for fname in candidates:
        if Path(fname).exists():
            found.append(fname)

    # Also scan for any .csv or .txt not already in list
    for p in Path(".").glob("*.csv"):
        if p.name not in found:
            found.append(p.name)
    for p in Path(".").glob("*.txt"):
        if p.name not in found and "requirements" not in p.name.lower():
            found.append(p.name)

    if not found:
        return {}

    print(f"  Found local files: {found}")
    for fname in found:
        if fname.endswith(".txt"):
            genes = read_bvbrc_txt(fname)
        else:
            genes = read_bvbrc_csv(fname)
        all_genes.update(genes)

    return all_genes


# ─────────────────────────────────────────────────────────────────────────────
# CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_cooccurrence(genome_genes: dict) -> np.ndarray:
    co  = np.zeros((N_GENES, N_GENES), dtype=np.float64)
    cnt = np.zeros(N_GENES, dtype=np.float64)
    n   = len(genome_genes)

    for genes in genome_genes.values():
        present = [GENE_INDEX.index(g) for g in genes if g in GENE_INDEX]
        for i in present:
            cnt[i] += 1
            for j in present:
                co[i, j] += 1
    for i in range(N_GENES):
        if cnt[i] > 0:
            co[i] /= cnt[i]

    print(f"\n  Gene prevalence across {n} real isolates:")
    for g_idx, gene in enumerate(GENE_INDEX):
        pct = cnt[g_idx] / max(1, n) * 100
        bar = "█" * int(pct / 3)
        print(f"    {gene:<20} {bar:<20} {pct:.1f}%  ({int(cnt[g_idx])} isolates)")

    return co


# ─────────────────────────────────────────────────────────────────────────────
# GNN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gnn_gene_probs() -> np.ndarray:
    from ai.gnn_inference import GNNInferenceEngine
    from simulation.amr_model import AMRSimulationModel

    engine = GNNInferenceEngine.load("ai/checkpoints/best_model.pt")
    if engine is None:
        print("  WARNING: no checkpoint found — using untrained model (predictions will be random)")
        engine = GNNInferenceEngine.load_untrained()

    acc = np.zeros(N_GENES, dtype=np.float64)
    n   = 0

    for scenario in ["ecoli_cipro", "klebsiella_carbapenem",
                     "pakistan_crisis", "xdr_acinetobacter"]:
        print(f"  {scenario}...", end=" ", flush=True)
        model = AMRSimulationModel(
            scenario=scenario, initial_bacteria=100,
            seed=999, enable_logging=False
        )
        for _ in range(20): model.step()
        for abk in model.active_antibiotic_keys:
            model.apply_antibiotic(abk, concentration=1.5, mode="uniform")
        for step in range(40):
            model.step()
            if not model.running: break
            if step % 5 != 0: continue
            state  = model.get_full_state()
            result = engine.predict(state, threshold=0.0)
            if result and result.get("model_ready"):
                for g_idx, gene in enumerate(GENE_INDEX):
                    acc[g_idx] += result["gene_transfer_probs"].get(gene, 0.0)
                n += 1
        print("done")

    if n > 0:
        acc /= n

    print(f"\n  GNN mean transfer probs ({n} predictions averaged):")
    for g_idx, gene in enumerate(GENE_INDEX):
        bar = "█" * int(acc[g_idx] * 200)
        print(f"    {gene:<20} {bar:<20} {acc[g_idx]:.4f}")

    return acc


# ─────────────────────────────────────────────────────────────────────────────
# SPEARMAN TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_spearman(gnn: np.ndarray, real: np.ndarray) -> dict:
    prev  = np.diag(real)
    valid = (gnn > 0) & (prev > 0)

    if valid.sum() < 4:
        return {
            "rho": 0.0, "p_value": 1.0,
            "n_genes": int(valid.sum()), "valid_genes": [],
            "interpretation": "Not enough genes with data in both vectors (need >= 4).",
        }

    rho, p = spearmanr(gnn[valid], prev[valid])
    genes  = [GENE_INDEX[i] for i in range(N_GENES) if valid[i]]

    print(f"\n  Per-gene comparison (GNN transfer prob vs real prevalence):")
    print(f"  {'Gene':<20} {'GNN Prob':>10} {'Real Prev':>10}")
    print(f"  {'-'*43}")
    for i in range(N_GENES):
        if valid[i]:
            print(f"  {GENE_INDEX[i]:<20} {gnn[i]:>10.4f} {prev[i]:>10.4f}")

    if rho > 0.5 and p < 0.05:
        interp = (f"SIGNIFICANT POSITIVE (rho={rho:.3f}, p={p:.4f}) — "
                  "GNN predictions align with real clinical gene prevalence. "
                  "Update paper Section 4.4 with this result.")
    elif rho > 0.3 and p < 0.10:
        interp = (f"MODERATE (rho={rho:.3f}, p={p:.4f}) — "
                  "Preliminary evidence. Report with caveats in paper.")
    else:
        interp = (f"NOT SIGNIFICANT (rho={rho:.3f}, p={p:.4f}) — "
                  "Keep negative-result paragraph in paper Section 4.4 as written.")

    return {
        "rho": float(rho), "p_value": float(p),
        "n_genes": int(valid.sum()), "valid_genes": genes,
        "interpretation": interp,
        "gnn_probs": {GENE_INDEX[i]: float(gnn[i]) for i in range(N_GENES)},
        "real_prev": {GENE_INDEX[i]: float(prev[i]) for i in range(N_GENES)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  BV-BRC External Validation")
    print("="*60)

    Path("ai/checkpoints").mkdir(parents=True, exist_ok=True)
    all_genes = {}

    # ── Load local CSV/TXT files ──────────────────────────────────────────
    print("\n[1/4] Loading local BV-BRC data files...")
    all_genes = load_local_files()

    if len(all_genes) < 20:
        print(f"\n  Only {len(all_genes)} genomes loaded from local files.")
        print("  To get data:")
        print("  1. E. coli:    https://www.bv-brc.org/view/Taxonomy/562#view_tab=amr")
        print("                 Click Download → CSV → save as: ecoli_amr.csv")
        print("  2. Klebsiella: https://www.bv-brc.org/view/Taxonomy/573#view_tab=amr")
        print("                 Click Download → CSV → save as: klebsiella_amr.csv")
        print("  3. Put both files in your project root (same folder as main.py)")
        print("  4. Run this script again")
        print("\n  If the files are already here, check their column headers match BV-BRC format.")
        print("  Print first 3 lines of your file:")
        for p in list(Path(".").glob("*.csv")) + list(Path(".").glob("*.txt")):
            print(f"\n  --- {p.name} ---")
            with open(p, encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if i >= 3: break
                    print(f"  {line.rstrip()[:120]}")
        sys.exit(1)

    print(f"\n  Total isolates loaded: {len(all_genes)}")

    # ── Co-occurrence ─────────────────────────────────────────────────────
    print(f"\n[2/4] Computing gene co-occurrence matrix...")
    cooccur = compute_cooccurrence(all_genes)

    # ── GNN predictions ───────────────────────────────────────────────────
    print("\n[3/4] Running GNN inference across 4 simulation scenarios...")
    gnn_probs = compute_gnn_gene_probs()

    # ── Spearman ─────────────────────────────────────────────────────────
    print("\n[4/4] Spearman rank correlation test...")
    result = run_spearman(gnn_probs, cooccur)

    print(f"\n{'='*60}")
    print(f"  RESULT: {result['interpretation']}")
    print(f"{'='*60}")

    output = {
        "n_real_isolates": len(all_genes),
        "spearman_result": result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "BV-BRC local CSV download",
    }
    out = "ai/checkpoints/patric_validation.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {out}")
    print(f"\n  VALUES FOR PAPER:")
    print(f"  rho = {result['rho']:.4f}")
    print(f"  p   = {result['p_value']:.4f}")
    print(f"  n_genes    = {result['n_genes']}")
    print(f"  n_isolates = {len(all_genes)}")

    if result['rho'] > 0.4 and result['p_value'] < 0.05:
        print(f"\n  SIGNIFICANT — send me these numbers and I will update the LaTeX.")
    else:
        print(f"\n  NOT SIGNIFICANT — paper Section 4.4 already handles this correctly.")
        print(f"  Send me all outputs (baselines + calibrate) and I will write the LaTeX.")


if __name__ == "__main__":
    main()