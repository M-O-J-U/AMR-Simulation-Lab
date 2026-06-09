# AMR Simulation Lab

**Antimicrobial Resistance Agent-Based Simulation + GNN Resistance Gene Transfer Predictor**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-92%20passing-brightgreen.svg)]()
[![AUROC](https://img.shields.io/badge/AUROC-0.9934-blue.svg)]()

**Abdul Moiz Muhammad** — Independent Researcher, Pakistan  
ORCID: [0009-0006-2795-5271](https://orcid.org/0009-0006-2795-5271) | 
[LinkedIn](https://linkedin.com/in/abdul-moiz-muhammad) |
[Paper preprint](https://doi.org/10.5281/zenodo.20347832)

---

## What this is

A complete end-to-end research pipeline:

1. **Agent-based simulation** of bacterial populations — each bacterium is a live agent with SOS response, HGT conjugation, biofilm formation, persister switching, and real pharmacodynamics
2. **GNN resistance predictor** — Graph Attention Network trained on simulation-generated HGT events, achieving AUROC 0.9934 across 8 clinically critical resistance genes
3. **REST API + live frontend** — FastAPI server with WebSocket streaming, dark sci-fi canvas UI
4. **92 automated tests** — 5-layer test suite from data integrity to science validation

All resistance gene parameters from [CARD](https://card.mcmaster.ca/). Antibiotic breakpoints from [EUCAST 2023](https://www.eucast.org/).

---

## Results

| Gene | AUROC | Clinical relevance |
|------|-------|-------------------|
| acrAB-tolC | 0.998 | MDR efflux pump, SOS-inducible |
| tetM | 0.997 | Transposon-borne, global spread |
| blaTEM-1 | 0.996 | Most common AMR gene globally |
| blaCTX-M-15 | 0.994 | Dominant ESBL in South Asia |
| gyrA_S83L | 0.992 | Ciprofloxacin resistance |
| blaKPC-2 | 0.991 | Carbapenem resistance |
| blaNDM-1 | 0.989 | Pakistan-origin, pandemic spread |
| mcr-1 | 0.989 | Last-resort colistin resistance |

**Macro AUROC: 0.9934 | AUPRC: 0.0602 (50× random baseline) | ECE: 0.0068**

Compared against Logistic Regression (AUROC 0.9746), Random Forest (0.9896), and frequency baseline (0.5000).

---

## Project structure

```
amr_sim/
├── main.py                      # Entry point — all commands
├── requirements.txt
│
├── data/
│   └── card_loader.py           # CARD resistance genes, germ profiles, antibiotic profiles
│
├── core/
│   └── bacterium_agent.py       # Mesa Agent — SOS, HGT, biofilm, persister, chemotaxis
│
├── simulation/
│   ├── amr_model.py             # Mesa Model — grid, diffusion, nutrient, HGT recording
│   └── sim_logger.py            # Structured JSON + human-readable logging
│
├── ai/
│   ├── feature_engineering.py   # 36-dim node features, 8-dim edge features, graph construction
│   ├── gnn_model.py             # AMRResistanceGNN — NodeEncoder, EdgeEncoder, 3×GAT, EdgeHead
│   ├── gnn_trainer.py           # Training loop — AdamW, cosine LR, early stopping, metrics
│   ├── gnn_inference.py         # Live inference engine — MC Dropout, treatment advisory
│   ├── resistance_analytics.py  # MIC estimation, treatment recommendation, population genetics
│   ├── baselines.py             # LR, RF, frequency baseline comparison
│   ├── threshold_calibration.py # PR curves, ECE, optimal threshold search
│   └── external_validation.py   # PATRIC/BV-BRC co-occurrence validation
│
├── api/
│   └── server.py                # FastAPI — 15+ REST endpoints + WebSocket + GNN endpoints
│
├── frontend/
│   └── index.html               # Dark canvas UI — 5 render modes, live events, HGT flashes
│
├── tests/
│   ├── test_simulation.py       # 50 tests — data integrity, agent biology, model dynamics
│   └── test_gnn.py              # 42 tests — feature engineering, GNN architecture, inference
│
└── run_patric_validation.py     # Standalone BV-BRC external validation script
```

---

## Quickstart

```bash
git clone https://github.com/M-O-J-U/amr-simulation-lab
cd amr-simulation-lab
pip install -r requirements.txt

# Validate simulation biology (4 checks, ~30 seconds)
python main.py validate

# Run headless simulation, print population table
python main.py headless --scenario pakistan_crisis --steps 50

# Train the GNN (requires GPU, ~45 min full / ~5 min quick)
python main.py train-gnn --full --epochs 60
python main.py train-gnn --epochs 10          # quick test

# Run baseline comparison
python main.py baselines --full

# Calibrate decision thresholds
python main.py calibrate

# Start API server + open frontend
python main.py server
# → open frontend/index.html in browser

# Run all 92 tests
python main.py test
```

---

## Simulation scenarios

| Key | Description |
|-----|-------------|
| `validation` | E. coli — no antibiotics (growth validation) |
| `ecoli_cipro` | E. coli + Ciprofloxacin |
| `klebsiella_carbapenem` | Klebsiella pneumoniae + Meropenem |
| `xdr_acinetobacter` | XDR Acinetobacter + Colistin (last resort) |
| `mrsa_hospital` | MRSA + Vancomycin |
| `pakistan_crisis` | E. coli + Klebsiella + Cipro + Meropenem |
| `multi_species` | E. coli + Klebsiella + Pseudomonas + Cipro |

---

## API endpoints

```
GET  /state              Full simulation state
GET  /stats              Lightweight stats
POST /step               Advance N steps
POST /apply_antibiotic   Apply antibiotic (uniform/gradient/spot/zone)
POST /remove_antibiotic  Clear antibiotic
POST /spawn_bacteria     Add bacteria mid-simulation
POST /reset              Reset to scenario

GET  /gnn/status         Model loaded, n predictions, avg latency
POST /gnn/predict        GNN inference on current state
POST /gnn/predict_mc     MC Dropout — with uncertainty estimates
POST /gnn/advisory       Treatment advisory (GNN + analytics)
POST /gnn/train          Trigger training in background
GET  /gnn/results        Last training results JSON

GET  /analytics/mic      MIC distribution for current population
GET  /analytics/diversity Shannon diversity index
GET  /analytics/recommend Treatment recommendation
GET  /analytics/emergence Resistance emergence prediction
GET  /analytics/logs      Structured simulation log (last 50 entries)

WS   /ws                 WebSocket real-time stream
```

---

## Requirements

```
Python 3.10+
mesa>=3.0
torch>=2.0
torch-geometric>=2.3
fastapi>=0.100
uvicorn
scikit-learn
scipy
numpy
```

Full list in `requirements.txt`. GPU optional but recommended for training.

---

## Bacteria simulated

| Species | WHO Priority | Key feature |
|---------|-------------|-------------|
| *Escherichia coli* | HIGH | ~70% ciprofloxacin-resistant in Pakistan |
| *Klebsiella pneumoniae* | CRITICAL | NDM-1 origin, high biofilm |
| *Acinetobacter baumannii* | CRITICAL | XDR, 95% biofilm potential |
| *Pseudomonas aeruginosa* | CRITICAL | Highest mutation rate, intrinsic MDR |
| *S. aureus* (MRSA) | HIGH | Vancomycin is primary treatment |

---

## Resistance genes modelled

All parameters from CARD v3.2 (McArthur et al. 2023, *Nucleic Acids Research*):

`blaTEM-1` · `blaCTX-M-15` · `blaKPC-2` · `blaNDM-1` · `mexAB-oprM` · `acrAB-tolC` · `gyrA_S83L` · `mcr-1` · `tetM` · `vanA`

---

## Simulation biology validation

| Test | Simulated | Published |
|------|-----------|-----------|
| E. coli doubling time | 30→120 cells / 20 steps | ~20 min in rich media |
| Ciprofloxacin killing at 6×MBC | 30% reduction / 20 steps | 1–2 log kill / 6 h |
| gyrA-resistant vs susceptible survival | 13× higher | 10–100× at therapeutic conc. |
| blaTEM-1 HGT spread | 20→150 carriers / 25 steps | 10⁻³–10⁻⁵ per cell/gen |

---

## Citation

If you use this work, please cite:

```bibtex
@article{muhammad2026amrgnn,
  author  = {Muhammad, Abdul Moiz},
  title   = {{AMRResistanceGNN}: Predicting Horizontal Gene Transfer of
             Antimicrobial Resistance Genes via Graph Attention Networks
             on Agent-Based Simulation Data},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review. Preprint: https://doi.org/10.5281/zenodo.20347832}
}
```

---

## Related work

- **Paper 1 (companion simulation paper):** [Zenodo](https://doi.org/10.5281/zenodo.19982081)
- **CARD database:** McArthur et al. (2023), *Nucleic Acids Research*
- **Graph Attention Networks:** Veličković et al. (2018), ICLR
- **Persister cells:** Lewis (2010), *Annual Review of Microbiology*
- **Biofilm resistance:** Hoiby et al. (2010), *Int J Antimicrob Agents*

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Pakistan has one of the world's highest AMR burdens. NDM-1 was first described
in a patient with Pakistan/India connections. This project is directly motivated
by the South Asian AMR crisis.*
