# AMR Simulation Lab 🦠

**Agent-Based Simulation of Antimicrobial Resistance Dynamics**

[![Python](https://img.shields.io/badge/Python-3.12%2B%20(tested%203.14)-blue?logo=python)](https://python.org)
[![Mesa](https://img.shields.io/badge/Mesa-ABM-orange)](https://mesa.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)](https://fastapi.tiangolo.com)
[![WHO Priority](https://img.shields.io/badge/WHO%20Priority%20Pathogens-CRITICAL-red)](https://www.who.int/publications/i/item/WHO-EMP-IAU-2017.12)

A scientifically grounded, interactive agent-based simulation of antimicrobial resistance (AMR) dynamics. Bacteria live, grow, mutate, form biofilms, transfer resistance genes via horizontal gene transfer (HGT), and die — all driven by real resistance mechanisms from the [CARD database](https://card.mcmaster.ca/).

Built as part of a research pipeline targeting **AMR AI for South Asian clinical contexts**, with particular relevance to Pakistan's documented AMR burden (WHO GLASS 2022).

---

## Demo

> Each dot is a living bacterium. Red outlines = resistance genes. Purple flashes = HGT events (gene transfer). Amber halos = biofilm. Red glow = SOS stress response.

```
[E. coli baseline]     →  [+Ciprofloxacin]     →  [Resistance emergence]
  ·  · · ·  ·              · ·  · ·                   ● ● ●  ●
 · · · · · · ·           · ·[̲C̲I̲P̲R̲O̲]· ·              ●●●●●●●●
 · · · · · ·  ·           · · · ·  ·                  ●●● ●●●
 green = susceptible       cyan heatmap               red outline = gyrA_S83L
```

---

## Features

### Biological Accuracy
- **5 WHO Priority Pathogens**: E. coli, Klebsiella pneumoniae, Acinetobacter baumannii, Pseudomonas aeruginosa, MRSA
- **6 Antibiotics** with real EUCAST breakpoints: Ciprofloxacin, Meropenem, Colistin, Vancomycin, Ampicillin, Tetracycline
- **10 Resistance Genes** from CARD: blaTEM-1, blaCTX-M-15, blaKPC-2, blaNDM-1, mexAB-oprM, acrAB-tolC, gyrA_S83L, mcr-1, tetM, vanA

### AI/ML Mechanisms
| Mechanism | Real Biology Reference | Implementation |
|-----------|----------------------|----------------|
| SOS Response | Radman 1975; Simmons et al. 2008 | State machine, 8× mutation rate increase |
| Horizontal Gene Transfer | Aine et al. 2018 | Conjugation-based, distance-limited |
| Adaptive Mutation | Eyre-Walker & Keightley 2007 | Gaussian fitness landscape |
| Persister Cells | Lewis 2010, Nat Rev Micro | Stochastic switching, bet-hedging |
| Biofilm Formation | Hoiby et al. 2010 | Density-threshold, 60% protection |
| Chemotaxis | Berg & Brown 1972 | Gradient-based movement |
| PK/PD Killing | Regoes et al. 2004 | Hill equation, MBC/MIC parameters |
| Fitness Landscape | Andersson & Hughes 2010 | Resistance gene cost model |

### Simulation Scenarios
| Scenario | Organisms | Treatment | Clinical Context |
|----------|-----------|-----------|-----------------|
| `validation` | E. coli | None | Growth kinetics baseline |
| `ecoli_cipro` | E. coli | Ciprofloxacin | UTI/sepsis (>70% resistant in Pakistan) |
| `klebsiella_carbapenem` | K. pneumoniae | Meropenem | NDM-1 ICU infections |
| `xdr_acinetobacter` | A. baumannii | Colistin | Last-resort hospital scenario |
| `mrsa_hospital` | MRSA | Vancomycin | Hospital-acquired infection |
| `pakistan_crisis` | E. coli + Klebsiella | Cipro + Meropenem | Pakistan AMR crisis |
| `multi_species` | 3 species | Ciprofloxacin | Ecological competition |

---

## Architecture

```
amr_sim/
├── data/
│   └── card_loader.py          # CARD resistance genes, germ profiles, antibiotic pharmacodynamics
├── core/
│   └── bacterium_agent.py      # Mesa BacteriumAgent — full biology (SOS, HGT, biofilm, mutation)
├── simulation/
│   ├── amr_model.py            # Mesa MultiGrid model — environment, antibiotic diffusion/decay grids, step logic, statistics
│   └── sim_logger.py           # Structured run logging to logs/
├── ai/
│   ├── resistance_analytics.py # MIC estimation, treatment recommendation, population genetics
│   ├── feature_engineering.py  # Simulation state → graph features for the GNN
│   ├── gnn_model.py            # AMRResistanceGNN architecture
│   ├── gnn_trainer.py          # GNN training
│   ├── gnn_inference.py        # GNN inference engine used by the API
│   ├── baselines.py, external_validation.py, threshold_calibration.py,
│   │   gnn_ablation.py, gnn_multiseed.py   # paper evaluation scripts
│   └── checkpoints/            # Trained models and result JSONs
├── api/
│   └── server.py               # FastAPI REST server (no WebSocket; the frontend polls GET /state)
├── frontend/
│   └── index.html              # Single-file Canvas UI; polls the REST API every 2.5 s
├── tests/
│   ├── test_simulation.py      # Data, agent biology, model dynamics, analytics, science validation, reproducibility
│   ├── test_gnn.py             # GNN feature engineering, model, inference
│   └── test_api.py             # REST contract tests (FastAPI TestClient) for every endpoint the frontend calls
├── main.py                     # Entry point
└── requirements.txt
```

Antibiotics are not agents: each active drug is a concentration grid on the
model, diffused and decayed every step in `AMRSimulationModel._diffuse_antibiotics()`,
using per-drug `diffusion_rate` / `decay_rate` from `data/card_loader.py`.

---

## Quickstart

### 1. Install dependencies
```bash
git clone https://github.com/M-O-J-U/amr-simulation
cd amr-simulation
pip install -r requirements.txt
```

### 2. Headless run (no server, just terminal output)
```bash
python main.py headless --scenario ecoli_cipro --steps 40 --seed 42
```

### 3. Start the API server, then open the UI
```bash
python main.py server            # http://127.0.0.1:8000 (localhost only)
# then open frontend/index.html in your browser (it is not opened for you);
# pick a scenario in the UI and press Reset.
# Interactive API docs: http://127.0.0.1:8000/docs
```
`--host 0.0.0.0` exposes the server to your local network; it has no
authentication and allows cross-origin requests, so only do that deliberately.

### 4. Run tests
```bash
python main.py test
# or
pytest            # pytest.ini limits collection to tests/
```

### 5. Other commands
```bash
python main.py validate          # 4 quick biology checks
python main.py --help            # train-gnn, baselines, calibrate, ablation-gnn, multiseed, validate-external
```

---

## API Reference

```bash
# Get full state (bacteria positions, heatmaps, stats)
GET  /state

# Advance simulation
POST /step            {"n_steps": 5}

# Apply antibiotic
POST /apply_antibiotic {
  "antibiotic_key": "ciprofloxacin",
  "concentration": 1.5,
  "mode": "uniform",   # uniform | gradient | spot | zone
  "center": [40, 30],  # optional; "spot" only — without it, spot adds no drug
  "radius": 10         # optional; "spot" only
}

# Remove antibiotic (simulate drug clearance)
POST /remove_antibiotic {"antibiotic_key": "ciprofloxacin"}

# Spawn bacteria mid-simulation
POST /spawn_bacteria  {"germ_key": "klebsiella_pneumoniae", "count": 30}

# Reset to scenario
POST /reset           {"scenario": "pakistan_crisis", "initial_bacteria": 100, "seed": 42}   # seed optional

# Pause / resume (while paused, POST /step is a no-op)
POST /pause
POST /resume

# GNN transfer prediction and advisory
GET  /gnn/status
POST /gnn/predict     {"threshold": 0.31, "max_nodes": 300, "max_edge_distance": 3}
POST /gnn/advisory    {"available_antibiotics": ["ciprofloxacin", "meropenem"]}

# Population analytics
GET  /analytics/mic?antibiotic_key=ciprofloxacin
GET  /analytics/diversity
GET  /analytics/recommend
```

There is no WebSocket endpoint. The frontend polls `GET /state` every 2.5 s
and after each action, and drives Play by POSTing `/step` on a timer.
Interactive docs: `http://localhost:8000/docs` while the server is running.

---

## Validation Protocol

To confirm the simulation matches real biology:

**Step 1 — Growth validation** (scenario: `validation`)
- Run 50 steps without antibiotics
- Expected: logistic growth curve, doubling ~8–10 steps (E. coli)

**Step 2 — Antibiotic killing** (scenario: `ecoli_cipro`)
- Apply ciprofloxacin at MBC (0.5 µg/mL) at step 15
- Expected: >80% population reduction within 20 steps

**Step 3 — Resistance selection** (same scenario)
- Continue to step 60
- Expected: gyrA_S83L frequency increases; survivors are predominantly resistant

**Step 4 — HGT spread** (scenario: `pakistan_crisis`)
- Watch HGT event log
- Expected: blaCTX-M-15 and blaNDM-1 spread from Klebsiella to E. coli

**Step 5 — Biofilm protection** (scenario: `xdr_acinetobacter`)
- Apply colistin to dense culture
- Expected: biofilm bacteria survive at higher rate than planktonic cells

---

## Data Sources

| Source | Usage |
|--------|-------|
| [CARD v3.2](https://card.mcmaster.ca/download) | Resistance gene ontology, mechanisms, drug class mappings |
| [WHO GLASS 2022](https://www.who.int/publications/i/item/9789240062894) | Pakistan AMR surveillance data, species prevalence |
| [EUCAST 2023 Breakpoints](https://www.eucast.org/clinical_breakpoints/) | MIC susceptible/resistant thresholds for all antibiotics |
| [WHO Priority Pathogens 2017](https://www.who.int/publications/i/item/WHO-EMP-IAU-2017.12) | Pathogen prioritization (CRITICAL/HIGH/MEDIUM) |
| [NCBI AMR Reference](https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/) | Gene sequences and nomenclature |

---

## Research Directions

This codebase is the foundation for:

1. **GNN-Based Resistance Prediction** — Graph Neural Network on PATRIC genomic data predicting which resistance genes will emerge given a starting genotype and antibiotic pressure
2. **NDM-1 Spread Modeling** — Calibrate HGT rates to real Pakistan clinical isolate data to produce epidemiologically valid spread curves
3. **Combination Therapy Optimization** — Use the AI analytics engine to find antibiotic combinations that minimize resistance selection
4. **Visual Biomarker Integration** — Link to FaceFuel/TriModal pipeline for clinical presentation prediction from bacterial phenotype imaging

---

## Citation

If you use this simulation in academic work, please cite:

```bibtex
@software{amr_simulation_2026,
  author    = {Abdul Moiz Muhammad},
  title     = {AMR Simulation Lab: Agent-Based Simulation of Antimicrobial Resistance Dynamics},
  year      = {2026},
  url       = {https://github.com/M-O-J-U/amr-simulation},
  note      = {ORCID: 0009-0006-2795-5271}
}
```

---

## License

No license file is included yet, so no license is granted at this time.

---

## Author

**Abdul Moiz Muhammad**  
BS Artificial Intelligence, COMSATS University Islamabad (2026)  
ORCID: [0009-0006-2795-5271](https://orcid.org/0009-0006-2795-5271)  
GitHub: [github.com/M-O-J-U](https://github.com/M-O-J-U)  
LinkedIn: [linkedin.com/in/abdul-moiz-muhammad](https://linkedin.com/in/abdul-moiz-muhammad)





### Run Sequence
# GPU Run Instructions — Final Submission-Grade Numbers

Run these on your RTX 4070 machine, in this **exact order**. The order is
not a suggestion — `baselines --full` will now hard-fail (not silently
substitute fake numbers) if `calibrate` and `train-gnn` haven't run first,
because of fixes made this session that removed three separate hardcoded
fabrication paths.

## Prerequisites

```powershell
cd "C:\Users\mojua\Desktop\AMR Simulation Lab"
python main.py test
```
This must show 0 failed before you do anything else (189 passed as of
2026-09-29; it was 114 when these instructions were written, before
`tests/test_api.py` and later GNN tests were added). If it doesn't,
something didn't sync correctly from this session's fixes — stop and
re-sync the files, don't proceed to training on a broken pipeline.

## Step 1 — Train the GNN (full config, DEFAULT_CONFIG: 4 scenarios × 3 seeds × 80 steps, 60 epochs)

```powershell
python main.py train-gnn --full --epochs 60
```

Expected runtime: on CPU here, the same config's smaller cousin (2 seeds,
45 epochs) took the better part of 300+ seconds and didn't fully converge
in one sitting — with your GPU and the full 3-seed dataset, expect
somewhere in the 10-30 minute range depending on how many epochs it
actually takes before early stopping (patience=12) kicks in. Let it run
to completion; don't interrupt it.

This produces:
- `ai/checkpoints/best_model.pt`
- `ai/checkpoints/training_results.json`

**Do not skip this even if you already have a checkpoint from an earlier
session** — that checkpoint was trained on the leaky 36/8-dim features
and buggy labels. It is incompatible with the current 35/5-dim model
architecture and will fail to load or, if it somehow loads, will be
scored against the wrong ground truth.

## Step 2 — Threshold calibration (must run before baselines)

```powershell
python main.py calibrate
```

This loads `best_model.pt` from Step 1, computes per-gene F1-optimal
thresholds, and writes `ai/checkpoints/calibration_results.json`.

Check the printed summary — specifically the "Genes with test-set
positives: X/10" line. If X is less than 6-7, the DEFAULT_CONFIG dataset
may still be too small for some rarer genes (mexAB-oprM, vanA are
expected to show 0 positives — that's correct, they're species-restricted
and shouldn't transfer in these scenarios). If common genes like
blaTEM-1 or acrAB-tolC show 0 positives, something is wrong — stop and
flag it before proceeding.

## Step 3 — Baselines + ablation (requires Steps 1 and 2 to have completed)

```powershell
python main.py baselines --full
```

This will:
1. Re-collect the same DEFAULT_CONFIG simulation data
2. Train Frequency/Logistic Regression/Random Forest baselines
3. Read the GNN's AUROC/AUPRC from `training_results.json` (Step 1) and
   F1/precision/recall from `calibration_results.json` (Step 2) —
   it will **raise an error and refuse to run** if either file is
   missing, rather than substituting a fabricated number. This is
   intentional; do not "fix" this by adding a fallback value.
4. Run the ablation study (feature-group importance)

Produces `ai/checkpoints/baseline_results.json` and
`ai/checkpoints/full_comparison.json`.

**Important — the split-consistency fix**: `split_dataset()` was fixed
this session to use a seeded local RNG (`split_seed`, default 42 from
`DEFAULT_CONFIG`) instead of Python's unseeded global `random` module.
This means Steps 1 and 3, despite being separate process invocations,
will now evaluate the GNN and the baselines against the **same held-out
test set** — this was NOT true before the fix, and silently wasn't true
in earlier sessions' reported numbers either. Do not modify
`split_seed` between steps or the comparison becomes invalid again.

## Step 4 — Verify cross-file consistency (copy-paste this exactly)

```powershell
python -c "
import json
with open('ai/checkpoints/training_results.json') as f: tr = json.load(f)
with open('ai/checkpoints/calibration_results.json') as f: cal = json.load(f)
with open('ai/checkpoints/baseline_results.json') as f: bl = json.load(f)

tr_auroc = tr['test_metrics']['auroc_macro']
bl_auroc = bl['gnn_ours']['auroc_macro']
print('AUROC match:', abs(tr_auroc - bl_auroc) < 1e-9, '|', tr_auroc, 'vs', bl_auroc)

cal_f1 = cal['summary']['macro_f1_at_recommended_threshold']
bl_f1 = bl['gnn_ours']['f1_macro']
print('F1 match:', abs(cal_f1 - bl_f1) < 1e-9, '|', cal_f1, 'vs', bl_f1)
"
```

Both lines must print `True`. If either prints `False`, do not proceed to
writing the paper with these numbers — something in the pipeline broke
between steps and needs to be diagnosed before the numbers can be trusted.

## Step 5 (optional, best-effort) — External validation

```powershell
python main.py validate-external --real
```

This attempts to pull real BV-BRC genomic data for comparison. It may
fail or return inconclusive results — this happened in the prior session
and is a known open limitation, not something you need to force to a
positive result. If it fails, the paper should state this as a limitation
honestly (see prior discussion) rather than being blocked on it.

## What "done" looks like

- `python main.py test` → 0 failed
- Step 4's two `True` checks both pass
- You have four JSON files in `ai/checkpoints/` that all agree with each
  other because they're computed from the same pipeline and the same
  held-out test set — not because any of them were hand-edited to match.

Send me the final printed AUROC/AUPRC/F1 numbers from Steps 1-3 plus the
Step 4 verification output once this completes, and I'll write the
resubmission-ready paper draft with real, cross-consistent numbers instead
of the reduced-scope CPU numbers currently sitting in this repo.