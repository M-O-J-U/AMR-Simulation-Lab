# Reseeded results (2026-09-30 10:44:47)

Biology lab_v2; scenarios ['ecoli_cipro', 'klebsiella_carbapenem', 'pakistan_crisis', 'xdr_acinetobacter', 'mrsa_hospital']; dosing {'dose': 0.25, 'dose_duration': 5}.
GNN hparams {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2}; edge features ZEROED; min_epochs 20; RF hparams {'n_estimators': 300, 'max_depth': 8}.
Model seeds [0, 1, 2, 3, 4]; ablation seeds [0, 1, 2]; device cuda.
Dataset sha256 `ee83ff385ca15ec9…`: 390 graph pairs, 1,707,498 edges, 1,443 positives; per gene {'blaTEM-1': 5, 'blaCTX-M-15': 480, 'blaKPC-2': 94, 'blaNDM-1': 177, 'mexAB-oprM': 0, 'acrAB-tolC': 19, 'gyrA_S83L': 34, 'mcr-1': 179, 'tetM': 420, 'vanA': 35}.

**Headline macro** = mean over ['blaTEM-1', 'blaCTX-M-15', 'blaKPC-2', 'blaNDM-1', 'acrAB-tolC', 'gyrA_S83L', 'mcr-1', 'tetM'] with test positives. Excluded: ['mexAB-oprM']. Reported separately: ['vanA'].

## Model comparison (test set, mean ± SD over seeds)

| Model | Headline AUROC | Headline AUPRC | All-gene AUROC | F1 (calibrated) |
|---|---|---|---|---|
| AMRResistanceGNN | 0.9805 ± 0.0012 (n=5) | 0.0316 ± 0.0293 (n=5) | 0.9828 ± 0.0011 (n=5) | 0.0782 ± 0.0342 (n=5) |
| GNN without message passing | 0.9774 ± 0.0006 (n=5) | 0.0230 ± 0.0115 (n=5) | 0.9801 ± 0.0005 (n=5) | 0.0649 ± 0.0094 (n=5) |
| Random Forest | 0.7822 ± 0.0276 (n=5) | 0.0043 ± 0.0011 (n=5) | 0.7594 ± 0.0270 (n=5) | 0.0214 ± 0.0139 (n=5) |
| Logistic Regression | 0.7096 ± 0.0359 (n=5) | 0.0016 ± 0.0007 (n=5) | 0.6959 ± 0.0358 (n=5) | 0.0035 ± 0.0012 (n=5) |
| Frequency baseline | 0.5000 ± 0.0000 (n=5) | 0.0001 ± 0.0000 (n=5) | 0.5000 ± 0.0000 (n=5) | 0.0002 ± 0.0000 (n=5) |

gnn_minus_rf (headline AUROC, paired by seed): 0.1984 ± 0.0281 (n=5); first better on 5/5 seeds.
gnn_minus_graph_free (headline AUROC, paired by seed): 0.0031 ± 0.0014 (n=5); first better on 5/5 seeds.
GNN ECE: 0.0012 ± 0.0003 (n=5).

## Per-gene AUROC

| Gene | GNN | RF | note |
|---|---|---|---|
| blaTEM-1 | n/a | n/a |  |
| blaCTX-M-15 | 0.9642 ± 0.0030 (n=5) | 0.9578 ± 0.0058 (n=5) |  |
| blaKPC-2 | 0.9564 ± 0.0021 (n=5) | 0.6558 ± 0.2193 (n=5) |  |
| blaNDM-1 | 0.9755 ± 0.0039 (n=5) | 0.9439 ± 0.0234 (n=5) |  |
| mexAB-oprM | n/a | n/a | EXCLUDED |
| acrAB-tolC | 0.9986 ± 0.0018 (n=5) | 0.5000 ± 0.0000 (n=5) |  |
| gyrA_S83L | 0.9977 ± 0.0006 (n=5) | 0.5000 ± 0.0000 (n=5) |  |
| mcr-1 | 0.9908 ± 0.0011 (n=5) | 0.9606 ± 0.0277 (n=5) |  |
| tetM | 0.9805 ± 0.0033 (n=5) | 0.9570 ± 0.0065 (n=5) |  |
| vanA | 0.9988 ± 0.0002 (n=5) | 0.5998 ± 0.2231 (n=5) | SEPARATE - simplified mechanism |

**vanA (reported separately):** SIMPLIFIED MECHANISM: in the simulation vanA arises de novo in MRSA and spreads MRSA-to-MRSA; the documented route is interspecies transfer of Tn1546 from Enterococcus faecalis (Weigel et al. 2003, doi:10.1126/science.1090956), with the first VRSA isolates arising independently (Clark et al. 2005, doi:10.1128/AAC.49.1.470-472.2005). Enterococcus is not modelled.

## GNN feature-group ablation (retrained per condition, headline AUROC)

| Condition | AUROC | Δ vs all features (paired) |
|---|---|---|
| All features (full model) | 0.9807 ± 0.0007 (n=3) | 0.0000 ± 0.0000 (n=3) |
| No genomic genes | 0.9258 ± 0.0329 (n=3) | -0.0549 ± 0.0331 (n=3) |
| No physiological | 0.9801 ± 0.0007 (n=3) | -0.0006 ± 0.0002 (n=3) |
| No behavioral | 0.9781 ± 0.0022 (n=3) | -0.0026 ± 0.0022 (n=3) |
| No spatial position | 0.9806 ± 0.0016 (n=3) | -0.0001 ± 0.0014 (n=3) |
| No population | 0.9794 ± 0.0011 (n=3) | -0.0013 ± 0.0004 (n=3) |
| No species | 0.9806 ± 0.0026 (n=3) | -0.0001 ± 0.0023 (n=3) |
| No gram stain | 0.9799 ± 0.0011 (n=3) | -0.0008 ± 0.0011 (n=3) |
| No antibiotic exposure | 0.9808 ± 0.0007 (n=3) | 0.0001 ± 0.0001 (n=3) |

Reproducibility check (|headline AUROC(main run) - (ablation 'All features' rerun)|, same data and torch_seed): max |Δ| = 0.00021132301757420535.

Per-seed training (GNN): seed 0: 32 epochs, best @ 20; seed 1: 38 epochs, best @ 32; seed 2: 37 epochs, best @ 25; seed 3: 33 epochs, best @ 26; seed 4: 27 epochs, best @ 15
