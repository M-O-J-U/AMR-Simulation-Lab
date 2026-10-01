# Reseeded results (2026-10-01 08:51:24)

Biology lab_v2; scenarios ['ecoli_cipro', 'klebsiella_carbapenem', 'pakistan_crisis', 'xdr_acinetobacter', 'mrsa_hospital']; dosing {'dose': 0.25, 'dose_duration': 5}.
GNN hparams {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2}; edge features ZEROED; min_epochs 20; RF hparams {'n_estimators': 300, 'max_depth': 8}.
Model seeds [0, 1, 2, 3, 4]; ablation seeds [0, 1, 2]; device cuda.
Dataset sha256 `7492f6e14601ed8f…`: 1300 graph pairs, 6,029,316 edges, 4,763 positives; per gene {'blaTEM-1': 24, 'blaCTX-M-15': 1446, 'blaKPC-2': 536, 'blaNDM-1': 554, 'mexAB-oprM': 0, 'acrAB-tolC': 23, 'gyrA_S83L': 109, 'mcr-1': 766, 'tetM': 1230, 'vanA': 75}.

**Headline macro** = mean over ['blaTEM-1', 'blaCTX-M-15', 'blaKPC-2', 'blaNDM-1', 'acrAB-tolC', 'gyrA_S83L', 'mcr-1', 'tetM'] with test positives. Excluded: ['mexAB-oprM']. Reported separately: ['vanA'].

## Model comparison (test set, mean ± SD over seeds)

| Model | Headline AUROC | Headline AUPRC | All-gene AUROC | F1 (calibrated) |
|---|---|---|---|---|
| AMRResistanceGNN | 0.9676 ± 0.0076 (n=5) | 0.0145 ± 0.0057 (n=5) | 0.9551 ± 0.0120 (n=5) | 0.0679 ± 0.0125 (n=5) |
| GNN without message passing | 0.9611 ± 0.0019 (n=5) | 0.0100 ± 0.0017 (n=5) | 0.9535 ± 0.0043 (n=5) | 0.0609 ± 0.0117 (n=5) |
| Random Forest | 0.8711 ± 0.0366 (n=5) | 0.0042 ± 0.0006 (n=5) | 0.8756 ± 0.0343 (n=5) | 0.0198 ± 0.0041 (n=5) |
| Logistic Regression | 0.7911 ± 0.0336 (n=5) | 0.0013 ± 0.0001 (n=5) | 0.8012 ± 0.0301 (n=5) | 0.0029 ± 0.0005 (n=5) |
| Frequency baseline | 0.5000 ± 0.0000 (n=5) | 0.0001 ± 0.0000 (n=5) | 0.5000 ± 0.0000 (n=5) | 0.0002 ± 0.0000 (n=5) |

gnn_minus_rf (headline AUROC, paired by seed): 0.0965 ± 0.0405 (n=5); first better on 5/5 seeds.
gnn_minus_graph_free (headline AUROC, paired by seed): 0.0065 ± 0.0086 (n=5); first better on 3/5 seeds.
GNN ECE: 0.0012 ± 0.0003 (n=5).

## Per-gene AUROC

| Gene | GNN | RF | note |
|---|---|---|---|
| blaTEM-1 | n/a | n/a |  |
| blaCTX-M-15 | 0.9821 ± 0.0024 (n=5) | 0.9534 ± 0.0121 (n=5) |  |
| blaKPC-2 | 0.9761 ± 0.0045 (n=5) | 0.9816 ± 0.0024 (n=5) |  |
| blaNDM-1 | 0.9860 ± 0.0054 (n=5) | 0.9786 ± 0.0075 (n=5) |  |
| mexAB-oprM | n/a | n/a | EXCLUDED |
| acrAB-tolC | 0.8890 ± 0.0506 (n=5) | 0.5000 ± 0.0000 (n=5) |  |
| gyrA_S83L | 0.9822 ± 0.0049 (n=5) | 0.7742 ± 0.2510 (n=5) |  |
| mcr-1 | 0.9738 ± 0.0018 (n=5) | 0.9503 ± 0.0057 (n=5) |  |
| tetM | 0.9844 ± 0.0010 (n=5) | 0.9596 ± 0.0045 (n=5) |  |
| vanA | 0.8669 ± 0.0504 (n=5) | 0.9071 ± 0.0429 (n=5) | SEPARATE - simplified mechanism |

**vanA (reported separately):** SIMPLIFIED MECHANISM: in the simulation vanA arises de novo in MRSA and spreads MRSA-to-MRSA; the documented route is interspecies transfer of Tn1546 from Enterococcus faecalis (Weigel et al. 2003, doi:10.1126/science.1090956), with the first VRSA isolates arising independently (Clark et al. 2005, doi:10.1128/AAC.49.1.470-472.2005). Enterococcus is not modelled.

## GNN feature-group ablation (retrained per condition, headline AUROC)

| Condition | AUROC | Δ vs all features (paired) |
|---|---|---|
| All features (full model) | 0.9687 ± 0.0086 (n=3) | 0.0000 ± 0.0000 (n=3) |
| No genomic genes | 0.9245 ± 0.0151 (n=3) | -0.0442 ± 0.0071 (n=3) |
| No physiological | 0.9738 ± 0.0057 (n=3) | 0.0051 ± 0.0125 (n=3) |
| No behavioral | 0.9724 ± 0.0084 (n=3) | 0.0037 ± 0.0017 (n=3) |
| No spatial position | 0.9739 ± 0.0026 (n=3) | 0.0052 ± 0.0083 (n=3) |
| No population | 0.9703 ± 0.0084 (n=3) | 0.0016 ± 0.0026 (n=3) |
| No species | 0.9641 ± 0.0023 (n=3) | -0.0046 ± 0.0102 (n=3) |
| No gram stain | 0.9690 ± 0.0005 (n=3) | 0.0003 ± 0.0091 (n=3) |
| No antibiotic exposure | 0.9693 ± 0.0079 (n=3) | 0.0006 ± 0.0011 (n=3) |

Reproducibility check (|headline AUROC(main run) - (ablation 'All features' rerun)|, same data and torch_seed): max |Δ| = 0.0015948251341019182.

Per-seed training (GNN): seed 0: 34 epochs, best @ 22; seed 1: 39 epochs, best @ 27; seed 2: 34 epochs, best @ 22; seed 3: 26 epochs, best @ 14; seed 4: 20 epochs, best @ 11
