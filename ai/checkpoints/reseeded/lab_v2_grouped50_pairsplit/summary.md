# Reseeded results (2026-10-01 12:28:40)

Biology lab_v2; scenarios ['ecoli_cipro', 'klebsiella_carbapenem', 'pakistan_crisis', 'xdr_acinetobacter', 'mrsa_hospital']; dosing {'dose': 0.25, 'dose_duration': 5}.
GNN hparams {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2}; edge features ZEROED; min_epochs 20; RF hparams {'n_estimators': 300, 'max_depth': 8}.
Model seeds [0, 1, 2, 3, 4]; ablation seeds [0, 1, 2]; device cuda.
Dataset sha256 `7492f6e14601ed8f…`: 1300 graph pairs, 6,029,316 edges, 4,763 positives; per gene {'blaTEM-1': 24, 'blaCTX-M-15': 1446, 'blaKPC-2': 536, 'blaNDM-1': 554, 'mexAB-oprM': 0, 'acrAB-tolC': 23, 'gyrA_S83L': 109, 'mcr-1': 766, 'tetM': 1230, 'vanA': 75}.

**Headline macro** = mean over ['blaTEM-1', 'blaCTX-M-15', 'blaKPC-2', 'blaNDM-1', 'acrAB-tolC', 'gyrA_S83L', 'mcr-1', 'tetM'] with test positives. Excluded: ['mexAB-oprM']. Reported separately: ['vanA'].

## Model comparison (test set, mean ± SD over seeds)

| Model | Headline AUROC | Headline AUPRC | All-gene AUROC | F1 (calibrated) |
|---|---|---|---|---|
| AMRResistanceGNN | 0.9852 ± 0.0008 (n=5) | 0.0189 ± 0.0054 (n=5) | 0.9796 ± 0.0025 (n=5) | 0.0575 ± 0.0092 (n=5) |
| GNN without message passing | 0.9793 ± 0.0025 (n=5) | 0.0097 ± 0.0010 (n=5) | 0.9692 ± 0.0026 (n=5) | 0.0377 ± 0.0038 (n=5) |
| Random Forest | 0.8280 ± 0.0330 (n=5) | 0.0037 ± 0.0002 (n=5) | 0.8201 ± 0.0429 (n=5) | 0.0153 ± 0.0036 (n=5) |
| Logistic Regression | 0.7852 ± 0.0375 (n=5) | 0.0016 ± 0.0003 (n=5) | 0.7779 ± 0.0389 (n=5) | 0.0034 ± 0.0008 (n=5) |
| Frequency baseline | 0.5000 ± 0.0000 (n=5) | 0.0001 ± 0.0000 (n=5) | 0.5000 ± 0.0000 (n=5) | 0.0001 ± 0.0000 (n=5) |

gnn_minus_rf (headline AUROC, paired by seed): 0.1572 ± 0.0323 (n=5); first better on 5/5 seeds.
gnn_minus_graph_free (headline AUROC, paired by seed): 0.0059 ± 0.0029 (n=5); first better on 5/5 seeds.
GNN ECE: 0.0009 ± 0.0001 (n=5).

## Per-gene AUROC

| Gene | GNN | RF | note |
|---|---|---|---|
| blaTEM-1 | 0.9985 ± 0.0018 (n=5) | 0.5000 ± 0.0000 (n=5) |  |
| blaCTX-M-15 | 0.9838 ± 0.0012 (n=5) | 0.9626 ± 0.0079 (n=5) |  |
| blaKPC-2 | 0.9847 ± 0.0018 (n=5) | 0.9715 ± 0.0044 (n=5) |  |
| blaNDM-1 | 0.9921 ± 0.0004 (n=5) | 0.9681 ± 0.0081 (n=5) |  |
| mexAB-oprM | n/a | n/a | EXCLUDED |
| acrAB-tolC | 0.9774 ± 0.0157 (n=5) | 0.5000 ± 0.0000 (n=5) |  |
| gyrA_S83L | 0.9710 ± 0.0082 (n=5) | 0.7840 ± 0.2595 (n=5) |  |
| mcr-1 | 0.9879 ± 0.0012 (n=5) | 0.9763 ± 0.0026 (n=5) |  |
| tetM | 0.9863 ± 0.0015 (n=5) | 0.9617 ± 0.0045 (n=5) |  |
| vanA | 0.9349 ± 0.0211 (n=5) | 0.7568 ± 0.2344 (n=5) | SEPARATE - simplified mechanism |

**vanA (reported separately):** SIMPLIFIED MECHANISM: in the simulation vanA arises de novo in MRSA and spreads MRSA-to-MRSA; the documented route is interspecies transfer of Tn1546 from Enterococcus faecalis (Weigel et al. 2003, doi:10.1126/science.1090956), with the first VRSA isolates arising independently (Clark et al. 2005, doi:10.1128/AAC.49.1.470-472.2005). Enterococcus is not modelled.

## GNN feature-group ablation (retrained per condition, headline AUROC)

| Condition | AUROC | Δ vs all features (paired) |
|---|---|---|
| All features (full model) | 0.9850 ± 0.0011 (n=3) | 0.0000 ± 0.0000 (n=3) |
| No genomic genes | 0.9751 ± 0.0045 (n=3) | -0.0099 ± 0.0035 (n=3) |
| No physiological | 0.9865 ± 0.0017 (n=3) | 0.0015 ± 0.0006 (n=3) |
| No behavioral | 0.9837 ± 0.0032 (n=3) | -0.0013 ± 0.0023 (n=3) |
| No spatial position | 0.9855 ± 0.0016 (n=3) | 0.0006 ± 0.0007 (n=3) |
| No population | 0.9857 ± 0.0009 (n=3) | 0.0008 ± 0.0010 (n=3) |
| No species | 0.9859 ± 0.0007 (n=3) | 0.0010 ± 0.0006 (n=3) |
| No gram stain | 0.9819 ± 0.0042 (n=3) | -0.0031 ± 0.0034 (n=3) |
| No antibiotic exposure | 0.9853 ± 0.0005 (n=3) | 0.0003 ± 0.0012 (n=3) |

Reproducibility check (|headline AUROC(main run) - (ablation 'All features' rerun)|, same data and torch_seed): max |Δ| = 0.001199340658960768.

Per-seed training (GNN): seed 0: 28 epochs, best @ 16; seed 1: 33 epochs, best @ 21; seed 2: 35 epochs, best @ 23; seed 3: 32 epochs, best @ 20; seed 4: 34 epochs, best @ 29
