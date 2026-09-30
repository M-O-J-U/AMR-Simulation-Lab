# Reseeded results (2026-09-30 03:57:47)

Model seeds [0, 1, 2, 3, 4]; ablation seeds [0, 1, 2]; device cuda; torch 2.11.0+cu128; features {'node': 35, 'edge': 5}.
Dataset sha256 `d10a94d592b50366…`: 312 graph pairs, 1,220,942 edges, 1,011 edge-gene positives.

## Model comparison (test set, mean ± SD over seeds)

| Model | AUROC | AUPRC | F1 (calibrated) |
|---|---|---|---|
| AMRResistanceGNN | 0.6802 ± 0.0381 (n=5) | 0.0006 ± 0.0001 (n=5) | 0.0027 ± 0.0009 (n=5) |
| Random Forest | 0.7739 ± 0.0446 (n=5) | 0.0040 ± 0.0006 (n=5) | 0.0190 ± 0.0047 (n=5) |
| Logistic Regression | 0.7104 ± 0.0344 (n=5) | 0.0013 ± 0.0004 (n=5) | 0.0035 ± 0.0004 (n=5) |
| Frequency baseline | 0.5000 ± 0.0000 (n=5) | 0.0001 ± 0.0000 (n=5) | 0.0002 ± 0.0000 (n=5) |

GNN − RF AUROC (paired by seed): -0.0937 ± 0.0285 (n=5); GNN better on 0/5 seeds.
GNN ECE: 0.0004 ± 0.0004 (n=5).

## Per-gene AUROC (GNN vs RF)

| Gene | GNN | RF |
|---|---|---|
| blaTEM-1 | n/a | n/a |
| blaCTX-M-15 | 0.7584 ± 0.0798 (n=5) | 0.9466 ± 0.0091 (n=5) |
| blaKPC-2 | 0.7197 ± 0.2028 (n=5) | 0.7375 ± 0.1346 (n=5) |
| blaNDM-1 | 0.8647 ± 0.0756 (n=5) | 0.9295 ± 0.0337 (n=5) |
| mexAB-oprM | n/a | n/a |
| acrAB-tolC | 0.6971 ± 0.1280 (n=5) | 0.5837 ± 0.1873 (n=5) |
| gyrA_S83L | 0.3136 ± 0.2029 (n=5) | 0.6984 ± 0.2717 (n=5) |
| mcr-1 | 0.8455 ± 0.0740 (n=5) | 0.9223 ± 0.0753 (n=5) |
| tetM | 0.5626 ± 0.2087 (n=5) | 0.5991 ± 0.2216 (n=5) |
| vanA | n/a | n/a |

## GNN feature-group ablation (retrained per condition)

| Condition | AUROC | Δ vs all features (paired) |
|---|---|---|
| All features (full model) | 0.7027 ± 0.0273 (n=3) | 0.0000 ± 0.0000 (n=3) |
| No genomic genes | 0.6642 ± 0.0766 (n=3) | -0.0385 ± 0.0744 (n=3) |
| No physiological | 0.6733 ± 0.0373 (n=3) | -0.0294 ± 0.0142 (n=3) |
| No behavioral | 0.7057 ± 0.0323 (n=3) | 0.0030 ± 0.0075 (n=3) |
| No spatial position | 0.7874 ± 0.1179 (n=3) | 0.0846 ± 0.1373 (n=3) |
| No population | 0.7832 ± 0.1512 (n=3) | 0.0805 ± 0.1487 (n=3) |
| No species | 0.8572 ± 0.1350 (n=3) | 0.1545 ± 0.1341 (n=3) |
| No gram stain | 0.7166 ± 0.0479 (n=3) | 0.0138 ± 0.0208 (n=3) |
| No antibiotic exposure | 0.7026 ± 0.0271 (n=3) | -0.0001 ± 0.0013 (n=3) |
| No edge features | 0.8577 ± 0.1032 (n=3) | 0.1550 ± 0.1157 (n=3) |

Reproducibility check (|AUROC(main run) - AUROC(ablation 'All features' rerun)|, same data and torch_seed): max |Δ| = 5.35963402348294e-06.
