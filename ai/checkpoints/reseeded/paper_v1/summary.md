# Reseeded results (2026-09-29 17:31:37)

Model seeds [0, 1, 2, 3, 4]; ablation seeds [0, 1, 2]; device cuda; torch 2.11.0+cu128; features {'node': 35, 'edge': 5}.
Dataset sha256 `cd5237c3df10719a…`: 301 graph pairs, 820,346 edges, 1,222 edge-gene positives.

## Model comparison (test set, mean ± SD over seeds)

| Model | AUROC | AUPRC | F1 (calibrated) |
|---|---|---|---|
| AMRResistanceGNN | 0.9261 ± 0.0211 (n=5) | 0.0071 ± 0.0011 (n=5) | 0.0280 ± 0.0052 (n=5) |
| Random Forest | 0.9226 ± 0.0263 (n=5) | 0.0074 ± 0.0023 (n=5) | 0.0270 ± 0.0070 (n=5) |
| Logistic Regression | 0.8466 ± 0.0406 (n=5) | 0.0029 ± 0.0002 (n=5) | 0.0066 ± 0.0005 (n=5) |
| Frequency baseline | 0.5000 ± 0.0000 (n=5) | 0.0002 ± 0.0000 (n=5) | 0.0004 ± 0.0000 (n=5) |

GNN − RF AUROC (paired by seed): 0.0035 ± 0.0125 (n=5); GNN better on 3/5 seeds.
GNN ECE: 0.0013 ± 0.0002 (n=5).

## Per-gene AUROC (GNN vs RF)

| Gene | GNN | RF |
|---|---|---|
| blaTEM-1 | 0.9756 ± 0.0149 (n=5) | 0.9611 ± 0.0214 (n=5) |
| blaCTX-M-15 | 0.8587 ± 0.0204 (n=5) | 0.9101 ± 0.0118 (n=5) |
| blaKPC-2 | 0.9114 ± 0.1245 (n=5) | 0.8718 ± 0.0535 (n=5) |
| blaNDM-1 | 0.9429 ± 0.0045 (n=5) | 0.9514 ± 0.0151 (n=5) |
| mexAB-oprM | n/a | n/a |
| acrAB-tolC | 0.9699 ± 0.0101 (n=5) | 0.9696 ± 0.0118 (n=5) |
| gyrA_S83L | 0.8485 ± 0.0335 (n=5) | 0.8305 ± 0.1897 (n=5) |
| mcr-1 | 0.9754 ± 0.0064 (n=5) | 0.9635 ± 0.0116 (n=5) |
| tetM | n/a | n/a |
| vanA | n/a | n/a |

## GNN feature-group ablation (retrained per condition)

| Condition | AUROC | Δ vs all features (paired) |
|---|---|---|
| All features (full model) | 0.9218 ± 0.0282 (n=3) | 0.0000 ± 0.0000 (n=3) |
| No genomic genes | 0.7945 ± 0.0216 (n=3) | -0.1274 ± 0.0074 (n=3) |
| No physiological | 0.9266 ± 0.0166 (n=3) | 0.0047 ± 0.0135 (n=3) |
| No behavioral | 0.9237 ± 0.0245 (n=3) | 0.0018 ± 0.0046 (n=3) |
| No spatial position | 0.9331 ± 0.0097 (n=3) | 0.0112 ± 0.0190 (n=3) |
| No population | 0.9280 ± 0.0212 (n=3) | 0.0061 ± 0.0072 (n=3) |
| No species | 0.9327 ± 0.0290 (n=3) | 0.0109 ± 0.0023 (n=3) |
| No gram stain | 0.9190 ± 0.0321 (n=3) | -0.0028 ± 0.0041 (n=3) |
| No antibiotic exposure | 0.9224 ± 0.0275 (n=3) | 0.0006 ± 0.0007 (n=3) |
| No edge features | 0.9514 ± 0.0129 (n=3) | 0.0296 ± 0.0153 (n=3) |

Reproducibility check (|AUROC(main run) - AUROC(ablation 'All features' rerun)|, same data and torch_seed): max |Δ| = 0.00021922735154222828.
