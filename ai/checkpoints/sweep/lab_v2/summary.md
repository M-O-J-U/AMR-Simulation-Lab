# Hyperparameter sweep — lab_v2

Dataset sha256 `d10a94d592b50366…`. Selection on validation AUROC; final = selected config on test, seeds [0, 1, 2, 3, 4].

| Arm | Selected hyperparameters | Test AUROC | Test AUPRC |
|---|---|---|---|
| gnn_full | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2} | 0.9272 ± 0.1051 | 0.0127 ± 0.0089 |
| gnn_no_edge | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2} | 0.9715 ± 0.0042 | 0.0326 ± 0.0194 |
| graph_free | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 0} | 0.9561 ± 0.0030 | 0.0337 ± 0.0125 |
| rf | {'n_estimators': 300, 'max_depth': 8} | 0.7862 ± 0.0534 | 0.0041 ± 0.0009 |

gnn_full − gnn_no_edge (paired by seed): -0.0443 ± 0.1050; first better on 2/5 seeds.

gnn_full − graph_free (paired by seed): -0.0289 ± 0.1058; first better on 4/5 seeds.

gnn_full − rf (paired by seed): 0.1410 ± 0.1532; first better on 4/5 seeds.
