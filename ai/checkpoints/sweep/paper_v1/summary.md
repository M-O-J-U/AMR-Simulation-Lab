# Hyperparameter sweep — paper_v1

Dataset sha256 `cd5237c3df10719a…`. Selection on validation AUROC; final = selected config on test, seeds [0, 1, 2, 3, 4].

| Arm | Selected hyperparameters | Test AUROC | Test AUPRC |
|---|---|---|---|
| gnn_full | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2} | 0.9665 ± 0.0048 | 0.0108 ± 0.0014 |
| gnn_no_edge | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2} | 0.9673 ± 0.0035 | 0.0106 ± 0.0022 |
| graph_free | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 0} | 0.9546 ± 0.0084 | 0.0081 ± 0.0018 |
| rf | {'n_estimators': 300, 'max_depth': 8} | 0.9306 ± 0.0317 | 0.0073 ± 0.0027 |

gnn_full − gnn_no_edge (paired by seed): -0.0008 ± 0.0039; first better on 2/5 seeds.

gnn_full − graph_free (paired by seed): 0.0119 ± 0.0073; first better on 5/5 seeds.

gnn_full − rf (paired by seed): 0.0359 ± 0.0329; first better on 5/5 seeds.
