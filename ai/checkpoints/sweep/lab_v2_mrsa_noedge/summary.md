# Hyperparameter sweep — lab_v2_mrsa_noedge

Dataset sha256 `ee83ff385ca15ec9…`. Selection on validation `headline_auroc`; final = selected config on test, seeds [0, 1, 2, 3, 4].

| Arm | Selected hyperparameters | Test headline_auroc | Test AUPRC |
|---|---|---|---|
| gnn_no_edge | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 2} | 0.9805 ± 0.0012 | 0.0310 ± 0.0293 |
| graph_free_no_edge | {'hidden_dim': 128, 'lr': 0.001, 'n_layers': 0} | 0.9774 ± 0.0006 | 0.0230 ± 0.0115 |
| rf | {'n_estimators': 300, 'max_depth': 8} | 0.7822 ± 0.0276 | 0.0043 ± 0.0011 |

Per-seed: gnn_no_edge [0.9812, 0.981, 0.98, 0.9786, 0.9814]; graph_free_no_edge [0.9771, 0.9784, 0.9774, 0.9773, 0.9769]; rf [0.761, 0.8142, 0.759, 0.8101, 0.7664]

gnn_no_edge − graph_free_no_edge (paired by seed): 0.0030 ± 0.0013; first better on 5/5 seeds.

gnn_no_edge − rf (paired by seed): 0.1983 ± 0.0281; first better on 5/5 seeds.

graph_free_no_edge − rf (paired by seed): 0.1953 ± 0.0272; first better on 5/5 seeds.
