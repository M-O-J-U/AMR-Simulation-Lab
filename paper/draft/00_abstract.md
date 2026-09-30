<!--
DRAFT — Abstract. Written last, per plan. Status: awaiting owner review.

claims_to_numbers.md entries used: S1, S2, S2b, S4, U7, U10, U11, U13.
Proportions follow Discussion: the scope caveats get roughly as much room as the results,
because §6.1 and §6.4 are where the Discussion's weight sits.
No new numbers. Owner decision 2026-09-30: the retired 2016 review's 10-million projection is
excluded from this document entirely (the AMR burden is cited from the GBD papers only).
-->

# Abstract

Horizontal gene transfer moves antimicrobial resistance genes between bacteria, but individual
transfer events are hard to observe directly in natural communities, which makes supervised
learning on them difficult to set up. We use an agent-based simulation, in which every transfer
is recorded as it happens, to pose a task with exact labels: given a snapshot of a bacterial
population, predict for each directed cell-to-cell contact and each resistance gene whether
that gene is transferred in the next time window. We train a graph attention network on this
task and evaluate it against per-gene random forest and logistic regression baselines.

The network reaches a test AUROC of 0.9735 ± 0.0017 on the five genes for which our per-gene
random forest can be fitted, against 0.8950 ± 0.0386 for that baseline, and is ahead on all
five model seeds; its macro average over the eight genes in our evaluation policy is
0.9805 ± 0.0012. Because it is trained jointly on all gene outputs, it also yields evaluable
predictions for two genes that fall below the baselines' five-positive training threshold and
that those baselines therefore cannot fit at all. We attribute this coverage advantage to
parameter sharing across a multi-label output rather than to the graph: removing message
passing costs only 0.0031 ± 0.0014, and the genes a cell already carries dominate every other
feature group by an order of magnitude.

We are explicit about what these figures do not establish. The simulator transfers genes only
between cells of the same species, so the interspecies transfers that dominate the literature
on these genes are absent from the task, the labels and the evaluation alike; in our dataset
only 0.275% of contacts are cross-species, and none of them carries a transfer, so the model is
evaluated almost entirely on within-species pairs. The held-out split is taken over
snapshot pairs rather than over simulation runs, so it measures generalisation to unseen time
windows of runs seen in training, not to unseen simulations. Three of the eleven modelled genes
are moved by a mechanism that is not their real one, and several parameters that resemble
measurements are invented; all are labelled as such. No external validation against real
genomic data exists: our attempt failed because the isolate data we could obtain carries
resistance phenotypes rather than per-isolate gene calls. Under these conditions we read a high
AUROC as a statement about the difficulty of the task as posed rather than as evidence that the
model has learned conjugation. What we take to be transferable is the formulation — an exactly
labelled, per-contact, per-gene transfer task — and the finding that training one model across
all genes reaches rare targets that per-gene models cannot.
