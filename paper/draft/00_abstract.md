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

The network reaches a test AUROC of 0.9805 ± 0.0021 on the five genes our per-gene random
forest can fit in every seed, against 0.9647 ± 0.0031 for that baseline, ahead on all five
model seeds; its macro average over the eight genes in our evaluation policy is
0.9676 ± 0.0076. That comparison is the one result that held steady across both a threefold
increase in simulated data and a correction to the evaluation itself. The graph contributes
little: removing message passing between cells changes the headline by +0.0065 ± 0.0086 and is
positive in only three seeds of five, so we cannot distinguish its contribution from zero,
while the genes a cell already carries dominate every other feature group by an order of
magnitude.

We are explicit about what these figures do not establish. The simulator transfers genes only
between cells of the same species, so the interspecies transfers that dominate the literature
on these genes are absent from the task, the labels and the evaluation alike; in our dataset
only 0.107% of contacts are cross-species, and none of them carries a transfer, so the model is
evaluated almost entirely on within-species pairs. Our held-out split holds out whole
simulation runs, but those runs come from the same five scenarios and five species used in
training, so generalisation to unseen scenarios or species is untested. Three of the eleven
modelled genes
are moved by a mechanism that is not their real one, and several parameters that resemble
measurements are invented; all are labelled as such. No external validation against real
genomic data exists: our attempt failed because the isolate data we could obtain carries
resistance phenotypes rather than per-isolate gene calls. Under these conditions we read a high
AUROC as a statement about the difficulty of the task as posed rather than as evidence that the
model has learned conjugation. We also report a methodological result we think generalises
beyond this system: an earlier version of this work split the data by snapshot rather than by
simulation run, and that alone inflated the headline by 0.0176 AUROC and the two rarest genes
by roughly 0.09 — turning scores of 0.999 into 0.87–0.89. What we take to be transferable is
the formulation, an exactly labelled per-contact, per-gene transfer task, together with that
cautionary result.
