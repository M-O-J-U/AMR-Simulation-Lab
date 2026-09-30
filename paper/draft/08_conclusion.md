<!--
DRAFT — Section 8: Conclusion. Status: awaiting owner review.

claims_to_numbers.md entries used: S2, S2b, S4, S5, S8, U7, U11, U13. No new numbers.
Deliberately short, and deliberately does not upgrade any claim made earlier.
-->

# 8 Conclusion

We posed horizontal gene transfer prediction as a supervised problem at the level of individual
cells: given a snapshot of a simulated bacterial population, predict for each directed
cell-to-cell contact and each resistance gene whether that gene is transferred in the next time
window. Because the labels are events the simulator records as they fire, the ground truth is
exact — which is the property that motivated the formulation and is not available from
observational genomic data.

On this task a graph attention network trained jointly on all gene outputs reaches a test AUROC
of 0.9777 ± 0.0023 on the four genes our per-gene random forest baseline fits in every seed,
against 0.9548 ± 0.0124 for that baseline, and is ahead on every seed. It also gives stable
predictions for three further genes the baselines fit unreliably or not at all — an advantage
that comes from sharing parameters across a multi-label output rather than from the graph.

That last distinction is the paper's main methodological conclusion. Message passing contributes
a small but consistent 0.0031 ± 0.0014, and the genes a cell already carries dominate every
other feature group by an order of magnitude. On this task, most of the signal is available from
the two endpoint feature vectors, and what earns the architecture its place is joint training
across genes, not the propagation of information over the contact graph. We report that rather
than the more flattering alternative reading.

The scope of these numbers is narrow, and deliberately stated as such throughout. They are
internal to one simulator; no external validation exists, and our attempt at one failed for want
of gene-level calls in the data we could obtain. The simulator transfers genes only between
cells of the same species, so the interspecies transfers that dominate the literature on these
very genes are absent from the task, the labels and the evaluation alike. The held-out split is
taken over snapshot pairs rather than over runs, so it measures generalisation to unseen time
windows of seen simulations and not to unseen simulations. Three of the eleven genes are moved
by a mechanism that is not their real one, and several parameters that resemble measurements are
invented. Under those conditions, a high AUROC is better read as a statement about the
difficulty of the task as posed than as evidence of a model that understands conjugation.

What we think survives those caveats is the formulation and one design choice: an exactly
labelled, per-contact, per-gene transfer task is a usable testbed, and training one model across
all genes reaches rare targets that per-gene models cannot. The pipeline that produces these
results is seeded and deterministic across processes, with a frozen configuration guarded
byte-for-byte by a regression test, so the numbers can be reproduced and, more importantly,
can be moved by the changes set out in Section 7: splitting by run, adding interspecies
transfer, modelling per-gene mobility, and validating against real gene calls. We expect several of those
changes to lower the figures reported here, and regard that as the point of making them.
