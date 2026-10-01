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
of 0.9805 ± 0.0021 on the five genes our per-gene random forest baseline fits in every seed,
against 0.9647 ± 0.0031 for that baseline, ahead on every seed. That gap stayed within 0.003 of
itself across a threefold increase in data and a correction to the evaluation, which makes it
the most durable number we report.

The graph itself earns little. Message passing changes the headline by +0.0065 ± 0.0086 and is
positive in only three seeds of five, so we cannot distinguish it from zero; the genes a cell
already carries dominate every other feature group by an order of magnitude. On this task most
of the signal is available from the two endpoint feature vectors. We report that rather than the
more flattering alternative reading — and note that an earlier version of this work did report
the flattering reading, because its evaluation was wrong.

The scope of these numbers is narrow, and deliberately stated as such throughout. They are
internal to one simulator; no external validation exists, and our attempt at one failed for want
of gene-level calls in the data we could obtain. The simulator transfers genes only between
cells of the same species, so the interspecies transfers that dominate the literature on these
very genes are absent from the task, the labels and the evaluation alike. The held-out split
holds out whole simulation runs, but those runs come from the same five scenarios and five
species used in training, so generalisation to unseen scenarios or species remains untested. Three of the eleven genes are moved by a mechanism
that is not their real one, and several parameters that resemble measurements are invented. Under those conditions, a high AUROC is better read as a statement about the
difficulty of the task as posed than as evidence of a model that understands conjugation.

What we think survives those caveats is the formulation and one methodological result: an
exactly labelled, per-contact, per-gene transfer task is a usable testbed, and the cost of
splitting such a task by snapshot rather than by run is large and concentrated exactly where a
reader would least expect it — in the rarest, most eye-catching per-gene scores. The pipeline that produces these
results is seeded and deterministic across processes, with a frozen configuration guarded
byte-for-byte by a regression test, so the numbers can be reproduced and, more importantly,
can be moved by the changes set out in Section 7: adding interspecies transfer, modelling
per-gene mobility, validating against real gene calls, and testing unseen scenarios. One such
change has already been made and is reported here — splitting by run rather than by
snapshot — and it lowered almost every number in this paper. We expect several of those
changes to lower the figures reported here, and regard that as the point of making them.
