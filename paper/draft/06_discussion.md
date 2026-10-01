<!--
DRAFT — Section 6: Discussion. Status: awaiting owner review.

claims_to_numbers.md entries used: S1, S2, S2b, S4, S5, S6, S7, U4 (with caveat), U5, U7,
U10, U11, U12, U13. No new numbers; figures are restated only where the argument turns on
them.
Sources: paper/citations.md (keys in paper/refs.bib) — zhou2021, nguyen2026amrgnn,
liu2016mcr, dolejska2012, kim2022.

Thesis: the results support a claim about the task formulation and about joint multi-gene
training, and do NOT support a claim about graph structure. Written to avoid restating
Section 4.

What was §6.5 ("What we would change first") was expanded into its own Section 7 (Future
Work) on 2026-09-30 at the owner's request; the old §6.6 is now §6.5.
-->

# 6 Discussion

## 6.1 What the headline number does and does not demonstrate

A headline AUROC of 0.9676 ± 0.0076 invites a stronger reading than our experiments support,
so it is worth being precise about what produced it.

Three of our own results, taken together, suggest that most of the achievable performance
comes from the pair of endpoint feature vectors rather than from the surrounding contact
structure. Removing message passing entirely changes the headline by +0.0065 ± 0.0086 and is positive in
only three seeds of five, so its contribution cannot be distinguished from zero; a model with no
graph convolution at all retains essentially all of the performance. Zeroing feature groups one
at a time shows a single group that matters: the genes a cell already carries, at
0.0442 ± 0.0071, with no other group distinguishable from zero. And edge features are zeroed in the reported configuration
without measurable cost.

The likeliest explanation is that transfer in our simulator requires the donor to carry the gene
and the recipient to be able to acquire it, and both conditions are readable from the two
endpoint gene vectors. That would account for the genomic group dominating and for the contact
graph adding little.

It is worth ruling out a tempting but incorrect second explanation, since we entertained it
ourselves. Because transfer is intraspecies only, one might expect the negative class to be
padded with cross-species contacts that any model can reject from the species one-hot alone.
We measured this and it is not so: only 0.107% of contacts in our dataset are cross-species
(6,428 of 6,029,316), they arise in only one of the five scenarios, and none of them carries a
transfer. Four of the five scenarios contain a single species, and in the one mixed scenario the
two populations are seeded as separate spatial clusters that rarely come within contact range.
The intraspecies restriction therefore does *not* make the task easier by supplying easy
negatives; what it does instead is make the interspecies case effectively absent from the
evaluation, so the model is assessed almost entirely on within-species discrimination (§6.4).

Combined with a held-out split taken over snapshot pairs rather than over runs — so that test
windows come from runs the model trained on — the natural reading of 0.98 is that **the task as
we posed it is substantially easier than the phrase "predicting horizontal gene transfer"
suggests**, not that the model has acquired a deep representation of conjugation. Note that the
"easier" here rests on the label structure and the split, not on the class-balance argument we
just rejected.

We think this matters beyond our own paper. High discrimination on a simulator-derived task is
easy to report and hard to interpret, because the task's difficulty is set by modelling
decisions that are invisible in the metric. Our own pipeline supplies two cautionary examples,
and we think the second is the more useful contribution of this work.

The first is feature leakage: before we removed four inputs that were literal components of the
label-generating rule, the same architecture scored higher than it does now. The number improved
when the experiment got worse.

The second is subtler, and we would not have found it by inspection. An earlier version of this
work split the data by snapshot rather than by simulation run. Snapshots are taken every three
simulated steps, so the training and test sets contained near-duplicate views of the same
population — the same individual cells, three steps apart. Nothing about that is visible in a
metric. Rerunning the entire pipeline twice on identical data, once per split, isolates what it
was doing: the headline fell by 0.0176 AUROC, which is modest, **but the two rarest genes fell
by roughly 0.09 and 0.07, from 0.9986 and 0.9988 to 0.8890 and 0.8669** (§4.9).

The distribution of that error is the point. The inflation was not spread evenly: it concentrated
on the genes with the fewest held-out positive events — five each — which are exactly the genes
whose scores look most impressive and get quoted most readily. Under the broken split those two
genes topped the per-gene table at 0.999; under the correct one they sit at the bottom. An
earlier draft of this paper contained a figure whose stated message was that the genes with the
fewest examples scored highest. That was not a finding about rare-gene learnability. It was a
leakage signature, and we had written a caption explaining it as though it were a result.

The general lesson is not "group your splits", which is standard advice. It is that when a task
is generated by a simulator, the unit of independence is the *simulation run*, not the sample,
and that violating it does its worst damage in the least conspicuous place: the small-sample
rows of a per-gene table, where there is least evidence and the most eye-catching numbers.

## 6.2 What the results do support: joint training across genes

The result we consider most robust is also the one least dependent on the graph. Our per-gene
baselines cannot be fitted for genes with fewer than five positive examples in the training
subsample they draw. On a smaller dataset this bound on three genes and we presented it as a
contribution; at the present dataset size it binds on one, acrAB-tolC, whose score rests on
five held-out events. We have demoted the claim accordingly (§4.2, §5.1), and record here that
it was data volume, not any change of method, that dissolved it. That advantage follows from parameter sharing
across a multi-label output, not from message passing — the graph-free variant would inherit
it too.

This connects to a known weakness of the per-gene approach in the wider literature: a recent
review of machine learning for resistance prediction notes that such models typically treat
genes as independent predictors [kim2022]. Our setting makes the cost of that assumption
concrete and measurable, because the rare targets are exactly the ones a per-gene model cannot
reach. The mechanism is mundane — shared representations let common genes subsidise rare ones —
but for a problem where the interesting determinants are often the rare ones, it is the part of
our design we would keep.

What survives is narrower and worth stating precisely: a per-gene model cannot be built at all
below its positive threshold, whereas a jointly trained one always emits a prediction. That is
structurally true and will matter whenever the targets of interest are rare. It is not, on this
evidence, a quantified advantage — the one gene it still applies to is measured from five
held-out events, and it is a gene our simulator moves by a mechanism that is not its real one.

## 6.3 Relation to prior approaches

The closest prior work predicts horizontal transfer between *genomes*: Zhou et al. infer recent
transfer events from near-identical sequence shared by distantly related organisms and predict
the resulting network from functional gene content, finding that a random forest on functional
profiles performs best and that a graph convolutional network performs comparably, improving as
network topology is added [zhou2021]. Recent graph neural network work on resistance operates
at the isolate or patient level, classifying nodes that are whole isolates connected by genomic
distance [nguyen2026amrgnn].

Our formulation sits at a different scale — individual cells, their momentary contacts, and a
specific next time window — and the trade is explicit. Genome-level work has real data and
indirect labels, inferred from sequence similarity over evolutionary time. We have exact labels
and simulated data. Neither is a substitute for the other, and our numbers are not comparable
with theirs: different unit of prediction, different notion of an event, different data. We draw
no comparison of magnitudes anywhere in this paper.

It is worth noting that the direction of our own ablation echoes theirs. In their setting a
non-graph model matched or beat the graph model on functional features; in ours we cannot
distinguish the contribution of message passing from zero. Two studies at different scales both
finding limited benefit from graph structure is weak evidence, but it points the same way, and
we would rather record it than present message passing as more load-bearing than it is. We note
that our own earlier, uncorrected evaluation did make it look load-bearing and consistent.

## 6.4 The interspecies gap

The restriction we consider most limiting is that a gene never crosses a species boundary in
our simulator. The literature on the very genes we model is largely about transfers that do:
the mcr-1 plasmid conjugated into *Escherichia coli* and maintained in *Klebsiella pneumoniae*
and *Pseudomonas aeruginosa* [liu2016mcr], an NDM-1 plasmid conjugated from *Citrobacter
freundii* into *E. coli* [dolejska2012].

This is not a gap that a better model closes. It is a gap in the simulator, and it propagates
into the task, the labels and the evaluation: a model trained where cross-species contacts are
always negative has had no opportunity to learn anything about the case of greatest practical
interest, and our metrics cannot detect that it has not. Adding interspecies transfer means
deciding, per gene, which species boundaries it can cross and at what rate — which is a
modelling problem requiring sourced host-range information rather than a parameter to tune. We
regard it as the most valuable extension, and the one most likely to reduce the headline
numbers, which we would treat as the experiment working rather than failing. Section 7.3
sets out what it would take.

## 6.5 On calibration and precision–recall

Two secondary results deserve a brief, honest reading. Predicted probabilities are well
calibrated (expected calibration error 0.0012 ± 0.0003), which is useful if such a model were
ever to inform a ranking, but calibration on a simulator's own events says nothing about
calibration against reality. Headline AUPRC is far above the base rate yet has a standard
deviation nearly equal to its mean, so we treat it as evidence that the ranking carries
information and not as a stable performance figure. Under extreme class imbalance the
precision–recall summary is dominated by the handful of genes with the fewest positives, which
is precisely where our dataset is least informative.

Finally, the finding that local antibiotic exposure adds no measurable signal should not be
read as a claim about selection pressure. Under our protocol the drug is present for five of
eighty steps. The honest conclusion is that these features carry no signal in this protocol,
and that a protocol with sustained exposure would be needed to say anything more.
