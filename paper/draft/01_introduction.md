<!--
DRAFT — Section 1: Introduction. Status: awaiting owner review.
Sources: paper/claims_to_numbers.md entries S2, S2b, S4, S5, S8, U3, U4, U7.
Citations: [key] = BibTeX key in paper/refs.bib; each is audited in paper/citations.md
(2026-09-30). Two sentences in paragraph 1 were reworded during the audit (see citations.md
"Open items").
No numbers appear here that are not in claims_to_numbers.md.
-->

# 1 Introduction

Antimicrobial resistance (AMR) is a major and growing threat to the treatment of bacterial
infections [gbd2019amr; gbd2021amr]. A central route by which resistance spreads
between bacteria is horizontal gene transfer (HGT), in which resistance genes move from one cell
to another — for example by plasmid conjugation — rather than arising independently in each
lineage [vonwintersdorff2016; partridge2018]. Whether a transfer happens depends on local,
cell-level conditions such as contact between cells and the physiological state of the donor
[seoane2011; merkey2011]. In natural bacterial communities, however, there is a gap between
what is known about HGT from laboratory experiments and what is known from natural
environments [brito2021], and even reliably assigning mobile genetic elements to their host
cells in such communities has been difficult [yaffe2020].

Agent-based simulation offers a setting in which every transfer event is recorded. In an
agent-based (individual-based) model, each bacterium is an explicit agent that follows stated
rules [hellweger2016], and such models have been used to study plasmid transfer in spatially
structured populations [krone2007; merkey2011]. In our model these rules cover growth, death,
stress responses and gene exchange, so the ground truth of *which cell passed which gene to
which neighbour* is known exactly. This
makes it possible to pose a supervised learning problem that is not available from observational
data: given a snapshot of a bacterial population, predict which resistance genes will be
transferred along which cell–cell contacts in the next time window. Graph neural networks (GNNs)
are a natural fit for this problem, because the population can be represented as a graph of
cells (nodes) connected by spatial proximity (edges), and the prediction target is defined per
edge; we use graph attention networks [velickovic2018].

In this paper we build such a system and evaluate it with the aim of reporting only what the
numbers support. We develop an agent-based AMR simulation with explicit conjugation-style HGT,
and a graph attention network that predicts per-gene transfer on each directed cell–cell edge.
All reported results are within-simulation: the labels are the simulator's own recorded transfer
events, and we have not yet validated the predictions against real genomic data (Section 5).

**Scope.** Our simulator models *intraspecies* transfer only: a gene moves between two cells
of the same species, so a contact between cells of different species is a negative example by
construction. This is a substantial restriction rather than an incidental one, because
interspecies and intergenus transfer of several of the exact genes we model is documented —
the mcr-1 colistin-resistance plasmid was conjugated into *Escherichia coli* and maintained in
*Klebsiella pneumoniae* and *Pseudomonas aeruginosa* [liu2016mcr], and an NDM-1
carbapenemase plasmid was conjugated from *Citrobacter freundii* into an *E. coli* recipient
[dolejska2012]. The prediction task we report on is therefore easier than the corresponding
task in a community of mixed species. A second qualification applies to the evaluation
itself: our held-out split holds out whole simulation runs, so the figures below do measure
generalisation to unseen runs — but those runs come from the same five scenarios and five
species used in training, so generalisation to unseen scenarios or species is untested. Both
restrictions should be kept in mind when reading the figures below (Sections 3, 5 and 6).

Our contributions are:

1. **A per-edge, per-gene HGT prediction task grounded in recorded simulation events**, with a
   documented simulation whose biology exists in two explicitly versioned forms. Results are
   reported on the corrected version (Section 3).

2. **A GNN that outperforms a per-gene Random Forest baseline, under the evaluation described
   above.** On the five resistance genes our per-gene Random Forest baseline can be fitted to
   in every seed, the GNN reaches a test AUROC of 0.9805 ± 0.0021, against 0.9647 ± 0.0031 for
   the Random Forest, and is better on all 5 of 5 seeds (Section 4). Both models are evaluated
   on the same held-out runs, so this is a like-for-like comparison between the two approaches
   on this task. It is the one result that survived both a threefold increase in simulated data
   and a correction to our evaluation (Section 4.9), which is why we lead with it; it is not
   evidence about behaviour on unseen scenarios, unseen species, or interspecies transfer, none
   of which we test.

3. **A measurement of what a snapshot-level split costs on this kind of task.** An earlier
   version of this work split the data by snapshot rather than by simulation run, so snapshots
   three simulated steps apart — from the same run and the same founding population — sat on
   both sides of the split. Rerunning both splits on identical data shows that this inflated
   the headline by 0.0176 AUROC and the two rarest genes by roughly 0.09 and 0.07, turning
   scores of 0.999 into 0.87–0.89 (Section 4.9). We report this because the inflation is
   concentrated precisely where it is least visible and most flattering: on the rare genes
   whose near-perfect scores are the most quotable numbers a paper like this produces.

4. **An honest account of where the signal comes from.** Message passing over the contact graph
   shows no reliable benefit: +0.0065 ± 0.0086 AUROC, positive in only 3 of 5 seeds, a spread
   that includes zero. The genes a cell already carries are by far the most important input
   (−0.0442 ± 0.0071 when removed, with no other feature group distinguishable from zero), while
   edge features and local antibiotic exposure add no measurable signal. For antibiotic exposure
   this is under our dosing protocol, in which drug is present for only 5 of the 80 simulated
   steps (Section 4).

5. **A reproducible pipeline**: the simulation and training-data generation are seeded and
   deterministic across processes (verified byte-for-byte for the frozen `paper_v1`
   configuration), GPU training with the same seed reproduces test AUROC to within
   1.6 × 10⁻³, and every reported number is a mean ± standard deviation over independent model
   seeds (Section 3).
