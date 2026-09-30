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

Our contributions are:

1. **A per-edge, per-gene HGT prediction task grounded in recorded simulation events**, with a
   documented simulation whose biology exists in two explicitly versioned forms. Results are
   reported on the corrected version (Section 3).

2. **A GNN that outperforms a per-gene Random Forest baseline.** On the five resistance
   genes for which our per-gene Random Forest baseline can be trained, the GNN reaches a test
   AUROC of 0.9735 ± 0.0017, against 0.8950 ± 0.0386 for the Random Forest, and is better on all
   5 of 5 seeds (Section 4).

3. **Coverage of rare genes.** Our per-gene Random Forest and logistic-regression baselines, as
   implemented, need at least five positive examples in their training subsample, and so cannot
   be trained for two of the genes. The GNN, trained jointly on all genes, still gives evaluable
   predictions for them (acrAB-tolC and gyrA_S83L, with 19 and 34 positive transfer events in
   the whole dataset respectively; Section 4). Both of these genes are chromosomal in real
   bacteria, and our simulator moves them between neighbouring cells as a deliberate
   simplification; their transfer events are therefore not models of documented conjugative
   transfer, and we label them as such throughout (Sections 3 and 5).

4. **An honest account of where the signal comes from.** Message passing over the contact graph
   adds a small but consistent gain (+0.0031 ± 0.0014 AUROC, better on 5 of 5 seeds). The genes a
   cell already carries are by far the most important input, while edge features and local
   antibiotic exposure add no measurable signal in our setting. For antibiotic exposure this is
   under our dosing protocol, in which drug is present for only 5 of the 80 simulated steps
   (Section 4).

5. **A reproducible pipeline**: the simulation and training-data generation are seeded and
   deterministic across processes (verified byte-for-byte for the frozen `paper_v1`
   configuration), GPU training with the same seed reproduces test AUROC to within
   2.1 × 10⁻⁴, and every reported number is a mean ± standard deviation over independent model
   seeds (Section 3).
