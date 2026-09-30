<!--
DRAFT — Section 2: Related Work. Status: awaiting owner review.
Sources: audited references in paper/citations.md (keys match paper/refs.bib).
claims_to_numbers.md entries used: U7 (results are within-simulation; no external validation).
No numbers from our results appear in this section. No numbers from cited papers are quoted.
Code fact used (verified 2026-09-30): per-step transfer probabilities (`acquisition_prob`,
data/card_loader.py) carry no source and are not fitted to experimental data.
-->

# 2 Related Work

## 2.1 Individual-based models of bacteria and plasmid transfer

Individual-based (agent-based) models represent each microbial cell explicitly. They have been
used to study how the behaviour of individual cells gives rise to population-level patterns, from
biofilms to phage–CRISPR dynamics [hellweger2016]. General-purpose frameworks include iDynoMiCS,
for individual-based modelling of biofilms [lardon2011], and BSim, an agent-based tool for
bacterial populations in systems and synthetic biology [gorochowski2012].

Several studies model plasmid transfer specifically.
- Krone et al. present an individual-based lattice model of plasmid transfer and persistence
  in spatially structured populations, and compare its predictions with agar-surface
  experiments [krone2007].
- Merkey et al. extend an individual-based biofilm model with plasmid carriage and transfer
  by individual cells. They use it to show, in silico, that growth-dependent conjugation
  limits plasmid invasion of biofilms [merkey2011].
- Seoane et al. use an individual-based experimental framework to estimate the main
  parameters governing conjugation at the single-cell scale [seoane2011].
- At a much coarser scale, the VERA model simulates the spread of resistance between people,
  including transfer of resistance determinants from commensal gut bacteria to a pathogen
  [glushchenko2019].

These models use simulation to test mechanistic hypotheses about how plasmids and resistance
spread. We use simulation for a different purpose: as a source of exactly labelled transfer
events for training and evaluating a predictive model. This purpose carries a limitation that
the work above does not share to the same degree. Krone et al., for example, compared their
model against experiments. Our simulator's per-step transfer probabilities are not fitted to
experimental conjugation data, and all of our results are within-simulation (Section 5).

## 2.2 Machine learning for resistance and gene-transfer prediction

Machine learning is increasingly used to predict antimicrobial resistance phenotypes from
bacterial gene content and genome composition. A recent review notes that such models
typically treat genes as independent predictors [kim2022].

Closer to our task, two lines of work predict the transfer of resistance genes rather than
the resistance phenotype.
- Ellabaan et al. identify putative horizontally transferred resistance genes and the
  gene-exchange networks that disseminate them across bacterial genomes. They then use the
  associated mobilisation elements to forecast where known resistance genes may spread
  [ellabaan2021]. The confirmatory analysis in that paper was later revised in an Author
  Correction.
- Zhou et al. predict a network of recent HGT events between genomes from their functional
  gene content [zhou2021]. An event is defined by near-identical DNA shared by distantly
  related organisms. A random forest on functional profiles gave the best performance. A
  baseline graph convolutional network performed similarly to it, and improved as
  network-topology information was added.

Both operate on real genomes. Transfers are therefore inferred from sequence similarity, at
the level of genomes and over evolutionary time. Our task is defined at the level of
individual cells and their contacts, over the next simulated time window, with labels taken
from recorded events. The two settings trade off in opposite directions. Genome-level work
has real data but indirect labels; ours has exact labels but simulated data.

## 2.3 Graph neural networks for antimicrobial resistance

Graph neural networks have recently been applied to resistance problems in which graph nodes
are isolates or patients.
- AMR-GNN treats each bacterial isolate as a node, connects isolates by genomic distance, and
  predicts resistance phenotypes by node classification [nguyen2026amrgnn].
- Donabauer et al. apply graph neural networks to time-dependent graphs of patient movements
  within a hospital to identify carriers of vancomycin-resistant enterococci [donabauer2025].
- The graph convolutional network of Zhou et al. (Section 2.2) operates on a network of
  genomes [zhou2021].

Our graph is at a finer scale. Nodes are individual simulated bacterial cells, edges are
spatial contacts between them, and the prediction target is defined per edge and per gene:
whether a given resistance gene is transferred along a given contact. We use graph attention
[velickovic2018] for this.

In the literature we reviewed, we did not find prior work that poses per-contact, per-gene HGT
prediction among individual cells as a supervised learning problem. We position our
contribution as bringing together the individual-based modelling of plasmid transfer
(Section 2.1) and graph-based learning for resistance (Sections 2.2–2.3). We do not claim to
identify a previously unstudied problem.
