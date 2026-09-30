<!--
DRAFT — Section 4: Results. Status: awaiting owner review.

claims_to_numbers.md entries used: S1, S2 (headline), S2b, S3, S4, S5, S6, S7, S8, S9,
U4 (with its caveat), U5 (why the 8-gene gap is not quoted), U6 (vanA separate), U8
(blaTEM-1 not evaluable), U10 (simplified mechanisms), U12 (per-gene counts are a seeding
artefact), U13 (what the held-out split measures).

Every figure is copied from claims_to_numbers.md. No new numbers are computed here, with one
exception flagged inline: per-gene positive counts for blaCTX-M-15, blaKPC-2, blaNDM-1 and
mcr-1 are NOT in claims_to_numbers.md, so they are not stated (open item, §4.6).
-->

# 4 Results

All figures are means and standard deviations over five independent model seeds (three for the
ablations), on the held-out split described in Section 3.3.3. That split is over snapshot
pairs rather than over runs, so throughout this section "test" means unseen time windows of
runs that contributed training data, not unseen runs, scenarios or species. Unless stated
otherwise, "headline AUROC" is the macro average over the eight genes fixed by the evaluation
policy of Section 3.7, one of which (blaTEM-1) has no positives in the test split, so the
macro is in practice over seven.

## 4.1 The GNN outperforms per-gene classical baselines

On the five genes for which the per-gene random forest can be trained — blaCTX-M-15,
blaKPC-2, blaNDM-1, mcr-1 and tetM — the GNN reaches a test AUROC of
**0.9735 ± 0.0017** against **0.8950 ± 0.0386** for the random forest. The GNN is ahead on
**all five of five seeds**. We treat this as the primary comparison, because it is the only
one in which both models are actually fitted to the same genes.

The GNN's own headline macro, over the eight genes of the evaluation policy, is
**0.9805 ± 0.0012**. Logistic regression reaches **0.7096 ± 0.0359** on the same headline
macro.

We deliberately do not quote a headline-macro *difference* between the GNN and the random
forest. The random forest's headline macro is depressed by the two genes it cannot train at
all, which are scored 0.5 by construction rather than by any prediction; a difference computed
across all eight genes would therefore combine a genuine performance gap with a coverage gap
and overstate the former. The two are reported separately: performance in this subsection, on
the five shared genes, and coverage in §4.2.

## 4.2 Coverage of genes the baselines cannot train on

Our per-gene random forest and logistic-regression baselines require at least five positive
examples for a gene within their 100,000-edge training subsample. Two genes fall below that
threshold, so neither baseline can be fitted for them and both are scored 0.5 by construction.
The GNN, which is trained jointly on all ten gene outputs, still produces evaluable
predictions for them:

| Gene | GNN test AUROC | Positive events in the whole dataset | Baselines |
|---|---|---|---|
| acrAB-tolC | 0.9986 ± 0.0018 | 19 | not trainable |
| gyrA_S83L | 0.9977 ± 0.0006 | 34 | not trainable |

Two qualifications belong with these figures, and neither is incidental.

First, the counts. Nineteen and thirty-four positive events across the whole dataset are very
few, and the standard deviations above are over model seeds on a fixed split, so they do not
reflect the sampling uncertainty that such counts imply. These AUROCs should not be read as
precise.

Second, the mechanism. Both of these genes are chromosomal in real bacteria and are moved
between neighbouring cells by the simulator as a deliberate simplification (Section 3.1.2).
Their transfer events are therefore not models of documented conjugative transfer. What this
result demonstrates is a property of the learning setup — that joint training across genes
yields usable predictions for targets too rare to fit individually — and not a biological
finding about these two genes.

With that scope stated, the coverage difference is a real advantage of the approach rather
than an artefact of the comparison: it follows from training one model on all genes instead of
one model per gene. It is a property of the baselines *as we implemented them*, namely
one-versus-rest with a five-positive requirement, and not of random forests in general. We did
not lower that threshold, since doing so would have changed the baseline rather than tested
it.

The GNN is evaluated on seven of the eight policy genes plus vanA separately. The eighth,
blaTEM-1, has five positive events in the entire dataset and none in the test split, so no
model of any kind can be scored on it; we report it as not evaluable rather than as a result.

## 4.3 Where the signal comes from

**Message passing contributes a small but consistent gain.** Removing the attention blocks, so
that node and edge encodings feed the prediction head directly, costs
**0.0031 ± 0.0014** headline AUROC (0.9805 with message passing against 0.9774 ± 0.0006
without). The direction is consistent across all five of five seeds. The effect is reliable
and small: the graph-free model retains almost all of the performance, so most of what the
model exploits is available from the two endpoint feature vectors alone rather than from the
surrounding contact structure.

**The genes a cell already carries dominate.** Zeroing feature groups one at a time, only the
genomic group matters appreciably: removing it costs **0.0549 ± 0.0331** headline AUROC
(three seeds). Every other group changes the headline by less than ±0.003, which is within
seed-to-seed variation and which we therefore do not interpret as an effect in either
direction.

**Local antibiotic exposure adds no measurable signal.** Zeroing the six drug-concentration
features changes the headline by **+0.0001 ± 0.0001** — that is, removing them does not hurt.
This deserves an explicit caveat rather than a conclusion about antibiotic pressure: under our
dosing protocol the drug is present for only five of the eighty simulated steps, so most
snapshots carry no exposure to measure. The correct reading is that these features carry no
signal *in this protocol*, not that antibiotic exposure is irrelevant to gene transfer.

## 4.4 Calibration and precision–recall

Predicted probabilities are well calibrated, with an expected calibration error of
**0.0012 ± 0.0003**.

Headline AUPRC is **0.0316 ± 0.0293**. This is far above the base rate of approximately
8.5 × 10⁻⁵ positives per edge-gene pair, so the ranking carries real information in absolute
terms; but the standard deviation is nearly as large as the mean, so the figure is unstable
across seeds and we do not treat it as a headline result. The instability is expected given
how few positives the rarer genes contribute.

## 4.5 Per-gene discrimination

| Gene | GNN test AUROC | Positive events in dataset |
|---|---|---|
| mcr-1 | 0.9908 ± 0.0011 | not recorded in our claims table (see below) |
| tetM | 0.9805 ± 0.0033 | 420 |
| blaNDM-1 | 0.9755 ± 0.0039 | not recorded |
| blaCTX-M-15 | 0.9642 ± 0.0030 | not recorded |
| blaKPC-2 | 0.9564 ± 0.0021 | not recorded |
| acrAB-tolC | 0.9986 ± 0.0018 | 19 — simplified mechanism |
| gyrA_S83L | 0.9977 ± 0.0006 | 34 — simplified mechanism |
| blaTEM-1 | not evaluable | 5, none in the test split |

**We offer no biological interpretation of the ordering in this table, because the variation
in it is confounded.** Two confounds are enough to account for it without appeal to gene
biology. The first is sample size: the number of positives per gene spans at least two orders
of magnitude, and the genes with the fewest positives are the ones with the most extreme
AUROCs. The second is the seeding artefact of Section 3.1.2: no acquired gene exists at
initialisation, so every gene's count is set by how often the donor-free mutation branch
happens to seed it — which scales inversely with the size of its species' acquirable set — and
then by its invented per-step transfer probability. A gene's position in this table therefore
reflects the simulator's seeding code and its sample size, not the mobility of the
corresponding real gene. We report the values for completeness and for comparison against
future versions of the pipeline, and draw no per-gene conclusions from them.

*Open item:* per-gene positive counts for blaCTX-M-15, blaKPC-2, blaNDM-1 and mcr-1 are not
recorded in `paper/claims_to_numbers.md`. They should be read out of the results file and
added there before submission, so that every AUROC in this table can be quoted with its count.

## 4.6 vanA, reported separately

vanA reaches a test AUROC of 0.9988 on 35 positive events. It is excluded from the headline
macro and reported here on its own, because the mechanism the simulator gives it is a
simplification in two respects: the gene arises and spreads within MRSA, whereas the
documented route is interspecies transfer of Tn1546 from *Enterococcus faecalis*, and
*Enterococcus* is not modelled at all. This figure is a discrimination measurement on
simulated events, not evidence about vancomycin resistance transfer.

mexAB-oprM is excluded entirely: it is present only as an intrinsic gene, appears in no
species' acquirable set, and therefore has zero transfer events by construction. There is
nothing to predict and no score to report.

## 4.7 Reproducibility

Data generation and training are deterministic across processes. Identical simulated states
and identical training pairs are produced under different Python hash seeds, which a
regression test enforces byte-for-byte for the frozen `paper_v1` configuration. Retraining the
same seed on a GPU reproduces test AUROC to within 2.1 × 10⁻⁴. Every figure in this section is
a mean and standard deviation over independent model seeds, with the data held fixed, so the
reported spreads capture variation from initialisation and training order and not from data
generation.

## 4.8 Summary

The GNN discriminates simulated per-edge, per-gene transfer events well (headline AUROC
0.9805 ± 0.0012) and beats a per-gene random forest on the genes that baseline can be fitted
to (0.9735 ± 0.0017 against 0.8950 ± 0.0386, on five of five seeds). Its advantage in coverage
— usable predictions for genes too rare for the per-gene baselines — follows from joint
training. Most of the signal is carried by the genes a cell already holds; message passing adds
a small, consistent increment; edge features and local drug exposure add nothing measurable in
this setting. What none of these figures establish is stated in Section 5: no external
validation, a held-out split that does not test unseen runs, and an intraspecies-only
transfer model.
