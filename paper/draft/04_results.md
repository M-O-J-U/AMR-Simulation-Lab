<!--
DRAFT — Section 4: Results. Status: awaiting owner review.

claims_to_numbers.md entries used: S1, S2 (headline), S2b, S3, S4, S5, S6, S7, S8, S9,
U4 (with its caveat), U5 (why the 8-gene gap is not quoted), U6 (vanA separate), U8
(blaTEM-1 not evaluable), U10 (simplified mechanisms), U12 (per-gene counts are a seeding
artefact), U13 (what the held-out split measures).

Every figure is copied from claims_to_numbers.md. No new numbers are computed here. (Per-gene
positive counts for all ten genes were added to claims_to_numbers.md on 2026-09-30, read from
the committed results file; an earlier draft of §4.5 left four of them blank.)
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

On the four genes the per-gene random forest can be trained on **in every seed** —
blaCTX-M-15, blaNDM-1, mcr-1 and tetM — the GNN reaches a test AUROC of
**0.9777 ± 0.0023** against **0.9548 ± 0.0124** for the random forest. The GNN is ahead on
**all five of five seeds**. We treat this as the primary comparison, because it is the only
one in which both models are actually fitted to the same genes in every run.

The qualifier "in every seed" is doing real work here, and an earlier version of this analysis
got it wrong. The baselines draw their 100,000-edge training subsample independently per seed,
so a gene's trainability is a per-seed property rather than a fixed one. blaKPC-2, with 94
positive events dataset-wide, clears the five-positive threshold in only two of the five
seeds; in the other three the random forest scores it 0.5 by construction. Including it in
this comparison — as we initially did — folds that construction artefact into the baseline's
score, which is precisely what §4.1's closing paragraph says must not be done. It inflated the
gap roughly threefold, from 0.023 to 0.079, and the tell was the baseline's implausibly large
standard deviation, which turned out to be bimodal across seeds. blaKPC-2 is reported under
coverage in §4.2 instead.

The GNN's own headline macro, over the eight genes of the evaluation policy, is
**0.9805 ± 0.0012**. Logistic regression reaches **0.7096 ± 0.0359** on the same headline
macro.

We deliberately do not quote a headline-macro *difference* between the GNN and the random
forest. The random forest's headline macro is depressed by the genes it cannot reliably train,
which are scored 0.5 by construction rather than by any prediction; a difference computed
across all eight genes would therefore combine a genuine performance gap with a coverage gap
and overstate the former. The two are reported separately: performance in this subsection, on
the four genes both models always fit, and coverage in §4.2.

## 4.2 Coverage of genes the baselines cannot reliably train on

Our per-gene random forest and logistic-regression baselines require at least five positive
examples for a gene within their 100,000-edge training subsample, and that subsample is drawn
independently for each seed. Trainability is therefore a per-seed property, and it comes in
degrees. The GNN, trained jointly on all ten gene outputs, produces stable predictions across
all five seeds for every gene below:

| Gene | Positives | Baselines trainable | GNN test AUROC | Random forest |
|---|---|---|---|---|
| acrAB-tolC* | 19 | 0 of 5 seeds | 0.9986 ± 0.0018 | 0.5 by construction, always |
| gyrA_S83L* | 34 | 0 of 5 seeds | 0.9977 ± 0.0006 | 0.5 by construction, always |
| blaKPC-2 | 94 | 2 of 5 seeds | 0.9564 ± 0.0021 | 0.9611 and 0.8179 where it trains |

\* transfer modelled by a simplified mechanism (Section 3.1.2).

We state the tiers separately because they are not the same claim. For acrAB-tolC and
gyrA_S83L the baseline is simply unavailable. For blaKPC-2 it is *unreliable*: it fits in two
seeds out of five, and where it does fit it returns 0.9611 and 0.8179 — a spread of 0.14 on
the same gene and the same data, against the GNN's 0.9564 ± 0.0021. The practical difference
between a model that sometimes cannot be built and one that is built but varies this much is
smaller than it looks, and blaKPC-2 is the cleaner illustration of the point, because unlike
the other two it carries no mechanism caveat: it is a genuine plasmid-borne gene, mobilised by
Tn4401 in reality as well as in our simulator.

Three qualifications belong with these figures, and none is incidental.

First, the counts. Nineteen, thirty-four and ninety-four positive events across the whole
dataset are few, and the standard deviations above are over model seeds on a fixed split, so
they do not reflect the sampling uncertainty that such counts imply. These AUROCs should not
be read as precise.

Second, the mechanism, for two of the three. acrAB-tolC and gyrA_S83L are chromosomal in real
bacteria and are moved between neighbouring cells by the simulator as a deliberate
simplification (Section 3.1.2), so their transfer events are not models of documented
conjugative transfer. What their result demonstrates is a property of the learning setup — that
joint training across genes yields usable predictions for targets too rare to fit
individually — and not a biological finding about those genes. blaKPC-2 carries no such caveat.

Third, this is a claim about our baselines as implemented, not about random forests. A
different baseline design — a single multi-label forest, or a lower positive threshold — would
face this differently, and we did not lower the threshold precisely because doing so would
change the baseline rather than test it.

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
| acrAB-tolC | 0.9986 ± 0.0018 | 19 — simplified mechanism |
| gyrA_S83L | 0.9977 ± 0.0006 | 34 — simplified mechanism |
| mcr-1 | 0.9908 ± 0.0011 | 179 |
| tetM | 0.9805 ± 0.0033 | 420 |
| blaNDM-1 | 0.9755 ± 0.0039 | 177 |
| blaCTX-M-15 | 0.9642 ± 0.0030 | 480 |
| blaKPC-2 | 0.9564 ± 0.0021 | 94 |
| blaTEM-1 | not evaluable | 5, none in the test split |

**We offer no biological interpretation of the ordering in this table, because the variation
in it is confounded.** Two confounds account for it without any appeal to gene biology.

The first is **sample size, and it runs opposite to the direction that would make the table
interpretable**. The positive counts span from 19 to 480, and the two highest AUROCs in the
table belong to the two genes with the fewest positives (acrAB-tolC, 19; gyrA_S83L, 34), while
the gene with by far the most positives (blaCTX-M-15, 480) sits near the bottom. A per-gene
AUROC estimated from 19 positives is not comparable to one estimated from 480, and reading the
ordering as a difficulty ranking would inverse-rank the genes by how much evidence supports
each estimate.

The second is the **seeding artefact** of Section 3.1.2. No acquired gene exists at
initialisation, so each gene's count is set by how readily the donor-free mutation branch seeds
it and by how early that happens, after which spread amplifies the result. These counts are not
a monotonic function of the per-step transfer probabilities: blaKPC-2, blaNDM-1 and mcr-1 are
available to the same species with probabilities of 0.02, 0.015 and 0.02, yet record 94, 177
and 179 events. A roughly two-fold spread among otherwise matched genes is left unexplained by
any parameter of the model, which is the clearest available indication that these counts are
high-variance outcomes of the seeding process rather than stable properties of each gene.

A gene's position in this table therefore reflects the simulator's seeding code, the timing of
a chance event, and the amount of evidence behind its estimate — not the mobility of the
corresponding real gene. We report the values for completeness and for comparison against
future versions of the pipeline, and draw no per-gene conclusions from them.

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
0.9805 ± 0.0012) and beats a per-gene random forest on the four genes that baseline fits in
every seed (0.9777 ± 0.0023 against 0.9548 ± 0.0124, on five of five seeds). Its advantage in
coverage — stable predictions for genes the per-gene baselines fit unreliably or not at
all — follows from joint training. Most of the signal is carried by the genes a cell already holds; message passing adds
a small, consistent increment; edge features and local drug exposure add nothing measurable in
this setting. What none of these figures establish is stated in Section 5: no external
validation, a held-out split that does not test unseen runs, and an intraspecies-only
transfer model.
