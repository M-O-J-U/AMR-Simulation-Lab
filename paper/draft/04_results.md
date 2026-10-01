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
ablations), on the held-out split described in Section 3.3.3. That split holds out whole
simulation runs, so throughout this section "test" means runs the model never trained on. It
does not mean unseen scenarios or unseen species: the ten held-out runs are drawn from the same
five scenarios as the training runs.

**These numbers replace an earlier set, and the replacement is itself a result.** Every figure
in this section was previously computed on a split taken over snapshot *pairs*, which placed
snapshots three steps apart — from the same run, the same founding population and the same
seed — on both sides of the split. Correcting that cost 0.0176 AUROC on the headline and far
more on the rare genes (§4.9). We report the corrected numbers throughout and state the size of
each change where it matters, rather than quietly substituting them. Unless stated
otherwise, "headline AUROC" is the macro average over the eight genes fixed by the evaluation
policy of Section 3.7, one of which (blaTEM-1) has no positives in the test split, so the
macro is in practice over seven.

## 4.1 The GNN outperforms per-gene classical baselines

On the five genes the per-gene random forest can be fitted to **in every seed** —
blaCTX-M-15, blaKPC-2, blaNDM-1, mcr-1 and tetM — the GNN reaches a test AUROC of
**0.9805 ± 0.0021** against **0.9647 ± 0.0031** for the random forest. The GNN is ahead on
**all five of five seeds**. We treat this as the primary comparison, because it is the only
one in which both models are actually fitted to the same genes in every run.

This comparison is the most robust result in the paper. The gap is 0.0158 here; on the earlier,
three-times-smaller dataset it was 0.0229, and on that same smaller dataset under the
uncorrected split it was also 0.0204. A gap stable to within 0.003 across both a 3.3-fold
increase in data and a correction to the evaluation itself is the one claim we would expect to
survive further scrutiny.

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
**0.9676 ± 0.0076** (previously 0.9805 ± 0.0012 under the uncorrected split). Logistic
regression reaches **0.7911 ± 0.0336** on the same headline macro, and the random forest
0.8711 ± 0.0366.

We deliberately do not quote a headline-macro *difference* between the GNN and the random
forest. The random forest's headline macro is depressed by the genes it cannot reliably train,
which are scored 0.5 by construction rather than by any prediction; a difference computed
across all eight genes would therefore combine a genuine performance gap with a coverage gap
and overstate the former. The two are reported separately: performance in this subsection, on
the four genes both models always fit, and coverage in §4.2.

## 4.2 Coverage of genes the baselines cannot be fitted to: a weaker finding than we first reported

Our per-gene baselines require at least five positive examples for a gene within the training
subsample they draw, and that subsample is drawn independently for each seed. On a smaller
dataset this excluded three genes to varying degrees, and we presented the resulting coverage
advantage as a contribution. **At the present dataset size it does not support that billing,
and we have demoted it.**

| Gene | Held-out positives | Baseline fitted | GNN test AUROC |
|---|---|---|---|
| acrAB-tolC* | 5 | **0 of 5 seeds** | 0.8890 ± 0.0506 |
| gyrA_S83L* | 24 | 3 of 5 seeds | 0.9822 ± 0.0049 |
| vanA* | 5 | 5 of 5 seeds | 0.8669 ± 0.0504 |
| blaKPC-2 | 61 | 5 of 5 seeds | 0.9761 ± 0.0045 |

\* transfer modelled by a simplified mechanism (Section 3.1.2).

Two things changed. First, tripling the data let the baselines fit almost everything: vanA went
from one seed in five to all five, blaKPC-2 from two to all five, gyrA_S83L from none to three.
Only **acrAB-tolC** is still never fitted. The coverage advantage was therefore substantially a
property of the data regime, not of the method, and it shrinks as data grows.

Second, the one gene that still carries it is estimated from **five** held-out positive events,
at 0.8890 ± 0.0506. That spread is over model seeds on a fixed split; the sampling uncertainty
attached to five events is far larger and is not shown. We do not think a single gene, measured
that imprecisely, can carry a headline claim.

What survives is a structural observation, and we state it as such rather than as a result: a
per-gene model cannot be built at all below its positive threshold, whereas a model trained
jointly across genes always emits a prediction, whatever that prediction is worth. In this
dataset that distinction binds for one gene out of ten. We return to it in Section 5 as a
limitation of the comparison rather than an advantage of the method.

The GNN is evaluated on seven of the eight policy genes plus vanA separately. The eighth,
blaTEM-1, has 24 positive events in the dataset but none in the held-out split, so no model of
any kind can be scored on it; we report it as not evaluable rather than as a result.

## 4.3 Where the signal comes from

**Message passing shows no reliable benefit.** Removing the attention blocks, so that node and
edge encodings feed the prediction head directly, changes the headline by
**+0.0065 ± 0.0086** (0.9676 with message passing against 0.9611 ± 0.0019 without). The
standard deviation exceeds the mean, and the per-seed values are +0.0188, −0.0005, +0.0052,
+0.0112 and −0.0022: **positive in three seeds of five, negative in two.** On this evidence we
cannot distinguish the contribution of the contact graph from zero.

We previously reported this as a small but consistent gain, +0.0031 ± 0.0014 and positive in
all five seeds. That consistency did not survive the corrected split. It is worth being precise
about what this does and does not mean: we are not claiming message passing is useless in
general, only that in this task, at this scale, with this architecture, we have no evidence
that it helps. Since the graph-free variant retains essentially all of the performance, almost
everything the model exploits is available from the two endpoint feature vectors alone.

**The genes a cell already carries dominate.** Zeroing feature groups one at a time, only the
genomic group matters: removing it costs **0.0442 ± 0.0071** headline AUROC (three seeds).
**No other group is distinguishable from zero** — every one has a standard deviation at least
as large as its own mean, the largest being species (−0.0046 ± 0.0102), spatial position
(+0.0052 ± 0.0083) and physiological (+0.0051 ± 0.0125). The estimate for the genomic group is
substantially tighter than the one we reported previously (−0.0549 ± 0.0331), so this
conclusion is now better supported than it was, not worse.

**Local antibiotic exposure adds no measurable signal.** Zeroing the six drug-concentration
features changes the headline by **+0.0006 ± 0.0011** — that is, removing them does not hurt.
This deserves an explicit caveat rather than a conclusion about antibiotic pressure: under our
dosing protocol the drug is present for only five of the eighty simulated steps, so most
snapshots carry no exposure to measure. The correct reading is that these features carry no
signal *in this protocol*, not that antibiotic exposure is irrelevant to gene transfer.

## 4.4 Calibration and precision–recall

Predicted probabilities are well calibrated, with an expected calibration error of
**0.0012 ± 0.0003**, unchanged by the correction.

Headline AUPRC is **0.0145 ± 0.0057** (previously 0.0316 ± 0.0293). This is far above the base
rate of approximately 7.9 × 10⁻⁵ positives per edge-gene pair, so the ranking carries real information in absolute
terms; but the standard deviation is nearly as large as the mean, so the figure is unstable
across seeds and we do not treat it as a headline result. The instability is expected given
how few positives the rarer genes contribute.

## 4.5 Per-gene discrimination

| Gene | GNN test AUROC | Positives in the held-out split |
|---|---|---|
| blaNDM-1 | 0.9860 ± 0.0054 | 16 |
| tetM | 0.9844 ± 0.0010 | 311 |
| gyrA_S83L* | 0.9822 ± 0.0049 | 24 |
| blaCTX-M-15 | 0.9821 ± 0.0024 | 212 |
| blaKPC-2 | 0.9761 ± 0.0045 | 61 |
| mcr-1 | 0.9738 ± 0.0018 | 217 |
| acrAB-tolC* | 0.8890 ± 0.0506 | 5 |
| vanA* | 0.8669 ± 0.0504 | 5 |
| blaTEM-1 | not evaluable | 0 |

\* transfer modelled by a simplified mechanism (Section 3.1.2).

**We offer no biological interpretation of this ordering.** Two confounds account for it
without any appeal to gene biology.

The first is **evidence**. The two genes at the bottom are estimated from five held-out
positive events each, and they carry standard deviations an order of magnitude wider than
every other row. Those spreads are over model seeds on a fixed split and therefore understate
the true uncertainty, which at five events is dominated by sampling. Among the genes with
tens or hundreds of held-out positives there is no clear relationship between count and score:
blaNDM-1 scores highest on 16 events, mcr-1 lowest of that group on 217.

The second is the **seeding artefact** of Section 3.1.2, and tripling the dataset tested it
directly. If per-gene counts were a stable property of each gene, every gene's count would have
grown with the dataset, by 3.53×. Instead they grew by between **1.21× (acrAB-tolC, 19 to 23)
and 5.70× (blaKPC-2, 94 to 536)** — a 4.7-fold spread in what should have been a constant.
The counts are high-variance outcomes of when a gene happens to seed, not properties of the
gene.

A gene's position in this table therefore reflects the simulator's seeding code, the timing of
a chance event, and the amount of evidence behind its estimate. We report the values for
completeness and draw no per-gene conclusions from them.

## 4.6 vanA, reported separately

vanA reaches a test AUROC of 0.8669 ± 0.0504 on five held-out positive events. It is excluded from the headline
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
same seed on a GPU reproduces test AUROC to within 1.6 × 10⁻³. (That tolerance was
2.1 × 10⁻⁴ on the earlier, easier split; the wider spread accompanies the harder task rather
than any change to seeding.) Every figure in this section is
a mean and standard deviation over independent model seeds, with the data held fixed, so the
reported spreads capture variation from initialisation and training order and not from data
generation.

## 4.8 Summary

The GNN discriminates simulated per-edge, per-gene transfer events well (headline AUROC
0.9676 ± 0.0076) and beats a per-gene random forest on the five genes that baseline fits in
every seed (0.9805 ± 0.0021 against 0.9647 ± 0.0031, on five of five seeds) — the one result
that held steady across both a threefold increase in data and a correction to the evaluation.
Most of the signal is carried by the genes a cell already holds. Message passing cannot be
distinguished from zero, and edge features and local drug exposure add nothing measurable.
The coverage advantage we previously highlighted largely dissolved once there was more data,
and now rests on a single gene measured from five held-out events. What none of these figures
establish is stated in Section 5: no external validation, no test of unseen scenarios or
species, and an intraspecies-only transfer model.

## 4.9 What the split correction cost

Because the correction and the larger dataset arrived together, we ran both arms on identical
data so the two effects could be separated. Three result sets are therefore comparable: the
original 15-run set under a snapshot-level split, the 50-run set under the same snapshot-level
split, and the 50-run set under the run-level split used throughout this paper.

| | 15 runs, snapshot split | 50 runs, snapshot split | 50 runs, run split |
|---|---|---|---|
| Headline AUROC | 0.9805 ± 0.0012 | 0.9852 ± 0.0008 | **0.9676 ± 0.0076** |
| acrAB-tolC | 0.9986 ± 0.0018 | 0.9774 ± 0.0157 | **0.8890 ± 0.0506** |
| vanA | 0.9988 ± 0.0002 | 0.9349 ± 0.0211 | **0.8669 ± 0.0504** |
| Message passing | +0.0031, 5/5 seeds | +0.0059, 5/5 seeds | **+0.0065, 3/5 seeds** |

Reading across the first two columns isolates the effect of more data; reading across the last
two isolates the effect of the split, with the data held identical. More data raised the
headline slightly (+0.0047). **The split correction lowered it by 0.0176**, and lowered the two
rarest genes by roughly 0.09 and 0.07 — an order of magnitude more. The reason is structural:
those genes have five held-out positive events, and under a snapshot-level split their
near-duplicates, three simulated steps away in the same run, sat in the training set.

This is the most useful result in the paper for anyone building a similar system, and
Section 6.1 treats it as such.
