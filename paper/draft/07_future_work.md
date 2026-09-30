<!--
DRAFT — Section 7: Future Work. Status: awaiting owner review.
Expanded from what was Discussion §6.5, per owner instruction 2026-09-30.

claims_to_numbers.md entries this section proposes to change: S1-S10 (item 1 replaces the
figures), S2b (item 5 removes its current demonstration genes), S8 (item 1 needs regeneration),
U6, U7, U10, U11, U12, U13.

Compute figures: the reference run's own recorded `elapsed_s` is 3,510 s (~1 h) on one CUDA
GPU (torch 2.11, cu128) for five model seeds plus baselines and three-seed ablations. Data
generation was measured at ~8.5 s per run for pakistan_crisis with the committed protocol.
Hyperparameter-sweep time is NOT recorded in the committed sweep files, so no figure is given
for it. Effort estimates in person-days are the authors' estimates, not measurements, and are
labelled as such.
-->

# 7 Future Work

Each item below states what it would change about this paper's claims and roughly what it
costs. Items 1 and 5 are the two we would do first, for different reasons: item 1 is cheap and
changes how every number should be read, and item 5 removes a simplification that currently
undercuts the result we most want to stand.

Two costs recur, so we state them once. A full regeneration and retrain of the reference
result set took **3,510 s (about one hour)** on a single CUDA GPU, as recorded in the results
file, plus a few minutes of data generation. That is cheap enough that none of these items is
blocked by compute; what they cost is design work, literature work, and in three cases the
invalidation of every number in Section 4. Person-day figures are our estimates, not
measurements.

## 7.1 Split by run or by scenario, not by snapshot pair

**What it changes.** This is the only item that changes the paper's numbers without changing
the simulator, and the only one that changes what those numbers *mean* rather than what they
measure. Every figure in Section 4 is currently computed on a split over snapshot pairs, so it
reports generalisation to unseen time windows of runs seen in training (U13). A grouped split
would replace S1–S9 with figures that answer the question a reader assumes is being answered.
We expect them to be lower; that is the point.

**What it takes.** The code change is small. `split_dataset` shuffles pairs; it would group by
run instead. The run identity is not currently in a graph's metadata — which carries step,
scenario, node and edge counts — so `collect_training_snapshots` would need to stamp each pair
with its scenario and data seed first. That is perhaps half a day.

**The real obstacle is that fifteen runs is too few to group.** Five scenarios × three data
seeds, split 70/15/15 by run, leaves about ten training runs and three test runs, which is too
coarse for a stable estimate and would make the reported standard deviations largely a
statement about which three runs landed in the test fold. Doing this properly means generating
more runs — on the order of ten data seeds per scenario, so fifty runs — and reporting either
leave-one-scenario-out (five folds, which also measures transfer to an unseen population
composition) or leave-one-seed-out. At roughly ten seconds per run, the extra data generation
is minutes; the retraining is five folds × five model seeds, so a handful of hours at the
recorded rate.

**Estimated effort:** 2–3 person-days plus a few hours of compute. **Consequence:** S1–S9 are
superseded; U13 is resolved.

## 7.2 Per-gene transfer mobility

**What it changes.** The simulator applies one mechanism — conjugation between same-species
neighbours — to every gene, regardless of how that gene actually moves. Fixing this would
resolve U10 and the vanA half of U6, and would let per-gene results be read as something other
than an artefact.

**It would also remove the demonstration behind our coverage claim, which is why we rank it
second.** The two genes in S2b, acrAB-tolC and gyrA_S83L, are exactly the two whose transfer is
a simplification. If gyrA_S83L becomes mutation-only and acrAB-tolC becomes
regulation-only — which is what the sources in §5.2 imply — then both drop out of the transfer
label set, and the finding that joint training reaches genes the per-gene baselines cannot fit
needs a new demonstration. blaTEM-1 is the natural candidate (5 positives, currently not
evaluable; see item 6), but the claim would have to be re-earned rather than carried over. We
would rather discover that now than have a reader discover it later.

**What it takes.** The design work is deciding what mechanisms to model and how each one moves:
a conjugative plasmid (what the model already does), a conjugative transposon such as the Tn916
that carries tet(M), a chromosomal point mutation that should not transfer at all, a
regulatory-overexpression route for resident efflux systems, and a mobile genomic island such
as SCCmec for mecA. Each needs a rate and a host range with a source rather than a chosen
number. The code change is contained — `_attempt_hgt` already consults a per-gene
`non_transferable` set, so the extension is a per-gene mechanism tag and a branch per
mechanism.

**Estimated effort:** 1–2 person-weeks, most of it sourcing per-gene mechanism parameters,
plus a regeneration. **Consequence:** U10 resolved, U6 partly resolved, S2b needs
re-demonstration, all of Section 4 regenerated.

## 7.3 Interspecies transfer

**What it changes.** This is the largest scope restriction in the paper. Transfer is currently
intraspecies only, so the interspecies case that dominates the literature on these genes is
absent from the task, the labels and the evaluation (U11), and our measurements cannot detect
that absence — only 0.275% of contacts in our dataset even cross a species boundary (S10).
Adding it would make the task match the phenomenon the paper names, and would very likely lower
the headline figures, since the model would face a class of positives it has never seen.

**What it takes.** Not a parameter sweep. It requires deciding, per gene, which species
boundaries it can cross and at what relative rate, which is a literature question: plasmid
incompatibility groups and documented host ranges, the kind of evidence cited in §5.2 for
mcr-1, NDM-1 and tet(M). It also interacts with scenario design, because four of the five
current scenarios contain a single species and the fifth seeds its two species as separate
spatial clusters that seldom touch (S10) — so scenarios with genuinely mixed, spatially
interleaved populations would be needed for interspecies contacts to be common enough to learn
from. That is a scenario-design change as much as a mechanism change.

**Estimated effort:** 2–3 person-weeks, dominated by host-range sourcing and scenario redesign,
plus a regeneration. **Consequence:** U11 resolved; S10 no longer describes the dataset; every
figure in Section 4 regenerated on a harder task.

## 7.4 External validation against real gene calls

**What it changes.** U7 currently says no external validation exists. Our attempt failed for a
data reason: the isolate collections we obtained carry resistance phenotypes, not per-isolate
gene calls, so gene prevalences collapse and the rank correlation is undefined. Re-annotating
assemblies with a resistance-gene caller such as ResFinder or AMRFinderPlus would supply the
gene-level ground truth the pipeline needs.

**What it would and would not validate, which we think matters more than the effort.** A static
collection of annotated genomes contains no transfer events, so it cannot validate the
predictor. What it can validate is the *simulator*: whether the gene prevalences and
co-occurrence patterns our populations produce resemble those in real isolates of the same
species. That is worth doing — it is the difference between a simulator nobody has checked and
one whose output distributions have been compared against reality — but it should be reported as
validation of the data-generating process, not of the model, and we would be explicit about
that. Validating the *predictions* would need observed transfer events, which means either
longitudinal sampling with strain tracking or transfer networks inferred from sequence
similarity, as in the genome-level work discussed in §6.3. We regard that as a separate study.

**Estimated effort:** about 1.5 person-weeks for the simulator-distribution comparison, matching
our earlier scope for the re-annotation work, and substantially more for anything that validates
the predictor. **Consequence:** U7 partly resolved, with the scope of the resolution stated
precisely.

## 7.5 A defined step duration, and sourced pharmacokinetics

**What it changes.** No simulated step is assigned a real duration anywhere in the code, and
that single omission blocks every pharmacological comparison: per-drug decay rates cannot be
checked against measured half-lives, the damage function's shape parameters cannot be checked
at all, and the dosing protocol is necessarily arbitrary rather than a modelled course of
treatment (§5.3). Fixing it would let several invented parameters be replaced with sourced ones.

**What it takes.** A calibration decision and then literature work. The natural anchor is
growth: the model already assigns each species a doubling time in steps, so fixing a real
doubling time for one species fixes the step. That choice is itself a claim needing a source and
a stated growth condition, since doubling times vary widely with medium and temperature. Given a
step duration, per-drug half-lives convert to per-step decay rates directly, and the dosing
protocol can be restated as a real concentration over a real interval. We would treat the whole
item as a biology change requiring sourced values rather than chosen ones.

**Estimated effort:** 3–5 person-days of sourcing and calibration, plus a regeneration.
**Consequence:** several entries in §5.3 move from invented to sourced; the antibiotic-exposure
result (U4) becomes testable under a realistic course rather than one lasting five of eighty
steps.

## 7.6 A seeding mechanism with a biological counterpart

**What it changes.** Acquired genes currently enter the population through a mutation branch
that adds a random gene from a species' acquirable set with no donor at all. Nothing in reality
corresponds to this, and it is the reason the per-gene positive counts cannot be interpreted
(U12) and the reason blaTEM-1 is not evaluable (U8). Replacing it — with a small seeded carrier
population, or an environmental reservoir from which genes are acquired — would make the
per-gene counts a modelled quantity rather than an artefact, and would let the counts be set
deliberately instead of falling out of pool sizes and chance.

**What it takes.** Small code change, real design thought about what the reservoir represents,
and a regeneration. It pairs naturally with item 2, since both concern how genes enter and move.

**Estimated effort:** 3–5 person-days plus a regeneration. **Consequence:** U12 resolved; U8
likely resolved; S9 becomes interpretable; per-gene counts become a designed property of the
experiment.

## 7.7 Summary

| # | Item | Effort (estimated) | Claims affected |
|---|---|---|---|
| 1 | Group the split by run or scenario | 2–3 days + hours of compute | S1–S9 superseded; U13 resolved |
| 2 | Per-gene transfer mobility | 1–2 weeks + regeneration | U10 resolved; U6 partly; **S2b needs re-demonstration** |
| 3 | Interspecies transfer | 2–3 weeks + regeneration | U11 resolved; S10 obsolete; Section 4 regenerated |
| 4 | External validation via gene calls | ~1.5 weeks (simulator only) | U7 partly resolved, scope stated |
| 5 | Defined step duration + sourced PK | 3–5 days + regeneration | §5.3 entries sourced; U4 testable |
| 6 | Seeding with a biological counterpart | 3–5 days + regeneration | U12 resolved; U8 likely; S9 interpretable |

Items 2, 3, 5 and 6 all change the simulator and therefore invalidate every figure in
Section 4; they would sensibly be batched into one regeneration rather than run separately.
Item 1 is independent of all of them and should be done first, because until it is, none of the
reported figures answers the question a reader is most likely to think it answers.
