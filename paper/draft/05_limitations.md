<!--
DRAFT — Section 5: Limitations. Status: awaiting owner review.

claims_to_numbers.md entries used: U5, U6, U7, U8, U10, U11, U12, U13, S2b (scope of the
coverage claim), plus the dataset header block (dosing protocol, invented parameters).
Sources: paper/citations.md, paper/gene_mechanism_audit.md (keys in paper/refs.bib).

Also covered here, from paper/decisions_log.md: the invented dosing protocol and unvalidated
decay rates, the Hill-function parameters, the single-simulator point, the residual biology
issues (A. baumannii pumps, blaTEM-1 confined to E. coli), and the leakage-fix history as a
methods note rather than a buried footnote.

No numbers appear that are not in claims_to_numbers.md.
-->

# 5 Limitations

We group these by what they limit: the claims the numbers support (§5.1), the biology the
simulator implements (§5.2), the parameters it uses (§5.3), and the history of the pipeline
itself (§5.4). Several are consequences of decisions we would make differently in a future
version, and we say so where that is the case rather than presenting each as an inherent
constraint.

## 5.1 What the evaluation does and does not establish

**No external validation.** Every number we report is internal to the simulator: the labels
are its own recorded transfer events, and the model is never tested against real genomic data.
We attempted such a validation and it is not functional. The public isolate collections we
obtained are antimicrobial-resistance *phenotype* tables with no per-isolate gene calls, so the
comparison degenerates — gene prevalences collapse to 1.0 or 0.0 and the rank correlation is
undefined. A meaningful comparison requires re-annotating assemblies with a resistance-gene
caller to obtain the gene-level calls the model predicts over. We scope that as future work
and make no claim of external validity here. Consequently nothing in this paper should be read
as a clinical or epidemiological prediction.

**The held-out split does not test unseen simulations.** We split graph pairs, not simulation
runs. Each run contributes roughly twenty-six snapshot pairs three steps apart, and those pairs
are distributed across training, validation and test, so a test pair typically comes from a run
whose other windows were trained on — sharing its founding population, its random seed and
largely the same individual cells. The reported figures therefore measure generalisation to
unseen *time windows within seen runs*, and not to unseen runs, unseen scenarios or unseen
species. This inflates them relative to a grouped split. A by-run or by-scenario split is the
correct design, and we did not adopt it; doing so would change every number in Section 4 and
requires a full regeneration, which we identify as the single most important change for a
future version.

**The comparison is between our implementations, not between model families.** Our per-gene
random forest and logistic-regression baselines are one-versus-rest with a five-positive
training requirement. Two genes fall below it and are scored 0.5 by construction. This is a
property of how we implemented those baselines, not a limitation of random forests, and we do
not claim otherwise. We kept the threshold rather than lowering it, because lowering it would
have produced a different baseline rather than a fairer test of this one — but the consequence
is that the coverage difference in §4.2 speaks to a design choice we made about the baselines
as much as to the GNN.

For the same reason we do not quote a macro-averaged difference between the GNN and the random
forest across all eight policy genes. Such a difference would combine a performance gap on the
genes both models fit with a coverage gap on the genes only one model fits, and would
substantially overstate the former. The two are reported separately throughout.

**One gene cannot be evaluated at all.** blaTEM-1 has five positive events in the entire
dataset and none in the test split, so no model can be scored on it. The headline macro,
nominally over eight genes, is in practice over seven. This is not a property of the gene: the
simulator makes blaTEM-1 available only to *E. coli*, although it is in fact common in
*Klebsiella pneumoniae* — one survey found it in the great majority of its carbapenemase-
producing *K. pneumoniae* isolates [cuzon2010] — and widening its availability would very
likely make it evaluable. We report it as not evaluable rather than reporting a figure.

**Per-gene results are not interpretable as biology.** As set out in §4.5, per-gene AUROCs are
confounded both by sample size, which spans from 19 to 480 positives and runs opposite to the
AUROC ordering, and by the seeding mechanism described below. We report them as discrimination
measurements only.

**Reported spreads are over model seeds, not over data.** Every mean and standard deviation in
Section 4 varies model initialisation and training order with the dataset held fixed. They
therefore describe the stability of training, not sampling uncertainty in the dataset, and they
understate total uncertainty — most sharply for the genes with the fewest positive events,
where the sampling component is largest and is not represented at all.

## 5.2 What the simulated biology omits or simplifies

**Transfer is intraspecies only.** A gene never crosses a species boundary in our simulator, so
every cross-species contact is a negative example by construction. This excludes the route that
matters most in the literature on these genes: the mcr-1 colistin-resistance plasmid was
conjugated into *Escherichia coli* and maintained in *Klebsiella pneumoniae* and *Pseudomonas
aeruginosa* [liu2016mcr]; an NDM-1 carbapenemase plasmid was conjugated from *Citrobacter
freundii* into an *E. coli* recipient [dolejska2012]; Tn916-borne tet(M) transferred from
*Lactococcus lactis* to *Enterococcus faecalis* both in vitro and in a rat gut
[boguslawska2009].

The consequence is one of absence rather than of class balance, and it is worth stating
precisely because the intuitive version is wrong. One might expect this restriction to flood the
negative class with cross-species pairs that are rejectable from species identity alone. It does
not: only 0.275% of contacts in our dataset are cross-species (4,696 of 1,707,498), they occur in
only one of the five scenarios, and none of them carries a transfer. Four scenarios contain a
single species, and in the mixed scenario the two populations are seeded as separate spatial
clusters that seldom come within contact range. What follows instead is that the interspecies
case is effectively absent from the data: the model is evaluated almost entirely on
within-species pairs, it has not been shown to do anything at all on interspecies transfer, and
our metrics cannot register that omission. That is the case of greatest practical interest, and
we have no evidence about it either way.

**Three of the eleven genes move by a mechanism that is not their real one.** We label these
simplified mechanisms rather than leave them implicit.

- *gyrA_S83L* is a chromosomal target-site mutation. Quinolone resistance that is genuinely
  plasmid-borne is carried by a different set of determinants — qnr proteins, a
  quinolone-modifying acetyltransferase, and mobile efflux pumps — not by mobilised *gyrA*
  [hooper2015; strahilevitz2009]. Our simulator both mutates it into existence, which is right,
  and transfers it between neighbours, which is not. Additionally, the gene object carries
  CARD's *Escherichia coli gyrA* entry while being available to three species in the model; we
  therefore describe it generically as a *gyrA* target-site mutation and treat the specific
  variant modelled as unverified across those species.
- *acrAB-tolC* is a chromosomal RND efflux system, ubiquitous in the Gram-negative species
  that carry it, and clinical resistance arises through overexpression rather than through
  acquiring the genes [li2015efflux]. The simulator transfers it between cells of a species
  that already universally possesses it. We note for completeness that plasmid-borne RND pump
  genes do exist [li2015efflux], so transferring *an* efflux pump is not in itself
  unrealistic; what is unrealistic is moving a species' own resident system between its cells.
- *vanA* arises and spreads within MRSA in our model, whereas the documented route is
  interspecies transfer of Tn1546 from *Enterococcus faecalis* to *Staphylococcus aureus*
  [weigel2003; clark2005], and *Enterococcus* is not modelled. vanA is reported separately from
  the headline throughout.

These three genes are exactly the ones whose results are most easily over-read: two of them
are the rare genes behind the coverage claim in §4.2, and the third posts the highest AUROC of
any gene. Their figures measure the learning setup, not resistance biology.

Two further genes are handled by *not* transferring them, which is correct but is itself a
simplification: mexAB-oprM exists only as an intrinsic gene and so has no events to predict,
and mecA is marked non-transferable because SCCmec mobilisation is a mechanism the simulator
does not implement and conjugation is the wrong substitute for it.

**A residual species-level error we did not correct.** *Acinetobacter baumannii* carries
mexAB-oprM and acrAB-tolC as intrinsic genes, whereas its characterised RND efflux systems are
AdeABC, AdeIJK and AdeFGH [coyne2011]; MexAB-OprM belongs to *P. aeruginosa*. We traced the
consequences and could identify no effect on the reported numbers — neither gene is in
*A. baumannii*'s acquirable set, so neither can spread among those cells, and the only drug in
the scenario that features this species is one that acrAB-tolC does not cover. We nonetheless
record it rather than correcting it silently, because correcting it would change the dataset.

**Genes appear from nowhere, and this shapes the dataset.** No acquired gene is present at
initialisation, so each one enters the population through a mutation branch that adds a
uniformly chosen gene from the species' acquirable set with no donor at all. Real acquired genes
must come from somewhere; this route exists only to seed the simulation. Because every gene's
first carrier arises this way, the number of transfer events per gene is an outcome of the
seeding code and of chance rather than of gene biology, as §4.5 sets out — and among three
genes matched on species availability and near-matched on transfer probability, event counts
still differ roughly two-fold.

**Other simplifications carried from the simulator.** Cells with no resistance genes receive
neither biofilm nor persister protection, because protection is computed from gene-derived
resistance before those modifiers are applied; this affects the species that begin with no
intrinsic genes. Only bacteria are modelled, so no plasmid host range, no phage, and no
environmental reservoir appears. Species composition is fixed per scenario.

## 5.3 Invented and unvalidated parameters

Several quantities that look like measurements are not, and we label them explicitly rather
than let a reader assume provenance from context.

- **Per-gene transfer probabilities** (0.01–0.05 per step) have no cited source, and neither
  does the rule that a cell attempts transfer on a randomly chosen 30% of its steps.
- **The biofilm conjugation multipliers** are motivated by the general finding that biofilm
  matrices raise conjugation frequency, but the specific multipliers we use are chosen, not
  measured.
- **The antibiotic damage function's shape parameters** — a maximum effect of 0.3 and a Hill
  coefficient of 2 — are hard-coded, with the half-maximal concentration set to each drug's
  minimum bactericidal concentration. Only the last of these derives from drug data.
- **The dosing protocol** — 0.25 µg/mL cleared after five steps — is an invented protocol. It
  is expressed in simulation steps because no real duration is assigned to a step.
- **Per-drug decay rates** are unvalidated against pharmacokinetic data. Since no step
  duration is defined, they cannot even be compared with published half-lives. Assigning a
  real step duration and sourcing per-drug half-lives is identified future work.

The absence of a defined step duration is the common thread: it means no rate in the simulator
can currently be checked against a measured rate. It also bears on the antibiotic-exposure
ablation in §4.3, where drug is present for only five of eighty steps, so the finding that
exposure features add no measurable signal holds for this protocol and should not be
generalised.

## 5.4 Pipeline history that bears on interpretation

We record two corrections because they change what earlier figures meant, not merely how large
they were.

Labels were originally inferred by comparing genomes between consecutive snapshots. That
conflates transfer with independent mutation and can attribute a gene's appearance to a
neighbour that donated nothing. Labels now come only from events recorded at the moment a
transfer fires.

Four model inputs were direct components of the rule that generates the label: the
same-species edge flag and the raw edge distance, both hard preconditions for transfer; a
transferable-gene count that mirrored the eligibility test; and the SOS flag, the literal term
that doubles the transfer probability. A model given these features can partly read the label
off its own inputs. All four were removed, taking node features from 36 to 35 dimensions and
edge features from 8 to 5. We verified which layout produced the earlier results by rebuilding
the architecture under both and matching the parameter counts recorded in the training logs.
Figures produced before these two corrections are not comparable with those reported here, and
we do not report them. We also note that a file referenced in the source as containing the
full leakage audit is not present in the repository, so the account in Section 3.4 is the
record.

Finally, a single simulator produced every result in this paper. The task, the feature
definitions and the labels all derive from one implementation of one set of modelling
decisions, many of them listed above. Performance figures obtained this way describe a model's
fit to that implementation. Whether the approach transfers to another simulator, let alone to
real populations, is untested, and §5.1's first point applies: no external validation exists
yet.
