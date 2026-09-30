<!--
DRAFT — Section 3: Methods. Status: awaiting owner review.

claims_to_numbers.md entries used: dataset header block (reference result set, protocol,
dataset SHA-256 and composition), S8 (reproducibility), U6 (vanA simplified), U10 (gyrA_S83L
and acrAB-tolC simplified), U11 (intraspecies only), U12 (seeding artefact), U1 (retired
numbers / leakage-fix provenance), U9 (default hyperparameters superseded).
Sources: paper/citations.md and paper/gene_mechanism_audit.md (keys in paper/refs.bib).

Every parameter value here was read from the code, not from a docstring: several docstrings
in ai/ are stale (see paper/decisions_log.md, 2026-09-30 code-comment findings).
No number appears that is not in claims_to_numbers.md or read directly from code.
-->

# 3 Methods

## 3.1 Agent-based simulation

The simulator is an agent-based model built on Mesa 3.5.1. Bacteria occupy an 80 × 60
multi-agent grid on which several cells may share a position. Each simulated step, every
living cell updates its energy, its stress response and its persister state, takes antibiotic
damage, attempts mutation, attempts gene transfer, updates its biofilm status, moves and
attempts division, in that order.

Five bacterial species are modelled (*Escherichia coli*, *Klebsiella pneumoniae*,
*Acinetobacter baumannii*, *Pseudomonas aeruginosa* and methicillin-resistant
*Staphylococcus aureus*), each with its own doubling time, baseline fitness, baseline mutation
rate, biofilm propensity, motility, intrinsic resistance genes and a set of genes it may
acquire. Ten resistance genes are modelled, drawn from CARD, spanning β-lactamases
(blaTEM-1, blaCTX-M-15, blaKPC-2, blaNDM-1), efflux systems (mexAB-oprM, acrAB-tolC),
target alteration (gyrA_S83L, mcr-1), ribosomal protection (tetM) and glycopeptide resistance
(vanA); the corrected biology adds mecA as an eleventh. Six antibiotics are modelled
(ciprofloxacin, meropenem, colistin, vancomycin, ampicillin, tetracycline), each as a
concentration field over the grid that diffuses and decays.

### 3.1.1 Antibiotic action

A cell's protection against a drug is combined multiplicatively over its genes, as
independent probabilities, and is then raised by biofilm membership and by the persister
state. For a bactericidal drug, damage accrues each step as a sigmoid (Hill-type) function of
the locally effective concentration,

  damage = E_max · C^n / (EC50^n + C^n),

with EC50 set to the drug's minimum bactericidal concentration and, in the code,
E_max = 0.3 and n = 2. Damage accumulates; a cell begins dying at 0.7 and dies at 1.0.
Bacteriostatic drugs delay division instead of causing damage. **E_max and the Hill
coefficient are invented parameters:** neither is taken from a cited source, and because no
real duration is assigned to a simulated step, neither the damage rate nor the per-drug decay
rates can be compared against measured pharmacokinetics. We report them as simulator
settings, not as pharmacological estimates.

### 3.1.2 Horizontal gene transfer, and the four ways a cell gains a gene

Transfer is modelled as conjugation between neighbouring cells. A cell attempts transfer on a
randomly chosen 30% of its steps. When it does, it considers every cell in its immediate
(Moore radius 1) neighbourhood and, for each gene it carries, transfers that gene with
probability

  p = acquisition_prob(gene) × (2 if the donor's SOS response is active) × m_biofilm,

where m_biofilm is 4 when both cells are in a biofilm, 1.5 when one is, and 1 otherwise, and
p is capped at 0.95. The recipient must not already carry the gene, and the gene must be in
the recipient species' acquirable set. The biofilm multipliers follow the observation that
biofilm matrices raise conjugation frequency; **the per-gene `acquisition_prob` values
(0.01–0.05 per step) and the 30% attempt rate are invented parameters with no cited source.**

Three restrictions define the task, and all three must be read alongside the results:

1. **Transfer is intraspecies only.** A donor skips any neighbour of a different species, so
   every cross-species contact is a negative example by construction. Real transfer of several
   of these exact genes is documented across species and genera: the mcr-1 plasmid was
   conjugated into *E. coli* and maintained in *K. pneumoniae* and *P. aeruginosa*
   [liu2016mcr]; an NDM-1 plasmid was conjugated from *Citrobacter freundii* into an *E. coli*
   recipient [dolejska2012]; Tn916-borne tet(M) transferred from *Lactococcus lactis* to
   *Enterococcus faecalis* in vitro and in a rat gut [boguslawska2009]. The task we evaluate
   is therefore easier than transfer prediction in a mixed-species community.
2. **Two genes are moved by a mechanism that is not their real one.** gyrA_S83L is a
   chromosomal target-site mutation, and quinolone resistance that *is* plasmid-borne is
   carried by a different set of genes (qnr proteins, a modifying enzyme, mobile efflux pumps)
   rather than by mobilised gyrA [hooper2015; strahilevitz2009]. acrAB-tolC is a chromosomal
   RND efflux system, ubiquitous in the Gram-negative species that carry it, and resistance
   arises by overexpression rather than by gaining the genes [li2015efflux]. The simulator
   nonetheless transfers both between neighbours. We label their transfer events a
   **simplified mechanism** and do not present them as models of conjugative transfer.
   (Plasmid-borne RND pump genes do exist [li2015efflux]; what the model does that reality
   does not is move a species' own resident AcrAB-TolC between cells of that species.)
   vanA is simplified in a different way: it arises and spreads within MRSA, whereas the
   documented route is interspecies transfer of Tn1546 from *E. faecalis* [weigel2003;
   clark2005], and *Enterococcus* is not modelled. vanA is reported separately throughout.
   For gyrA_S83L we use the generic description "a gyrA target-site mutation": the gene object
   carries CARD's *E. coli* gyrA entry but is available to three species in the model, and we
   did not establish that the S83L substitution specifically is the dominant variant in all
   three, so the particular variant modelled should be treated as unverified.
3. **Two genes are never transferred, correctly.** mexAB-oprM appears only as an intrinsic
   gene and is in no species' acquirable set, so it has no transfer events by construction and
   is excluded from evaluation. mecA is marked non-transferable because its real mobilisation
   is by SCCmec, a mechanism the model does not implement; reusing conjugation for it would be
   the wrong mechanism.

Besides transfer, a cell can gain a gene by inheriting it at division, by starting with its
species' intrinsic genes, or through a mutation step. The mutation step has a branch that adds
a **uniformly random gene from the species' acquirable set with no donor at all**. This last
route has no biological counterpart — an acquired gene must come from somewhere — and exists
only to seed genes into the population. It matters for interpreting the results, because no
acquired gene is present at initialisation, so every gene's first carrier arises this way.
How readily a gene seeds therefore depends on the size of its species' acquirable set, on
whether it has an additional seeding route, and on how many scenarios contain a species that
can carry it; and however it seeds, the resulting count is then amplified by spread, so that
the earlier a gene happens to seed, the more events it accumulates. **The number of transfer
events per gene is consequently an outcome of the seeding code and of chance rather than of
gene epidemiology.** Consistent with that, three of the genes in our dataset share the same
species availability and near-identical transfer probabilities yet differ roughly two-fold in
their event counts. We therefore do not interpret per-gene differences biologically anywhere
in this paper.

Only recorded transfer events produce the supervised labels; genes gained by the other three
routes produce no label.

## 3.2 Two versioned biologies

The simulation's biology exists in two explicitly versioned forms, and we report on the
corrected one.

**`paper_v1`** is the original configuration, frozen. A regression test checks that its
simulated states and its generated training pairs are byte-identical to the pre-versioning
code under more than one Python hash seed.

**`lab_v2`** is the corrected configuration and the one all reported results use. Its
differences from `paper_v1`, each with the source that motivated it, are:

| Change | Basis |
|---|---|
| MRSA's intrinsic gene is mecA rather than tetM | CARD ARO:3000617 describes mecA (PBP2a) as commonly associated with MRSA; CARD ARO:3000186 describes tet(M) as found on transposable elements, and EUCAST Expected Resistant Phenotypes v1.2 does not list *S. aureus* as expected-resistant to tetracyclines |
| acrAB-tolC removed from MRSA's acquirable set | AcrAB-TolC is a Gram-negative tripartite system (CARD ARO:3000384) |
| mecA is non-transferable | SCCmec mobilisation is not modelled (§3.1.2) |
| *K. pneumoniae*'s blanket intrinsic acrAB-tolC replaced by species-level intrinsic resistance to ampicillin only | EUCAST Expected Resistant Phenotypes v1.2 rule 1.7 lists the *K. pneumoniae* complex as expected resistant to ampicillin/amoxicillin and ticarcillin, not to ciprofloxacin or tetracycline; ticarcillin is not simulated |
| Antibiotic diffusion conserves total drug | Bug fix: the original applied a kernel that already summed to one and then rescaled the field, removing drug every step, which contradicts a diffusion process |
| mecA fitness cost 0.275 | Ender et al. 2004, comparing the strains RA120 and BB255 |

A known limitation carried by both versions is that decay rates are unvalidated against real
pharmacokinetics, for the reason given in §3.1.1.

We also note two residual issues we did not change, because changing them would alter the
dataset: *A. baumannii* carries mexAB-oprM and acrAB-tolC as intrinsic genes although its
characterised RND systems are AdeABC, AdeIJK and AdeFGH [coyne2011]; and blaTEM-1 is available
only to *E. coli*, although it is common in *K. pneumoniae* [cuzon2010].

## 3.3 Prediction task and dataset

### 3.3.1 Graphs and labels

From a simulation snapshot we build a graph whose nodes are living cells (subsampled to at
most 300 per snapshot with a seeded generator) and whose edges join pairs of cells within
three grid cells of one another, added in both directions. Each node carries a 35-dimensional
feature vector: ten binary gene-presence flags; five physiological values (fitness, energy,
stress level, accumulated antibiotic damage, normalised age); two behavioural flags (biofilm
membership, persister state); normalised position; three population descriptors (local
density, generation, offspring count); a five-way species one-hot; a two-way Gram stain
indicator; and the local concentration of each of the six drugs. Each edge carries five
features: normalised shared-gene count, absolute fitness difference, a both-in-biofilm flag,
absolute stress-level difference, and a coarsened proximity band with three levels
(same cell / within three cells / farther).

The edge radius of three is deliberately wider than the transfer radius of one. If the graph
contained only transfer-eligible pairs, the existence of an edge would itself identify
eligibility, which would put the label-generating gate back into the graph's topology.
Including near-but-ineligible pairs under the same coarse proximity band prevents that.

The target is defined per directed edge and per gene: did a recorded transfer of that gene
occur from the source cell to the target cell within the snapshot window? Labels come from the
simulator's own event log, recorded at the moment a transfer fires, and never from comparing
genomes between snapshots. This distinction is necessary because a cell can gain a gene by
mutation in the same window in which an eligible neighbour happens to carry it; inferring
transfer from genome differences would label that as a transfer from a cell that had nothing
to do with it.

### 3.3.2 Data generation

Each run initialises 120 cells, steps 15 times, then applies its scenario's drugs uniformly
and continues to 80 steps, taking a snapshot every three steps and pairing consecutive
snapshots. Under the corrected biology the dose is 0.25 µg/mL, cleared after five further
steps; this time-limited course is an invented protocol, expressed in steps because no real
step duration is defined. Five scenarios are used — `ecoli_cipro`,
`klebsiella_carbapenem`, `pakistan_crisis`, `xdr_acinetobacter` and `mrsa_hospital` — each
with three data seeds.

The resulting dataset (SHA-256 `ee83ff385ca15ec9…`) contains 390 graph pairs,
1,707,498 edges and 1,443 positive transfer events. Positives are distributed very unevenly
across genes — blaCTX-M-15 480, tetM 420, mcr-1 179, blaNDM-1 177, blaKPC-2 94, vanA 35,
gyrA_S83L 34, acrAB-tolC 19, blaTEM-1 5 and mexAB-oprM 0 — for the structural reason given in
§3.1.2, not for any biological reason. Against ten gene outputs the positive rate is
approximately 8.5 × 10⁻⁵ per edge-gene pair.

### 3.3.3 Splitting, and a caveat about it

Graph pairs are shuffled with a fixed seed and split 70% / 15% / 15% into training,
validation and test sets. The same split seed is used by the neural model, by the baselines
and by calibration, so that all of them are evaluated on the same held-out edges; this was not
originally the case, and correcting it is part of the history in §3.4.

**The split is over graph pairs, not over simulation runs.** Because each run contributes
roughly 26 pairs taken three steps apart, snapshots from one run — sharing its founding
population, its seed and largely the same cells — can be distributed across training,
validation and test. The held-out set therefore measures generalisation to new time windows
of runs the model has seen, not to unseen runs, scenarios or species. This inflates the
reported figures relative to a grouped split, and we state it here rather than leaving it to
be inferred (see also Section 5, and Section 7.1 for what fixing it would involve).

## 3.4 Development history that affects the numbers

Two corrections separate the numbers reported here from earlier ones, and we record them
because they change what the figures mean rather than merely improving them.

**Label definition.** Labels were originally inferred by comparing genomes between snapshots,
which conflates transfer with independent mutation and produces positives attributable to a
cell that did not donate anything. Labels now come from recorded events only.

**Feature leakage.** Four inputs were direct components of the rule that generates the label:
the same-species edge flag and the raw edge distance (both hard preconditions for transfer),
a transferable-gene count that mirrored the eligibility test almost exactly, and the SOS flag
(the literal term that doubles the transfer probability). All four were removed. Node features
went from 36 to 35 dimensions and edge features from 8 to 5. Raw distance was replaced by the
coarsened proximity band described in §3.3.1, and the transferable-gene count by a plain
shared-gene count. We confirmed which layout produced the earlier results by rebuilding the
architecture under both: the pre-correction layout reproduces a parameter count of 349,370 and
the corrected layout 349,162, matching the counts logged by the respective runs. Results
obtained before these two corrections are not comparable with those reported here and are not
reported. An audit file referenced in a source comment as holding the full leakage analysis is
not present in the repository; §3.4 is the account of record.

Separately, results obtained with the original default hyperparameters are superseded by the
tuned configuration of §3.5, selected on the validation split.

## 3.5 Model

The model is a graph attention network [velickovic2018]. Node features are encoded by one
small linear encoder per feature group — genomic, physiological, behavioural, spatial,
population, species, Gram stain, drug exposure — whose outputs are concatenated and projected
to the hidden width, so that groups with different scales and meanings are not mixed by a
single projection. Edge features are encoded separately by a two-layer network.

Each message-passing block applies multi-head graph attention that takes the encoded edge
vector as an edge attribute, followed by a residual connection and layer normalisation, then a
position-wise feed-forward network with a further residual connection and normalisation. For
each directed edge, a prediction head receives the source representation, the target
representation and the encoded edge vector, and emits one logit per gene.

The configuration used for all reported results, selected on the validation split, is: hidden
width 128, edge encoding width 64, two attention blocks, four heads, dropout 0.15. Training
uses binary cross-entropy with a positive-class weight of 15 to offset the extreme imbalance,
AdamW at learning rate 10⁻³ with weight decay 10⁻⁴ and cosine annealing, gradient clipping at
1.0, batch size 8, and at most 60 epochs. Early stopping watches validation AUROC with
patience 12 and a minimum improvement of 10⁻³, but is suppressed for the first 20 epochs.

That warm-up is not cosmetic. Without it, one seed stopped at epoch 13 because validation
AUROC had peaked at epoch 1 — on an essentially untrained model — and had not yet recovered
from a subsequent dip; the retained checkpoint was worse on every gene. The behaviour was
reproducible on retraining, and the warm-up resolves it: that seed reaches the same range as
the others, while a normally-behaved seed is unchanged. With the warm-up in place, every seed
in the reported runs trained for 27 to 38 epochs with its best epoch between 15 and 32.

**Edge features are zeroed in all reported runs.** Tuned models with and without edge features
were indistinguishable on the frozen biology, so the reported configuration does not use them.
This should not be read as evidence that edge attributes are harmful; the one earlier result
that appeared to show a cost was the early-stopping artefact described above.

## 3.6 Baselines and ablations

**Per-gene classical baselines.** Random forest and logistic regression are trained
one-versus-rest per gene on the flattened concatenation of the two endpoint node vectors and
the edge vector, using a seeded 100,000-edge subsample of the training edges. The random
forest uses 300 trees, maximum depth 8 and balanced subsampling. A gene with fewer than five
positives in that subsample cannot be trained and is scored 0.5 by construction. We keep this
cutoff as it is: lowering it for the comparison would change the baseline rather than test it.

Because the subsample is drawn independently for each seed, **trainability is a per-seed
property**: two genes fall below the threshold in every seed, and a third falls below it in
three seeds of five. The headline comparison is therefore computed on the four genes the
random forest fits in *every* seed, so that no 0.5-by-construction score enters it. The genes
it fits unreliably or never are reported separately under coverage, as a property of the
approaches rather than folded into the same number.

**Message-passing ablation.** The same architecture with the attention blocks removed, so that
node and edge encodings feed the prediction head directly, isolates the contribution of
message passing.

**Feature-group ablation.** Groups of node features are zeroed one at a time to measure each
group's contribution.

## 3.7 Evaluation

Genes are handled according to a fixed, pre-recorded policy. mexAB-oprM is excluded, having no
transfer events by construction. vanA is evaluated but always reported separately and labelled
a simplified mechanism. The headline macro average is taken over the remaining eight genes.
One of those eight, blaTEM-1, has no positives in the test split (five in the whole dataset),
so no model can be scored on it and the macro average is effectively over seven; we say so
wherever the figure appears rather than implying eight evaluable genes.

We report AUROC as the primary metric, since the positive rate makes accuracy uninformative,
alongside AUPRC and expected calibration error. Every figure is a mean and standard deviation
over five independent model seeds, with three seeds for the ablations; the seeds vary model
initialisation and training, not the data. Per-gene AUROCs are reported as discrimination
figures only, always with their positive counts, and are not interpreted as reflecting the
relative mobility of real genes, for the reason given in §3.1.2.

Data generation and training are seeded and deterministic across processes: identical
simulated states and identical training pairs are produced under different Python hash seeds,
which a regression test enforces for the frozen biology. Retraining the same seed on a GPU
reproduces test AUROC to within 2.1 × 10⁻⁴.
