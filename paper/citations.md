# Citation audit (new paper)

Written 2026-09-30. Each entry was checked for (a) existence, authors, venue, year and DOI
against PubMed or the publisher/arXiv page, and (b) whether the source's own text supports
the specific claim it is cited for. **Verification level** says what was read: *full text*
(PMC), *abstract* (PubMed abstract only; full text paywalled), or *publisher page*. Quotes
are short excerpts kept here as evidence; they are not for reuse in the paper.
BibTeX keys match `paper/refs.bib`.

Rules applied (same standard as the SLURP citation audits): no citation without a checked
source; a claim a source does not clearly support is reworded or flagged, never kept on the
strength of the citation's existence.

---

## Introduction placeholders

### [CITE: AMR burden] → `gbd2019amr`, `gbd2021amr`

- **gbd2019amr** — Antimicrobial Resistance Collaborators (Murray CJL, Ikuta KS, Sharara F,
  et al.). Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis.
  *Lancet* 2022;399(10325):629–655. doi:10.1016/S0140-6736(21)02724-0. PMID 35065702.
  Level: publisher page + abstract.
  Supports "a major threat": Interpretation — "AMR is a leading cause of death around the
  world, with the highest burdens in low-resource settings."
- **gbd2021amr** — GBD 2021 Antimicrobial Resistance Collaborators. Global burden of bacterial
  antimicrobial resistance 1990–2021: a systematic analysis with forecasts to 2050.
  *Lancet* 2024;404(10459):1199–1226. doi:10.1016/S0140-6736(24)01867-1. PMID 39299261.
  Level: publisher page + abstract.
  Supports "growing": the reference forecast has deaths attributable to AMR rising
  from 1.14 million in 2021 to 1.91 million in 2050 (abstract). Those numbers are **not**
  used in the paper text.

**O'Neill 2016 — recommendation: replace, don't supplement.**
- The "10 million deaths a year by 2050" projection first appeared in the AMR Review's 2014
  paper. It is restated in the final report: O'Neill J. *Tackling drug-resistant infections
  globally: final report and recommendations.* Review on Antimicrobial Resistance, 2016
  (amr-review.org).
- de Kraker, Stewardson & Harbarth (*PLoS Med* 2016;13(11):e1002184,
  doi:10.1371/journal.pmed.1002184; full text read) criticise that projection:
  - It rests on scenarios for which "there is no empirical data supporting any of these
    scenarios".
  - It reports no confidence intervals.
  - It was never peer reviewed.
- The GBD papers are peer reviewed, give uncertainty intervals, and are more recent.
- The Introduction quotes no burden number, so it needs no O'Neill citation.
- **DECIDED 2026-09-30 (owner): excluded entirely.** This version of the paper does not cite
  O'Neill 2016 or mention the 10-million projection anywhere. The AMR burden is cited from the
  GBD papers only. de Kraker 2016 is retained in this audit as the reason for the exclusion,
  not as a citation in the paper.

### [CITE: review of HGT in AMR spread] → `vonwintersdorff2016`, `partridge2018`

- **vonwintersdorff2016** — von Wintersdorff CJH, Penders J, van Niekerk JM, Mills ND,
  Majumder S, van Alphen LB, Savelkoul PHM, Wolffs PFG. Dissemination of antimicrobial
  resistance in microbial ecosystems through horizontal gene transfer. *Front Microbiol*
  2016;7:173. doi:10.3389/fmicb.2016.00173. PMID 26925045. Level: abstract (open access).
  - Supports HGT as a route of resistance spread: "pathogenic bacteria can acquire
    resistance via horizontal gene transfer (HGT)".
  - Supports conjugation as the example: "conjugation is thought to have the greatest
    influence on the dissemination of ARGs".
- **partridge2018** — Partridge SR, Kwong SM, Firth N, Jensen SO. Mobile genetic elements
  associated with antimicrobial resistance. *Clin Microbiol Rev* 2018;31(4):e00088-17.
  doi:10.1128/CMR.00088-17. PMID 30068738. Level: abstract.
  - Supports "rather than arising independently": bacteria resist "often by acquiring
    preexisting resistance determinants from the bacterial gene pool".
  - Plasmids and integrative conjugative elements "play a central role in facilitating
    horizontal genetic exchange".

### [CITE: difficulty of observing conjugation in situ] → **claim reworded**; `brito2021`, `yaffe2020`

- **The original sentence is not clearly supported as worded.** "These individual transfer
  events are difficult to observe directly in real bacterial populations" — no source found
  states this directly.
- The obvious candidate, Sørensen et al. 2005 (*Nat Rev Microbiol* 3:700–710,
  doi:10.1038/nrmicro1232), reports the opposite trend in its abstract: single-cell
  detection of donors, recipients and transconjugants "has provided a new platform for HGT
  studies". **Not used.**
- **Replacement wording**, supported by the sources below: a gap between laboratory and
  natural-environment knowledge of HGT, and difficulty assigning mobile elements to host
  cells in natural communities. See the updated `01_introduction.md`.
- **brito2021** — Brito IL. Examining horizontal gene transfer in microbial communities.
  *Nat Rev Microbiol* 2021;19(7):442–453. doi:10.1038/s41579-021-00534-7. PMID 33846600.
  Level: abstract (paywalled).
  - "there is a surprising disconnect between what we know from laboratory experiments and
    what we know from natural environments, such as the human gut microbiome."
- **yaffe2020** — Yaffe E, Relman DA. Tracking microbial evolution in the human gut using
  Hi-C reveals extensive horizontal gene transfer, persistence and adaptation.
  *Nat Microbiol* 2020;5(2):343–353. doi:10.1038/s41564-019-0625-0. PMID 31873203.
  Level: full text (PMC6992475).
  - The abstract says "reliable assignment of mobile genetic elements to their microbial
    hosts in natural communities such as the human gut microbiota is lacking".
  - The main text calls it "elusive".

### Uncited sentence found during the audit: "Where and when such transfers happen is determined by local, cell-level conditions…"

- This was a biological claim with no citation. It is now reworded ("depends on") and cited.
- **seoane2011** — Seoane J, Yankelevich T, Dechesne A, Merkey B, Sternberg C, Smets BF.
  An individual-based approach to explain plasmid invasion in bacterial populations.
  *FEMS Microbiol Ecol* 2011;75(1):17–27. doi:10.1111/j.1574-6941.2010.00994.x.
  PMID 21091520. Level: abstract.
  - Supports physiological state: "pWW0 conjugation occurs mainly at advanced stages of the
    growth cycle and that nongrowing cells … do not display conjugal activity".
  - Supports contact: "cell-to-cell contact mechanics".
- **merkey2011** (below): conjugation depends on donor growth rate. It is modelled, not
  measured, so it is cited only alongside Seoane.
- Caveat: both concern specific plasmids (pWW0 TOL in *P. putida*; a generic model plasmid).
  "Depends on" is supported. A general law ("is determined by") is not.

### [CITE: ABM of bacterial populations / AMR] → `hellweger2016`, `krone2007`, `merkey2011`

- **hellweger2016** — Hellweger FL, Clegg RJ, Clark JR, Plugge CM, Kreft J-U. Advancing
  microbial sciences by individual-based modelling. *Nat Rev Microbiol* 2016;14(7):461–471.
  doi:10.1038/nrmicro.2016.62. PMID 27265769. Level: abstract.
  - Supports individual-based models of microbes as an established approach, built from
    single-cell observations.
- **krone2007** — Krone SM, Lu R, Fox R, Suzuki H, Top EM. Modelling the spatial dynamics of
  plasmid transfer and persistence. *Microbiology* 2007;153(Pt 8):2803–2816.
  doi:10.1099/mic.0.2006/004531-0. PMID 17660444. Level: abstract (PMC2613009).
  - Supports an individual-based lattice model of plasmid transfer in spatially structured
    populations, compared against agar-surface experiments.
- **merkey2011** — Merkey BV, Lardon LA, Seoane JM, Kreft J-U, Smets BF. Growth dependence of
  conjugation explains limited plasmid invasion in biofilms: an individual-based modelling
  study. *Environ Microbiol* 2011;13(9):2435–2452. doi:10.1111/j.1462-2920.2011.02535.x.
  PMID 21906217. Level: abstract.
  - Supports extending an IbM "to include the dynamics of plasmid carriage and transfer by
    individual cells".
- The sentence was split so that the list "growth, death, stress responses and gene
  exchange" is attributed to **our** model, not to the cited works.

### [CITE: GNN foundations] → `velickovic2018`

- **velickovic2018** — Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y.
  Graph attention networks. *ICLR* 2018. arXiv:1710.10903. Level: arXiv page.
  - Supports GAT, which our model uses: layers "in which nodes are able to attend over their
    neighborhoods' features".

---

## Related Work sources (additional)

- **kim2022** — Kim JI, Maguire F, Tsang KK, Gouliouris T, Peacock SJ, McAllister TA,
  McArthur AG, Beiko RG. Machine learning for antimicrobial resistance prediction: current
  practice, limitations, and clinical perspective. *Clin Microbiol Rev* 2022;35(3):e0017921.
  doi:10.1128/cmr.00179-21. PMID 35612324. Level: abstract.
  - "machine learning (ML) is increasingly being used to predict resistance … based on gene
    content and genome composition".
  - "ML models typically treat genes as independent predictors".
- **nguyen2026amrgnn** — Nguyen H-A, Peleg AY, Wisniewski JA, et al. (14 authors; last
  author Macesic N). AMR-GNN: a multi-representation graph neural network framework to enable
  genomic antimicrobial resistance prediction. *Nat Commun* 2026;17:3555.
  doi:10.1038/s41467-026-69934-8. PMID 41792137. Level: full text (PMC13087051).
  - "AMR-GNN predicts AMR for each isolate by treating each isolate as a node within a graph
    and performing node classification."
  - Edges come from SNP / FCGR distance matrices.
  - **Naming note:** "AMR-GNN" is close to our class name `AMRResistanceGNN`. Avoid "AMR-GNN"
    as our model's name in the paper.
- **donabauer2025** — Donabauer G, Rath A, Caplunik-Pratsch A, et al. AI modeling for outbreak
  prediction: a graph-neural-network approach for identifying vancomycin-resistant
  enterococcus carriers. *PLOS Digit Health* 2025;4(4):e0000821.
  doi:10.1371/journal.pdig.0000821. PMID 40208871. Level: abstract.
  - GNNs on time-dependent hospital-movement graphs classify patients as VRE carriers.
- **zhou2021** — Zhou H, Beltrán JF, Brito IL. Functions predict horizontal gene transfer and
  the emergence of antibiotic resistance. *Sci Adv* 2021;7(43):eabj5056.
  doi:10.1126/sciadv.abj5056. PMID 34678056. Level: full text (PMC8535800).
  - **Closest prior work.** ML predicts a genome-level HGT network.
  - An edge is a "recent HGT event" between distantly related organisms (<97% 16S
    similarity) sharing ≥500 bp at ≥99% identity.
  - Models: logistic regression, random forest and a graph convolutional network.
  - RF on functional (KO) profiles is best. The baseline GCN was "akin to the RF model" and
    improved with added network-topology input.
  - **Their AUROCs are not comparable to ours** (a different task, unit and data). Do not
    put them side by side.
- **ellabaan2021** — Ellabaan MMH, Munck C, Porse A, Imamovic L, Sommer MOA. Forecasting the
  dissemination of antibiotic resistance genes across bacterial genomes. *Nat Commun*
  2021;12:2435. doi:10.1038/s41467-021-22757-1. PMID 33893312. Level: abstract.
  - **Author Correction (Nat Commun 2026;17, doi:10.1038/s41467-026-74295-3; full text
    read).** Following Godron et al., 30 of the 34 "confirmed" predicted mobilisations
    likely came from contaminant sequences.
  - **Cite only the approach** (predicting future ARG dissemination from gene-exchange
    networks and mobilisation elements). Never cite the 34/94 confirmation figure.
- **lardon2011** — Lardon LA, Merkey BV, Martins S, Dötsch A, Picioreanu C, Kreft J-U,
  Smets BF. iDynoMiCS: next-generation individual-based modelling of biofilms.
  *Environ Microbiol* 2011;13(9):2416–2434. doi:10.1111/j.1462-2920.2011.02414.x.
  PMID 21410622. Level: abstract.
  - Supports iDynoMiCS as a common individual-based biofilm modelling framework.
- **gorochowski2012** — Gorochowski TE, Matyjaszkiewicz A, Todd T, et al. BSim: an agent-based
  tool for modeling bacterial populations in systems and synthetic biology. *PLoS One*
  2012;7(8):e42790. doi:10.1371/journal.pone.0042790. PMID 22936991. Level: abstract.
  - Supports BSim as an agent-based tool for bacterial populations.
- **glushchenko2019** — Glushchenko OE, Prianichnikov NA, Olekhnovich EI, et al. VERA:
  agent-based modeling transmission of antibiotic resistance between human pathogens and gut
  microbiota. *Bioinformatics* 2019;35(19):3803–3811. doi:10.1093/bioinformatics/btz154.
  PMID 30825306. Level: abstract.
  - An agent-based model at the human-population level, including HGT of resistance
    determinants from commensals to a pathogen.

## Full-text upgrade pass (2026-09-30)

At the owner's request, full-text access was attempted for the six sources behind U10 and
U11 via legitimate open-access routes only (PMC, Europe PMC, publisher pages, institutional
repositories). No paywall-circumvention service was used.

| Source | Outcome | Route |
|---|---|---|
| Boguslawska 2009 | **upgraded to full text** | PMC2753074 |
| Hooper & Jacoby 2015 | **upgraded to full text** | PMC4626314 |
| Li, Plesiat & Nikaido 2015 | **upgraded to full text** | PMC4402952 |
| Dolejska 2012 | remains abstract | OUP paywall; the claim we cite it for is stated in the abstract ("plasmid DNA purified from the pNDM-CIT *Escherichia coli* J53 transconjugant") |
| Liu 2016 | remains abstract | Elsevier paywall. The Bristol research portal lists metadata only, with no accepted manuscript. No legitimate open copy found |
| Weigel 2003 | remains abstract | *Science* paywall; no legitimate open copy found. (A CDC Stacks hit was a different 2007 paper) |

What the new full texts confirmed, beyond the abstracts:

- **Hooper & Jacoby 2015** states directly that plasmid-mediated quinolone resistance is due
  to Qnr proteins, the AAC(6')-Ib-cr acetyltransferase and mobile efflux pumps (QepA, OqxAB) —
  i.e. *gyrA itself is not the mobilised element*, which is the exact basis of U10.
  It also independently supports the acrAB-tolC half of U10: resistance follows from
  regulatory mutation raising pump expression ("mutations in the MarR regulator result in both
  an increase in [acrAB] expression as well as a decrease in [ompF] expression").
  It further supports the A. baumannii note in `gene_mechanism_audit.md` §3.6 (AdeIJK
  constitutive, AdeABC/AdeFGH overexpressed) independently of Coyne 2011.
- **Li 2015** confirms all three points it is cited for, including the caveat we chose to
  state: "Plasmid-borne efflux pump genes (including those for RND pumps) have increasingly
  been identified."

## Open items for the owner
1. O'Neill 2016: replace (recommended) or keep with de Kraker 2016 + GBD 2021.
2. The reworded in-situ sentence and the "determined by" → "depends on" change in the
   Introduction. Please review both, since they change wording as well as citations.
3. The abstract-only verifications (Brito 2021, Partridge 2018, Hellweger 2016, Merkey
   2011, Seoane 2011, Kim 2022, Donabauer 2025, Lardon 2011, Gorochowski 2012, Glushchenko
   2019). Each is cited only for what its abstract states. Before submission, re-check
   against the full text if you have institutional access.
