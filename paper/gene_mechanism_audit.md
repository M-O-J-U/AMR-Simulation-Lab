# Transfer-mechanism audit: all 11 simulated resistance genes

Written 2026-09-30, at the owner's request, before drafting Methods. The aim is to catch
every mismatch between a gene's real mobility and what the simulator does with it, in one
pass, rather than one per drafting session.

**Method.** For each gene: (1) read the code path that can add it to a cell
(`core/bacterium_agent.py`), (2) read which species may carry it
(`data/card_loader.py`, `data/biology.py`), (3) check its real mechanism against a primary
source or authoritative review, verifying the source's own text, not just its existence
(same standard as `paper/citations.md`). **No code was changed.** All sources are logged in
§4; BibTeX keys are to be added to `paper/refs.bib` when Methods cites them.

## 1 What the simulator can do, in full

There are exactly four ways a cell can gain a resistance gene:

| # | Route | Code | Notes |
|---|---|---|---|
| R1 | Inherited at birth | `BacteriumAgent.__init__`, `inherited_genes` | Vertical. |
| R2 | Present from the start | `__init__`: `resistance_genes = set(profile.natural_resistances)` | Only `natural_resistances`. **No acquired gene is present at t=0.** |
| R3 | Conjugation-style transfer from a neighbour | `_attempt_hgt` | The only route that records an HGT event, i.e. the only route that produces the labels the GNN is trained on. |
| R4 | Appearance with no donor | `_attempt_mutation` | Two sub-cases: a `gyrA_S83L`-specific branch (roll < 0.6, then p=0.3), and a generic branch (0.6 ≤ roll < 0.85) that picks a **uniformly random gene from the species' acquirable pool** and adds it. Plus an `acrAB-tolC`-specific route in `_update_sos_response` (p=0.15 on SOS activation). |

R3 has three restrictions that matter for this audit:
- **same species only** (`if recipient.species != self.species: continue`),
- the gene must be in the **recipient's** `acquired_resistance_pool`,
- the gene must not be in `non_transferable` (lab_v2: `mecA` only).

Otherwise **R3 applies the same conjugation-style mechanism to every gene**, at a per-step
probability of `acquisition_prob` (× 2 if the donor's SOS response is active, × 1.5 or × 4
for biofilm), regardless of whether that gene is mobile in reality.

## 2 Per-gene findings

Verdict key: **OK** = HGT is the right kind of mechanism; **SIMPLIFIED** = chromosomal in
reality but transferred by the model; **N/A** = the model never transfers it.

| Gene | Real mechanism (source) | What the sim does | Verdict |
|---|---|---|---|
| **blaTEM-1** | Plasmid-encoded acquired β-lactamase; found on plasmids alongside other bla genes in K. pneumoniae from 5 countries (Cuzon 2010 describes blaTEM-1 among "acquired and plasmid-encoded genes", 81.3% of 16 KPC-2 isolates) | R3 + R4; E. coli only | **OK** (mechanism). See §3.5 for the species restriction |
| **blaCTX-M-15** | Mobilised from chromosomal *Kluyvera* genes onto mobile elements that drive "rapid and efficient inter-replicon and cell-to-cell dissemination" (D'Andrea 2013) | R3 + R4; E. coli, Klebsiella, A. baumannii | **OK** |
| **blaKPC-2** | "The blaKPC-2 gene was always associated with 1 of the Tn4401 isoforms (a, b, or c)", on plasmids of different incompatibility groups (Cuzon 2010) | R3 + R4; Klebsiella, A. baumannii, P. aeruginosa | **OK** |
| **blaNDM-1** | Carried on conjugative plasmids of several scaffolds; e.g. an IncFII plasmid in E. coli ST131 (Bonnin 2012) and an IncHI1 plasmid from *Citrobacter freundii* conjugated into an E. coli recipient (Dolejska 2012) | R3 + R4; Klebsiella, A. baumannii, P. aeruginosa | **OK** |
| **mcr-1** | The first plasmid-mediated polymyxin resistance: "mobilised to an E coli recipient at a frequency of 10⁻¹ to 10⁻³ cells per recipient cell by conjugation" (Liu 2016) | R3 + R4; Klebsiella, A. baumannii, P. aeruginosa | **OK** |
| **tetM** | Carried on Tn916, a "broad-host, conjugative transposon" (Devirgiliis 2009); transferred between species in vitro and in vivo (Boguslawska 2009) | R3 + R4; E. coli, MRSA | **OK** (a conjugative transposon, not a plasmid; the model does not distinguish) |
| **gyrA_S83L** | A **chromosomal point mutation** in a drug target. Hooper & Jacoby 2015: resistance is by "mutation and acquisition of resistance-conferring genes", where target mutations are in GyrA/ParE, while *plasmid*-encoded quinolone resistance is a **different set of genes** (Qnr, a modifying enzyme, mobile efflux pumps) — not mobilised gyrA. Strahilevitz 2009 likewise describes PMQR as qnr / aac(6')-Ib-cr / oqxAB / qepA | R4 point-mutation branch (correct) **and R3** (not correct) | **SIMPLIFIED** — logged as U10 |
| **acrAB-tolC** | A chromosomal RND efflux system: "chromosomally encoded drug efflux mechanisms that are ubiquitous in these bacteria", AcrAB-TolC named as the clinically relevant example (Li 2015). Resistance arises by **overexpression**, typically via regulatory mutation, not by gaining the genes | R3 (not correct) + R4 + the SOS route (a crude stand-in for overexpression) | **SIMPLIFIED** — logged as U10. **Caveat, stated honestly:** Li 2015 also notes "Plasmid-borne efflux pump genes (including those for RND pumps) have increasingly been identified", so transfer of an RND pump is not impossible in general. What is wrong is transferring *the resident, ubiquitous* AcrAB-TolC of E. coli between E. coli cells |
| **vanA** | Tn1546 on plasmids, documented route *E. faecalis* → *S. aureus* (Weigel 2003; Clark 2005). *Enterococcus* is not modelled | R3 + R4; MRSA only, so MRSA→MRSA | **SIMPLIFIED** — already U6; reported separately |
| **mexAB-oprM** | Chromosomal RND system of *P. aeruginosa* (Li 2015; CARD ARO:3000386) | Intrinsic (R2) for P. aeruginosa and A. baumannii. **In no species' acquirable pool, so R3 and R4 can never add it** → 0 transfer events by construction | **N/A** — correctly never transferred; excluded from evaluation. But see §3.6 |
| **mecA** (lab_v2 only) | PBP2a on SCCmec, a mobile genomic island, not a conjugative plasmid (CARD ARO:3000617) | Intrinsic to MRSA; in `non_transferable`, and `acquisition_prob = 0.0` | **N/A** — correctly never transferred; already documented in `data/biology.py` |

**Count:** 6 of 11 genes are mobile in the right kind of way. 3 are simplified
(gyrA_S83L, acrAB-tolC, vanA). 2 are never transferred, correctly.

## 3 Cross-cutting findings

These apply to every gene and were not previously recorded. (3.1) and (3.2) are the
substantive new ones.

### 3.1 Transfer is restricted to the same species; real transfer crosses species and genera

`_attempt_hgt` skips any neighbour of a different species. Every source above documents
the opposite for these genes:
- the mcr-1 plasmid was conjugated into E. coli and "maintained in K pneumoniae and
  Pseudomonas aeruginosa" (Liu 2016);
- Tn916-borne tet(M) transferred from *Lactococcus lactis* to *Enterococcus faecalis*,
  i.e. between genera, in vitro and in a rat gut (Boguslawska 2009);
- an NDM-1 IncHI1 plasmid from *Citrobacter freundii* was conjugated into an E. coli
  recipient (Dolejska 2012);
- blaCTX-M-15 spread involves both E. coli and K. pneumoniae high-risk clones
  (D'Andrea 2013).

So the simulator excludes the route that matters most epidemiologically. This makes the
prediction task easier in a specific way: a cross-species contact is a guaranteed negative,
and species is a node feature. **It must be disclosed in Methods and Limitations.** It is
not a bug in the sense of contradicting the code's stated intent, but it is a strong
simplification that the paper cannot leave implicit.

### 3.2 Acquired genes appear with no donor, and this sets the per-gene positive counts

No acquired gene is present at t=0 (R2 covers `natural_resistances` only), so the first
carrier of every acquired gene arises through R4 — a gene appearing in a cell from nothing.
Real acquired genes must come from somewhere, so R4 has no biological counterpart; it is a
seeding device.

Its side effect is that **the distribution of positives across genes is a property of the
seeding code, not of anything resembling real epidemiology.** R4's generic branch picks
uniformly from the species' acquirable pool, so a gene's seeding rate scales with
1/(pool size) of the species that can carry it, and its subsequent spread scales with
`acquisition_prob`. This predicts the observed counts in the reference dataset:

| Gene | Pool(s) it is in | Extra seeding route | `acquisition_prob` | Positives |
|---|---|---|---|---|
| tetM | MRSA (lab_v2 pool of 2), E. coli (of 5) | — | 0.05 (highest) | 420 |
| vanA | MRSA (of 2) | — | 0.01 | 35 |
| gyrA_S83L | E. coli (of 5), Klebsiella (of 5), P. aeruginosa (of 4) | dedicated point-mutation branch | 0.01 | 34 |
| acrAB-tolC | E. coli (of 5) | SOS route, p=0.15 | 0.03 | 19 |
| blaTEM-1 | E. coli (of 5) only | — | 0.04 | **5** |
| mexAB-oprM | no pool | — | 0.025 | 0 |

This explains U8 (blaTEM-1 not evaluable) mechanically: it is confined to one species'
five-gene pool with no extra seeding route, so it rarely seeds, and rarely seeded means
rarely transferred. It also means **S9's per-gene ordering must not be read as biological
insight** — a gene's number of positives, and hence how well it can be learned, is set by
the seeding code.

### 3.3 Per-step transfer probabilities are invented

`acquisition_prob` (0.01–0.05 per step) carries no source, and no step duration is defined
anywhere in the code, so these cannot be compared with measured conjugation frequencies
(Liu 2016's 10⁻¹–10⁻³ per recipient, for instance). Already logged 2026-09-30; repeated here
so the gene table is self-contained. Methods must label them invented (CLAUDE.md rule 7).

### 3.4 `gyrA_S83L` is the E. coli variant, applied to three species

`data/card_loader.py` cites CARD ARO:3003294, which the code comment correctly identifies as
the *E. coli* gyrA entry. The same gene object is in Klebsiella's and P. aeruginosa's pools.
Target-site numbering and the favoured substitution are not identical across these species
(Hooper & Jacoby 2015 describe the mutations as lying in a localised domain of GyrA, without
a single universal substitution). **Flagged, not settled:** I did not find a source
establishing that S83L specifically is the dominant variant in all three. Methods should
describe it generically ("a gyrA target-site mutation") or restrict the claim to E. coli.

### 3.5 blaTEM-1 is confined to E. coli, though it is common in Klebsiella

Cuzon 2010 found blaTEM-1 in 81.3% of 16 KPC-2-producing K. pneumoniae isolates. In the
model only E. coli may carry it. A pool gap, interacting with 3.2 to make blaTEM-1
unevaluable. Fixing it would change the dataset, so: **report, do not change.**

### 3.6 A. baumannii carries the wrong efflux pumps

A. baumannii has `natural_resistances = ["mexAB-oprM", "acrAB-tolC"]` in both biology
versions. Coyne 2011 (a review devoted to efflux in *Acinetobacter*) states that RND systems
are the most prevalent in multiresistant A. baumannii and names them **AdeABC** (overexpressed
via adeRS mutation), **AdeIJK** ("intrinsic to this species") and **AdeFGH** — not MexAB-OprM,
which is a *P. aeruginosa* system, and not AcrAB-TolC. The mexAB-oprM half was already
flagged on 2026-09-30; the acrAB-tolC half is new, and it matters more, because acrAB-tolC
*is* transferable in the model: every A. baumannii is born with a gene it should not have,
in a scenario (`xdr_acinetobacter`) that is in the training set. It cannot spread there,
though, because acrAB-tolC is not in A. baumannii's acquirable pool, so no A. baumannii can
receive it. Its only effect is intrinsic protection against ciprofloxacin, tetracycline and
ampicillin — and `xdr_acinetobacter` doses colistin, which acrAB-tolC does not cover.
**Net effect on the reference numbers: none that I can identify.** Still needs disclosure.

## 4 Sources used (all verified against the source's own text)

Verification level: *abstract* = PubMed abstract only, full text paywalled.

| Key | Reference | Level |
|---|---|---|
| cuzon2010 | Cuzon G, Naas T, Truong H, et al. Worldwide diversity of *Klebsiella pneumoniae* that produce β-lactamase blaKPC-2 gene. *Emerg Infect Dis* 2010;16(9):1349–1356. doi:10.3201/eid1609.091389 | abstract |
| dandrea2013 | D'Andrea MM, Arena F, Pallecchi L, Rossolini GM. CTX-M-type β-lactamases: a successful story of antibiotic resistance. *Int J Med Microbiol* 2013;303(6-7):305–317. doi:10.1016/j.ijmm.2013.02.008 | abstract |
| bonnin2012 | Bonnin RA, Poirel L, Carattoli A, Nordmann P. Characterization of an IncFII plasmid encoding NDM-1 from *Escherichia coli* ST131. *PLoS One* 2012;7(4):e34752. doi:10.1371/journal.pone.0034752 | abstract |
| dolejska2012 | Dolejska M, Villa L, Poirel L, Nordmann P, Carattoli A. Complete sequencing of an IncHI1 plasmid encoding the carbapenemase NDM-1… *J Antimicrob Chemother* 2013;68(1):34–39. doi:10.1093/jac/dks357 | abstract |
| liu2016mcr | Liu Y-Y, Wang Y, Walsh TR, et al. Emergence of plasmid-mediated colistin resistance mechanism MCR-1 in animals and human beings in China. *Lancet Infect Dis* 2016;16(2):161–168. doi:10.1016/S1473-3099(15)00424-7 | abstract |
| devirgiliis2009 | Devirgiliis C, Coppola D, Barile S, Colonna B, Perozzi G. Characterization of the Tn916 conjugative transposon in a food-borne strain of *Lactobacillus paracasei*. *Appl Environ Microbiol* 2009;75(12):3866–3871. doi:10.1128/AEM.00589-09 | abstract |
| boguslawska2009 | Boguslawska J, Zycka-Krzesinska J, Wilcks A, Bardowski J. Intra- and interspecies conjugal transfer of Tn916-like elements from *Lactococcus lactis* in vitro and in vivo. *Appl Environ Microbiol* 2009;75(19):6352–6360. doi:10.1128/AEM.00470-09 | abstract |
| hooper2015 | Hooper DC, Jacoby GA. Mechanisms of drug resistance: quinolone resistance. *Ann N Y Acad Sci* 2015;1354(1):12–31. doi:10.1111/nyas.12830 | abstract |
| strahilevitz2009 | Strahilevitz J, Jacoby GA, Hooper DC, Robicsek A. Plasmid-mediated quinolone resistance: a multifaceted threat. *Clin Microbiol Rev* 2009;22(4):664–689. doi:10.1128/CMR.00016-09 | abstract |
| li2015efflux | Li X-Z, Plésiat P, Nikaido H. The challenge of efflux-mediated antibiotic resistance in Gram-negative bacteria. *Clin Microbiol Rev* 2015;28(2):337–418. doi:10.1128/CMR.00117-14 | abstract |
| coyne2011 | Coyne S, Courvalin P, Périchon B. Efflux-mediated antibiotic resistance in *Acinetobacter* spp. *Antimicrob Agents Chemother* 2011;55(3):947–953. doi:10.1128/AAC.01388-10 | abstract |

Already in `paper/claims_to_numbers.md` (U6): Weigel 2003 doi:10.1126/science.1090956;
Clark 2005 doi:10.1128/AAC.49.1.470-472.2005. CARD: ARO:3000386 (MexAB-OprM),
ARO:3000617 (mecA), ARO:3003294 (E. coli gyrA), ARO:3000384 (AcrAB-TolC).

## 5 What this changes

Nothing in the code, and nothing in the reference numbers. What it changes is what the
paper must say:

1. **U10 stands and is now sourced** (gyrA_S83L, acrAB-tolC as simplified mechanisms), with
   the honest caveat that plasmid-borne RND pump genes do exist (§2, acrAB-tolC row).
2. **New: U11** — same-species-only transfer (§3.1). Methods + Limitations.
3. **New: U12** — the per-gene positive counts are an artefact of the seeding code (§3.2).
   This bears directly on S9 and on how S2b's rare-gene framing is worded.
4. **§3.4 is unresolved** and needs either a generic description or a narrowed claim.
5. §3.3, §3.5, §3.6 are disclosures with no effect on the numbers I can identify.
6. The 6 genes in the **OK** rows need no caveat beyond the shared ones.
