"""
Expected (species-level) resistance for the 5 simulated species x 6 antibiotics.

USED ONLY BY ANALYTICS (ai/resistance_analytics.py, and the GNN treatment
advisory through it). The simulation does NOT read this table: killing,
fitness and HGT in core/ and simulation/ are unchanged, so simulated
trajectories and the paper's numbers are unaffected. It exists so that
MIC estimates and treatment recommendations stop reporting, e.g., E. coli as
100% susceptible to vancomycin just because it carries no acquired gene.

Only "R" (resistance expected without testing) is recorded. Absence from the
table means "no expected resistance recorded", NOT "expected susceptible".

Sources (checked 2026-09-29 against the primary documents):

  [ERP]  EUCAST Expected Resistant Phenotypes, v1.2, January 2023
         (formerly "Intrinsic Resistance and Unusual Phenotypes" v3.3).
         https://www.eucast.org/bacteria/important-additional-information/expected-phenotypes/
         EUCAST's definition: >90% of isolates resistant irrespective of origin.
  [BP14] EUCAST Clinical Breakpoint Tables v14.0, valid from 2024-01-01,
         Staphylococcus spp. section. Applies to the "MRSA" profile only; this
         is the phenotype that DEFINES MRSA (acquired mecA/mecC), not a species-
         wide expected phenotype of S. aureus.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from data.card_loader import GERM_PROFILES


@dataclass(frozen=True)
class ExpectedResistance:
    basis: str      # "expected_phenotype" (EUCAST ERP) or "defining_phenotype" (MRSA)
    source: str     # citation key + exact rule/note


_ERP_T1 = "[ERP] Table 1 header: Enterobacterales expected resistant to glycopeptides"
_ERP_T2 = "[ERP] Table 2 header: non-fermentative Gram-negatives expected resistant to glycopeptides"

EXPECTED_RESISTANCE: Dict[str, Dict[str, ExpectedResistance]] = {
    "e_coli": {
        # E. coli has no species row in ERP Table 1; only the Enterobacterales-wide header applies.
        "vancomycin": ExpectedResistance("expected_phenotype", _ERP_T1),
    },
    "klebsiella_pneumoniae": {
        "vancomycin": ExpectedResistance("expected_phenotype", _ERP_T1),
        "ampicillin": ExpectedResistance("expected_phenotype",
            "[ERP] Table 1 rule 1.7 (Klebsiella pneumoniae complex): ampicillin/amoxicillin R"),
    },
    "acinetobacter_baumannii": {
        "vancomycin": ExpectedResistance("expected_phenotype", _ERP_T2),
        "ampicillin": ExpectedResistance("expected_phenotype",
            "[ERP] Table 2 rule 2.1 (A. baumannii/pittii/nosocomialis): ampicillin/amoxicillin R"),
        "tetracycline": ExpectedResistance("expected_phenotype",
            "[ERP] Table 2 rule 2.1 + footnote 2: 'Acinetobacter is resistant to tetracycline "
            "and doxycycline' (less so to minocycline/tigecycline)"),
    },
    "pseudomonas_aeruginosa": {
        "vancomycin": ExpectedResistance("expected_phenotype", _ERP_T2),
        "ampicillin": ExpectedResistance("expected_phenotype",
            "[ERP] Table 2 rule 2.7 (P. aeruginosa): ampicillin/amoxicillin R"),
        "tetracycline": ExpectedResistance("expected_phenotype",
            "[ERP] Table 2 rule 2.7 (P. aeruginosa): tetracyclines R"),
    },
    "mrsa": {
        "colistin": ExpectedResistance("expected_phenotype",
            "[ERP] Table 4 header: Gram-positive bacteria expected resistant to polymyxin B/colistin"),
        "ampicillin": ExpectedResistance("defining_phenotype",
            "[BP14] Staphylococcus, penicillins note 1/A: 'Isolates that test resistant to "
            "cefoxitin are resistant to all penicillins'"),
        "meropenem": ExpectedResistance("defining_phenotype",
            "[BP14] Staphylococcus, carbapenems note 1/A: susceptibility 'is inferred from the "
            "cefoxitin susceptibility' (MRSA = cefoxitin resistant)"),
    },
}

_SPECIES_TO_KEY = {g.species: k for k, g in GERM_PROFILES.items()}


def germ_key_for(species_or_key: Optional[str]) -> Optional[str]:
    """Accept a germ key ('e_coli') or full species name ('Escherichia coli')."""
    if species_or_key in GERM_PROFILES:
        return species_or_key
    return _SPECIES_TO_KEY.get(species_or_key)


def expected_resistance(species_or_key: Optional[str],
                        antibiotic_key: str) -> Optional[ExpectedResistance]:
    key = germ_key_for(species_or_key)
    if key is None:
        return None
    return EXPECTED_RESISTANCE.get(key, {}).get(antibiotic_key)


def is_expected_resistant(species_or_key: Optional[str], antibiotic_key: str) -> bool:
    return expected_resistance(species_or_key, antibiotic_key) is not None
