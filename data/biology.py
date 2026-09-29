"""
Versioned biology for the simulation.

  PAPER_V1  The biology that produced the AMRResistanceGNN paper, FROZEN. It
            is literally the same table objects as data/card_loader.py, and
            tests/test_paper_v1_frozen.py proves its trajectories and GNN
            training pairs are byte-identical to the pre-versioning code.
            AMRSimulationModel defaults to it, so the paper pipeline
            (ai/*.py, main.py) is unchanged.

  LAB_V2    Corrected biology for the interactive lab UI (the API server's
            default). Every change from paper_v1 is listed in
            LAB_V2_CHANGES with its source. Currently:
              - MRSA carries mecA (CARD ARO:3000617) as its intrinsic gene,
                replacing tetM, which CARD describes as mobile/acquired.
              - acrAB-tolC removed from MRSA's acquirable pool (AcrAB-TolC is
                a Gram-negative tripartite system; CARD ARO:3000237 / 3000384).
              - mecA is non-transferring (see KNOWN LIMITATION below).
            Klebsiella and the antibiotic diffusion step are NOT yet changed
            in lab_v2 (pending decisions; see CLAUDE.md).

KNOWN LIMITATION: SCCmec-style mobilisation of mecA is not modelled. The
model's only horizontal transfer mechanism is conjugation-style HGT between
neighbours, which is the wrong mechanism for SCCmec, so mecA is marked
non-transferring rather than transferred by it.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from data.card_loader import (
    RESISTANCE_GENES, GERM_PROFILES, ANTIBIOTIC_PROFILES,
    ResistanceGene, GermProfile, AntibioticProfile,
)

LAB_V2_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "lab_v2_config.json")


@dataclass(frozen=True)
class Biology:
    name: str
    genes: Dict[str, ResistanceGene]
    germs: Dict[str, GermProfile]
    antibiotics: Dict[str, AntibioticProfile]
    non_transferable: FrozenSet[str] = frozenset()
    warnings: List[str] = field(default_factory=list)

    def germ(self, key: str) -> GermProfile:
        if key not in self.germs:
            raise ValueError(f"Unknown germ: {key}. Available: {list(self.germs)}")
        return self.germs[key]

    def antibiotic(self, key: str) -> AntibioticProfile:
        if key not in self.antibiotics:
            raise ValueError(f"Unknown antibiotic: {key}. Available: {list(self.antibiotics)}")
        return self.antibiotics[key]

    def summary(self) -> dict:
        return {"name": self.name, "warnings": list(self.warnings)}


# ─────────────────────────────────────────────────────────────────────────────
# PAPER_V1 — frozen: the very same objects card_loader exposes
# ─────────────────────────────────────────────────────────────────────────────

PAPER_V1 = Biology(
    name="paper_v1",
    genes=RESISTANCE_GENES,
    germs=GERM_PROFILES,
    antibiotics=ANTIBIOTIC_PROFILES,
)


# ─────────────────────────────────────────────────────────────────────────────
# LAB_V2
# ─────────────────────────────────────────────────────────────────────────────

LAB_V2_CHANGES = [
    ("MRSA intrinsic genes: ['tetM'] -> ['mecA']",
     "CARD ARO:3000617 (mecA: PBP2a, target replacement; penicillins, cephalosporins, "
     "carbapenems, monobactams; 'commonly associated with MRSA'). CARD ARO:3000186 "
     "(tet(M) 'found on transposable DNA elements'); EUCAST Expected Resistant "
     "Phenotypes v1.2 Table 4 does not list S. aureus as expected-resistant to tetracyclines."),
    ("MRSA acquirable pool: acrAB-tolC removed",
     "CARD ARO:3000237 (TolC: subunit of efflux complexes 'in Gram negative bacteria'); "
     "CARD ARO:3000384 (AcrAB-TolC: RND efflux system 'in Gram-negative bacteria')."),
    ("mecA does not transfer by the model's conjugation-style HGT",
     "Known limitation: SCCmec mobilisation is not modelled (wrong mechanism to reuse)."),
]


def load_lab_v2_config(path: str = LAB_V2_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def build_lab_v2(**overrides) -> Biology:
    """Build LAB_V2 from data/lab_v2_config.json, with keyword overrides
    (e.g. build_lab_v2(mecA_fitness_cost=0.05)) for experiments/tests."""
    cfg = {**load_lab_v2_config(), **overrides}
    warnings = []

    cost = cfg.get("mecA_fitness_cost")
    if cost is None:
        warnings.append("mecA fitness cost is UNSET (awaiting the values measured by "
                        "Ender et al. 2004); no mecA fitness cost is applied.")
        cost = 0.0
    if not (0.0 <= float(cost) < 1.0):
        raise ValueError(f"mecA_fitness_cost must be in [0, 1), got {cost}")

    mecA = ResistanceGene(
        card_id="ARO:3000617",
        name="mecA",
        mechanism="antibiotic target replacement",
        drug_classes=["penicillin", "cephalosporin", "carbapenem", "monobactam"],
        acquisition_prob=0.0,   # never transferred: SCCmec mobilisation not modelled
        fitness_cost=float(cost),
        description="PBP2a (low-affinity penicillin-binding protein) on SCCmec; defines MRSA. "
                    "CARD ARO:3000617. Fitness cost: see data/lab_v2_config.json.",
    )
    genes = {**RESISTANCE_GENES, "mecA": mecA}

    mrsa = dataclasses.replace(
        GERM_PROFILES["mrsa"],
        natural_resistances=["mecA"],
        acquired_resistance_pool=[g for g in GERM_PROFILES["mrsa"].acquired_resistance_pool
                                  if g != "acrAB-tolC"],
    )
    germs = {**GERM_PROFILES, "mrsa": mrsa}

    return Biology(
        name="lab_v2",
        genes=genes,
        germs=germs,
        antibiotics=ANTIBIOTIC_PROFILES,
        non_transferable=frozenset({"mecA"}),
        warnings=warnings,
    )


LAB_V2 = build_lab_v2()

BIOLOGIES = {"paper_v1": PAPER_V1, "lab_v2": LAB_V2}
