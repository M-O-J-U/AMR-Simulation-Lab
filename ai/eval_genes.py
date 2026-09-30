"""
Which genes enter the headline (macro) metrics, and why. Decided with the
project owner on 2026-09-30.

The model still predicts all 10 genes (GENE_INDEX, unchanged architecture);
this only controls how results are aggregated and reported.

  EXCLUDED  never evaluated
  SEPARATE  evaluated, but reported on its own line with a label, never
            folded into the headline macro
"""
import math
from typing import Dict, Tuple

from ai.feature_engineering import GENE_INDEX

EXCLUDED: Dict[str, str] = {
    "mexAB-oprM": ("intrinsic chromosomal efflux system (CARD ARO:3000386: 'expressed in "
                   "P. aeruginosa'); no species can acquire it in the model, so it has zero "
                   "transfer events by construction"),
}

SEPARATE: Dict[str, str] = {
    "vanA": ("SIMPLIFIED MECHANISM: in the simulation vanA arises de novo in MRSA and spreads "
             "MRSA-to-MRSA; the documented route is interspecies transfer of Tn1546 from "
             "Enterococcus faecalis (Weigel et al. 2003, doi:10.1126/science.1090956), with "
             "the first VRSA isolates arising independently (Clark et al. 2005, "
             "doi:10.1128/AAC.49.1.470-472.2005). Enterococcus is not modelled."),
}

HEADLINE_GENES = [g for g in GENE_INDEX if g not in EXCLUDED and g not in SEPARATE]


def _finite(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def headline_macro(per_gene: Dict[str, float]) -> Tuple[float, int]:
    """Mean over HEADLINE_GENES that have a finite value (i.e. test positives).
    Returns (mean or nan, number of genes averaged)."""
    vals = [per_gene[g] for g in HEADLINE_GENES if _finite(per_gene.get(g))]
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)
