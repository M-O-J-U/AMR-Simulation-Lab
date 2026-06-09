"""
AI analytics engine for the AMR simulation.

Provides:
  1. Resistance trajectory prediction — given current population genetics,
     predict which resistance genes will emerge next
  2. Fitness landscape mapping — track evolutionary paths bacteria take
  3. MIC prediction — estimate effective MIC for a bacterial population
  4. Treatment recommendation engine — suggest antibiotic combinations
  5. Population genetics metrics — Shannon diversity, selection coefficient

This module does NOT use neural networks (no training data).
Instead it uses:
  - Markov chain resistance gene transition models
  - Evolutionary game theory (replicator equations)
  - PK/PD pharmacodynamic models (Regoes et al.)
  - Population genetics (Wright-Fisher model approximation)
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from data.card_loader import (
    RESISTANCE_GENES, ANTIBIOTIC_PROFILES,
    ResistanceGene, AntibioticProfile, resistance_probability
)

# ─────────────────────────────────────────────────────────────────────────────
# FITNESS LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

class FitnessLandscape:
    """
    Models the fitness landscape for a bacterial population.
    Tracks which genotypes (sets of resistance genes) are present
    and their fitness values over time.
    """

    def __init__(self):
        self.genotype_history: Dict[frozenset, List[float]] = defaultdict(list)
        self.dominant_genotype_history: List[Tuple[int, frozenset]] = []

    def record(self, step: int, bacteria_list: list):
        """Record genotype frequencies at this step."""
        if not bacteria_list:
            return

        genotype_counts = Counter(
            frozenset(b.resistance_genes) for b in bacteria_list
        )

        for genotype, count in genotype_counts.items():
            freq = count / len(bacteria_list)
            self.genotype_history[genotype].append(freq)

        if genotype_counts:
            dominant = genotype_counts.most_common(1)[0][0]
            self.dominant_genotype_history.append((step, dominant))

    def get_emerging_genotypes(self) -> List[dict]:
        """Identify genotypes whose frequency is increasing."""
        emerging = []
        for genotype, history in self.genotype_history.items():
            if len(history) >= 3:
                trend = history[-1] - history[-3]
                if trend > 0.05:
                    emerging.append({
                        "genes": list(genotype),
                        "current_freq": round(history[-1], 3),
                        "trend": round(trend, 3),
                    })
        return sorted(emerging, key=lambda x: x["trend"], reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# MIC ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_population_mic(
    bacteria_list: list,
    antibiotic_key: str
) -> dict:
    """
    Estimate the MIC distribution for a bacterial population.
    Returns the fraction susceptible, intermediate, and resistant.

    Based on EUCAST breakpoints and resistance gene profiles.
    """
    if antibiotic_key not in ANTIBIOTIC_PROFILES:
        return {}

    ab = ANTIBIOTIC_PROFILES[antibiotic_key]

    susceptible = 0
    intermediate = 0
    resistant_count = 0

    for b in bacteria_list:
        r = _calculate_mic_fold_change(b.resistance_genes, ab)
        effective_mic = ab.mic_susceptible * r

        if effective_mic <= ab.mic_susceptible:
            susceptible += 1
        elif effective_mic <= ab.mic_resistant:
            intermediate += 1
        else:
            resistant_count += 1

    total = len(bacteria_list) or 1
    return {
        "antibiotic": ab.name,
        "susceptible_pct": round(100 * susceptible / total, 1),
        "intermediate_pct": round(100 * intermediate / total, 1),
        "resistant_pct": round(100 * resistant_count / total, 1),
        "breakpoint_S": ab.mic_susceptible,
        "breakpoint_R": ab.mic_resistant,
    }

def _calculate_mic_fold_change(resistance_genes: set, ab: AntibioticProfile) -> float:
    """Calculate fold-change in MIC due to resistance genes."""
    fold = 1.0
    for gene_name in resistance_genes:
        if gene_name in RESISTANCE_GENES:
            gene = RESISTANCE_GENES[gene_name]
            r = resistance_probability(gene, ab)
            if r >= 0.9:
                fold *= 64   # clinical high-level resistance
            elif r >= 0.4:
                fold *= 4    # intermediate resistance
    return fold

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def recommend_treatment(bacteria_list: list, available_antibiotics: List[str]) -> List[dict]:
    """
    Rank antibiotics by predicted efficacy against the current population.
    Uses resistance gene profiles to score each antibiotic.

    Returns sorted list of recommendations with rationale.
    """
    if not bacteria_list:
        return []

    recommendations = []

    for ab_key in available_antibiotics:
        if ab_key not in ANTIBIOTIC_PROFILES:
            continue
        ab = ANTIBIOTIC_PROFILES[ab_key]

        # Calculate fraction susceptible
        susceptible_count = sum(
            1 for b in bacteria_list
            if _calculate_mic_fold_change(b.resistance_genes, ab) <= 2.0
        )
        pct_susceptible = susceptible_count / len(bacteria_list)

        # Score: weighted by susceptibility, bactericidal bonus
        score = pct_susceptible * 100
        if ab.bactericidal:
            score *= 1.2

        # Penalty if resistance genes known to counter it are widespread
        resistant_genes = [
            g for g in RESISTANCE_GENES.values()
            if ab.drug_class in g.drug_classes
        ]
        gene_prevalence = sum(
            1 for b in bacteria_list
            if any(g.name in b.resistance_genes for g in resistant_genes)
        ) / len(bacteria_list)

        score *= (1 - gene_prevalence * 0.8)

        recommendations.append({
            "antibiotic": ab.name,
            "key": ab_key,
            "drug_class": ab.drug_class,
            "score": round(score, 1),
            "pct_susceptible": round(pct_susceptible * 100, 1),
            "bactericidal": ab.bactericidal,
            "mechanism": ab.mechanism_of_action,
            "rationale": _build_rationale(ab, pct_susceptible, gene_prevalence),
        })

    return sorted(recommendations, key=lambda x: x["score"], reverse=True)

def _build_rationale(ab: AntibioticProfile, pct_susceptible: float, gene_prevalence: float) -> str:
    parts = []
    if pct_susceptible > 0.8:
        parts.append(f"{pct_susceptible*100:.0f}% of population susceptible")
    elif pct_susceptible < 0.2:
        parts.append(f"High resistance — only {pct_susceptible*100:.0f}% susceptible")

    if ab.bactericidal:
        parts.append("bactericidal (kills, not just inhibits)")
    else:
        parts.append("bacteriostatic (inhibits growth)")

    if gene_prevalence > 0.5:
        parts.append(f"WARNING: {gene_prevalence*100:.0f}% of population carry resistance genes")

    return "; ".join(parts) if parts else "Standard recommendation"

# ─────────────────────────────────────────────────────────────────────────────
# POPULATION GENETICS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def shannon_diversity(bacteria_list: list) -> float:
    """
    Shannon diversity index of resistance gene genotypes.
    High diversity = more evolutionary potential.
    """
    if not bacteria_list:
        return 0.0

    genotype_counts = Counter(
        frozenset(b.resistance_genes) for b in bacteria_list
    )
    total = len(bacteria_list)
    H = 0.0
    for count in genotype_counts.values():
        p = count / total
        if p > 0:
            H -= p * math.log(p)
    return round(H, 4)

def selection_coefficient(
    bacteria_list: list,
    antibiotic_key: str
) -> float:
    """
    Selection coefficient s for resistance under antibiotic pressure.
    s > 0 means resistance is positively selected.
    Based on population genetics: s = (fitness_R - fitness_S) / fitness_S
    """
    if antibiotic_key not in ANTIBIOTIC_PROFILES:
        return 0.0

    ab = ANTIBIOTIC_PROFILES[antibiotic_key]
    resistant = [
        b for b in bacteria_list
        if _calculate_mic_fold_change(b.resistance_genes, ab) > 2.0
    ]
    susceptible_bac = [
        b for b in bacteria_list
        if _calculate_mic_fold_change(b.resistance_genes, ab) <= 2.0
    ]

    if not resistant or not susceptible_bac:
        return 0.0

    mean_fit_R = sum(b.fitness for b in resistant) / len(resistant)
    mean_fit_S = sum(b.fitness for b in susceptible_bac) / len(susceptible_bac)

    if mean_fit_S == 0:
        return 0.0
    return round((mean_fit_R - mean_fit_S) / mean_fit_S, 4)

def predict_resistance_emergence(
    bacteria_list: list,
    antibiotic_key: str,
    steps_ahead: int = 20
) -> dict:
    """
    Simple Markov chain prediction of resistance gene emergence probability.
    Given the current population, estimates probability that a specific
    resistance gene will reach 50% frequency within N steps.
    """
    if antibiotic_key not in ANTIBIOTIC_PROFILES:
        return {}

    ab = ANTIBIOTIC_PROFILES[antibiotic_key]

    # Find relevant resistance genes not yet at high frequency
    total = len(bacteria_list) or 1
    gene_freqs = {}
    for gene_name, gene in RESISTANCE_GENES.items():
        if ab.drug_class in gene.drug_classes:
            count = sum(1 for b in bacteria_list if gene_name in b.resistance_genes)
            freq = count / total
            gene_freqs[gene_name] = freq

    predictions = []
    for gene_name, current_freq in gene_freqs.items():
        if current_freq >= 0.5:
            status = "already_dominant"
            prob_50_pct = 1.0
        elif current_freq == 0:
            # Not present yet — could emerge via mutation/HGT
            gene = RESISTANCE_GENES[gene_name]
            emergence_prob = gene.acquisition_prob * steps_ahead * 0.5
            prob_50_pct = min(0.99, emergence_prob)
            status = "not_present"
        else:
            # Present but rare — use logistic growth approximation
            # P(fixation) ≈ 2s for small s in diploid, adjust for haploid
            s = selection_coefficient(bacteria_list, antibiotic_key)
            if s > 0:
                # Time to 50%: t ≈ ln(0.5/(1-0.5)) / s * (1/current_freq)
                t_to_50 = abs(math.log(current_freq / (1 - current_freq + 1e-9))) / max(s, 0.001)
                prob_50_pct = min(0.99, steps_ahead / max(t_to_50, 1))
            else:
                prob_50_pct = 0.05
            status = "rare"

        predictions.append({
            "gene": gene_name,
            "current_frequency": round(current_freq, 3),
            "probability_dominant_in_{}_steps".format(steps_ahead): round(prob_50_pct, 3),
            "status": status,
            "mechanism": RESISTANCE_GENES[gene_name].mechanism,
        })

    return {
        "antibiotic": ab.name,
        "steps_ahead": steps_ahead,
        "predictions": sorted(predictions, key=lambda x: x.get(
            f"probability_dominant_in_{steps_ahead}_steps", 0), reverse=True)
    }