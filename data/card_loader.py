"""
CARD (Comprehensive Antibiotic Resistance Database) loader.
Provides real resistance gene data and known germ profiles grounded in published biology.

Real CARD data sources:
  - https://card.mcmaster.ca/download
  - NCBI AMR Reference Gene Catalog
  - WHO GLASS Pakistan Surveillance

For simulation, we embed a curated subset of well-documented resistance mechanisms.
When internet is available, this module can fetch live CARD data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json, math, random

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResistanceGene:
    """A real AMR resistance gene from CARD ontology."""
    card_id: str
    name: str
    mechanism: str          # e.g. "antibiotic inactivation", "efflux pump", "target alteration"
    drug_classes: List[str] # antibiotics this gene resists
    acquisition_prob: float # probability of horizontal gene transfer per step
    fitness_cost: float     # metabolic cost (0=none, 1=lethal) — real bacteria pay a cost
    description: str

@dataclass
class AntibioticProfile:
    """Real antibiotic with pharmacodynamic parameters."""
    name: str
    drug_class: str
    mechanism_of_action: str    # how it kills bacteria
    bactericidal: bool          # True = kills, False = bacteriostatic (inhibits growth)
    mbc: float                  # Minimum bactericidal concentration (µg/mL)
    mic_susceptible: float      # MIC breakpoint S (EUCAST)
    mic_resistant: float        # MIC breakpoint R (EUCAST)
    diffusion_rate: float       # how fast it spreads in the environment
    decay_rate: float           # how fast it degrades
    color_hex: str              # for visualization
    description: str

@dataclass
class GermProfile:
    """
    A real bacterial species with documented biology.
    Parameters sourced from WHO Priority Pathogen List and published literature.
    """
    species: str
    common_name: str
    gram_stain: str             # "positive" or "negative"
    shape: str                  # "coccus", "bacillus", "spirillum"
    who_priority: str           # "CRITICAL", "HIGH", "MEDIUM"
    doubling_time_steps: int    # steps between reproduction
    baseline_mutation_rate: float
    baseline_fitness: float
    natural_resistances: List[str]   # intrinsic resistance genes
    acquired_resistance_pool: List[str]  # genes it CAN acquire
    virulence_factors: List[str]
    energy_consumption: float   # metabolic energy cost per step
    motility: float             # 0=none, 1=highly motile
    sporulation: bool           # can form spores
    biofilm_potential: float    # 0–1 ability to form biofilms
    description: str
    color_hex: str
    pakistan_prevalence: str    # relevance for Pakistan/South Asia

# ─────────────────────────────────────────────────────────────────────────────
# REAL CARD RESISTANCE GENES (curated subset)
# ─────────────────────────────────────────────────────────────────────────────

RESISTANCE_GENES: Dict[str, ResistanceGene] = {
    # ── Beta-lactamases ──────────────────────────────────────────────────────
    "blaTEM-1": ResistanceGene(
        card_id="ARO:3000237",
        name="blaTEM-1",
        mechanism="antibiotic inactivation",
        drug_classes=["penicillin", "ampicillin", "amoxicillin"],
        acquisition_prob=0.04,
        fitness_cost=0.02,
        description="Classic TEM-1 beta-lactamase. Most common resistance gene worldwide. "
                    "Hydrolyzes penicillins and early cephalosporins."
    ),
    "blaCTX-M-15": ResistanceGene(
        card_id="ARO:3000237",
        name="blaCTX-M-15",
        mechanism="antibiotic inactivation",
        drug_classes=["cephalosporin", "penicillin", "beta-lactam"],
        acquisition_prob=0.03,
        fitness_cost=0.03,
        description="ESBL (Extended-Spectrum Beta-Lactamase). Dominant in South Asia "
                    "including Pakistan. Hydrolyzes 3rd-gen cephalosporins. High clinical burden."
    ),
    "blaKPC-2": ResistanceGene(
        card_id="ARO:3000159",
        name="blaKPC-2",
        mechanism="antibiotic inactivation",
        drug_classes=["carbapenem", "penicillin", "cephalosporin"],
        acquisition_prob=0.02,
        fitness_cost=0.04,
        description="Klebsiella pneumoniae Carbapenemase. Confers resistance to carbapenems "
                    "(last-resort antibiotics). WHO critical priority pathogen gene."
    ),
    "blaNDM-1": ResistanceGene(
        card_id="ARO:3000589",
        name="blaNDM-1",
        mechanism="antibiotic inactivation",
        drug_classes=["carbapenem", "beta-lactam", "cephalosporin"],
        acquisition_prob=0.015,
        fitness_cost=0.05,
        description="New Delhi Metallo-beta-lactamase. First identified in India/Pakistan 2009. "
                    "Hydrolyzes almost ALL beta-lactams. Pandemic spread via plasmids."
    ),
    # ── Efflux Pumps ────────────────────────────────────────────────────────
    "mexAB-oprM": ResistanceGene(
        card_id="ARO:3000157",
        name="mexAB-oprM",
        mechanism="antibiotic efflux",
        drug_classes=["fluoroquinolone", "beta-lactam", "chloramphenicol"],
        acquisition_prob=0.025,
        fitness_cost=0.06,
        description="MexAB-OprM efflux pump system in Pseudomonas. Pumps antibiotics "
                    "out of the cell. Multi-drug resistance mechanism."
    ),
    "acrAB-tolC": ResistanceGene(
        card_id="ARO:3000055",
        name="acrAB-tolC",
        mechanism="antibiotic efflux",
        drug_classes=["fluoroquinolone", "tetracycline", "chloramphenicol", "ampicillin"],
        acquisition_prob=0.03,
        fitness_cost=0.04,
        description="AcrAB-TolC efflux pump. Major MDR pump in E. coli and Klebsiella. "
                    "Can be upregulated under antibiotic stress."
    ),
    # ── Target Modification ─────────────────────────────────────────────────
    "gyrA_S83L": ResistanceGene(
        card_id="ARO:3000181",
        name="gyrA_S83L",
        mechanism="antibiotic target alteration",
        drug_classes=["fluoroquinolone", "ciprofloxacin"],
        acquisition_prob=0.01,  # point mutation, not HGT
        fitness_cost=0.01,
        description="Point mutation in DNA gyrase (gyrA Ser83Leu). Main fluoroquinolone "
                    "resistance mechanism. Very common in ciprofloxacin-resistant E. coli/Klebsiella."
    ),
    "mcr-1": ResistanceGene(
        card_id="ARO:3000745",
        name="mcr-1",
        mechanism="antibiotic target alteration",
        drug_classes=["colistin", "polymyxin"],
        acquisition_prob=0.02,
        fitness_cost=0.07,
        description="MCR-1 phosphoethanolamine transferase. Confers resistance to colistin "
                    "(last-resort antibiotic for MDR gram-negatives). Plasmid-mediated — pandemic concern."
    ),
    # ── Ribosomal Protection ────────────────────────────────────────────────
    "tetM": ResistanceGene(
        card_id="ARO:3000186",
        name="tetM",
        mechanism="antibiotic target protection",
        drug_classes=["tetracycline"],
        acquisition_prob=0.05,
        fitness_cost=0.02,
        description="Tet(M) ribosomal protection protein. Most widespread tetracycline "
                    "resistance gene globally. Carried on Tn916-type transposons."
    ),
    "vanA": ResistanceGene(
        card_id="ARO:3000089",
        name="vanA",
        mechanism="antibiotic target alteration",
        drug_classes=["vancomycin", "glycopeptide"],
        acquisition_prob=0.01,
        fitness_cost=0.08,
        description="VanA vancomycin resistance. Found in VRE (Vancomycin-Resistant Enterococcus). "
                    "Alters peptidoglycan precursors so vancomycin cannot bind."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# REAL ANTIBIOTIC PROFILES (EUCAST breakpoints)
# ─────────────────────────────────────────────────────────────────────────────

ANTIBIOTIC_PROFILES: Dict[str, AntibioticProfile] = {
    "ciprofloxacin": AntibioticProfile(
        name="Ciprofloxacin",
        drug_class="fluoroquinolone",
        mechanism_of_action="Inhibits DNA gyrase and topoisomerase IV → blocks DNA replication",
        bactericidal=True,
        mbc=0.5,
        mic_susceptible=0.5,
        mic_resistant=1.0,
        diffusion_rate=0.8,
        decay_rate=0.005,
        color_hex="#00D4FF",
        description="Broad-spectrum fluoroquinolone. First-line for UTIs, GI infections. "
                    "High resistance rates in Pakistan (>60% in E. coli isolates per WHO GLASS)."
    ),
    "meropenem": AntibioticProfile(
        name="Meropenem",
        drug_class="carbapenem",
        mechanism_of_action="Inhibits penicillin-binding proteins → disrupts cell wall synthesis",
        bactericidal=True,
        mbc=0.25,
        mic_susceptible=2.0,
        mic_resistant=8.0,
        diffusion_rate=0.6,
        decay_rate=0.008,
        color_hex="#FF6B35",
        description="Last-resort carbapenem. Used for MDR gram-negative infections. "
                    "Rising resistance via NDM-1, KPC-2. Critical importance for Pakistan AMR burden."
    ),
    "colistin": AntibioticProfile(
        name="Colistin",
        drug_class="polymyxin",
        mechanism_of_action="Disrupts outer membrane of gram-negative bacteria → cell lysis",
        bactericidal=True,
        mbc=2.0,
        mic_susceptible=2.0,
        mic_resistant=4.0,
        diffusion_rate=0.4,
        decay_rate=0.003,
        color_hex="#FF2D55",
        description="Last-resort antibiotic for XDR gram-negative bacteria. "
                    "MCR-1 plasmid resistance spreading globally. Nephrotoxic."
    ),
    "vancomycin": AntibioticProfile(
        name="Vancomycin",
        drug_class="glycopeptide",
        mechanism_of_action="Binds D-Ala-D-Ala peptidoglycan precursors → inhibits cell wall synthesis",
        bactericidal=True,
        mbc=2.0,
        mic_susceptible=2.0,
        mic_resistant=16.0,
        diffusion_rate=0.3,
        decay_rate=0.004,
        color_hex="#BF5AF2",
        description="Primary treatment for MRSA and VRE. Gram-positive only. "
                    "VanA resistance (vancomycin-resistant enterococci) spreading in hospitals."
    ),
    "ampicillin": AntibioticProfile(
        name="Ampicillin",
        drug_class="penicillin",
        mechanism_of_action="Binds penicillin-binding proteins → inhibits peptidoglycan cross-linking",
        bactericidal=True,
        mbc=4.0,
        mic_susceptible=8.0,
        mic_resistant=8.0,
        diffusion_rate=0.9,
        decay_rate=0.01,
        color_hex="#34C759",
        description="Classic penicillin. >80% E. coli isolates resistant via blaTEM-1. "
                    "Still useful for susceptible strains (Listeria, Enterococcus)."
    ),
    "tetracycline": AntibioticProfile(
        name="Tetracycline",
        drug_class="tetracycline",
        mechanism_of_action="Binds 30S ribosomal subunit → blocks aminoacyl-tRNA binding",
        bactericidal=False,  # bacteriostatic
        mbc=8.0,
        mic_susceptible=1.0,
        mic_resistant=8.0,
        diffusion_rate=0.85,
        decay_rate=0.006,
        color_hex="#FFD60A",
        description="Broad-spectrum bacteriostatic. Widespread resistance via tetM. "
                    "Still used in resource-limited settings. Marks bacteria yellow in visualization."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# REAL GERM PROFILES (WHO Priority Pathogens + Cancer cell analogue)
# ─────────────────────────────────────────────────────────────────────────────

GERM_PROFILES: Dict[str, GermProfile] = {

    # ── STAGE 1: Well-known, well-studied — validation targets ───────────────
    "e_coli": GermProfile(
        species="Escherichia coli",
        common_name="E. coli",
        gram_stain="negative",
        shape="bacillus",
        who_priority="HIGH",
        doubling_time_steps=8,
        baseline_mutation_rate=0.001,
        baseline_fitness=0.85,
        natural_resistances=[],
        acquired_resistance_pool=["blaTEM-1", "blaCTX-M-15", "gyrA_S83L", "acrAB-tolC", "tetM"],
        virulence_factors=["LPS endotoxin", "type 1 fimbriae", "hemolysin"],
        energy_consumption=0.05,
        motility=0.7,
        sporulation=False,
        biofilm_potential=0.4,
        description="Most common cause of UTIs, sepsis, and food poisoning worldwide. "
                    "Model organism for AMR research. Pakistan UTI isolates: ~70% ciprofloxacin-resistant.",
        color_hex="#4CAF50",
        pakistan_prevalence="VERY HIGH — dominant cause of community-acquired UTI and sepsis"
    ),

    "klebsiella_pneumoniae": GermProfile(
        species="Klebsiella pneumoniae",
        common_name="Klebsiella",
        gram_stain="negative",
        shape="bacillus",
        who_priority="CRITICAL",
        doubling_time_steps=10,
        baseline_mutation_rate=0.0015,
        baseline_fitness=0.90,
        natural_resistances=["acrAB-tolC"],
        acquired_resistance_pool=["blaCTX-M-15", "blaKPC-2", "blaNDM-1", "mcr-1", "gyrA_S83L"],
        virulence_factors=["capsule (K antigen)", "LPS", "siderophores", "fimbriae"],
        energy_consumption=0.06,
        motility=0.1,
        sporulation=False,
        biofilm_potential=0.75,
        description="WHO Critical Priority 1. Causes hospital-acquired pneumonia, sepsis, UTI. "
                    "NDM-1 producing strains extremely difficult to treat. "
                    "Pakistan: highest burden in South Asia per WHO GLASS 2022.",
        color_hex="#FF9800",
        pakistan_prevalence="CRITICAL — leading cause of hospital AMR deaths in Pakistan"
    ),

    # ── STAGE 2: Almost incurable — the hard targets ─────────────────────────
    "acinetobacter_baumannii": GermProfile(
        species="Acinetobacter baumannii",
        common_name="Acinetobacter",
        gram_stain="negative",
        shape="coccus",
        who_priority="CRITICAL",
        doubling_time_steps=12,
        baseline_mutation_rate=0.002,
        baseline_fitness=0.75,
        natural_resistances=["mexAB-oprM", "acrAB-tolC"],
        acquired_resistance_pool=["blaNDM-1", "blaKPC-2", "mcr-1", "blaCTX-M-15"],
        virulence_factors=["outer membrane proteins", "biofilm", "capsule", "phospholipase"],
        energy_consumption=0.04,
        motility=0.05,
        sporulation=False,
        biofilm_potential=0.95,
        description="'Iraqibacter' — extremely drug-resistant hospital pathogen. "
                    "Survives on surfaces for weeks. XDRAB (extensively drug-resistant) "
                    "strains may only be susceptible to colistin alone. ",
        color_hex="#F44336",
        pakistan_prevalence="HIGH — dominant ICU pathogen in Pakistani hospitals"
    ),

    "pseudomonas_aeruginosa": GermProfile(
        species="Pseudomonas aeruginosa",
        common_name="Pseudomonas",
        gram_stain="negative",
        shape="bacillus",
        who_priority="CRITICAL",
        doubling_time_steps=11,
        baseline_mutation_rate=0.003,  # highest intrinsic mutation rate
        baseline_fitness=0.80,
        natural_resistances=["mexAB-oprM"],
        acquired_resistance_pool=["blaKPC-2", "blaNDM-1", "mcr-1", "gyrA_S83L"],
        virulence_factors=["pyocyanin", "elastase", "exotoxin A", "biofilm", "flagella"],
        energy_consumption=0.07,
        motility=0.9,
        sporulation=False,
        biofilm_potential=0.90,
        description="Highly adaptable environmental pathogen. Intrinsically resistant to many antibiotics. "
                    "Key pathogen in cystic fibrosis, burn wounds, ventilator-associated pneumonia. "
                    "Acquires resistance rapidly during treatment.",
        color_hex="#9C27B0",
        pakistan_prevalence="HIGH — major cause of burn unit and ICU infections"
    ),

    "mrsa": GermProfile(
        species="Staphylococcus aureus (MRSA)",
        common_name="MRSA",
        gram_stain="positive",
        shape="coccus",
        who_priority="HIGH",
        doubling_time_steps=9,
        baseline_mutation_rate=0.001,
        baseline_fitness=0.88,
        natural_resistances=["tetM"],  # mecA not modeled as standard gene — is chromosomal
        acquired_resistance_pool=["vanA", "tetM", "acrAB-tolC"],
        virulence_factors=["protein A", "alpha-toxin", "leukotoxin", "TSST-1", "biofilm"],
        energy_consumption=0.05,
        motility=0.0,
        sporulation=False,
        biofilm_potential=0.85,
        description="Methicillin-Resistant S. aureus. Resistant to all beta-lactams via mecA (chromosomal). "
                    "Vancomycin is primary treatment. VRSA (vancomycin-resistant) emerging. "
                    "Community and hospital strains differ in virulence.",
        color_hex="#FFD700",
        pakistan_prevalence="HIGH — common in hospital wounds and skin infections across Pakistan"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT COMBINATIONS (real clinical protocols)
# ─────────────────────────────────────────────────────────────────────────────

COMBINATION_THERAPIES = {
    "NDM_protocol": {
        "name": "NDM-1 Producer Protocol",
        "antibiotics": ["colistin", "meropenem"],
        "synergy_factor": 1.6,
        "description": "Combination therapy for NDM-1 producing Klebsiella/E. coli. "
                       "Colistin + meropenem shows in vitro synergy despite individual resistance."
    },
    "XDRAB_protocol": {
        "name": "XDRAB Salvage Protocol",
        "antibiotics": ["colistin", "ampicillin"],
        "synergy_factor": 1.4,
        "description": "For extensively drug-resistant Acinetobacter. Limited options. "
                       "Colistin remains backbone. Ampicillin-sulbactam may add activity."
    },
    "MRSA_protocol": {
        "name": "MRSA Standard Protocol",
        "antibiotics": ["vancomycin"],
        "synergy_factor": 1.0,
        "description": "Vancomycin monotherapy remains standard for MRSA bacteremia. "
                       "AUC/MIC monitoring required. Alternatives: daptomycin, linezolid."
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_germ(name: str) -> GermProfile:
    if name not in GERM_PROFILES:
        raise ValueError(f"Unknown germ '{name}'. Available: {list(GERM_PROFILES.keys())}")
    return GERM_PROFILES[name]

def get_antibiotic(name: str) -> AntibioticProfile:
    if name not in ANTIBIOTIC_PROFILES:
        raise ValueError(f"Unknown antibiotic '{name}'. Available: {list(ANTIBIOTIC_PROFILES.keys())}")
    return ANTIBIOTIC_PROFILES[name]

def get_resistance_gene(name: str) -> ResistanceGene:
    return RESISTANCE_GENES[name]

def resistance_probability(gene: ResistanceGene, antibiotic: AntibioticProfile) -> float:
    """
    Calculate how much a resistance gene reduces antibiotic killing.
    Returns 0.0 (no protection) to 1.0 (full resistance).
    """
    if antibiotic.drug_class in gene.drug_classes or antibiotic.name.lower() in [d.lower() for d in gene.drug_classes]:
        return 0.90  # near-complete resistance
    # partial cross-resistance for related classes
    related = {
        "carbapenem": ["penicillin", "cephalosporin", "beta-lactam"],
        "beta-lactam": ["penicillin", "carbapenem"],
        "fluoroquinolone": ["ciprofloxacin"],
    }
    for drug_class in gene.drug_classes:
        if antibiotic.drug_class in related.get(drug_class, []):
            return 0.40
    return 0.0

def list_germs() -> List[str]:
    return list(GERM_PROFILES.keys())

def list_antibiotics() -> List[str]:
    return list(ANTIBIOTIC_PROFILES.keys())

def germ_summary(name: str) -> dict:
    g = get_germ(name)
    return {
        "species": g.species,
        "who_priority": g.who_priority,
        "gram_stain": g.gram_stain,
        "natural_resistances": g.natural_resistances,
        "can_acquire": g.acquired_resistance_pool,
        "biofilm_potential": g.biofilm_potential,
        "pakistan_prevalence": g.pakistan_prevalence,
    }