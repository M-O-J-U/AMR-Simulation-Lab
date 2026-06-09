"""
BacteriumAgent — Mesa 3.x agent modeling a single bacterial cell.

AI mechanisms:
  1. Adaptive resistance — SOS response upregulates efflux pumps under stress
  2. Horizontal Gene Transfer (HGT) — plasmid conjugation between neighbors
  3. Mutation engine — point mutations + gene acquisition
  4. Biofilm formation — collective protection above density threshold
  5. Persister switching — bet-hedging dormancy under antibiotic stress
  6. Energy metabolism — nutrient consumption, starvation death
  7. Chemotaxis — motile bacteria move toward nutrients, away from antibiotics

Biology references: Simmons 2008 (SOS), Lewis 2010 (persisters),
Hoiby 2010 (biofilm), Andersson 2010 (fitness costs), Radman 1975 (SOS mutation rate)
"""

import math, random, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from mesa import Agent
from data.card_loader import (
    GermProfile, AntibioticProfile,
    RESISTANCE_GENES, resistance_probability
)

# ── Constants ─────────────────────────────────────────────────────────────────
BIOFILM_DENSITY_THRESHOLD  = 4
BIOFILM_PROTECTION_FACTOR  = 0.60   # Hoiby et al. 2010
SOS_MUTATION_MULTIPLIER    = 8.0    # Radman 1975
CONJUGATION_DISTANCE       = 1
MAX_AGE_STEPS              = 200
ENERGY_GAIN_PER_STEP       = 0.15
DEATH_THRESHOLD            = 0.0

class BacterialState(Enum):
    GROWING  = "growing"
    STRESSED = "stressed"
    DORMANT  = "dormant"
    BIOFILM  = "biofilm"
    DYING    = "dying"
    DEAD     = "dead"

@dataclass
class MutationEvent:
    step: int
    gene: str
    fitness_delta: float
    description: str

class BacteriumAgent(Agent):

    def __init__(self, model, profile: GermProfile, position: Tuple[int,int],
                 generation: int = 0, parent_id: Optional[int] = None,
                 inherited_genes: Optional[Set[str]] = None,
                 inherited_mutations: int = 0):
        super().__init__(model)

        self.cell_id          = str(uuid.uuid4())[:8]
        self.profile          = profile
        self.species          = profile.species
        self.generation       = generation
        self.parent_id        = parent_id
        self.age              = 0
        self.birth_step       = model.steps

        self._init_pos = position  # set properly by grid.place_agent
        self.state            = BacterialState.GROWING
        self.energy           = random.uniform(0.5, 0.9)

        self.resistance_genes: Set[str] = set(profile.natural_resistances)
        if inherited_genes:
            self.resistance_genes.update(inherited_genes)

        self.total_mutations  = inherited_mutations
        self.mutation_history: List[MutationEvent] = []
        self.fitness          = profile.baseline_fitness
        self._recalculate_fitness()

        self.sos_active       = False
        self.sos_duration     = 0
        self.stress_level     = 0.0

        self.in_biofilm       = False
        self.is_persister     = False
        self.persister_probability = 0.001

        self.steps_since_division = 0
        self.division_threshold   = profile.doubling_time_steps
        self.antibiotic_damage    = 0.0

        self.antibiotics_survived: List[str] = []
        self.hgt_events    = 0
        self.offspring_count = 0
        self.local_density = 0

    def _recalculate_fitness(self):
        cost = sum(RESISTANCE_GENES[g].fitness_cost
                   for g in self.resistance_genes if g in RESISTANCE_GENES)
        self.fitness = max(0.05, self.profile.baseline_fitness - cost)

    def get_resistance_to(self, antibiotic: AntibioticProfile) -> float:
        if not self.resistance_genes:
            return 0.0
        resistances = []
        for gn in self.resistance_genes:
            if gn in RESISTANCE_GENES:
                r = resistance_probability(RESISTANCE_GENES[gn], antibiotic)
                if r > 0:
                    resistances.append(r)
        if not resistances:
            base = 0.0
        else:
            combined = 1.0
            for r in resistances:
                combined *= (1.0 - r)
            base = 1.0 - combined
        if self.in_biofilm:
            base = 1.0 - (1.0 - base) * (1.0 - BIOFILM_PROTECTION_FACTOR)
        if self.is_persister:
            base = max(base, 0.95)
        return min(base, 0.999)

    def _update_sos_response(self, local_ab_conc: float):
        if local_ab_conc > 0.3 and not self.is_persister:
            self.stress_level = min(1.0, self.stress_level + local_ab_conc * 0.1)
            if self.stress_level > 0.4 and not self.sos_active:
                self.sos_active = True
                self.state = BacterialState.STRESSED
                if "acrAB-tolC" in self.profile.acquired_resistance_pool:
                    if random.random() < 0.15:
                        self.resistance_genes.add("acrAB-tolC")
                        self._recalculate_fitness()
        elif local_ab_conc < 0.1:
            self.stress_level = max(0, self.stress_level - 0.02)
            if self.stress_level < 0.2 and self.sos_active:
                self.sos_active = False
                if self.state == BacterialState.STRESSED:
                    self.state = BacterialState.GROWING
        if self.sos_active:
            self.sos_duration += 1

    def _attempt_hgt(self):
        if not self.resistance_genes or self.state == BacterialState.DEAD:
            return
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=CONJUGATION_DISTANCE)
        for recipient in neighbors:
            if not isinstance(recipient, BacteriumAgent):
                continue
            if recipient.state == BacterialState.DEAD:
                continue
            if recipient.species != self.species:
                continue
            for gn in list(self.resistance_genes):
                if gn not in RESISTANCE_GENES:
                    continue
                gene = RESISTANCE_GENES[gn]
                hgt_prob = gene.acquisition_prob * (2.0 if self.sos_active else 1.0)
                if (gn not in recipient.resistance_genes and
                    gn in recipient.profile.acquired_resistance_pool and
                    random.random() < hgt_prob):
                    recipient.resistance_genes.add(gn)
                    recipient._recalculate_fitness()
                    recipient.hgt_events += 1
                    self.hgt_events += 1
                    self.model.record_hgt_event(
                        donor_id=self.unique_id,
                        recipient_id=recipient.unique_id,
                        gene=gn, step=self.model.steps, position=self.pos)

    def _attempt_mutation(self):
        rate = self.profile.baseline_mutation_rate
        if self.sos_active:
            rate *= SOS_MUTATION_MULTIPLIER
        if random.random() > rate:
            return
        self.total_mutations += 1
        roll = random.random()
        if roll < 0.6:
            # Point mutation
            if ("gyrA_S83L" not in self.resistance_genes and
                "gyrA_S83L" in self.profile.acquired_resistance_pool and
                random.random() < 0.3):
                self.resistance_genes.add("gyrA_S83L")
                self._recalculate_fitness()
                self.mutation_history.append(MutationEvent(
                    self.model.steps, "gyrA_S83L",
                    -RESISTANCE_GENES["gyrA_S83L"].fitness_cost,
                    "Point mutation → fluoroquinolone resistance"))
            else:
                delta = random.gauss(-0.01, 0.02)
                self.fitness = max(0.01, min(1.0, self.fitness + delta))
        elif roll < 0.85:
            # Gene acquisition via mutation
            available = [g for g in self.profile.acquired_resistance_pool
                         if g not in self.resistance_genes]
            if available:
                gn = random.choice(available)
                self.resistance_genes.add(gn)
                self._recalculate_fitness()
                self.mutation_history.append(MutationEvent(
                    self.model.steps, gn,
                    -RESISTANCE_GENES[gn].fitness_cost,
                    f"Gene acquisition: {RESISTANCE_GENES[gn].mechanism}"))

    def _check_persister_switch(self, ab_conc: float):
        if not self.is_persister:
            p = self.persister_probability * (1 + 5 * self.stress_level)
            if random.random() < p:
                self.is_persister = True
                self.state = BacterialState.DORMANT
        else:
            if ab_conc < 0.05 and random.random() < 0.1:
                self.is_persister = False
                self.state = BacterialState.GROWING

    def _update_biofilm_status(self):
        cell_agents = [a for a in self.model.grid.get_cell_list_contents([self.pos])
                       if isinstance(a, BacteriumAgent) and a.state != BacterialState.DEAD]
        self.local_density = len(cell_agents)
        if (self.local_density >= BIOFILM_DENSITY_THRESHOLD and
                self.profile.biofilm_potential > 0.3 and not self.in_biofilm):
            if random.random() < self.profile.biofilm_potential * 0.1:
                self.in_biofilm = True
                self.state = BacterialState.BIOFILM
        elif self.local_density < 2 and self.in_biofilm:
            self.in_biofilm = False
            if self.state == BacterialState.BIOFILM:
                self.state = BacterialState.GROWING

    def _attempt_division(self):
        if self.state in [BacterialState.DYING, BacterialState.DEAD, BacterialState.DORMANT]:
            return
        if self.energy < 0.3:
            return
        effective_threshold = int(self.profile.doubling_time_steps / max(0.1, self.fitness))
        if self.sos_active:
            effective_threshold = int(effective_threshold * 1.5)
        self.steps_since_division += 1
        if self.steps_since_division < effective_threshold:
            return

        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty = [p for p in neighbors if self.model.grid.is_cell_empty(p)]
        target = random.choice(empty) if empty else self.pos

        daughter = BacteriumAgent(
            model=self.model, profile=self.profile, position=target,
            generation=self.generation + 1, parent_id=self.unique_id,
            inherited_genes=set(self.resistance_genes),
            inherited_mutations=self.total_mutations)
        if self.sos_active and random.random() < 0.05:
            daughter._attempt_mutation()

        self.model.grid.place_agent(daughter, target)
        self.model.population_counter[self.species] = \
            self.model.population_counter.get(self.species, 0) + 1

        self.offspring_count += 1
        self.steps_since_division = 0
        self.energy *= 0.5

    def _update_energy(self):
        nutrient = self.model.get_nutrient_at(self.pos)
        gain = ENERGY_GAIN_PER_STEP * nutrient * self.fitness
        cost = self.profile.energy_consumption
        if self.sos_active: cost *= 1.5
        if self.in_biofilm:  cost *= 0.7
        self.energy = max(0.0, min(1.0, self.energy + gain - cost))
        self.model.consume_nutrient(self.pos, cost * 0.5)

    def _move(self):
        if (self.profile.motility < 0.1 or self.in_biofilm or
                self.state == BacterialState.DEAD or
                random.random() > self.profile.motility):
            return
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        best_pos, best_score = self.pos, -999
        for p in neighbors:
            score = self.model.get_nutrient_at(p) * 2.0 - self.model.get_total_antibiotic_at(p) * 3.0
            if score > best_score:
                best_score, best_pos = score, p
        if best_pos != self.pos:
            self.model.grid.move_agent(self, best_pos)
            self.pos = best_pos

    def _take_antibiotic_damage(self):
        total_damage = 0.0
        for ab_name, conc in self.model.get_antibiotics_at(self.pos).items():
            if conc < 0.01:
                continue
            ab = self.model.antibiotic_profiles[ab_name]
            resistance = self.get_resistance_to(ab)
            effective_conc = conc * (1.0 - resistance)
            if ab.bactericidal:
                emax, ec50, n = 0.3, ab.mbc, 2.0
                damage = emax * (effective_conc**n) / (ec50**n + effective_conc**n)
                total_damage += damage
            else:
                self.steps_since_division = max(
                    self.steps_since_division,
                    int(self.division_threshold * 0.7 * effective_conc))

        if total_damage > 0:
            self.antibiotic_damage = min(1.0, self.antibiotic_damage + total_damage)
            if self.antibiotic_damage >= 0.7 and self.state != BacterialState.DYING:
                self.state = BacterialState.DYING
            if self.antibiotic_damage >= 1.0:
                self.state = BacterialState.DEAD

    def step(self):
        if self.state == BacterialState.DEAD:
            return
        self.age += 1
        if self.age > MAX_AGE_STEPS and random.random() < 0.05:
            self.state = BacterialState.DEAD
            return

        local_ab = self.model.get_total_antibiotic_at(self.pos)
        self._update_energy()
        if self.energy <= DEATH_THRESHOLD:
            self.state = BacterialState.DEAD
            return
        self._update_sos_response(local_ab)
        self._check_persister_switch(local_ab)
        self._take_antibiotic_damage()
        if self.state == BacterialState.DEAD:
            return
        self._attempt_mutation()
        if random.random() < 0.3:
            self._attempt_hgt()
        self._update_biofilm_status()
        self._move()
        self._attempt_division()

    def to_dict(self) -> dict:
        # find species key
        from data.card_loader import GERM_PROFILES
        sp_key = next((k for k,v in GERM_PROFILES.items() if v.species == self.species), "unknown")
        return {
            "id": self.unique_id,
            "cell_id": self.cell_id,
            "species": self.species,
            "species_key": sp_key,
            "pos": list(self.pos),
            "state": self.state.value,
            "age": self.age,
            "generation": self.generation,
            "energy": round(self.energy, 3),
            "fitness": round(self.fitness, 3),
            "stress_level": round(self.stress_level, 3),
            "antibiotic_damage": round(self.antibiotic_damage, 3),
            "resistance_genes": list(self.resistance_genes),
            "gene_count": len(self.resistance_genes),
            "in_biofilm": self.in_biofilm,
            "sos_active": self.sos_active,
            "is_persister": self.is_persister,
            "total_mutations": self.total_mutations,
            "hgt_events": self.hgt_events,
            "offspring_count": self.offspring_count,
            "local_density": self.local_density,
            "color_hex": self.profile.color_hex,
            "gram_stain": self.profile.gram_stain,
            "shape": self.profile.shape,
        }