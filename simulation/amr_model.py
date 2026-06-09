"""
AMRSimulationModel — Mesa 3.x MultiGrid model.
"""

import math, random, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa import DataCollector

from core.bacterium_agent import BacteriumAgent, BacterialState
from data.card_loader import (
    GERM_PROFILES, ANTIBIOTIC_PROFILES,
    GermProfile, AntibioticProfile, get_germ, get_antibiotic
)
from simulation.sim_logger import SimLogger, LogLevel

DEFAULT_WIDTH  = 80
DEFAULT_HEIGHT = 60
MAX_POPULATION = 3000
NUTRIENT_REGEN = 0.02

class HGTEvent:
    __slots__ = ["step","donor_id","recipient_id","gene","position"]
    def __init__(self, step, donor_id, recipient_id, gene, position):
        self.step, self.donor_id = step, donor_id
        self.recipient_id, self.gene, self.position = recipient_id, gene, position

class AMRSimulationModel(Model):

    def __init__(self, scenario="validation", width=DEFAULT_WIDTH,
                 height=DEFAULT_HEIGHT, initial_bacteria=80,
                 nutrient_level=0.9, seed=None, enable_logging=True):
        super().__init__(seed=seed)

        self.sim_scenario = scenario
        self.width      = width
        self.height     = height
        self.current_step = 0
        self.running    = True
        self.paused     = False
        self.enable_logging = enable_logging

        # Get or create logger (silent mode when logging disabled)
        if enable_logging:
            try:
                self.logger = SimLogger.get()
            except Exception:
                self.logger = None
        else:
            self.logger = None

        self.grid = MultiGrid(width, height, torus=False)

        self.population_counter: Dict[str,int] = defaultdict(int)
        self.antibiotic_grids:   Dict[str, np.ndarray] = {}
        self.antibiotic_profiles: Dict[str, AntibioticProfile] = {}
        self.nutrient_grid = np.full((width, height), nutrient_level, dtype=np.float32)

        self.hgt_events: List[HGTEvent] = []
        self.hgt_events_this_step: List[HGTEvent] = []
        self.event_log: List[dict] = []

        # Resistance emergence tracking (to log first-time events)
        self._gene_emergence_logged: set = set()
        self._prev_pop = 0

        self.active_germ_keys: List[str] = []
        self.active_antibiotic_keys: List[str] = []

        self._setup_scenario(scenario, initial_bacteria)

        self.datacollector = DataCollector(model_reporters={
            "Total_Bacteria":    lambda m: m.count_living_bacteria(),
            "Resistant_Bacteria":lambda m: m.count_resistant_bacteria(),
            "Biofilm_Bacteria":  lambda m: m.count_biofilm_bacteria(),
            "Persister_Cells":   lambda m: m.count_persisters(),
            "HGT_Events_Total":  lambda m: len(m.hgt_events),
            "Avg_Resistance_Genes": lambda m: m.avg_resistance_genes(),
            "Avg_Fitness":       lambda m: m.avg_fitness(),
        })

        self._log_event("simulation_start",
            f"Scenario '{scenario}' | {initial_bacteria} bacteria | {width}x{height} grid")

        if self.logger:
            self.logger.log_simulation_start(
                scenario=scenario,
                germ_names=self.active_germ_keys,
                antibiotic_names=self.active_antibiotic_keys,
                initial_pop=initial_bacteria,
                grid_size=(width, height),
                seed=seed,
            )

    # ── Scenario setup ────────────────────────────────────────────────────────
    def _setup_scenario(self, scenario, initial_bacteria):
        configs = {
            "validation":            {"germs":["e_coli"],                                           "antibiotics":[]},
            "ecoli_cipro":           {"germs":["e_coli"],                                           "antibiotics":["ciprofloxacin"]},
            "klebsiella_carbapenem": {"germs":["klebsiella_pneumoniae"],                            "antibiotics":["meropenem"]},
            "xdr_acinetobacter":     {"germs":["acinetobacter_baumannii"],                          "antibiotics":["colistin"]},
            "mrsa_hospital":         {"germs":["mrsa"],                                             "antibiotics":["vancomycin"]},
            "pakistan_crisis":       {"germs":["e_coli","klebsiella_pneumoniae"],                   "antibiotics":["ciprofloxacin","meropenem"]},
            "multi_species":         {"germs":["e_coli","klebsiella_pneumoniae","pseudomonas_aeruginosa"], "antibiotics":["ciprofloxacin"]},
        }
        cfg = configs.get(scenario, configs["validation"])
        self.active_germ_keys = cfg["germs"]
        self.active_antibiotic_keys = cfg["antibiotics"]
        self.sim_scenario_description = cfg.get("description", scenario)

        per_species = initial_bacteria // max(1, len(self.active_germ_keys))
        for gk in self.active_germ_keys:
            self._spawn_bacteria_cluster(get_germ(gk), per_species)

        for abk in self.active_antibiotic_keys:
            ab = get_antibiotic(abk)
            self.antibiotic_profiles[abk] = ab
            self.antibiotic_grids[abk] = np.zeros((self.width, self.height), dtype=np.float32)

    def _spawn_bacteria_cluster(self, profile: GermProfile, count: int):
        cx = random.randint(self.width//4, 3*self.width//4)
        cy = random.randint(self.height//4, 3*self.height//4)
        radius = max(5, int(math.sqrt(count) * 1.5))
        spawned = 0
        for _ in range(count * 10):
            if spawned >= count: break
            angle = random.uniform(0, 2*math.pi)
            r = random.gauss(0, radius/3)
            x = max(0, min(self.width-1,  int(cx + r*math.cos(angle))))
            y = max(0, min(self.height-1, int(cy + r*math.sin(angle))))
            agent = BacteriumAgent(model=self, profile=profile, position=(x,y))
            self.grid.place_agent(agent, (x,y))
            self.population_counter[profile.species] += 1
            spawned += 1

    # ── Antibiotic management ─────────────────────────────────────────────────
    def apply_antibiotic(self, antibiotic_key, concentration=1.0, mode="uniform",
                         center=None, radius=None):
        if antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise ValueError(f"Unknown antibiotic: {antibiotic_key}")
        ab = get_antibiotic(antibiotic_key)
        if antibiotic_key not in self.antibiotic_grids:
            self.antibiotic_profiles[antibiotic_key] = ab
            self.antibiotic_grids[antibiotic_key] = np.zeros((self.width,self.height), dtype=np.float32)
            if antibiotic_key not in self.active_antibiotic_keys:
                self.active_antibiotic_keys.append(antibiotic_key)
        g = self.antibiotic_grids[antibiotic_key]
        if mode == "uniform":
            g += concentration
        elif mode == "gradient":
            for x in range(self.width):
                g[x,:] += concentration * (1.0 - x/self.width)
        elif mode == "spot" and center:
            cx, cy = center
            r = radius or 10
            for x in range(max(0,cx-r), min(self.width,cx+r)):
                for y in range(max(0,cy-r), min(self.height,cy+r)):
                    d = math.sqrt((x-cx)**2+(y-cy)**2)
                    if d <= r:
                        g[x,y] += concentration * (1-d/r)
        elif mode == "zone":
            g[:self.width//2,:] += concentration
        self.antibiotic_grids[antibiotic_key] = np.clip(g, 0.0, 5.0)

        msg = f"{ab.name} applied ({mode}, conc={concentration:.2f}) | MBC={ab.mbc} µg/mL"
        self._log_event("antibiotic_applied", msg)
        if self.logger:
            self.logger.log_antibiotic_applied(
                self.current_step, antibiotic_key, concentration, mode,
                self.count_living_bacteria())

    def remove_antibiotic(self, key):
        if key in self.antibiotic_grids:
            self.antibiotic_grids[key][:] = 0.0
            self._log_event("antibiotic_removed", f"{key} cleared from environment")
            if self.logger:
                self.logger.log_antibiotic_removed(
                    self.current_step, key, self.count_living_bacteria())

    def get_antibiotics_at(self, pos) -> Dict[str,float]:
        x,y = pos
        return {n: float(g[x,y]) for n,g in self.antibiotic_grids.items() if g[x,y]>0.001}

    def get_total_antibiotic_at(self, pos) -> float:
        x,y = pos
        return sum(float(g[x,y]) for g in self.antibiotic_grids.values())

    def add_antibiotic_at(self, pos, ab: AntibioticProfile, amount: float):
        key = ab.name.lower().replace(" ","_")
        if key in self.antibiotic_grids:
            x,y = pos
            self.antibiotic_grids[key][x,y] = min(5.0, self.antibiotic_grids[key][x,y]+amount)

    # ── Nutrients ─────────────────────────────────────────────────────────────
    def get_nutrient_at(self, pos) -> float:
        x,y = pos; return float(self.nutrient_grid[x,y])

    def consume_nutrient(self, pos, amount):
        x,y = pos; self.nutrient_grid[x,y] = max(0.0, self.nutrient_grid[x,y]-amount)

    def _regenerate_nutrients(self):
        self.nutrient_grid = np.minimum(1.0, self.nutrient_grid + NUTRIENT_REGEN)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    def _diffuse_antibiotics(self):
        kernel = np.array([[0.05,0.10,0.05],[0.10,0.40,0.10],[0.05,0.10,0.05]], dtype=np.float32)
        for key, grid in self.antibiotic_grids.items():
            if not np.any(grid > 0.001): continue
            ab = self.antibiotic_profiles[key]
            new_grid = np.zeros_like(grid)
            for x in range(self.width):
                for y in range(self.height):
                    if grid[x,y] < 0.001: continue
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            nx,ny = x+dx, y+dy
                            if 0<=nx<self.width and 0<=ny<self.height:
                                new_grid[nx,ny] += grid[x,y]*kernel[dx+1,dy+1]*ab.diffusion_rate
            new_grid *= (1.0 - ab.decay_rate)
            self.antibiotic_grids[key] = np.clip(new_grid, 0.0, 5.0)

    # ── HGT recording ─────────────────────────────────────────────────────────
    def record_hgt_event(self, donor_id, recipient_id, gene, step, position):
        ev = HGTEvent(step, donor_id, recipient_id, gene, position)
        self.hgt_events.append(ev)
        self.hgt_events_this_step.append(ev)
        if len(self.hgt_events) > 1000:
            self.hgt_events = self.hgt_events[-500:]

    # ── Statistics ────────────────────────────────────────────────────────────
    def _living(self):
        return [a for a in self.agents if isinstance(a, BacteriumAgent)
                and a.state != BacterialState.DEAD]

    def count_living_bacteria(self) -> int:
        return len(self._living())

    def count_dead_bacteria(self) -> int:
        return sum(1 for a in self.agents
                   if isinstance(a,BacteriumAgent) and a.state==BacterialState.DEAD)

    def count_resistant_bacteria(self) -> int:
        return sum(1 for a in self._living()
                   if len(a.resistance_genes) > len(a.profile.natural_resistances))

    def count_biofilm_bacteria(self) -> int:
        return sum(1 for a in self._living() if a.in_biofilm)

    def count_persisters(self) -> int:
        return sum(1 for a in self._living() if a.is_persister)

    def avg_resistance_genes(self) -> float:
        living = self._living()
        return sum(len(a.resistance_genes) for a in living) / max(1, len(living))

    def avg_fitness(self) -> float:
        living = self._living()
        return sum(a.fitness for a in living) / max(1, len(living))

    def get_species_counts(self) -> Dict[str,int]:
        counts = defaultdict(int)
        for a in self._living():
            counts[a.profile.species] += 1
        return dict(counts)

    def get_resistance_gene_distribution(self) -> Dict[str,int]:
        dist = defaultdict(int)
        for a in self._living():
            for g in a.resistance_genes:
                dist[g] += 1
        return dict(dist)

    # ── Structured logging checks ─────────────────────────────────────────────
    def _check_science_events(self):
        """Check for biologically significant events and log them."""
        if not self.logger:
            return
        living = self._living()
        total = len(living)
        if total == 0:
            return

        # Log step summary
        self.logger.log_step_summary(
            step=self.current_step,
            total=total,
            resistant=self.count_resistant_bacteria(),
            biofilm=self.count_biofilm_bacteria(),
            persisters=self.count_persisters(),
            hgt_total=len(self.hgt_events),
            avg_fitness=self.avg_fitness(),
            avg_genes=self.avg_resistance_genes(),
            species_counts=self.get_species_counts(),
            gene_dist=self.get_resistance_gene_distribution(),
        )

        # Log first emergence of each resistance gene
        gene_dist = self.get_resistance_gene_distribution()
        for gene, count in gene_dist.items():
            freq = count / total
            tag = f"{gene}_{self.sim_scenario}"
            if tag not in self._gene_emergence_logged:
                if freq >= 0.01:   # 1% threshold = "emerged"
                    self._gene_emergence_logged.add(tag)
                    species = max(self.get_species_counts(), key=self.get_species_counts().get, default="unknown")
                    self.logger.log_resistance_emerged(
                        self.current_step, gene, species, count, total)

        # Population milestones
        milestones = [100, 250, 500, 1000, 1500, 2000]
        for m in milestones:
            tag = f"pop_{m}"
            if tag not in self._gene_emergence_logged:
                if self._prev_pop < m <= total:
                    self._gene_emergence_logged.add(tag)
                    self.logger.log_population_milestone(
                        self.current_step, f"population reached {m}", total)
                elif self._prev_pop > m >= total and total > 0:
                    self.logger.log_population_milestone(
                        self.current_step, f"population dropped below {m} under treatment", total)

        self._prev_pop = total

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _remove_dead_agents(self):
        dead = [a for a in self.agents
                if isinstance(a,BacteriumAgent) and a.state==BacterialState.DEAD]
        for a in dead:
            self.grid.remove_agent(a)
            a.remove()

    def _cull_weakest(self, n):
        living = sorted(self._living(), key=lambda a: a.fitness)
        for a in living[:n]:
            a.state = BacterialState.DEAD
            self.grid.remove_agent(a)
            a.remove()

    # ── Internal event log (for API/frontend) ─────────────────────────────────
    def _log_event(self, event_type, message, data=None):
        self.event_log.append({"step":self.current_step,"type":event_type,"message":message,"data":data or {}})
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-100:]

    # ── State snapshot ────────────────────────────────────────────────────────
    def get_full_state(self) -> dict:
        bacteria_data = [a.to_dict() for a in self._living()]

        ab_heatmaps = {}
        for key, grid in self.antibiotic_grids.items():
            ds = grid[::2, ::2]
            ab_heatmaps[key] = {
                "data": ds.tolist(), "max": float(ds.max()),
                "color": self.antibiotic_profiles[key].color_hex,
                "name":  self.antibiotic_profiles[key].name,
            }

        nutrient_ds = self.nutrient_grid[::2, ::2]
        recent_hgt = [{"step":e.step,"donor":e.donor_id,"recipient":e.recipient_id,
                       "gene":e.gene,"pos":list(e.position)} for e in self.hgt_events[-50:]]

        stats = {
            "step": self.current_step,
            "total_bacteria": len(bacteria_data),
            "resistant_bacteria": self.count_resistant_bacteria(),
            "biofilm_bacteria":   self.count_biofilm_bacteria(),
            "persister_cells":    self.count_persisters(),
            "hgt_total":          len(self.hgt_events),
            "avg_resistance_genes": round(self.avg_resistance_genes(), 3),
            "avg_fitness":          round(self.avg_fitness(), 3),
            "species_counts":       self.get_species_counts(),
            "gene_distribution":    self.get_resistance_gene_distribution(),
            "antibiotic_active":    list(self.antibiotic_grids.keys()),
            "scenario":             self.sim_scenario,
            "scenario_description": getattr(self,"sim_scenario_description",""),
        }
        return {
            "bacteria": bacteria_data, "antibiotic_heatmaps": ab_heatmaps,
            "nutrient_heatmap": nutrient_ds.tolist(),
            "hgt_events": recent_hgt, "stats": stats,
            "event_log": self.event_log[-20:],
            "grid_width": self.width, "grid_height": self.height,
        }

    # ── Main step ─────────────────────────────────────────────────────────────
    def step(self):
        if self.paused: return
        self.hgt_events_this_step = []
        self.current_step += 1

        if self.count_living_bacteria() > MAX_POPULATION:
            self._cull_weakest(MAX_POPULATION // 4)

        agent_list = list(self.agents)
        random.shuffle(agent_list)
        for a in agent_list:
            if isinstance(a, BacteriumAgent):
                a.step()

        self._diffuse_antibiotics()
        self._regenerate_nutrients()

        if self.current_step % 5 == 0:
            self._remove_dead_agents()
        if self.current_step % 3 == 0:
            self.datacollector.collect(self)

        # HGT burst log
        if self.hgt_events_this_step:
            genes = set(e.gene for e in self.hgt_events_this_step)
            positions = [list(e.position) for e in self.hgt_events_this_step]
            self._log_event("hgt_burst",
                f"{len(self.hgt_events_this_step)} HGT events — genes: {genes}")
            if self.logger:
                self.logger.log_hgt_burst(
                    self.current_step, len(self.hgt_events_this_step),
                    genes, positions)

        # Science event checks every 5 steps
        if self.current_step % 5 == 0:
            self._check_science_events()

        # Extinction
        if self.count_living_bacteria() == 0:
            self._log_event("extinction", "All bacteria eliminated.")
            if self.logger:
                self.logger.log_extinction(self.current_step, "antibiotic treatment")
            self.running = False