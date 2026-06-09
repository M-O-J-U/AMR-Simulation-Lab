"""
AMR Simulation Logger — writes structured logs during simulation and test runs.

Produces TWO files per session:
  1. logs/amr_sim_TIMESTAMP.log   — human-readable, color-coded terminal mirror
  2. logs/amr_sim_TIMESTAMP.json  — structured JSON for analysis / paper figures

Log levels:
  SCIENCE   — biologically meaningful events (HGT, SOS, resistance emergence)
  MILESTONE — population thresholds, extinction, treatment applied
  STEP      — per-step summary (only every N steps to avoid noise)
  WARNING   — unexpected simulation states
  ERROR     — exceptions / bad data

Philosophy: a researcher should be able to read the .log file and understand
exactly what the simulation did, in plain English, without looking at code.
"""

import json
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# LOG LEVELS
# ─────────────────────────────────────────────────────────────────────────────

class LogLevel(Enum):
    SCIENCE   = "SCIENCE"    # biological events — most important
    MILESTONE = "MILESTONE"  # simulation state changes
    STEP      = "STEP"       # periodic step summaries
    TEST      = "TEST"       # test pass/fail results
    WARNING   = "WARNING"
    ERROR     = "ERROR"


# ANSI colors for terminal output
_COLORS = {
    LogLevel.SCIENCE:   "\033[95m",   # magenta
    LogLevel.MILESTONE: "\033[96m",   # cyan
    LogLevel.STEP:      "\033[37m",   # light gray
    LogLevel.TEST:      "\033[92m",   # green
    LogLevel.WARNING:   "\033[93m",   # yellow
    LogLevel.ERROR:     "\033[91m",   # red
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SimLogger:
    """
    Singleton-style logger for the AMR simulation.
    Call SimLogger.get() to get the active instance.
    Call SimLogger.start_session(...) at the beginning of each run.
    """

    _instance: Optional["SimLogger"] = None

    def __init__(self, log_dir: str = "logs", session_name: str = "amr_sim",
                 step_log_interval: int = 5, print_to_terminal: bool = True):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file  = self.log_dir / f"{session_name}_{timestamp}.log"
        self.json_file = self.log_dir / f"{session_name}_{timestamp}.json"

        self.step_log_interval = step_log_interval
        self.print_to_terminal = print_to_terminal

        self.session_name = session_name
        self.session_start = time.time()
        self.entries: List[Dict[str, Any]] = []

        # Counters
        self.hgt_count      = 0
        self.mutation_count = 0
        self.sos_count      = 0
        self.extinction_step: Optional[int] = None

        # Open log file
        self._log_fh = open(self.log_file, "w", encoding="utf-8")

        # Write header
        self._write_header(session_name, timestamp)

    @classmethod
    def get(cls) -> "SimLogger":
        if cls._instance is None:
            cls._instance = SimLogger()
        return cls._instance

    @classmethod
    def start_session(cls, session_name: str = "amr_sim",
                      log_dir: str = "logs",
                      step_interval: int = 5,
                      print_to_terminal: bool = True) -> "SimLogger":
        """Start a new logging session. Call this at the start of each run."""
        if cls._instance and cls._instance._log_fh:
            cls._instance.close()
        cls._instance = SimLogger(
            log_dir=log_dir,
            session_name=session_name,
            step_log_interval=step_interval,
            print_to_terminal=print_to_terminal,
        )
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────────
    # CORE LOG METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def log(self, level: LogLevel, message: str,
            step: Optional[int] = None, data: Optional[Dict] = None):
        """Write a single log entry."""
        elapsed = time.time() - self.session_start
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(elapsed, 2),
            "level":     level.value,
            "step":      step,
            "message":   message,
            "data":      data or {},
        }
        self.entries.append(entry)

        # Human-readable line
        step_str  = f"[step {step:04d}]" if step is not None else "         "
        time_str  = f"[{elapsed:6.1f}s]"
        color     = _COLORS.get(level, "")
        level_str = f"{color}{_BOLD}{level.value:<9}{_RESET}"
        line      = f"{time_str} {step_str} {level_str}  {message}"

        self._log_fh.write(line.replace(_RESET, "").replace(_BOLD, "")
                           .replace(color, "") + "\n")
        self._log_fh.flush()

        if self.print_to_terminal:
            print(line)

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS — called from model/agents
    # ─────────────────────────────────────────────────────────────────────────

    def log_simulation_start(self, scenario: str, germ_names: List[str],
                             antibiotic_names: List[str], initial_pop: int,
                             grid_size: tuple, seed: Optional[int] = None):
        self.log(LogLevel.MILESTONE,
            f"Simulation started — scenario: '{scenario}' | "
            f"germs: {germ_names} | antibiotics: {antibiotic_names} | "
            f"population: {initial_pop} | grid: {grid_size[0]}×{grid_size[1]}",
            step=0,
            data={"scenario": scenario, "germs": germ_names,
                  "antibiotics": antibiotic_names, "initial_pop": initial_pop,
                  "grid": list(grid_size), "seed": seed})

    def log_step_summary(self, step: int, total: int, resistant: int,
                         biofilm: int, persisters: int, hgt_total: int,
                         avg_fitness: float, avg_genes: float,
                         species_counts: Dict[str, int],
                         gene_dist: Dict[str, int]):
        if step % self.step_log_interval != 0:
            return
        pct_r = (resistant / max(1, total)) * 100
        top_gene = max(gene_dist, key=gene_dist.get) if gene_dist else "none"
        self.log(LogLevel.STEP,
            f"Pop: {total:>4} | Resistant: {resistant:>4} ({pct_r:4.1f}%) | "
            f"Biofilm: {biofilm:>3} | Persisters: {persisters:>3} | "
            f"HGT events: {hgt_total:>4} | Avg fitness: {avg_fitness:.3f} | "
            f"Avg resistance genes: {avg_genes:.2f} | Dominant gene: {top_gene}",
            step=step,
            data={"total": total, "resistant": resistant, "biofilm": biofilm,
                  "persisters": persisters, "hgt_total": hgt_total,
                  "avg_fitness": avg_fitness, "avg_genes": avg_genes,
                  "species_counts": species_counts, "gene_dist": gene_dist})

    def log_antibiotic_applied(self, step: int, antibiotic: str,
                               concentration: float, mode: str,
                               population: int):
        self.log(LogLevel.MILESTONE,
            f"ANTIBIOTIC APPLIED — {antibiotic} | concentration: {concentration:.2f} µg/mL | "
            f"mode: {mode} | population at time of treatment: {population}",
            step=step,
            data={"antibiotic": antibiotic, "concentration": concentration,
                  "mode": mode, "population": population})

    def log_antibiotic_removed(self, step: int, antibiotic: str, population: int):
        self.log(LogLevel.MILESTONE,
            f"ANTIBIOTIC CLEARED — {antibiotic} removed from environment | "
            f"surviving population: {population}",
            step=step,
            data={"antibiotic": antibiotic, "population": population})

    def log_hgt_burst(self, step: int, count: int, genes: set, positions: list):
        self.hgt_count += count
        self.log(LogLevel.SCIENCE,
            f"HGT BURST — {count} horizontal gene transfer event(s) | "
            f"genes transferred: {sorted(genes)} | "
            f"total HGT events this session: {self.hgt_count}",
            step=step,
            data={"count": count, "genes": sorted(genes),
                  "positions": positions[:5], "session_total": self.hgt_count})

    def log_resistance_emerged(self, step: int, gene: str, species: str,
                               carriers: int, total_pop: int):
        pct = (carriers / max(1, total_pop)) * 100
        self.log(LogLevel.SCIENCE,
            f"RESISTANCE EMERGED — gene '{gene}' now in {carriers}/{total_pop} "
            f"({pct:.1f}%) of {species} | "
            f"mechanism: {self._gene_mechanism(gene)}",
            step=step,
            data={"gene": gene, "species": species, "carriers": carriers,
                  "total_pop": total_pop, "frequency": pct})

    def log_sos_activation(self, step: int, agent_id: int, species: str,
                           stress_level: float):
        self.sos_count += 1
        if self.sos_count <= 10 or self.sos_count % 50 == 0:
            self.log(LogLevel.SCIENCE,
                f"SOS RESPONSE ACTIVATED — cell #{agent_id} ({species}) | "
                f"stress level: {stress_level:.2f} | "
                f"effect: mutation rate ×8, efflux pump upregulation | "
                f"session SOS count: {self.sos_count}",
                step=step,
                data={"agent_id": agent_id, "species": species,
                      "stress_level": stress_level, "session_count": self.sos_count})

    def log_biofilm_formed(self, step: int, pos: tuple, species: str,
                           cell_count: int):
        self.log(LogLevel.SCIENCE,
            f"BIOFILM COLONY — {species} forming biofilm at {pos} | "
            f"local density: {cell_count} cells | "
            f"effect: antibiotic penetration reduced by 60% (Hoiby 2010)",
            step=step,
            data={"pos": list(pos), "species": species, "density": cell_count})

    def log_persister_formed(self, step: int, agent_id: int, species: str):
        self.log(LogLevel.SCIENCE,
            f"PERSISTER CELL — #{agent_id} ({species}) entered dormancy | "
            f"effect: 95% antibiotic tolerance, metabolically silent | "
            f"will resume growth when antibiotic is removed",
            step=step,
            data={"agent_id": agent_id, "species": species})

    def log_population_milestone(self, step: int, milestone: str,
                                 population: int, data: dict = None):
        self.log(LogLevel.MILESTONE,
            f"POPULATION MILESTONE — {milestone} | current population: {population}",
            step=step, data={"milestone": milestone, "population": population, **(data or {})})

    def log_extinction(self, step: int, cause: str):
        self.extinction_step = step
        self.log(LogLevel.MILESTONE,
            f"EXTINCTION — all bacteria eliminated at step {step} | cause: {cause}",
            step=step, data={"step": step, "cause": cause})

    def log_treatment_recommendation(self, step: int,
                                     recommendations: List[Dict]):
        if not recommendations:
            return
        best = recommendations[0]
        self.log(LogLevel.SCIENCE,
            f"TREATMENT RECOMMENDATION — Best option: {best['antibiotic']} "
            f"(score {best['score']:.1f}, {best['pct_susceptible']:.0f}% susceptible) | "
            f"Rationale: {best.get('rationale', 'N/A')}",
            step=step, data={"recommendations": recommendations[:3]})

    def log_mutation(self, step: int, agent_id: int, gene: str,
                     mutation_type: str, species: str):
        self.mutation_count += 1
        if self.mutation_count <= 5 or self.mutation_count % 100 == 0:
            self.log(LogLevel.SCIENCE,
                f"MUTATION — {mutation_type} in {species} cell #{agent_id} | "
                f"gene: {gene} | mechanism: {self._gene_mechanism(gene)} | "
                f"session mutation count: {self.mutation_count}",
                step=step,
                data={"agent_id": agent_id, "gene": gene,
                      "type": mutation_type, "species": species,
                      "session_count": self.mutation_count})

    # ─────────────────────────────────────────────────────────────────────────
    # TEST LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def log_test_result(self, test_name: str, passed: bool,
                        details: str = "", duration_ms: float = 0):
        status = "PASS" if passed else "FAIL"
        color  = "\033[92m" if passed else "\033[91m"
        self.log(LogLevel.TEST,
            f"{color}{status}{_RESET}  {test_name:<65} "
            f"({duration_ms:.0f}ms){' — ' + details if details else ''}",
            data={"test": test_name, "passed": passed,
                  "details": details, "duration_ms": duration_ms})

    def log_test_suite_summary(self, passed: int, failed: int,
                               total: int, duration_s: float):
        pct = (passed / max(1, total)) * 100
        status = "ALL PASS" if failed == 0 else f"{failed} FAILED"
        self.log(LogLevel.MILESTONE,
            f"TEST SUITE COMPLETE — {passed}/{total} passed ({pct:.0f}%) | "
            f"{status} | duration: {duration_s:.2f}s",
            data={"passed": passed, "failed": failed, "total": total,
                  "duration_s": duration_s})

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION CLOSE
    # ─────────────────────────────────────────────────────────────────────────

    def log_session_summary(self, final_population: int,
                            total_steps: int, final_gene_dist: Dict[str, int]):
        elapsed = time.time() - self.session_start
        self.log(LogLevel.MILESTONE,
            f"SESSION COMPLETE — {total_steps} steps | "
            f"final population: {final_population} | "
            f"total HGT events: {self.hgt_count} | "
            f"total mutations logged: {self.mutation_count} | "
            f"SOS activations: {self.sos_count} | "
            f"elapsed: {elapsed:.1f}s",
            data={"total_steps": total_steps, "final_population": final_population,
                  "hgt_total": self.hgt_count, "mutation_total": self.mutation_count,
                  "sos_total": self.sos_count, "elapsed_s": elapsed,
                  "final_gene_distribution": final_gene_dist,
                  "extinction_step": self.extinction_step})

    def close(self):
        """Flush and save the JSON log."""
        summary = {
            "session": self.session_name,
            "log_file": str(self.log_file),
            "entries_count": len(self.entries),
            "hgt_events": self.hgt_count,
            "mutations_logged": self.mutation_count,
            "sos_activations": self.sos_count,
            "entries": self.entries,
        }
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self._write_footer()
        self._log_fh.close()

        if self.print_to_terminal:
            print(f"\n  Log saved: {self.log_file}")
            print(f"  JSON data: {self.json_file}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _write_header(self, session_name: str, timestamp: str):
        header = (
            f"\n{'='*80}\n"
            f"  AMR SIMULATION LOG\n"
            f"  Session  : {session_name}\n"
            f"  Started  : {timestamp}\n"
            f"  Log file : {self.log_file}\n"
            f"  JSON data: {self.json_file}\n"
            f"{'='*80}\n\n"
            f"  Log levels:\n"
            f"    SCIENCE   — biologically significant events\n"
            f"    MILESTONE — simulation state changes\n"
            f"    STEP      — periodic population summary\n"
            f"    TEST      — test pass/fail results\n"
            f"    WARNING   — unexpected states\n"
            f"\n{'='*80}\n\n"
        )
        self._log_fh.write(header)
        self._log_fh.flush()

    def _write_footer(self):
        footer = (
            f"\n{'='*80}\n"
            f"  SESSION ENDED\n"
            f"  Total entries : {len(self.entries)}\n"
            f"  HGT events    : {self.hgt_count}\n"
            f"  Mutations     : {self.mutation_count}\n"
            f"  SOS responses : {self.sos_count}\n"
            f"  Elapsed       : {time.time() - self.session_start:.1f}s\n"
            f"{'='*80}\n"
        )
        self._log_fh.write(footer)

    def _gene_mechanism(self, gene: str) -> str:
        mechanisms = {
            "blaTEM-1":    "beta-lactamase (penicillin hydrolysis)",
            "blaCTX-M-15": "ESBL (extended-spectrum cephalosporin hydrolysis)",
            "blaKPC-2":    "carbapenemase (carbapenem hydrolysis)",
            "blaNDM-1":    "metallo-beta-lactamase (broad spectrum hydrolysis)",
            "mexAB-oprM":  "efflux pump (antibiotic export)",
            "acrAB-tolC":  "efflux pump (multi-drug export)",
            "gyrA_S83L":   "DNA gyrase mutation (fluoroquinolone target alteration)",
            "mcr-1":       "phosphoethanolamine transferase (colistin target alteration)",
            "tetM":        "ribosomal protection (tetracycline displacement)",
            "vanA":        "peptidoglycan remodeling (vancomycin target alteration)",
        }
        return mechanisms.get(gene, "unknown mechanism")

    # ─────────────────────────────────────────────────────────────────────────
    # QUICK ACCESS — for use from model without importing instance
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_log_files(log_dir: str = "logs") -> List[Path]:
        """Return all log files sorted by newest first."""
        log_path = Path(log_dir)
        if not log_path.exists():
            return []
        return sorted(log_path.glob("*.log"), reverse=True)

    @staticmethod
    def get_latest_json(log_dir: str = "logs") -> Optional[Dict]:
        """Load the most recent JSON log."""
        log_path = Path(log_dir)
        if not log_path.exists():
            return None
        jsons = sorted(log_path.glob("*.json"), reverse=True)
        if not jsons:
            return None
        with open(jsons[0]) as f:
            return json.load(f)