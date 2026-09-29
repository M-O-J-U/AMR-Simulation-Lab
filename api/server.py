"""
AMR Simulation — FastAPI REST server.

Rebuilt 2026-09-29: the original server source was lost (this file had been
overwritten with a copy of ai/gnn_inference.py). The routes below are
reconstructed from the calls frontend/index.html makes, and are thin wrappers
around existing model / analytics / GNN methods. This module contains no
simulation logic and never touches the model RNG directly — every random draw
happens inside AMRSimulationModel methods, exactly as in headless runs.

Transport: REST only. The frontend polls GET /state (every 2.5 s, and after
each action). There is no WebSocket route.

Pause semantics: Play is driven by the frontend's own timer posting /step.
/pause and /resume set model.paused, but an explicit POST /step always
advances (see step()), so while paused the '+1 Step' button still works.

Endpoints
  GET  /state                       full simulation snapshot
  POST /step                        {"n_steps": int}
  POST /apply_antibiotic            {"antibiotic_key", "concentration", "mode", "center"?, "radius"?}
  POST /remove_antibiotic           {"antibiotic_key"}
  POST /spawn_bacteria              {"germ_key", "count"}
  POST /reset                       {"scenario", "initial_bacteria", "seed"?}
  POST /pause, POST /resume
  GET  /gnn/status
  POST /gnn/predict                 {"threshold", "max_nodes", "max_edge_distance"}
  POST /gnn/advisory                {"available_antibiotics": [...]}
  GET  /analytics/mic?antibiotic_key=...
  GET  /analytics/diversity
  GET  /analytics/recommend
"""

import math
import os
import sys
import threading
from typing import List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.amr_model import AMRSimulationModel
from data.card_loader import ANTIBIOTIC_PROFILES, GERM_PROFILES, get_germ
from ai.resistance_analytics import (
    estimate_population_mic, recommend_treatment, shannon_diversity,
)

# Scenario keys handled by AMRSimulationModel._setup_scenario. The model
# silently falls back to "validation" for unknown keys; the API rejects them
# instead so a typo can't masquerade as a different experiment.
SCENARIOS = (
    "validation", "ecoli_cipro", "klebsiella_carbapenem", "xdr_acinetobacter",
    "mrsa_hospital", "pakistan_crisis", "multi_species",
)

DEFAULT_SCENARIO = "validation"
DEFAULT_INITIAL_BACTERIA = 80    # matches the frontend's population slider default


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST BODIES (field names match frontend/index.html exactly)
# ─────────────────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    n_steps: int = Field(1, ge=1, le=100)

class ApplyAntibioticRequest(BaseModel):
    antibiotic_key: str
    concentration: float = Field(1.0, ge=0.0, le=5.0)
    mode: Literal["uniform", "gradient", "spot", "zone"] = "uniform"
    center: Optional[Tuple[int, int]] = None   # only used by mode="spot"
    radius: Optional[int] = Field(None, ge=1)

class RemoveAntibioticRequest(BaseModel):
    antibiotic_key: str

class SpawnRequest(BaseModel):
    germ_key: str
    count: int = Field(20, ge=1, le=500)

class ResetRequest(BaseModel):
    scenario: str = DEFAULT_SCENARIO
    initial_bacteria: int = Field(DEFAULT_INITIAL_BACTERIA, ge=1, le=1000)
    seed: Optional[int] = None

class GNNPredictRequest(BaseModel):
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    max_nodes: int = Field(300, ge=2, le=3000)
    max_edge_distance: int = Field(3, ge=1, le=20)

class GNNAdvisoryRequest(BaseModel):
    available_antibiotics: List[str] = Field(default_factory=lambda: list(ANTIBIOTIC_PROFILES))


def json_safe(obj):
    """Replace NaN/inf with None so strict JSON encoding doesn't 500.
    (e.g. best_model.pt's stored val_metrics include auroc_mexAB-oprM = NaN.)"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# SESSION — one live model, guarded by a lock
# ─────────────────────────────────────────────────────────────────────────────

class SimulationSession:
    """Holds the single live model. FastAPI runs sync endpoints in a thread
    pool, so every model access goes through `lock` to keep a /step from
    interleaving with a /state read or another /step."""

    def __init__(self, enable_logging: bool, checkpoint_path: Optional[str]):
        self.enable_logging = enable_logging
        self.checkpoint_path = checkpoint_path
        self.lock = threading.RLock()
        self._model: Optional[AMRSimulationModel] = None
        self._gnn = None
        self._gnn_trained = False
        self._gnn_lock = threading.Lock()

    @property
    def model(self) -> AMRSimulationModel:
        if self._model is None:
            self.reset(DEFAULT_SCENARIO, DEFAULT_INITIAL_BACTERIA, None)
        return self._model

    def reset(self, scenario: str, initial_bacteria: int, seed: Optional[int]):
        self._model = AMRSimulationModel(
            scenario=scenario, initial_bacteria=initial_bacteria,
            seed=seed, enable_logging=self.enable_logging,
        )

    def living_bacteria(self) -> list:
        return self.model._living()

    def gnn(self):
        """Lazily load the GNN (torch import + checkpoint load take seconds).
        Falls back to an untrained model if no checkpoint exists, which the
        frontend reports as 'Untrained - run train-gnn first'."""
        with self._gnn_lock:
            if self._gnn is None:
                from ai.gnn_inference import GNNInferenceEngine, DEFAULT_CHECKPOINT
                path = self.checkpoint_path or DEFAULT_CHECKPOINT
                engine = GNNInferenceEngine.load(path)
                self._gnn_trained = engine is not None
                self._gnn = engine or GNNInferenceEngine.load_untrained()
            return self._gnn


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app(enable_logging: bool = True,
               checkpoint_path: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="AMR Simulation Lab API", version="1.0")
    # The frontend is opened straight from disk (file://, origin "null") and
    # calls http://localhost:8000, so cross-origin requests must be allowed.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    session = SimulationSession(enable_logging, checkpoint_path)
    app.state.session = session

    # ── Simulation ───────────────────────────────────────────────────────────
    @app.get("/state")
    def get_state():
        with session.lock:
            return session.model.get_full_state()

    @app.post("/step")
    def step(req: StepRequest = StepRequest()):
        """An explicit step request always advances n_steps, even while paused
        (e.g. the '+1 Step' button after Pause). model.step() returns early when
        model.paused is set, so the flag is cleared for the duration of this
        request and restored afterwards; model.step() itself is unchanged."""
        with session.lock:
            m = session.model
            was_paused = m.paused
            m.paused = False
            try:
                for _ in range(req.n_steps):
                    m.step()
            finally:
                m.paused = was_paused
            return {"step": m.current_step, "paused": m.paused,
                    "running": m.running,
                    "total_bacteria": m.count_living_bacteria()}

    @app.post("/apply_antibiotic")
    def apply_antibiotic(req: ApplyAntibioticRequest):
        if req.antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise HTTPException(404, f"Unknown antibiotic: {req.antibiotic_key}")
        with session.lock:
            session.model.apply_antibiotic(
                req.antibiotic_key, concentration=req.concentration,
                mode=req.mode, center=req.center, radius=req.radius)
            return {"ok": True, "antibiotic_key": req.antibiotic_key,
                    "active": list(session.model.antibiotic_grids)}

    @app.post("/remove_antibiotic")
    def remove_antibiotic(req: RemoveAntibioticRequest):
        if req.antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise HTTPException(404, f"Unknown antibiotic: {req.antibiotic_key}")
        with session.lock:
            present = req.antibiotic_key in session.model.antibiotic_grids
            session.model.remove_antibiotic(req.antibiotic_key)
            return {"ok": True, "antibiotic_key": req.antibiotic_key,
                    "was_present": present}

    @app.post("/spawn_bacteria")
    def spawn_bacteria(req: SpawnRequest):
        if req.germ_key not in GERM_PROFILES:
            raise HTTPException(404, f"Unknown germ: {req.germ_key}")
        with session.lock:
            m = session.model
            before = m.count_living_bacteria()
            profile = get_germ(req.germ_key)
            m._spawn_bacteria_cluster(profile, req.count)
            spawned = m.count_living_bacteria() - before
            m._log_event("bacteria_spawned",
                         f"{spawned} {profile.species} spawned")
            return {"ok": True, "germ_key": req.germ_key, "spawned": spawned,
                    "total_bacteria": before + spawned}

    @app.post("/reset")
    def reset(req: ResetRequest = ResetRequest()):
        if req.scenario not in SCENARIOS:
            raise HTTPException(404, f"Unknown scenario: {req.scenario}")
        with session.lock:
            session.reset(req.scenario, req.initial_bacteria, req.seed)
            return {"ok": True, "scenario": req.scenario,
                    "total_bacteria": session.model.count_living_bacteria()}

    @app.post("/pause")
    def pause():
        with session.lock:
            session.model.paused = True
            return {"paused": True}

    @app.post("/resume")
    def resume():
        with session.lock:
            session.model.paused = False
            return {"paused": False}

    # ── GNN ──────────────────────────────────────────────────────────────────
    @app.get("/gnn/status")
    def gnn_status():
        engine = session.gnn()
        return json_safe({**engine.status(), "trained": session._gnn_trained})

    @app.post("/gnn/predict")
    def gnn_predict(req: GNNPredictRequest = GNNPredictRequest()):
        engine = session.gnn()
        with session.lock:
            state = session.model.get_full_state()
        return json_safe(engine.predict(state, threshold=req.threshold,
                                        max_nodes=req.max_nodes,
                                        max_edge_distance=req.max_edge_distance))

    @app.post("/gnn/advisory")
    def gnn_advisory(req: GNNAdvisoryRequest = GNNAdvisoryRequest()):
        engine = session.gnn()
        with session.lock:
            state = session.model.get_full_state()
        return json_safe(engine.treatment_advisory(state, req.available_antibiotics))

    # ── Analytics ────────────────────────────────────────────────────────────
    @app.get("/analytics/mic")
    def analytics_mic(antibiotic_key: str = "ciprofloxacin"):
        if antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise HTTPException(404, f"Unknown antibiotic: {antibiotic_key}")
        with session.lock:
            living = session.living_bacteria()
            # Empty body -> frontend shows "No data - start simulation first"
            return estimate_population_mic(living, antibiotic_key) if living else {}

    @app.get("/analytics/diversity")
    def analytics_diversity():
        with session.lock:
            living = session.living_bacteria()
            h = shannon_diversity(living)
        # Descriptive bands only; they mirror the frontend's colour thresholds
        # (H>2 red, H>1 amber). They are UI labels, not empirically derived cutoffs.
        if not living:
            text = "No living bacteria."
        elif h > 2:
            text = "High genotype diversity (H > 2): many distinct resistance-gene combinations."
        elif h > 1:
            text = "Moderate genotype diversity (1 < H <= 2)."
        else:
            text = "Low genotype diversity (H <= 1): population dominated by few genotypes."
        return {"shannon_diversity": h, "population": len(living),
                "interpretation": text}

    @app.get("/analytics/recommend")
    def analytics_recommend():
        with session.lock:
            living = session.living_bacteria()
            recs = recommend_treatment(living, list(ANTIBIOTIC_PROFILES))
        return {"recommendations": recs, "population": len(living)}

    return app


app = create_app()
