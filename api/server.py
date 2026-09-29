"""
AMR Simulation — FastAPI server: REST commands + WebSocket live state stream.

Rebuilt 2026-09-29: the original server source was lost (this file had been
overwritten with a copy of ai/gnn_inference.py). Routes are thin wrappers
around existing model / analytics / GNN methods. This module contains no
simulation logic and never touches the model RNG directly — every random draw
happens inside AMRSimulationModel methods, exactly as in headless runs.

Transport
  REST = commands (and GET /state as a polling fallback).
  WS /ws = live state stream: a snapshot on connect, then one diff frame per
  simulation step and per state-changing command. Protocol: api/stream.py.

Play is server-side: POST /resume starts a loop that steps the model on a
timer (POST /speed sets the rate), POST /pause stops it. POST /step always
advances n steps, playing or paused (the '+1 Step' button).

Endpoints
  GET  /state                       full snapshot (get_full_state() + parent_id)
  POST /step                        {"n_steps": int}
  POST /apply_antibiotic            {"antibiotic_key", "concentration", "mode", "center"?, "radius"?}
  POST /remove_antibiotic           {"antibiotic_key"}
  POST /spawn_bacteria              {"germ_key", "count"}
  POST /reset                       {"scenario", "initial_bacteria", "seed"?, "biology"?: lab_v2|paper_v1}
  POST /pause, POST /resume
  POST /speed                       {"speed": 1..20}
  WS   /ws                          live stream
  GET  /gnn/status
  POST /gnn/predict                 {"threshold", "max_nodes", "max_edge_distance"}
  POST /gnn/advisory                {"available_antibiotics": [...]}
  GET  /analytics/mic?antibiotic_key=...
  GET  /analytics/diversity
  GET  /analytics/recommend
"""

import asyncio
import contextlib
import json
import math
import os
import sys
import threading
from typing import List, Literal, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.amr_model import AMRSimulationModel
from data.card_loader import ANTIBIOTIC_PROFILES, GERM_PROFILES
from data.biology import BIOLOGIES
from ai.resistance_analytics import (
    estimate_population_mic, recommend_treatment, shannon_diversity,
)
from api.stream import StateTracker, Broadcaster, SNAPSHOT_MARKER, full_state

# Scenario keys handled by AMRSimulationModel._setup_scenario. The model
# silently falls back to "validation" for unknown keys; the API rejects them
# instead so a typo can't masquerade as a different experiment.
SCENARIOS = (
    "validation", "ecoli_cipro", "klebsiella_carbapenem", "xdr_acinetobacter",
    "mrsa_hospital", "pakistan_crisis", "multi_species",
)

DEFAULT_SCENARIO = "validation"
DEFAULT_INITIAL_BACTERIA = 80    # matches the frontend's population slider default
DEFAULT_SPEED = 5                # matches the frontend's speed slider default
# The lab runs the corrected biology; paper_v1 (frozen, what the paper used)
# stays selectable via POST /reset {"biology": "paper_v1"}. See data/biology.py.
DEFAULT_BIOLOGY = "lab_v2"


def play_timing(speed: int) -> Tuple[float, int]:
    """(seconds between ticks, steps per tick). Same mapping the frontend's
    old client-side play timer used, so Play feels the same as before."""
    return max(80, 650 - speed * 28) / 1000.0, max(1, math.ceil(speed / 5))


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
    biology: Literal["lab_v2", "paper_v1"] = DEFAULT_BIOLOGY

class SpeedRequest(BaseModel):
    speed: int = Field(DEFAULT_SPEED, ge=1, le=20)

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
    """Holds the single live model. REST handlers run in a thread pool and the
    play loop steps in a worker thread, so every model access goes through
    `lock`. Every mutation produces stream frames, which are published while
    the lock is still held so frame order always matches mutation order."""

    def __init__(self, enable_logging: bool, checkpoint_path: Optional[str],
                 broadcaster: Broadcaster):
        self.enable_logging = enable_logging
        self.checkpoint_path = checkpoint_path
        self.lock = threading.RLock()
        self.bus = broadcaster
        self.tracker = StateTracker()
        self.playing = False
        self.speed = DEFAULT_SPEED
        self._model: Optional[AMRSimulationModel] = None
        self._gnn = None
        self._gnn_trained = False
        self._gnn_lock = threading.Lock()

    # -- model lifecycle ------------------------------------------------------
    @property
    def model(self) -> AMRSimulationModel:
        with self.lock:
            if self._model is None:
                self.reset(DEFAULT_SCENARIO, DEFAULT_INITIAL_BACTERIA, None)
            return self._model

    def reset(self, scenario: str, initial_bacteria: int, seed: Optional[int],
              biology: str = DEFAULT_BIOLOGY):
        with self.lock:
            self._model = AMRSimulationModel(
                scenario=scenario, initial_bacteria=initial_bacteria,
                seed=seed, enable_logging=self.enable_logging,
                biology=BIOLOGIES[biology],
            )
            snap = self.tracker.rebase(self._model)
            snap["status"] = self.status()
            self.bus.publish([snap])

    # -- mutations (each publishes frames) ------------------------------------
    def step(self, n: int) -> List[dict]:
        """Advance n steps, playing or paused. model.step() is a no-op while
        model.paused is set, so the flag is cleared for the duration of this
        call and restored afterwards; model.step() itself is unchanged."""
        with self.lock:
            m = self.model
            frames = []
            was_paused = m.paused
            m.paused = False
            try:
                for _ in range(n):
                    m.step()
                    frames.append(self.tracker.diff_frame(m, stepped=True))
            finally:
                m.paused = was_paused
            self.bus.publish(frames)
            return frames

    def command(self, fn):
        """Run a non-stepping state change and publish its diff."""
        with self.lock:
            result = fn(self.model)
            self.bus.publish([self.tracker.diff_frame(self._model, stepped=False)])
            return result

    # -- play loop ------------------------------------------------------------
    def set_playing(self, playing: bool):
        with self.lock:
            self.playing = playing
            self.model.paused = not playing
            self.publish_status()

    def set_speed(self, speed: int):
        with self.lock:
            self.speed = speed
            self.publish_status()

    def play_tick(self) -> bool:
        """One tick of server-side Play. Returns False if play stopped."""
        with self.lock:
            if not self.playing:
                return False
            _, n = play_timing(self.speed)
            self.step(n)
            if not self._model.running:          # extinction: stop playing
                self.playing = False
                self.publish_status(reason="extinct")
                return False
            return True

    # -- stream helpers -------------------------------------------------------
    def status(self, reason: Optional[str] = None) -> dict:
        m = self._model
        s = {"type": "status", "playing": self.playing, "speed": self.speed,
             "paused": bool(m.paused) if m else True,
             "scenario": m.sim_scenario if m else None,
             "biology": m.biology.name if m else None,
             "step": m.current_step if m else 0}
        if reason:
            s["reason"] = reason
        return s

    def publish_status(self, reason: Optional[str] = None):
        self.bus.send_all(json.dumps(self.status(reason)))

    def snapshot_text(self, reason: str) -> str:
        with self.lock:
            self.model   # ensure a model (and a tracker chain) exists
            snap = self.tracker.snapshot_frame(reason)
            snap["status"] = self.status()
            return json.dumps(snap, separators=(",", ":"))

    def detail_text(self, bid: Optional[int]) -> str:
        with self.lock:
            return json.dumps(self.tracker.detail_frame(bid), separators=(",", ":"))

    def living_bacteria(self) -> list:
        return self.model._living()

    # -- GNN ------------------------------------------------------------------
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
               checkpoint_path: Optional[str] = None,
               queue_max: Optional[int] = None) -> FastAPI:
    bus = Broadcaster(**({"queue_max": queue_max} if queue_max else {}))
    session = SimulationSession(enable_logging, checkpoint_path, bus)

    async def play_loop():
        while True:
            if session.playing:
                interval, _ = play_timing(session.speed)
                started = asyncio.get_running_loop().time()
                await anyio.to_thread.run_sync(session.play_tick)
                elapsed = asyncio.get_running_loop().time() - started
                await asyncio.sleep(max(0.0, interval - elapsed))
            else:
                await asyncio.sleep(0.05)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        bus.attach(asyncio.get_running_loop())
        task = asyncio.create_task(play_loop())
        try:
            yield
        finally:
            session.playing = False
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="AMR Simulation Lab API", version="1.1", lifespan=lifespan)
    # The frontend is opened straight from disk (file://, origin "null") and
    # calls http://localhost:8000, so cross-origin requests must be allowed.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.session = session
    app.state.bus = bus

    # ── Simulation ───────────────────────────────────────────────────────────
    @app.get("/state")
    def get_state():
        with session.lock:
            return full_state(session.model)

    @app.post("/step")
    def step(req: StepRequest = StepRequest()):
        with session.lock:
            session.step(req.n_steps)
            m = session.model
            return {"step": m.current_step, "paused": m.paused,
                    "running": m.running,
                    "total_bacteria": m.count_living_bacteria()}

    @app.post("/apply_antibiotic")
    def apply_antibiotic(req: ApplyAntibioticRequest):
        if req.antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise HTTPException(404, f"Unknown antibiotic: {req.antibiotic_key}")
        def do(m):
            m.apply_antibiotic(req.antibiotic_key, concentration=req.concentration,
                               mode=req.mode, center=req.center, radius=req.radius)
            return {"ok": True, "antibiotic_key": req.antibiotic_key,
                    "active": list(m.antibiotic_grids)}
        return session.command(do)

    @app.post("/remove_antibiotic")
    def remove_antibiotic(req: RemoveAntibioticRequest):
        if req.antibiotic_key not in ANTIBIOTIC_PROFILES:
            raise HTTPException(404, f"Unknown antibiotic: {req.antibiotic_key}")
        def do(m):
            present = req.antibiotic_key in m.antibiotic_grids
            m.remove_antibiotic(req.antibiotic_key)
            return {"ok": True, "antibiotic_key": req.antibiotic_key,
                    "was_present": present}
        return session.command(do)

    @app.post("/spawn_bacteria")
    def spawn_bacteria(req: SpawnRequest):
        if req.germ_key not in GERM_PROFILES:
            raise HTTPException(404, f"Unknown germ: {req.germ_key}")
        def do(m):
            before = m.count_living_bacteria()
            profile = m.biology.germ(req.germ_key)
            m._spawn_bacteria_cluster(profile, req.count)
            spawned = m.count_living_bacteria() - before
            m._log_event("bacteria_spawned", f"{spawned} {profile.species} spawned")
            return {"ok": True, "germ_key": req.germ_key, "spawned": spawned,
                    "total_bacteria": before + spawned}
        return session.command(do)

    @app.post("/reset")
    def reset(req: ResetRequest = ResetRequest()):
        if req.scenario not in SCENARIOS:
            raise HTTPException(404, f"Unknown scenario: {req.scenario}")
        with session.lock:
            session.reset(req.scenario, req.initial_bacteria, req.seed, req.biology)
            return {"ok": True, "scenario": req.scenario, "biology": req.biology,
                    "total_bacteria": session.model.count_living_bacteria()}

    @app.post("/pause")
    def pause():
        session.set_playing(False)
        return {"paused": True, "playing": False}

    @app.post("/resume")
    def resume():
        session.set_playing(True)
        return {"paused": False, "playing": True}

    @app.post("/speed")
    def speed(req: SpeedRequest):
        session.set_speed(req.speed)
        interval, n = play_timing(req.speed)
        return {"speed": req.speed, "interval_ms": round(interval * 1000),
                "steps_per_tick": n}

    # ── Live stream ──────────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        client = bus.add()
        token = object()
        bus.request_snapshot(client)          # first message is always a snapshot

        async def sender():
            while True:
                item = await client.queue.get()
                if item is SNAPSHOT_MARKER:
                    item = await anyio.to_thread.run_sync(session.snapshot_text, "sync")
                await websocket.send_text(item)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                try:
                    msg = await websocket.receive_json()
                except (ValueError, KeyError):
                    continue                    # ignore malformed control messages
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "resync":
                    bus.request_snapshot(client)
                elif kind == "inspect":
                    bid = msg.get("id")
                    bid = bid if isinstance(bid, int) else None
                    with session.lock:
                        session.tracker.inspected[token] = bid
                    client.queue.put_nowait(
                        await anyio.to_thread.run_sync(session.detail_text, bid))
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            bus.remove(client)
            with session.lock:
                session.tracker.inspected.pop(token, None)

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
