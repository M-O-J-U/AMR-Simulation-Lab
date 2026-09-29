"""
Live state stream for WS /ws: snapshot + per-step diff frames.

Read-only with respect to the simulation: frames are built from
model.get_full_state() and agent attributes after the model has already
stepped. Nothing here calls the model RNG or changes model state, so
connecting clients cannot change a trajectory (tested in tests/test_stream.py).

Protocol (server -> client, JSON text frames)
  snapshot  {type, seq, reason, state, status}
            state = GET /state (full get_full_state() + parent_id per bacterium)
  diff      {type, seq, base_seq, step, stepped, n_bacteria,
             added:   [full bacterium dicts],
             removed: [ids],
             changed: [[id, {field: value}]],     # STREAM fields only
             detail:  {id: {DETAIL fields}},      # only for inspected ids
             stats, log: [new event_log entries], events: [...],
             heatmaps: {key: packed}, nutrient: packed | absent}
  status    {type, playing, paused, speed, scenario}   (no seq; out of band)
  detail    {type, seq, id, bacterium | null}           (reply to "inspect")

Client -> server (stream control only; simulation commands stay on REST)
  {"type": "resync"}            -> fresh snapshot
  {"type": "inspect", "id": N}  -> full detail for N in every diff (null clears)

Client apply rules
  - Ignore any frame with seq <= current seq.
  - A diff whose base_seq != current seq means a gap: send "resync".
  - If stepped: every surviving bacterium's age += 1 BEFORE applying
    `changed` (age is only sent when it did not advance by exactly 1).
  - DETAIL fields are only kept fresh for inspected ids; for other cells
    they are as of the last snapshot/added frame.
  - After applying, n_bacteria must equal the local count, else resync.
"""

import asyncio
import base64
import json
import threading
from typing import Dict, List, Optional

# Per-bacterium fields sent only for inspected cells (the rest are streamed).
# None of these are used for rendering; local_density alone changes on ~55%
# of agent-steps, so leaving it out is most of the saving.
DETAIL_FIELDS = ("stress_level", "antibiotic_damage", "offspring_count", "local_density")

QUEUE_MAX = 8          # per-client frames buffered before we give up and resnapshot
HGT_KEEP = 50          # get_full_state() reports the last 50 HGT events


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP PACKING
# ─────────────────────────────────────────────────────────────────────────────

def pack_grid(data: List[List[float]]) -> dict:
    """data[x][y] floats -> 8-bit, x-major, base64. Scaled to the grid's own
    max, so the quantisation step is max/255 (<=0.02 ug/mL at the 5.0 cap)."""
    w = len(data)
    h = len(data[0]) if w else 0
    scale = max((max(col) for col in data), default=0.0) if w else 0.0
    if scale > 0:
        q = bytes(min(255, int(round(v / scale * 255))) for col in data for v in col)
    else:
        q = bytes(w * h)
    return {"w": w, "h": h, "scale": scale, "b64": base64.b64encode(q).decode("ascii")}


def unpack_grid(p: dict) -> List[List[float]]:
    """Reference decoder (mirrors the JS one); used by tests."""
    raw = base64.b64decode(p["b64"])
    w, h, s = p["w"], p["h"], p["scale"]
    return [[raw[x * h + y] / 255 * s for y in range(h)] for x in range(w)]


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER — last published view + frame builder (call under session.lock)
# ─────────────────────────────────────────────────────────────────────────────

def full_state(model) -> dict:
    """GET /state payload: get_full_state() plus each bacterium's parent_id
    (read from the agent; to_dict() does not include it) and the active
    biology version with any warnings (data/biology.py)."""
    state = model.get_full_state()
    state["biology"] = model.biology.summary()
    parents = {a.unique_id: getattr(a, "parent_id", None) for a in model.agents}
    for b in state["bacteria"]:
        b["parent_id"] = parents.get(b["id"])
    return state


class StateTracker:
    def __init__(self):
        self.seq = 0
        self.inspected: Dict[object, Optional[int]] = {}   # client token -> id
        self._view: Dict[int, dict] = {}
        self._heat: Dict[str, dict] = {}
        self._nutrient: Optional[dict] = None
        self._last_log = None
        self._state: Optional[dict] = None

    # -- snapshots ------------------------------------------------------------
    def rebase(self, model) -> dict:
        """Start a new chain from the model's current state (after reset)."""
        self.seq += 1
        self._absorb(full_state(model))
        return self.snapshot_frame("reset")

    def snapshot_frame(self, reason: str) -> dict:
        return {"type": "snapshot", "seq": self.seq, "reason": reason,
                "state": json.loads(json.dumps(self._state))}

    def _absorb(self, state: dict):
        self._state = state
        self._view = {b["id"]: b for b in state["bacteria"]}
        self._heat = {k: pack_grid(v["data"]) for k, v in state["antibiotic_heatmaps"].items()}
        self._nutrient = pack_grid(state["nutrient_heatmap"])
        self._last_log = state["event_log"][-1] if state["event_log"] else None

    # -- diffs ----------------------------------------------------------------
    def diff_frame(self, model, stepped: bool) -> dict:
        prev_view = self._view
        state = full_state(model)
        view = {b["id"]: b for b in state["bacteria"]}

        added = [b for i, b in view.items() if i not in prev_view]
        removed = [i for i in prev_view if i not in view]
        changed, events = [], []
        for i, b in view.items():
            p = prev_view.get(i)
            if p is None:
                continue
            d = {}
            for k, v in b.items():
                if k in DETAIL_FIELDS or p.get(k) == v:
                    continue
                if k == "age" and stepped and v == p["age"] + 1:
                    continue           # client derives it
                d[k] = v
            if stepped and "age" not in d and b["age"] == p["age"]:
                d["age"] = b["age"]    # did NOT advance: must say so explicitly
            if d:
                changed.append([i, d])
                for flag, on, off in (("sos_active", "sos_on", "sos_off"),
                                      ("in_biofilm", "biofilm_on", "biofilm_off"),
                                      ("is_persister", "persister_on", "persister_off")):
                    if flag in d:
                        events.append({"kind": on if d[flag] else off, "id": i, "pos": b["pos"]})

        for i in removed:
            p = prev_view[i]
            events.append({"kind": "death", "id": i, "pos": p["pos"], "species": p["species"]})
        # birth/spawn carry only ids: position, species etc. are in `added`
        # (a mass-division step can have ~1500 births, so no duplication here).
        for b in added:
            if b.get("parent_id") is not None and stepped:
                events.append({"kind": "birth", "id": b["id"], "parent_id": b["parent_id"]})
            else:
                events.append({"kind": "spawn", "id": b["id"]})
        if stepped:
            for e in model.hgt_events_this_step:
                events.append({"kind": "hgt", "step": e.step, "donor": e.donor_id,
                               "recipient": e.recipient_id, "gene": e.gene,
                               "pos": list(e.position)})

        heat = {k: pack_grid(v["data"]) for k, v in state["antibiotic_heatmaps"].items()}
        heat_out = {}
        for k, v in heat.items():
            if self._heat.get(k) != v:
                meta = state["antibiotic_heatmaps"][k]
                heat_out[k] = {**v, "max": meta["max"], "color": meta["color"], "name": meta["name"]}
        nutrient = pack_grid(state["nutrient_heatmap"])

        log = state["event_log"]
        new_log = log
        if self._last_log is not None:
            for idx in range(len(log) - 1, -1, -1):
                if log[idx] is self._last_log:
                    new_log = log[idx + 1:]
                    break

        wanted = {i for i in self.inspected.values() if i is not None}
        detail = {i: {k: view[i][k] for k in DETAIL_FIELDS} for i in wanted if i in view}

        self.seq += 1
        frame = {
            "type": "diff", "seq": self.seq, "base_seq": self.seq - 1,
            "step": state["stats"]["step"], "stepped": stepped,
            "n_bacteria": len(view), "added": added, "removed": removed,
            "changed": changed, "detail": detail, "stats": state["stats"],
            "log": new_log, "events": events, "heatmaps": heat_out,
        }
        if nutrient != self._nutrient:
            frame["nutrient"] = nutrient
        # Commit the new view
        self._state = state
        self._view = view
        self._heat = heat
        self._nutrient = nutrient
        self._last_log = log[-1] if log else None
        return frame

    def detail_frame(self, bid: Optional[int]) -> dict:
        b = self._view.get(bid) if bid is not None else None
        return {"type": "detail", "seq": self.seq, "id": bid, "bacterium": b}


# ─────────────────────────────────────────────────────────────────────────────
# BROADCASTER — fan frames out to per-client bounded queues
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_MARKER = object()


class Client:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.overflows = 0


class Broadcaster:
    """publish() may be called from any thread (REST handlers run in a
    thread pool, the play loop steps in a worker thread). Frames are JSON-
    encoded once in the calling thread, then handed to the event loop with
    call_soon_threadsafe, which preserves publish order."""

    def __init__(self, queue_max: int = QUEUE_MAX):
        self.queue_max = queue_max
        self.clients: List[Client] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def publish(self, frames: List[dict]):
        if not frames or self.loop is None or not self.clients:
            return
        encoded = [(f["type"], json.dumps(f, separators=(",", ":"))) for f in frames]
        try:
            self.loop.call_soon_threadsafe(self._fanout, encoded)
        except RuntimeError:
            pass    # loop closed during shutdown

    def _fanout(self, encoded):
        """A client is 'slow' if frames from EARLIER batches are still unsent
        (backlog >= queue_max) when a new batch arrives. The current batch is
        always delivered whole otherwise — one POST /step with n_steps=100 is a
        single batch of 100 frames and must not force a healthy client to
        resync. Memory per client is bounded by queue_max + one batch."""
        for c in list(self.clients):
            if c.queue.qsize() >= self.queue_max:
                # Slow client: drop its backlog and this batch, resync once.
                c.overflows += 1
                self._clear(c)
                c.queue.put_nowait(SNAPSHOT_MARKER)
                continue
            for kind, text in encoded:
                if kind == "snapshot":
                    self._clear(c)
                c.queue.put_nowait(text)

    def request_snapshot(self, c: Client):
        self._clear(c)
        c.queue.put_nowait(SNAPSHOT_MARKER)

    def send_all(self, text: str):
        """Out-of-band messages (status) — never dropped, never trigger resync."""
        if self.loop is None:
            return
        def _put():
            for c in list(self.clients):
                c.queue.put_nowait(text)
        try:
            self.loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    @staticmethod
    def _clear(c: Client):
        while not c.queue.empty():
            c.queue.get_nowait()

    def add(self) -> Client:
        c = Client()
        self.clients.append(c)
        return c

    def remove(self, c: Client):
        if c in self.clients:
            self.clients.remove(c)
