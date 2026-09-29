"""
Tests for the WS /ws live stream (api/stream.py + api/server.py).

Core property: snapshot + every diff, applied by the client rules in
api/stream.py, reproduces GET /state exactly (heatmaps within the 8-bit
quantisation step; DETAIL fields for inspected cells only). RefClient below
mirrors the JavaScript client's apply logic in frontend/index.html.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from api.server import create_app, play_timing
from api.stream import (
    DETAIL_FIELDS, HGT_KEEP, Broadcaster, SNAPSHOT_MARKER, pack_grid, unpack_grid,
)
from simulation.amr_model import AMRSimulationModel

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE CLIENT (mirror of the JS apply rules)
# ─────────────────────────────────────────────────────────────────────────────

class RefClient:
    def __init__(self):
        self.seq = None
        self.state = None
        self.bmap = {}
        self.inspected = None
        self.gaps = 0
        self.frames = []

    def apply(self, f):
        self.frames.append(f)
        t = f["type"]
        if t == "snapshot":
            self.seq = f["seq"]
            self.state = f["state"]
            self.bmap = {b["id"]: b for b in self.state["bacteria"]}
        elif t == "diff":
            if self.seq is not None and f["seq"] <= self.seq:
                return "stale"
            if f["base_seq"] != self.seq:
                self.gaps += 1
                return "gap"
            for i in f["removed"]:
                del self.bmap[i]
            if f["stepped"]:
                for b in self.bmap.values():
                    b["age"] += 1
            for i, d in f["changed"]:
                self.bmap[i].update(d)
            for b in f["added"]:
                self.bmap[b["id"]] = b
            for i, d in f["detail"].items():
                self.bmap[int(i)].update(d)
            st = self.state
            st["stats"] = f["stats"]
            st["event_log"] = (st["event_log"] + f["log"])[-20:]
            hgt = [{"step": e["step"], "donor": e["donor"], "recipient": e["recipient"],
                    "gene": e["gene"], "pos": e["pos"]}
                   for e in f["events"] if e["kind"] == "hgt"]
            st["hgt_events"] = (st["hgt_events"] + hgt)[-HGT_KEEP:]
            for k, p in f["heatmaps"].items():
                st["antibiotic_heatmaps"][k] = {"data": unpack_grid(p), "max": p["max"],
                                                "color": p["color"], "name": p["name"]}
            if "nutrient" in f:
                st["nutrient_heatmap"] = unpack_grid(f["nutrient"])
            assert len(self.bmap) == f["n_bacteria"]
            self.seq = f["seq"]
        elif t == "detail":
            if f["bacterium"] is not None and f["id"] in self.bmap:
                self.bmap[f["id"]].update({k: f["bacterium"][k] for k in DETAIL_FIELDS})
        if self.state is not None:
            self.state["bacteria"] = list(self.bmap.values())
        return t


def assert_matches(ref: RefClient, truth: dict, inspected=None):
    got = ref.state
    for k in ("stats", "event_log", "hgt_events", "grid_width", "grid_height"):
        assert got[k] == truth[k], k
    assert [b["id"] for b in got["bacteria"]] == [b["id"] for b in truth["bacteria"]]
    for g, t in zip(got["bacteria"], truth["bacteria"]):
        for k, v in t.items():
            if k in DETAIL_FIELDS and g["id"] != inspected:
                continue
            assert g[k] == v, (g["id"], k, g[k], v)
    assert set(got["antibiotic_heatmaps"]) == set(truth["antibiotic_heatmaps"])
    for k, t in truth["antibiotic_heatmaps"].items():
        g = got["antibiotic_heatmaps"][k]
        assert (g["max"], g["color"], g["name"]) == (t["max"], t["color"], t["name"])
        _close_grid(g["data"], t["data"])
    _close_grid(got["nutrient_heatmap"], truth["nutrient_heatmap"])


def _close_grid(a, b):
    scale = max((max(c) for c in b), default=0.0)
    tol = scale / 255 / 2 + 1e-9
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    for ca, cb in zip(a, b):
        for va, vb in zip(ca, cb):
            assert abs(va - vb) <= tol


def recv(ws, ref, n):
    for _ in range(n):
        ref.apply(ws.receive_json())


def recv_until(ws, ref, pred, limit=500):
    for _ in range(limit):
        f = ws.receive_json()
        ref.apply(f)
        if pred(f):
            return f
    raise AssertionError("condition not reached")


@pytest.fixture
def client():
    app = create_app(enable_logging=False)
    with TestClient(app) as c:
        c.post("/reset", json={"scenario": "pakistan_crisis",
                               "initial_bacteria": 80, "seed": SEED})
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_first_message_is_snapshot_equal_to_state(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            f = ws.receive_json()
            ref.apply(f)
            assert f["type"] == "snapshot" and f["status"]["playing"] is False
            assert_matches(ref, client.get("/state").json())

    def test_state_has_parent_ids(self, client):
        client.post("/step", json={"n_steps": 10})
        bs = client.get("/state").json()["bacteria"]
        assert all("parent_id" in b for b in bs)
        born = [b for b in bs if b["parent_id"] is not None]
        assert born and all(b["generation"] >= 1 for b in born)

    def test_reset_broadcasts_snapshot(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/reset", json={"scenario": "validation", "initial_bacteria": 30})
            f = recv_until(ws, ref, lambda f: f["type"] == "snapshot")
            assert f["reason"] == "reset" and f["seq"] > ref.frames[0]["seq"]
            assert_matches(ref, client.get("/state").json())

    def test_resync_returns_snapshot(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 3})
            recv(ws, ref, 3)
            ws.send_json({"type": "resync"})
            f = ws.receive_json()
            assert f["type"] == "snapshot" and f["seq"] == ref.seq

    def test_malformed_control_messages_ignored(self, client):
        with client.websocket_connect("/ws") as ws:
            recv(ws, RefClient(), 1)
            ws.send_text("not json")
            ws.send_json(["list"])
            ws.send_json({"type": "nonsense"})
            ws.send_json({"type": "resync"})
            assert ws.receive_json()["type"] == "snapshot"


class TestDiffs:
    def test_one_diff_per_step_and_chain(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 5})
            recv(ws, ref, 5)
            diffs = ref.frames[1:]
            assert [d["type"] for d in diffs] == ["diff"] * 5
            assert [d["step"] for d in diffs] == [1, 2, 3, 4, 5]
            assert all(d["stepped"] for d in diffs) and ref.gaps == 0
            assert_matches(ref, client.get("/state").json())

    def test_long_mixed_session_reconstructs_exactly(self, client):
        """Steps, every command type, inspection and a reset: the client view
        must equal GET /state at the end (and at checkpoints)."""
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 6}); recv(ws, ref, 6)
            target = ref.state["bacteria"][0]["id"]
            ws.send_json({"type": "inspect", "id": target})
            recv_until(ws, ref, lambda f: f["type"] == "detail")
            client.post("/apply_antibiotic", json={"antibiotic_key": "colistin",
                        "concentration": 2.0, "mode": "gradient"}); recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 8}); recv(ws, ref, 8)
            client.post("/spawn_bacteria", json={"germ_key": "mrsa", "count": 25}); recv(ws, ref, 1)
            client.post("/apply_antibiotic", json={"antibiotic_key": "meropenem",
                        "concentration": 1.0, "mode": "spot", "center": [20, 20]}); recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 12}); recv(ws, ref, 12)
            alive = {b["id"] for b in client.get("/state").json()["bacteria"]}
            assert_matches(ref, client.get("/state").json(),
                           inspected=target if target in alive else None)
            client.post("/remove_antibiotic", json={"antibiotic_key": "colistin"}); recv(ws, ref, 1)
            client.post("/pause"); client.post("/step", json={"n_steps": 4})
            recv_until(ws, ref, lambda f: f["type"] == "diff" and f["step"] == 30)
            assert ref.gaps == 0
            alive = {b["id"] for b in client.get("/state").json()["bacteria"]}
            assert_matches(ref, client.get("/state").json(),
                           inspected=target if target in alive else None)

    def test_command_frames_do_not_advance_age(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/spawn_bacteria", json={"germ_key": "e_coli", "count": 5})
            f = ws.receive_json(); ref.apply(f)
            assert f["stepped"] is False and f["step"] == 0
            assert len(f["added"]) == 5
            assert {e["kind"] for e in f["events"]} == {"spawn"}
            assert_matches(ref, client.get("/state").json())

    def test_age_mostly_omitted(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 10}); recv(ws, ref, 10)
            n_changed = sum(len(f["changed"]) for f in ref.frames[1:])
            n_age = sum(1 for f in ref.frames[1:] for _, d in f["changed"] if "age" in d)
            assert n_changed > 0 and n_age / n_changed < 0.05
            assert_matches(ref, client.get("/state").json())

    def test_detail_fields_only_for_inspected(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            bid = ref.state["bacteria"][3]["id"]
            ws.send_json({"type": "inspect", "id": bid})
            d = recv_until(ws, ref, lambda f: f["type"] == "detail")
            assert d["id"] == bid and d["bacterium"]["id"] == bid
            client.post("/step", json={"n_steps": 3}); recv(ws, ref, 3)
            for f in ref.frames[-3:]:
                for _, ch in f["changed"]:
                    assert not set(ch) & set(DETAIL_FIELDS)
                assert set(f["detail"]) <= {str(bid)}
            ws.send_json({"type": "inspect", "id": None})
            recv_until(ws, ref, lambda f: f["type"] == "detail")
            client.post("/step", json={"n_steps": 1}); f = ws.receive_json()
            assert f["detail"] == {}

    def test_heatmaps_sent_only_when_changed(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 2}); recv(ws, ref, 2)
            # pakistan_crisis starts with zero-concentration drug grids -> unchanged
            assert all(f["heatmaps"] == {} for f in ref.frames[1:])
            client.post("/apply_antibiotic", json={"antibiotic_key": "ciprofloxacin",
                        "concentration": 1.0, "mode": "uniform"})
            f = ws.receive_json(); ref.apply(f)
            assert set(f["heatmaps"]) == {"ciprofloxacin"}
            assert_matches(ref, client.get("/state").json())

    def test_diffs_much_smaller_than_snapshots(self, client):
        client.post("/step", json={"n_steps": 25})
        with client.websocket_connect("/ws") as ws:
            snap = ws.receive_text()
            client.post("/step", json={"n_steps": 1})
            diff = ws.receive_text()
            assert len(diff) < len(snap) / 3


class TestEvents:
    def test_births_deaths_hgt_consistent(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            prev_ids = set(ref.bmap)
            prev_hgt = ref.state["stats"]["hgt_total"]
            client.post("/apply_antibiotic", json={"antibiotic_key": "ciprofloxacin",
                        "concentration": 1.0, "mode": "uniform"})
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 25})
            births = deaths = 0
            for _ in range(25):
                f = ws.receive_json()
                kinds = [e["kind"] for e in f["events"]]
                assert sorted(e["id"] for e in f["events"] if e["kind"] == "death") \
                    == sorted(f["removed"])
                for e in f["events"]:
                    if e["kind"] == "birth":
                        births += 1
                        assert e["parent_id"] in prev_ids or e["parent_id"] in \
                            {b["id"] for b in f["added"]}
                hgt_now = f["stats"]["hgt_total"]
                if hgt_now < 1000:   # model trims its list at 1000
                    assert kinds.count("hgt") == hgt_now - prev_hgt
                prev_hgt = hgt_now
                deaths += kinds.count("death")
                ref.apply(f)
                prev_ids = set(ref.bmap)
            assert births > 0 and deaths > 0

    def test_state_flag_events_match_changes(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            client.post("/step", json={"n_steps": 20}); recv(ws, ref, 20)
            for f in ref.frames[1:]:
                flips = {(i, k) for i, d in f["changed"] for k in d
                         if k in ("sos_active", "in_biofilm", "is_persister")}
                evs = {e["id"] for e in f["events"]
                       if e["kind"].split("_")[0] in ("sos", "biofilm", "persister")}
                assert {i for i, _ in flips} == evs


# ─────────────────────────────────────────────────────────────────────────────
# SERVER-SIDE PLAY
# ─────────────────────────────────────────────────────────────────────────────

class TestPlay:
    def test_timing_matches_old_client_mapping(self):
        assert play_timing(5) == (0.51, 1)
        assert play_timing(20) == (0.09, 4)
        assert play_timing(1) == (0.622, 1)

    def test_resume_streams_then_pause_stops(self, client):
        with client.websocket_connect("/ws") as ws:
            ref = RefClient()
            recv(ws, ref, 1)
            assert client.post("/speed", json={"speed": 20}).json()["steps_per_tick"] == 4
            client.post("/resume")
            recv_until(ws, ref, lambda f: f["type"] == "diff" and f["step"] >= 12)
            client.post("/pause")
            recv_until(ws, ref, lambda f: f["type"] == "status" and not f["playing"])
            truth = client.get("/state").json()
            assert ref.gaps == 0 and truth["stats"]["step"] == ref.state["stats"]["step"]
            assert_matches(ref, truth)
            # No further steps after pause
            client.get("/state")
            assert client.get("/state").json()["stats"]["step"] == truth["stats"]["step"]

    def test_play_trajectory_identical_to_manual_steps(self):
        """Server-side Play must not change what the sim computes."""
        def run(play):
            with TestClient(create_app(enable_logging=False)) as c:
                c.post("/reset", json={"scenario": "pakistan_crisis",
                                       "initial_bacteria": 80, "seed": SEED})
                with c.websocket_connect("/ws") as ws:
                    ref = RefClient(); recv(ws, ref, 1)
                    if play:
                        c.post("/speed", json={"speed": 20})
                        c.post("/resume")
                        recv_until(ws, ref, lambda f: f["type"] == "diff" and f["step"] >= 16)
                        c.post("/pause")
                        recv_until(ws, ref, lambda f: f["type"] == "status" and not f["playing"])
                    else:
                        c.post("/step", json={"n_steps": 16}); recv(ws, ref, 16)
                    s = c.get("/state").json()
                    n = s["stats"]["step"]
                    for b in s["bacteria"]:
                        b.pop("cell_id")
                    return n, s
        n_play, s_play = run(True)
        with TestClient(create_app(enable_logging=False)) as c:
            c.post("/reset", json={"scenario": "pakistan_crisis",
                                   "initial_bacteria": 80, "seed": SEED})
            c.post("/step", json={"n_steps": n_play})
            s_manual = c.get("/state").json()
            for b in s_manual["bacteria"]:
                b.pop("cell_id")
        assert s_play == s_manual

    def test_status_broadcast(self, client):
        with client.websocket_connect("/ws") as ws:
            recv(ws, RefClient(), 1)
            client.post("/speed", json={"speed": 7})
            f = ws.receive_json()
            assert f == {**f, "type": "status", "speed": 7, "playing": False}


# ─────────────────────────────────────────────────────────────────────────────
# STREAM DOES NOT PERTURB THE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def test_streaming_run_identical_to_headless(client):
    with client.websocket_connect("/ws") as ws:
        ref = RefClient(); recv(ws, ref, 1)
        ws.send_json({"type": "inspect", "id": ref.state["bacteria"][0]["id"]})
        recv_until(ws, ref, lambda f: f["type"] == "detail")
        client.post("/step", json={"n_steps": 5}); recv(ws, ref, 5)
        client.post("/apply_antibiotic", json={"antibiotic_key": "ciprofloxacin",
                    "concentration": 1.0, "mode": "uniform"}); recv(ws, ref, 1)
        client.post("/step", json={"n_steps": 15}); recv(ws, ref, 15)
        via = client.get("/state").json()

    m = AMRSimulationModel(scenario="pakistan_crisis", initial_bacteria=80,
                           seed=SEED, enable_logging=False)
    for _ in range(5): m.step()
    m.apply_antibiotic("ciprofloxacin", concentration=1.0, mode="uniform")
    for _ in range(15): m.step()
    direct = json.loads(json.dumps(m.get_full_state()))
    for s in (via, direct):
        for b in s["bacteria"]:
            b.pop("cell_id"); b.pop("parent_id", None)
    assert via["stats"]["total_bacteria"] > 0 and via == direct


# ─────────────────────────────────────────────────────────────────────────────
# BROADCASTER UNIT TESTS (slow-client behaviour can't be simulated through
# TestClient, whose transport never applies backpressure)
# ─────────────────────────────────────────────────────────────────────────────

class TestBroadcaster:
    def _drain(self, c):
        out = []
        while not c.queue.empty():
            out.append(c.queue.get_nowait())
        return out

    def test_large_batch_to_healthy_client_is_delivered_whole(self):
        async def go():
            b = Broadcaster(queue_max=3)
            c = b.add()
            b._fanout([("diff", f"d{i}") for i in range(10)])   # e.g. n_steps=10
            assert self._drain(c) == [f"d{i}" for i in range(10)] and c.overflows == 0
        asyncio.run(go())

    def test_slow_client_backlog_replaced_with_snapshot_marker(self):
        async def go():
            b = Broadcaster(queue_max=3)
            c = b.add()
            b._fanout([("diff", "d0"), ("diff", "d1"), ("diff", "d2")])   # not consumed
            b._fanout([("diff", "d3")])
            items = self._drain(c)
            assert items == [SNAPSHOT_MARKER] and c.overflows == 1
            b._fanout([("diff", "d4")])          # after recovery, diffs flow again
            assert self._drain(c) == ["d4"]
        asyncio.run(go())

    def test_broadcast_snapshot_clears_backlog(self):
        async def go():
            b = Broadcaster(queue_max=8)
            c = b.add()
            b._fanout([("diff", "d1"), ("diff", "d2"), ("snapshot", "S"), ("diff", "d3")])
            assert self._drain(c) == ["S", "d3"]
        asyncio.run(go())

    def test_pack_roundtrip_tolerance(self):
        grid = [[(x * 7 + y * 3) % 11 / 2.2 for y in range(30)] for x in range(40)]
        back = unpack_grid(pack_grid(grid))
        _close_grid(back, grid)
        zero = [[0.0] * 30 for _ in range(40)]
        assert unpack_grid(pack_grid(zero)) == zero
