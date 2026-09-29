"""
API contract tests for api/server.py (FastAPI TestClient, no network).

Covers every route frontend/index.html calls, with the request bodies it
actually sends, plus a check that driving a seeded model through the API
produces exactly the same trajectory as driving it directly — i.e. the
server adds no randomness and changes no step semantics.

Run with: python -m pytest tests/test_api.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from fastapi.testclient import TestClient

from api.server import create_app, SCENARIOS
from data.card_loader import ANTIBIOTIC_PROFILES, GERM_PROFILES, get_germ
from simulation.amr_model import AMRSimulationModel
from data.biology import LAB_V2

JSON = {"Content-Type": "application/json"}
SEED = 42

# Scenario -> species expected at step 0 (mirrors AMRSimulationModel._setup_scenario)
SCENARIO_SPECIES = {
    "validation":            {"e_coli"},
    "ecoli_cipro":           {"e_coli"},
    "klebsiella_carbapenem": {"klebsiella_pneumoniae"},
    "xdr_acinetobacter":     {"acinetobacter_baumannii"},
    "mrsa_hospital":         {"mrsa"},
    "pakistan_crisis":       {"e_coli", "klebsiella_pneumoniae"},
    "multi_species":         {"e_coli", "klebsiella_pneumoniae", "pseudomonas_aeruginosa"},
}


@pytest.fixture
def client():
    app = create_app(enable_logging=False)
    with TestClient(app) as c:
        r = c.post("/reset", json={"scenario": "pakistan_crisis",
                                   "initial_bacteria": 80, "seed": SEED})
        assert r.status_code == 200
        yield c


@pytest.fixture(scope="module")
def gnn_client():
    """Shared across GNN tests — loading torch + checkpoint takes seconds."""
    app = create_app(enable_logging=False)
    with TestClient(app) as c:
        c.post("/reset", json={"scenario": "pakistan_crisis",
                               "initial_bacteria": 80, "seed": SEED})
        c.post("/step", json={"n_steps": 5})
        yield c


def state(c):
    r = c.get("/state")
    assert r.status_code == 200
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# GET /state
# ─────────────────────────────────────────────────────────────────────────────

class TestState:
    def test_top_level_shape(self, client):
        s = state(client)
        for k in ("bacteria", "antibiotic_heatmaps", "nutrient_heatmap",
                  "hgt_events", "stats", "event_log", "grid_width", "grid_height"):
            assert k in s
        assert (s["grid_width"], s["grid_height"]) == (80, 60)

    def test_stats_fields_used_by_frontend(self, client):
        st = state(client)["stats"]
        for k in ("step", "scenario", "total_bacteria", "resistant_bacteria",
                  "hgt_total", "biofilm_bacteria", "persister_cells",
                  "avg_fitness", "avg_resistance_genes", "species_counts",
                  "gene_distribution"):
            assert k in st
        assert st["scenario"] == "pakistan_crisis"
        assert st["step"] == 0

    def test_bacterium_fields_used_by_frontend(self, client):
        b = state(client)["bacteria"][0]
        for k in ("id", "parent_id", "pos", "state", "shape", "color_hex", "gene_count",
                  "fitness", "energy", "sos_active", "in_biofilm", "species",
                  "resistance_genes", "generation", "age", "stress_level",
                  "antibiotic_damage", "is_persister", "total_mutations",
                  "hgt_events", "offspring_count"):
            assert k in b

    def test_default_session_created_lazily(self):
        with TestClient(create_app(enable_logging=False)) as c:
            st = state(c)["stats"]
            assert st["scenario"] == "validation"
            assert st["total_bacteria"] == 80


# ─────────────────────────────────────────────────────────────────────────────
# POST /step, /pause, /resume
# ─────────────────────────────────────────────────────────────────────────────

class TestStepPauseResume:
    def test_step_one(self, client):
        r = client.post("/step", json={"n_steps": 1}, headers=JSON)
        assert r.status_code == 200 and r.json()["step"] == 1
        assert state(client)["stats"]["step"] == 1

    def test_step_many(self, client):
        client.post("/step", json={"n_steps": 4})
        assert state(client)["stats"]["step"] == 4

    @pytest.mark.parametrize("n", [0, -1, 101])
    def test_step_rejects_out_of_range(self, client, n):
        assert client.post("/step", json={"n_steps": n}).status_code == 422

    def test_step_while_paused_advances_and_stays_paused(self, client):
        """'+1 Step' after Pause must step once; the pause flag survives."""
        assert client.post("/pause").json() == {"paused": True, "playing": False}
        r = client.post("/step", json={"n_steps": 1}).json()
        assert r["step"] == 1 and r["paused"] is True
        assert state(client)["stats"]["step"] == 1
        assert client.post("/pause").json()["paused"] is True   # stop play before manual steps
        client.post("/step", json={"n_steps": 2})
        assert state(client)["stats"]["step"] == 3

    def test_paused_step_matches_unpaused_step(self):
        """Stepping while paused must give the same trajectory as unpaused."""
        runs = []
        for pause in (False, True):
            with TestClient(create_app(enable_logging=False)) as c:
                c.post("/reset", json={"scenario": "pakistan_crisis",
                                       "initial_bacteria": 80, "seed": SEED})
                if pause:
                    c.post("/pause")
                for _ in range(6):
                    c.post("/step", json={"n_steps": 1})
                runs.append(_strip_cell_ids(state(c)))
        assert runs[0] == runs[1]


# ─────────────────────────────────────────────────────────────────────────────
# POST /apply_antibiotic, /remove_antibiotic
# ─────────────────────────────────────────────────────────────────────────────

class TestAntibiotics:
    @pytest.mark.parametrize("mode", ["uniform", "gradient", "zone"])
    def test_apply_modes(self, client, mode):
        r = client.post("/apply_antibiotic", json={
            "antibiotic_key": "colistin", "concentration": 1.0, "mode": mode})
        assert r.status_code == 200
        hm = state(client)["antibiotic_heatmaps"]["colistin"]
        assert hm["max"] > 0 and hm["name"] and hm["color"].startswith("#")

    def test_spot_without_center_is_noop(self, client):
        """Current model behaviour: mode='spot' needs a center; the frontend
        never sends one, so its 'spot' option adds no drug."""
        client.post("/apply_antibiotic", json={
            "antibiotic_key": "colistin", "concentration": 1.0, "mode": "spot"})
        assert state(client)["antibiotic_heatmaps"]["colistin"]["max"] == 0

    def test_spot_with_center(self, client):
        client.post("/apply_antibiotic", json={
            "antibiotic_key": "colistin", "concentration": 1.0, "mode": "spot",
            "center": [40, 30], "radius": 8})
        assert state(client)["antibiotic_heatmaps"]["colistin"]["max"] > 0

    def test_apply_unknown_antibiotic(self, client):
        r = client.post("/apply_antibiotic", json={
            "antibiotic_key": "not_a_drug", "concentration": 1.0, "mode": "uniform"})
        assert r.status_code == 404

    def test_apply_bad_mode(self, client):
        r = client.post("/apply_antibiotic", json={
            "antibiotic_key": "colistin", "concentration": 1.0, "mode": "everywhere"})
        assert r.status_code == 422

    def test_remove_clears_grid(self, client):
        client.post("/apply_antibiotic", json={
            "antibiotic_key": "ciprofloxacin", "concentration": 2.0, "mode": "uniform"})
        r = client.post("/remove_antibiotic", json={"antibiotic_key": "ciprofloxacin"})
        assert r.status_code == 200 and r.json()["was_present"] is True
        assert state(client)["antibiotic_heatmaps"]["ciprofloxacin"]["max"] == 0

    def test_clear_all_loop_like_frontend(self, client):
        """clearAllAB() posts all six keys, including ones not in the scenario."""
        for k in ["ciprofloxacin", "meropenem", "colistin",
                  "vancomycin", "ampicillin", "tetracycline"]:
            assert client.post("/remove_antibiotic",
                               json={"antibiotic_key": k}).status_code == 200

    def test_remove_unknown_antibiotic(self, client):
        r = client.post("/remove_antibiotic", json={"antibiotic_key": "not_a_drug"})
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# POST /spawn_bacteria
# ─────────────────────────────────────────────────────────────────────────────

class TestSpawn:
    @pytest.mark.parametrize("germ_key", sorted(GERM_PROFILES))
    def test_spawn_each_germ(self, client, germ_key):
        before = state(client)["stats"]["total_bacteria"]
        r = client.post("/spawn_bacteria", json={"germ_key": germ_key, "count": 20})
        assert r.status_code == 200 and r.json()["spawned"] == 20
        s = state(client)
        assert s["stats"]["total_bacteria"] == before + 20
        assert get_germ(germ_key).species in s["stats"]["species_counts"]
        assert s["event_log"][-1]["type"] == "bacteria_spawned"

    def test_spawn_unknown_germ(self, client):
        r = client.post("/spawn_bacteria", json={"germ_key": "not_a_germ", "count": 5})
        assert r.status_code == 404

    def test_spawn_rejects_zero(self, client):
        r = client.post("/spawn_bacteria", json={"germ_key": "e_coli", "count": 0})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /reset
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:
    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_reset_each_scenario(self, client, scenario):
        r = client.post("/reset", json={"scenario": scenario, "initial_bacteria": 60})
        assert r.status_code == 200
        st = state(client)["stats"]
        assert st["scenario"] == scenario and st["step"] == 0
        expected = {get_germ(k).species for k in SCENARIO_SPECIES[scenario]}
        assert set(st["species_counts"]) == expected

    def test_scenario_list_matches_frontend(self):
        html = open(os.path.join(os.path.dirname(__file__), "..",
                                 "frontend", "index.html"), encoding="utf-8").read()
        for sc in SCENARIOS:
            assert f'<option value="{sc}"' in html

    def test_reset_unknown_scenario(self, client):
        r = client.post("/reset", json={"scenario": "nope", "initial_bacteria": 50})
        assert r.status_code == 404

    def test_reset_clears_pause_and_step(self, client):
        client.post("/step", json={"n_steps": 3})
        client.post("/pause")
        client.post("/reset", json={"scenario": "validation", "initial_bacteria": 40})
        client.post("/step", json={"n_steps": 1})
        assert state(client)["stats"]["step"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY — the API must not perturb the simulation
# ─────────────────────────────────────────────────────────────────────────────

def _strip_cell_ids(s):
    """BacteriumAgent.cell_id comes from uuid4 (not the seeded model RNG), so
    it differs between any two runs, even two direct ones. It is a display
    label only — no simulation or feature code reads it — so it is excluded.
    parent_id is added by the API (read from the agent; to_dict() lacks it),
    so it is dropped too when comparing against a direct get_full_state()."""
    for b in s["bacteria"]:
        b.pop("cell_id")
        b.pop("parent_id", None)
    s.pop("biology", None)      # API adds the active biology version
    return s


class TestAPIMatchesHeadless:
    @pytest.mark.parametrize("scenario", ["pakistan_crisis", "multi_species"])
    def test_api_trajectory_identical_to_direct_model(self, client, scenario):
        client.post("/reset", json={"scenario": scenario,
                                    "initial_bacteria": 80, "seed": SEED})
        client.post("/step", json={"n_steps": 5})
        client.post("/apply_antibiotic", json={
            "antibiotic_key": "ciprofloxacin", "concentration": 0.3, "mode": "uniform"})
        client.post("/step", json={"n_steps": 10})
        # Read-only calls interleaved must not change anything
        client.get("/analytics/diversity")
        client.get("/analytics/recommend")
        client.post("/step", json={"n_steps": 5})
        via_api = state(client)

        m = AMRSimulationModel(scenario=scenario, initial_bacteria=80,
                               seed=SEED, enable_logging=False,
                               biology=LAB_V2)   # the server's default biology
        for _ in range(5): m.step()
        m.apply_antibiotic("ciprofloxacin", concentration=0.3, mode="uniform")
        for _ in range(15): m.step()
        direct = json.loads(json.dumps(m.get_full_state()))

        assert via_api["stats"]["total_bacteria"] > 0   # non-trivial comparison
        assert _strip_cell_ids(via_api) == _strip_cell_ids(direct)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_mic(self, client):
        r = client.get("/analytics/mic", params={"antibiotic_key": "ciprofloxacin"})
        d = r.json()
        assert r.status_code == 200
        for k in ("antibiotic", "susceptible_pct", "intermediate_pct",
                  "resistant_pct", "breakpoint_S", "breakpoint_R"):
            assert k in d
        assert abs(d["susceptible_pct"] + d["intermediate_pct"]
                   + d["resistant_pct"] - 100) < 0.5

    @pytest.mark.parametrize("ab", sorted(ANTIBIOTIC_PROFILES))
    def test_mic_each_antibiotic(self, client, ab):
        assert client.get("/analytics/mic", params={"antibiotic_key": ab}).status_code == 200

    def test_mic_unknown(self, client):
        assert client.get("/analytics/mic",
                          params={"antibiotic_key": "nope"}).status_code == 404

    def test_mic_empty_population_returns_empty(self, client, monkeypatch):
        session = client.app.state.session
        monkeypatch.setattr(session, "living_bacteria", lambda: [])
        assert client.get("/analytics/mic").json() == {}

    def test_diversity(self, client):
        d = client.get("/analytics/diversity").json()
        assert d["population"] == state(client)["stats"]["total_bacteria"]
        assert d["shannon_diversity"] >= 0 and d["interpretation"]

    def test_recommend(self, client):
        d = client.get("/analytics/recommend").json()
        recs = d["recommendations"]
        assert len(recs) == len(ANTIBIOTIC_PROFILES)
        for r in recs:
            for k in ("antibiotic", "score", "pct_susceptible", "rationale"):
                assert k in r
        assert [r["score"] for r in recs] == sorted((r["score"] for r in recs), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# GNN
# ─────────────────────────────────────────────────────────────────────────────

class TestGNN:
    def test_status(self, gnn_client):
        d = gnn_client.get("/gnn/status").json()
        assert d["model_ready"] is True
        assert d["trained"] is True     # ai/checkpoints/best_model.pt present
        assert "n_predictions_run" in d

    def test_status_nan_metrics_become_null(self, gnn_client):
        """best_model.pt stores auroc_mexAB-oprM = NaN; strict JSON can't carry
        NaN, so the API reports it as null instead of failing with a 500."""
        d = gnn_client.get("/gnn/status").json()
        assert d["val_metrics"]["auroc_mexAB-oprM"] is None

    def test_predict_with_frontend_body(self, gnn_client):
        r = gnn_client.post("/gnn/predict", json={
            "threshold": 0.31, "max_nodes": 300, "max_edge_distance": 3})
        d = r.json()
        assert r.status_code == 200 and "error" not in d
        for k in ("gene_transfer_probs", "n_nodes", "n_edges",
                  "n_predicted_transfers", "inference_time_ms"):
            assert k in d

    def test_predict_does_not_change_state(self, gnn_client):
        before = state(gnn_client)
        gnn_client.post("/gnn/predict", json={"threshold": 0.31})
        gnn_client.post("/gnn/advisory", json={})
        assert state(gnn_client) == before

    def test_advisory_with_frontend_body(self, gnn_client):
        r = gnn_client.post("/gnn/advisory", json={"available_antibiotics": [
            "ciprofloxacin", "meropenem", "colistin",
            "vancomycin", "ampicillin", "tetracycline"]})
        d = r.json()
        assert r.status_code == 200 and "error" not in d
        assert d["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        for k in ("imminent_resistance", "threatened_antibiotics",
                  "safe_antibiotics", "recommendations", "advisory_text"):
            assert k in d

    def test_untrained_fallback(self, tmp_path):
        app = create_app(enable_logging=False,
                         checkpoint_path=str(tmp_path / "missing.pt"))
        with TestClient(app) as c:
            d = c.get("/gnn/status").json()
            assert d["model_ready"] is True and d["trained"] is False


# ─────────────────────────────────────────────────────────────────────────────
# CORS — frontend is opened from file:// (Origin: null)
# ─────────────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_simple_request_allowed(self, client):
        r = client.get("/state", headers={"Origin": "null"})
        assert r.headers.get("access-control-allow-origin") == "*"

    def test_preflight_allowed(self, client):
        r = client.options("/step", headers={
            "Origin": "null", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 additions: /analytics/mic_all, /reference, frontend served at /ui/
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2Endpoints:
    def test_mic_all_matches_single(self, client):
        d = client.get("/analytics/mic_all").json()
        assert set(d["antibiotics"]) == set(ANTIBIOTIC_PROFILES)
        assert d["population"] == state(client)["stats"]["total_bacteria"]
        single = client.get("/analytics/mic", params={"antibiotic_key": "colistin"}).json()
        assert d["antibiotics"]["colistin"] == single

    def test_reference_reflects_active_biology(self, client):
        r = client.get("/reference").json()
        assert r["biology"] == "lab_v2"
        assert r["genes"]["mecA"]["transferable"] is False
        assert r["genes"]["blaNDM-1"]["card_id"] == "ARO:3000589"
        assert set(r["antibiotics"]) == set(ANTIBIOTIC_PROFILES)
        client.post("/reset", json={"scenario": "validation", "initial_bacteria": 5,
                                    "biology": "paper_v1"})
        r = client.get("/reference").json()
        assert r["biology"] == "paper_v1" and "mecA" not in r["genes"]

    def test_frontend_served(self, client):
        r = client.get("/ui/")
        assert r.status_code == 200 and "js/main.js" in r.text
        for path in ("/ui/js/render.js", "/ui/css/app.css", "/ui/vendor/uplot/uPlot.iife.min.js"):
            assert client.get(path).status_code == 200, path
        root = client.get("/", follow_redirects=False)
        assert root.status_code in (302, 307) and root.headers["location"] == "/ui/"
