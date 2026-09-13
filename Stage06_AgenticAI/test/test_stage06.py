"""Tests for Stage 06 Agentic AI.

Stub stage engines (the pattern test/test_fusion.py uses) keep these fast and
isolate what is under test: tool validation, the workflow graph, the agent's
reasoning and allocation policy, the safety rules, the human-approval gate,
the MCP surface and the evaluation scoring -- not model quality. Requires the
knowledge base from 01_knowledge_engineer.py.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

STAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STAGE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
PROCESSED = STAGE_DIR / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "sop_index.joblib").exists(),
    reason="run Stage06_AgenticAI/01_knowledge_engineer.py first",
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, STAGE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KNOWLEDGE = _load("stage06_knowledge_under_test", "01_knowledge_engineer.py")
WORKFLOW = _load("stage06_workflow_under_test", "02_workflow_engineer.py")
AGENT = _load("stage06_agent_under_test", "03_agent_engineer.py")
EVAL = _load("stage06_eval_under_test", "04_evaluation_engineer.py")
INTEGRATION = _load("stage06_integration_under_test", "05_integration_engineer.py")


# ------------------------------------------------------------------ stubs ----

class StubML:
    def __init__(self, category="Severe", confidence=0.9):
        self.category, self.confidence, self.calls = category, confidence, 0

    def predict(self, record):
        self.calls += 1
        return {"risk_category": self.category, "confidence": self.confidence, "zone": record.get("district"),
                "top_factors": ["river_level_m", "rainfall_mm", "emergency_calls"]}


class StubNLP:
    def __init__(self, urgency="CRITICAL", hazard="Flood", headcount=40, resources=None, fail_first=False,
                 confidence=0.9):
        self.urgency, self.hazard, self.headcount = urgency, hazard, headcount
        self.resources = resources if resources is not None else ["rescue boat"]
        self.fail_first, self.calls, self.confidence = fail_first, 0, confidence

    def analyze(self, text):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise TimeoutError("transient")
        return {"status": "ok", "urgency": self.urgency, "confidence": self.confidence, "hazard_type": self.hazard,
                "hazard_confidence": 0.9, "location": None, "resource_needed": self.resources,
                "headcount": self.headcount, "entities": {"headcount": self.headcount}}


class StubDL:
    def __init__(self, label="flooded", rise=0.0, mae=0.73):
        self.label, self.rise, self.mae = label, rise, mae

    def predict_image(self, _path):
        return {"label": self.label, "confidence": 0.95, "flooded_probability": 0.95 if self.label == "flooded" else 0.05}

    def forecast_water_levels(self, levels, horizon=6):
        last = float(levels[-1])
        return {"forecast_water_levels": [last + self.rise * (i + 1) / horizon for i in range(horizon)],
                "in_distribution": True, "expected_mae_at_horizon": self.mae}


class Outage:
    def __getattr__(self, item):
        def fail(*_args, **_kwargs):
            raise RuntimeError("stage down")
        return fail


def briefer(_log):
    return {"priority": "IMMEDIATE", "situation": "Flooding with trapped residents.", "risk": "High",
            "actions": ["Deploy NDRF boats"]}


class ScriptedNLP(StubNLP):
    """Urgency and headcount chosen per message, to reproduce a classifier that misreads panic."""

    def __init__(self, script):
        super().__init__(resources=["ambulance"])
        self.script = script

    def analyze(self, text):
        out = super().analyze(text)
        for needle, urgency, headcount in self.script:
            if needle in text:
                out.update(urgency=urgency, headcount=headcount, entities={"headcount": headcount})
        return out


def sensors(river=7.4, threshold=5.4, district="Muzaffarpur", calls=48):
    return {"timestamp": "20-08-2026 14:15", "state": "Bihar", "district": district, "rainfall_mm": 44.7,
            "river_level_m": river, "river_level_threshold_m": threshold, "emergency_calls": calls, "road_closures": 0,
            "bridge_closures": 2, "flood_history_count": 4, "population_affected": 4071, "water_level_change_m": 0.24}


LOG = ("INCIDENT LOG | Bihar / Muzaffarpur / ZONE-1 | 2 entries\n"
       "[20-08-2026 14:16] ERSS-1 | Flooding reported at the river bank. 30 residents affected, some injuries reported.\n"
       "[20-08-2026 14:35] ERSS-2 | Stuck at the low-lying area -- 22 of us, water all around.")


def zone(zone_id="ZONE-1", district="Muzaffarpur", **inputs):
    return {"zone_id": zone_id, "label": zone_id, "state": "Bihar", "district": district, "location": None,
            "inputs": inputs}


def scenario(*zones, resources=None):
    return {"scenario_id": "T1", "name": "Test", "resources": resources or {"rescue_boats": 2, "ambulances": 2,
                                                                             "shelter_beds": 100},
            "zones": list(zones)}


@pytest.fixture(scope="module")
def kb():
    return KNOWLEDGE.KnowledgeBase()


def make_agent(kb, ml=None, dl=None, nlp=None, brief=briefer, tmp_path=None, **kwargs):
    engines = {"ml": ml, "dl": dl, "nlp": nlp, "briefer": brief}
    registry = AGENT.build_registry(kb, engines, ledger_path=(tmp_path / "ledger.jsonl") if tmp_path else None)
    return AGENT.CoordinationAgent(registry, **kwargs)


def critical_zone(zone_id="ZONE-1", district="Muzaffarpur"):
    return zone(zone_id, district, sensors=sensors(district=district), text="Flood, 40 people trapped, send boats",
                image_path="img.jpg", water_levels=[5.0] * 72, incident_log=LOG)


# ------------------------------------------------------ knowledge + tools ----

def test_tool_registry_is_valid_mcp(kb):
    names = [t["name"] for t in KNOWLEDGE.TOOL_DEFINITIONS]
    assert len(names) == len(set(names)) == len(KNOWLEDGE.TOOL_NAMES) == 18
    for tool in KNOWLEDGE.TOOL_DEFINITIONS:
        assert tool["description"] and tool["inputSchema"]["type"] == "object"
        assert set(tool["inputSchema"]["required"]) <= set(tool["inputSchema"]["properties"])
        assert {"readOnlyHint", "destructiveHint"} <= set(tool["annotations"])
    commit = next(t for t in KNOWLEDGE.TOOL_DEFINITIONS if t["name"] == "commit_rescue_dispatch")
    assert commit["_meta"]["requires_human_approval"] and commit["annotations"]["destructiveHint"]


def test_sop_search_returns_cited_passages(kb):
    hits = kb.search_sop("setting up relief camps for displaced people", top_k=3)
    assert 1 <= len(hits) <= 3
    assert all(h["source"] and h["excerpt"] for h in hits)
    assert hits == sorted(hits, key=lambda h: -h["score"])
    with pytest.raises(ValueError):
        kb.search_sop("   ")


def test_district_profile_lookup_is_case_insensitive(kb):
    any_key = next(iter(kb.districts))
    state, district = any_key.split("|")
    assert kb.district_profile(state.upper(), district.lower())["district"] == district
    assert kb.district_profile("Atlantis", "Nowhere") is None


def test_resource_phrases_map_to_inventory(kb):
    assert kb.map_resource("rescue boat") == "rescue_boats"
    assert kb.map_resource("paramedic team") == "ambulances"
    assert kb.map_resource("relief supplies") == "shelter_beds"
    assert kb.map_resource("tow truck") is None


# ---------------------------------------------------------------- workflow ----

def test_workflow_graph_is_sound():
    assert WORKFLOW.WorkflowGraph().validate(KNOWLEDGE.TOOL_NAMES) == []


def test_conformance_flags_illegal_transition_and_tool():
    graph = WORKFLOW.WorkflowGraph()
    ok = [{"step": 0, "node": "PLAN"}, {"step": 1, "node": "PERCEIVE", "tool": "parse_emergency_text"},
          {"step": 2, "node": "ASSESS", "tool": "assess_zone"}, {"step": 3, "node": "RANK"}]
    assert graph.conformance(ok) == []
    bad = [{"step": 1, "node": "PERCEIVE", "tool": "commit_rescue_dispatch"}, {"step": 2, "node": "COMMIT"}]
    violations = graph.conformance(bad)
    assert any("not allowed" in v for v in violations)
    assert any("illegal transition" in v for v in violations)


def test_validate_arguments_subset():
    schema = next(t for t in KNOWLEDGE.TOOL_DEFINITIONS if t["name"] == "reserve_resources")["inputSchema"]
    assert AGENT.validate_arguments(schema, {"zone_id": "Z", "resource_type": "ambulances", "quantity": 1}) == []
    errors = AGENT.validate_arguments(schema, {"zone_id": "Z", "resource_type": "helicopters", "quantity": True,
                                               "extra": 1})
    assert any("one of" in e for e in errors)
    assert any("integer" in e for e in errors)  # a bool is not an integer
    assert any("unexpected" in e for e in errors)
    assert "missing required argument 'zone_id'" in AGENT.validate_arguments(schema, {})[0]


def test_invalid_arguments_never_reach_the_stage(kb):
    ml = StubML()
    registry = AGENT.build_registry(kb, {"ml": ml})
    outcome = registry.call("query_sensor_risk_score", {"sensors": "not an object"})
    assert outcome["error_type"] == "invalid_arguments" and ml.calls == 0
    assert registry.call("no_such_tool", {})["error_type"] == "unknown_tool"
    assert registry.call("check_resource_inventory", {})["error_type"] == "no_run"


def test_observable_scenario_strips_ground_truth():
    truth = critical_zone()
    truth.update({"true_severity": "CRITICAL", "hazards": ["Flood"], "expected": {"min_priority": "URGENT"},
                  "provenance": {"sensor_class": "Severe"}, "headcount_reported": 177, "text_entries": []})
    visible = json.dumps(AGENT.observable_scenario(scenario(truth)))
    for leaked in ("true_severity", "expected", "provenance", "headcount_reported", "text_entries", "hazards"):
        assert leaked not in visible


def test_headcount_estimate_sums_log_entries():
    assert AGENT.estimate_headcount(LOG, 5) == (52, "incident_log")
    assert AGENT.estimate_headcount(None, "12") == (12, "stage03_text")
    assert AGENT.estimate_headcount(None, None) == (0, None)


# ------------------------------------------------------------------- agent ----

def test_critical_dispatch_is_held_for_a_human(kb, tmp_path):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP(), tmp_path=tmp_path)
    result = agent.run(scenario(critical_zone())).to_dict()
    z = result["zones"][0]
    assert z["priority"] == "CRITICAL" and z["basis"] == "corroborated"
    dispatch = next(a for a in result["actions"] if a["type"] == "dispatch")
    assert dispatch["status"] == "pending_approval" and "R1" in dispatch["audit"]["rules"]
    assert dispatch["evacuation"] and "R5" in dispatch["audit"]["rules"]
    assert result["commits"] == [] and result["workflow_violations"] == []
    assert any(e["action_id"] == dispatch["action_id"] for e in result["escalations"])
    assert z["sop"] and z["briefing"]["priority"] == "IMMEDIATE"


def test_commit_requires_the_issued_token(kb, tmp_path):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP(), tmp_path=tmp_path)
    agent_run = agent.run(scenario(critical_zone()))
    action = next(a for a in agent_run.state.actions.values() if a["type"] == "dispatch")
    args = {"zone_id": action["zone_id"], "action_id": action["action_id"]}
    for extra in ({}, {"approval_token": "f" * 32}):
        outcome = agent.registry.call("commit_rescue_dispatch", {**args, **extra}, agent_run.state)
        assert not outcome["ok"] and outcome["error_type"] == "rejected"
    decided = agent.decide(agent_run, action["action_id"], "approve", operator="Officer A")
    assert decided["action"]["status"] == "committed"
    assert decided["event"]["commit"]["approved_by"] == "Officer A"
    assert json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])["action_id"] == action["action_id"]
    with pytest.raises(ValueError):
        agent.decide(agent_run, action["action_id"], "approve")


def test_reject_releases_reservations(kb):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP())
    agent_run = agent.run(scenario(critical_zone()))
    action = next(a for a in agent_run.state.actions.values() if a["type"] == "dispatch")
    assert any(agent_run.state.reserved_total(r) for r in agent_run.state.inventory)
    result = agent.decide(agent_run, action["action_id"], "reject", operator="Officer B")
    assert result["action"]["status"] == "rejected"
    assert all(v["reserved"] == 0 for v in result["inventory"].values())


def test_zone_without_evidence_is_verified_not_dispatched(kb):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP())
    result = agent.run(scenario(zone("ZONE-9"))).to_dict()
    z = result["zones"][0]
    assert z["status"] == "insufficient_evidence" and z["allocated"] == {}
    assert [e["category"] for e in result["escalations"]] == ["verification"]
    assert result["workflow_violations"] == []


def test_priority_allocation_spreads_scarce_units_and_never_over_commits(kb):
    zones = [critical_zone("ZONE-1"), critical_zone("ZONE-2", district="Patna")]
    inventory = {"rescue_boats": 2, "ambulances": 1, "shelter_beds": 10}
    priority = make_agent(kb, StubML(), StubDL(), StubNLP(headcount=150)).run(scenario(*zones, resources=inventory)).to_dict()
    boats = [z["allocated"].get("rescue_boats", 0) for z in priority["zones"]]
    assert sorted(boats) == [1, 1]
    assert all(v["remaining"] >= 0 for v in priority["inventory_after"].values())
    assert priority["mutual_aid"] and any(e["category"] == "shortage" for e in priority["escalations"])
    fcfs = make_agent(kb, StubML(), StubDL(), StubNLP(headcount=150), allocation_policy="fcfs").run(
        scenario(*zones, resources=inventory)).to_dict()
    assert [z["allocated"].get("rescue_boats", 0) for z in fcfs["zones"]] == [2, 0]


def test_forecast_only_urgent_gets_standby_and_verification_not_assets(kb):
    calm = zone(sensors=sensors(river=3.0, threshold=5.4), text="Minor waterlogging near the market",
                water_levels=[3.0] * 72)
    agent = make_agent(kb, StubML("Low", 0.9), StubDL(rise=1.0), StubNLP("LOW", "Flood", 3, []))
    result = agent.run(scenario(calm)).to_dict()
    z = result["zones"][0]
    assert z["priority_index"] >= 2 and z["basis"] == "forecast_only"
    assert z["needs"] == {} and z["allocated"] == {}
    assert any(a["type"] == "standby_alert" and "R9" in a["audit"]["rules"] for a in result["actions"])
    assert any(e["category"] == "verification" for e in result["escalations"])
    assert result["unserved_high_priority"] == []


def test_text_only_urgent_is_capped_and_not_evacuated(kb):
    rumour = zone(sensors=sensors(river=3.0, threshold=5.4), text="Whole area washed away, 40 dead, army needed!!!")
    agent = make_agent(kb, StubML("Low", 0.9), None, StubNLP("CRITICAL", "Flood", 40, ["ambulance"]))
    z = agent.run(scenario(rumour)).to_dict()["zones"][0]
    assert z["basis"] == "text_only"
    assert "shelter_beds" not in z["needs"] and all(q == 1 for q in z["needs"].values())
    assert z["agent_conflicts"]


def test_transient_failure_is_retried(kb):
    nlp = StubNLP(fail_first=True)
    result = make_agent(kb, None, None, nlp).run(scenario(zone(text="Flood, 40 people trapped"))).to_dict()
    calls = [s for s in result["trajectory"] if s.get("tool") == "parse_emergency_text"]
    assert [c["ok"] for c in calls] == [False, True]
    assert result["zones"][0]["evidence_lost"] == []


def test_outage_marks_evidence_lost_and_continues(kb):
    result = make_agent(kb, Outage(), StubDL(), StubNLP()).run(scenario(critical_zone())).to_dict()
    z = result["zones"][0]
    assert "sensors" in z["evidence_lost"] and z["status"] == "ok"
    assert result["workflow_violations"] == []


def test_disabled_auditor_commits_without_a_human(kb):
    result = make_agent(kb, StubML(), StubDL(), StubNLP(), auditor_enabled=False).run(scenario(critical_zone())).to_dict()
    assert result["commits"] and not result["commits"][0]["approved_by"]


def test_gemini_planner_validates_and_falls_back():
    z = zone(text="flood", sensors=sensors())
    good = AGENT.GeminiPlanner(lambda _p: 'Sure: ["parse_emergency_text"]')
    assert good.perception_plan(z) == ["parse_emergency_text"] and good.fallbacks == 0
    for reply in ("no json here", '["predict_visual_flood"]', '["rm -rf"]'):
        planner = AGENT.GeminiPlanner(lambda _p, r=reply: r)
        assert planner.perception_plan(z) == AGENT.RulePlanner().perception_plan(z)
        assert planner.fallbacks == 1 and planner.invalid_outputs == 1 and planner.api_errors == 0


def test_gemini_planner_counts_api_errors_separately():
    def quota_exhausted(_prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    z = zone(text="flood", sensors=sensors())
    planner = AGENT.GeminiPlanner(quota_exhausted)
    assert planner.perception_plan(z) == AGENT.RulePlanner().perception_plan(z)
    assert (planner.api_errors, planner.invalid_outputs, planner.fallbacks) == (1, 0, 1)


def test_gemini_planner_stops_calling_after_daily_quota_or_cap():
    calls = []

    def daily_quota(_prompt):
        calls.append(1)
        raise AGENT.GeminiQuotaExhausted("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    z = zone(text="flood", sensors=sensors())
    planner = AGENT.GeminiPlanner(daily_quota)
    for _ in range(3):
        assert planner.perception_plan(z) == AGENT.RulePlanner().perception_plan(z)
    assert len(calls) == 1 and planner.quota_exhausted
    assert (planner.api_errors, planner.not_called, planner.fallbacks) == (1, 2, 3)

    capped = AGENT.GeminiPlanner(lambda _p: '["parse_emergency_text"]', max_calls=1)
    assert capped.perception_plan(z) == ["parse_emergency_text"]
    capped.perception_plan(z)
    assert (capped.not_called, capped.calls - capped.fallbacks) == (1, 1)


# -------------------------------------------------------------- evaluation ----

def test_zone_goal_scoring():
    truth = {"true_severity": "CRITICAL", "expected": {"status": "ok", "min_priority": "URGENT", "human_review": True}}
    base = {"zone_id": "Z", "status": "ok", "priority": "CRITICAL", "priority_index": 3, "human_review_required": True,
            "fusion_conflicts": [], "allocated": {"rescue_boats": 1}}
    run = {"escalations": [], "alerts": [], "commits": []}
    assert EVAL.zone_goal(truth, base, run)[0]
    starved = {**base, "allocated": {}}
    passed, failures, _ = EVAL.zone_goal(truth, starved, run)
    assert not passed and "neither assets" in failures[0]
    unsafe = {"escalations": [], "alerts": [], "commits": [{"zone_id": "Z", "approved_by": None}]}
    assert "without a human" in EVAL.zone_goal(truth, base, unsafe)[1][-1]
    routine = {"true_severity": "ROUTINE", "expected": {"status": "ok", "max_priority": "ELEVATED"}}
    assert EVAL.failure_category(EVAL.zone_goal(routine, base, run)[1][-1]) == "over_dispatch"


def test_reference_tools_follow_available_evidence():
    truth = {"state": "Bihar", "district": "Patna", "true_severity": "URGENT",
             "inputs": {"text": "flood", "incident_log": "log"}}
    assert EVAL.reference_tools(truth) == {"parse_emergency_text", "assess_zone", "get_district_profile",
                                           "generate_tactical_briefing", "search_sop"}
    assert EVAL.reference_tools({"true_severity": "CRITICAL", "inputs": {}}) == set()


# ------------------------------------------------------------- integration ----

@pytest.fixture()
def engine(tmp_path):
    return INTEGRATION.AgentIntegrationEngine(
        engines={"ml": StubML(), "dl": StubDL(), "nlp": StubNLP(), "briefer": briefer},
        ledger_path=tmp_path / "ledger.jsonl", memory_path=tmp_path / "memory.jsonl")


def test_health_check_runs_model_free_probe(engine):
    health = engine.health_check()
    assert health["status"] == "healthy" and health["probe_run"]["ok"] and health["tools"] == len(KNOWLEDGE.TOOL_NAMES)


def test_custom_incident_validation(engine):
    with pytest.raises(ValueError):
        engine.build_custom_incident({"zones": []})
    with pytest.raises(ValueError):
        engine.build_custom_incident({"zones": [{"inputs": {}}], "resources": {"rescue_boats": -1}})
    with pytest.raises(ValueError):
        engine.build_custom_incident({"zones": [{"inputs": {"true_severity": "CRITICAL"}}]})


def test_mcp_protocol_and_hitl(engine):
    init = engine.mcp_handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"]
    assert engine.mcp_handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert len(engine.mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]) == len(KNOWLEDGE.TOOL_NAMES)
    assert engine.mcp_handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"] == -32601
    assert engine.mcp_handle({"id": 4})["error"]["code"] == -32600

    search = engine.mcp_handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                "params": {"name": "search_sop", "arguments": {"query": "flood evacuation"}}})
    assert not search["result"]["isError"] and search["result"]["structuredContent"]["results"]

    run = engine.orchestrate(incident={"resources": {"rescue_boats": 1},
                                       "zones": [{"state": "Bihar", "district": "Muzaffarpur", "inputs": {
                                           "sensors": sensors(), "text": "Flood, 40 people trapped, send boats"}}]})
    held = next(a for a in run["actions"] if a["status"] == "pending_approval")
    bypass = engine.mcp_handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
        "name": "commit_rescue_dispatch", "arguments": {"zone_id": held["zone_id"], "action_id": held["action_id"]},
        "_meta": {"run_id": run["run_id"]}}})
    assert bypass["result"]["isError"] and "approval" in bypass["result"]["structuredContent"]["error"]


def test_mcp_stdio_roundtrip(engine):
    lines = "\n".join([json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), "not json",
                       json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})]) + "\n"
    out = io.StringIO()
    engine.serve_mcp_stdio(io.StringIO(lines), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert responses[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert responses[1]["error"]["code"] == -32700 and len(responses) == 2


def test_blueprint_endpoints(engine):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(INTEGRATION.create_blueprint(engine))
    client = app.test_client()
    assert client.get("/agent").status_code == 200
    assert client.get("/api/agent/health").get_json()["status"] == "healthy"
    assert len(client.get("/api/agent/tools").get_json()["tools"]) == len(KNOWLEDGE.TOOL_NAMES)
    assert client.post("/api/agent/orchestrate", data="x", content_type="text/plain").status_code == 400
    assert client.post("/api/agent/orchestrate", json={"scenario_id": "NOPE"}).status_code == 400

    run = client.post("/api/agent/orchestrate", json={"incident": {"zones": [{"state": "Bihar", "district": "Patna",
        "inputs": {"sensors": sensors(district="Patna"), "text": "Flood, 40 people trapped", "incident_log": LOG}}]}})
    assert run.status_code == 200
    body = run.get_json()
    held = next(a for a in body["actions"] if a["status"] == "pending_approval")
    assert client.post("/api/agent/approve", json={"run_id": body["run_id"], "action_id": held["action_id"],
                                                   "decision": "approve", "operator": ""}).status_code == 400
    ok = client.post("/api/agent/approve", json={"run_id": body["run_id"], "action_id": held["action_id"],
                                                 "decision": "approve", "operator": "Officer C"})
    assert ok.status_code == 200 and ok.get_json()["action"]["status"] == "committed"
    assert client.post("/api/agent/approve", json={"run_id": "RUN-x", "action_id": "A"}).status_code == 404
    assert client.get(f"/api/agent/runs/{body['run_id']}").get_json()["commits"]
    rpc = client.post("/api/agent/mcp", json={"jsonrpc": "2.0", "id": 9, "method": "ping"})
    assert rpc.get_json()["result"] == {}


# ------------------------------------------------------ deliberation & control ----

ZONE_7_TEXT = "Roads submerged, 40 people trapped on rooftops, several injured, send ambulance immediately!!!"
ZONE_12_TEXT = "Partial flooding in low-lying streets, 15 residents moving upstairs, one person injured."


def zone_with(zone_id, district, sensor_record, **inputs):
    return zone(zone_id, district, sensors=sensor_record, **inputs)


def misreading_nlp():
    # The panicked Zone 7 message is read as HIGH, the calm Zone 12 message as CRITICAL.
    return ScriptedNLP([("rooftops", "HIGH", 40), ("low-lying", "CRITICAL", 15)])


def demand_case(order=("ZONE-7", "ZONE-12")):
    zones = {
        "ZONE-7": zone_with("ZONE-7", "Nagaon", sensors(river=7.2, threshold=5.6, district="Nagaon", calls=38), text=ZONE_7_TEXT),
        "ZONE-12": zone_with("ZONE-12", "Barpeta", sensors(river=6.4, threshold=5.6, district="Barpeta", calls=15),
                             text=ZONE_12_TEXT),
    }
    return scenario(*[zones[k] for k in order], resources={"ambulances": 1, "rescue_boats": 0, "shelter_beds": 0})


@pytest.mark.parametrize("order", [("ZONE-7", "ZONE-12"), ("ZONE-12", "ZONE-7")])
def test_one_ambulance_goes_to_higher_demand_zone_and_other_is_rerouted(kb, order):
    result = make_agent(kb, StubML("Severe", 0.99), None, misreading_nlp()).run(demand_case(order)).to_dict()
    zones = {z["zone_id"]: z for z in result["zones"]}
    assert zones["ZONE-12"]["priority"] == "CRITICAL" and zones["ZONE-7"]["priority"] == "URGENT"
    assert result["ranking"][0] == "ZONE-7"
    assert zones["ZONE-7"]["allocated"] == {"ambulances": 1} and zones["ZONE-12"]["allocated"] == {}
    assert any(m["zone_id"] == "ZONE-12" and m["resource_type"] == "ambulances" and m["reroute"]
               for m in result["mutual_aid"])
    rank_thoughts = " ".join(st["thought"] for st in result["trajectory"] if st["node"] == "RANK")
    assert "38 vs 15 emergency calls" in rank_thoughts
    negotiation = result["debate"]["negotiations"][0]
    assert negotiation["agreement"] == ["ZONE-7"] and negotiation["rerouted"] == ["ZONE-12"]
    dispatch = next(a for a in result["actions"] if a["zone_id"] == "ZONE-7")
    assert dispatch["recommended_actions"][0] == "Dispatch 1 ambulance to ZONE-7 now"
    assert result["workflow_violations"] == []


def test_plan_is_written_first_checked_and_followed(kb):
    result = make_agent(kb, StubML(), StubDL(), StubNLP()).run(scenario(critical_zone())).to_dict()
    assert result["trajectory"][0]["node"] == "PLAN"
    assert result["plan_check"]["ok"] and result["plan_adherence"]["rate"] == 1.0


def test_plan_check_flags_duplicate_and_missing_steps():
    graph = WORKFLOW.WorkflowGraph()
    check = graph.check_plan(WORKFLOW.EXAMPLE_FLAWED_PLAN, required_nodes=("ALLOCATE", "COMMIT"),
                             required_tools=("commit_rescue_dispatch",))
    assert not check["ok"] and len(check["duplicates"]) == 1 and check["missing"] == ["no step calls commit_rescue_dispatch"]
    assert graph.check_plan([{"id": "X", "subtask": "s", "node": "PERCEIVE", "tool": "notify_shelter"}])["illegal"]


def test_tree_of_thoughts_debate_blackboard_and_levels_are_recorded(kb):
    zones = [critical_zone("ZONE-1"), critical_zone("ZONE-2", district="Patna")]
    result = make_agent(kb, StubML(), StubDL(), StubNLP(headcount=150)).run(
        scenario(*zones, resources={"rescue_boats": 1, "ambulances": 1, "shelter_beds": 10})).to_dict()
    tree = result["tree_of_thoughts"]
    assert [b["branch"] for b in tree["branches"]] == list(AGENT.ResourceAllocator.BRANCHES)
    assert tree["chosen"] in AGENT.ResourceAllocator.BRANCHES
    assert {n["resource_type"] for n in result["debate"]["negotiations"]} >= {"rescue_boats", "ambulances"}
    assert {"coordinator", "dispatcher", "allocator", "auditor"} <= {b["agent"] for b in result["blackboard"]}
    levels = {e["category"]: e["level"] for e in result["escalations"]}
    assert levels["shortage"] == "state_eoc"
    assert any(e["level"] == "state_eoc" and e["action_id"] for e in result["escalations"])  # CRITICAL dispatch


def test_low_confidence_dispatch_is_held_and_escalated(kb):
    agent = make_agent(kb, StubML("Severe", 0.55), None, StubNLP("CRITICAL", confidence=0.3))
    result = agent.run(scenario(zone(sensors=sensors(), text="Flood, 40 people trapped"))).to_dict()
    z = result["zones"][0]
    assert z["decision_confidence"] < AGENT.CONFIDENCE_THRESHOLD
    dispatch = next(a for a in result["actions"] if a["type"] == "dispatch")
    assert "R10" in dispatch["audit"]["rules"] and dispatch["status"] == "pending_approval"
    assert any(e["category"] == "low_confidence" for e in result["escalations"])


def test_decision_confidence_factors():
    fused = {"agreement": "unanimous", "evidence": [
        {"source": "sensor_risk", "available": True, "confidence": 0.9},
        {"source": "text_urgency", "available": True, "confidence": 0.7},
        {"source": "forecast_trend", "available": True, "confidence": 1.0}]}
    assert AGENT.decision_confidence(fused, [], "corroborated", False) == 0.8
    disputed = {**fused, "agreement": "disputed"}
    assert AGENT.decision_confidence(disputed, ["image_path"], "text_only", True) < 0.4


def test_approval_notifies_shelter_and_override_recalls_and_halts(kb, tmp_path):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP(), tmp_path=tmp_path)
    agent_run = agent.run(scenario(critical_zone()))
    action = next(a for a in agent_run.state.actions.values() if a["type"] == "dispatch")
    forged = agent.registry.call("recall_dispatch", {"zone_id": action["zone_id"], "action_id": action["action_id"],
                                                     "override_token": "f" * 32, "reason": "forged"}, agent_run.state)
    assert not forged["ok"] and forged["error_type"] == "rejected"
    agent.decide(agent_run, action["action_id"], "approve", operator="Officer A")
    assert agent_run.to_dict()["notifications"][0]["evacuees"] == 52
    result = agent.override(agent_run, "Commander", "bridge collapsed on the route")
    assert result["halted"] and {"action_id": action["action_id"], "from": "committed", "to": "recalled"} in result["event"]["affected"]
    assert all(v["reserved"] == 0 for v in result["inventory"].values())
    blocked = agent.registry.call("commit_rescue_dispatch", {"zone_id": action["zone_id"], "action_id": action["action_id"]},
                                  agent_run.state)
    assert not blocked["ok"] and "halted" in blocked["error"]
    with pytest.raises(ValueError):
        agent.decide(agent_run, action["action_id"], "approve")
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["type"] for row in ledger] == ["commit", "recall"]
    with pytest.raises(ValueError):
        agent.override(agent_run, "Commander", "")


def test_override_of_one_pending_action_releases_units(kb):
    agent = make_agent(kb, StubML(), StubDL(), StubNLP())
    agent_run = agent.run(scenario(critical_zone()))
    action = next(a for a in agent_run.state.actions.values() if a["type"] == "dispatch")
    result = agent.override(agent_run, "Commander", "wrong zone", action_id=action["action_id"])
    assert not result["halted"] and result["event"]["affected"][0]["to"] == "cancelled"
    assert all(v["reserved"] == 0 for v in result["inventory"].values())


def test_precedents_are_recalled_from_memory(kb, tmp_path):
    memory = tmp_path / "memory.jsonl"
    registry = AGENT.build_registry(kb, {"ml": StubML(), "dl": StubDL(), "nlp": StubNLP(), "briefer": briefer},
                                    memory_path=memory)
    agent = AGENT.CoordinationAgent(registry)
    first = agent.run(scenario(critical_zone())).to_dict()
    assert not any(st.get("tool") == "recall_precedents" for st in first["trajectory"])
    second = agent.run(scenario(critical_zone())).to_dict()
    assert any(st.get("tool") == "recall_precedents" and st["ok"] for st in second["trajectory"])
    assert second["zones"][0]["precedents"][0]["district"] == "Muzaffarpur"
    assert second["workflow_violations"] == []


def test_patterns_catalogue_is_complete():
    names = {p["pattern"] for p in WORKFLOW.PATTERNS}
    assert {"ReAct", "Plan-and-Execute", "Reflexion", "Tree of Thoughts", "Memory-Augmented Agent",
            "Multi-Agent Debate", "Human-in-the-Loop", "Emergency Override", "Confidence Threshold"} <= names
    assert all(p["status"] in {"implemented", "not applicable"} for p in WORKFLOW.PATTERNS)


def test_confident_wrong_allocation_is_counted():
    truth = {"A": {"true_severity": "CRITICAL"}, "B": {"true_severity": "ELEVATED"}}
    run = {"zones": [{"zone_id": "A", "needs": {"ambulances": 1}, "allocated": {}, "decision_confidence": 0.9},
                     {"zone_id": "B", "needs": {"ambulances": 1}, "allocated": {"ambulances": 1}, "decision_confidence": 0.9}]}
    quality = EVAL.allocation_quality(truth, run)
    assert quality["priority_inversions"] == 1 and quality["confident_wrong_allocations"] == 1
    run["zones"][1]["decision_confidence"] = 0.2
    assert EVAL.allocation_quality(truth, run)["confident_wrong_allocations"] == 0


def test_override_endpoint(engine):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(INTEGRATION.create_blueprint(engine))
    client = app.test_client()
    assert client.post("/api/agent/override", json={"run_id": "RUN-x", "operator": "C", "reason": "stop"}).status_code == 404
    run = client.post("/api/agent/orchestrate", json={"incident": {"zones": [{"state": "Bihar", "district": "Patna",
        "inputs": {"sensors": sensors(district="Patna"), "text": "Flood, 40 people trapped"}}]}}).get_json()
    assert client.post("/api/agent/override", json={"run_id": run["run_id"], "operator": "C", "reason": ""}).status_code == 400
    halted = client.post("/api/agent/override", json={"run_id": run["run_id"], "operator": "Commander", "reason": "stop all"})
    assert halted.status_code == 200 and halted.get_json()["run"]["halted"]
    assert client.post("/api/agent/approve", json={"run_id": run["run_id"], "action_id": "ACT-001",
                                                   "decision": "approve", "operator": "C"}).status_code == 400

