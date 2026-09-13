"""Stage 06 Agentic AI -- Integration Engineer.

Deploys the coordination agent three ways:

  Flask blueprint   /agent (orchestration console) and /api/agent/*,
                    registered by app.py; standalone with --serve
  MCP server        every tool over the Model Context Protocol: JSON-RPC 2.0,
                    `initialize`, `tools/list`, `tools/call`, `ping`; over stdio
                    with --mcp, or over HTTP at POST /api/agent/mcp
  Python API        AgentIntegrationEngine.orchestrate() / decide()

Human-in-the-loop survives every surface: a held dispatch cannot be committed
through the console, the REST API or MCP without a human decision, because the
dispatch tool itself demands the token that only decide() issues.

Usage:
  python Stage06_AgenticAI/05_integration_engineer.py            # self-test
  python Stage06_AgenticAI/05_integration_engineer.py --serve    # console on :5006
  python Stage06_AgenticAI/05_integration_engineer.py --mcp      # MCP server on stdio
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
LEDGER_JSONL = OUTPUT_DIR / "dispatch_ledger.jsonl"
MEMORY_JSONL = OUTPUT_DIR / "agent_memory.jsonl"

logger = logging.getLogger("disaster_response.stage06")

MAX_RUNS = 50
MAX_CUSTOM_ZONES = 10
MAX_INVENTORY_UNITS = 10000
MCP_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
SERVER_INFO = {"name": "disaster-response-coordination-agent", "title": "Disaster Response Stage 06 tools",
               "version": "1.0.0"}

PROBE_INCIDENT = {
    "scenario_id": "PROBE", "name": "Health probe", "resources": {"rescue_boats": 1},
    "zones": [{"zone_id": "PROBE-1", "label": "Probe zone", "state": "Assam", "district": "Nagaon", "inputs": {}}],
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load_module("stage06_agent_engineer", BASE_DIR / "03_agent_engineer.py")
KNOWLEDGE = AGENT.KNOWLEDGE
WORKFLOW = AGENT.WORKFLOW
EVAL_REPORT_JSON = BASE_DIR / "data" / "outputs" / "evaluation" / "agent_eval_report.json"


class AgentIntegrationEngine:
    """App-facing wrapper: lazy model loading, run storage, human decisions, MCP."""

    def __init__(self, engines: dict[str, Any] | None = None, ledger_path: Path | None = LEDGER_JSONL,
                 memory_path: Path | None = MEMORY_JSONL) -> None:
        self._engines = engines
        self._kb = None
        self._agent = None
        self.ledger_path = ledger_path
        self.memory_path = memory_path
        self.stage_status: dict[str, str] = {}
        self.load_error: str | None = None
        self.runs: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

    # ---- wiring --------------------------------------------------------------

    def attach_engines(self, ml=None, dl=None, nlp=None, briefer=None) -> None:
        """Reuse adapters app.py already loaded instead of loading the models twice."""
        with self._lock:
            engines = {"ml": ml, "nlp": nlp, "dl": AGENT.wrap_dl_for_stage05(dl)}
            if briefer is None:
                try:
                    briefer = AGENT.load_briefer()
                except Exception as exc:
                    logger.warning("Stage 04 baseline briefer unavailable: %s", exc)
            engines["briefer"] = briefer
            self._engines = engines
            self.stage_status = {k: ("attached" if v is not None else "unavailable") for k, v in engines.items()}
            self._agent = None

    @property
    def kb(self):
        if self._kb is None:
            self._kb = KNOWLEDGE.KnowledgeBase()
        return self._kb

    @property
    def agent(self):
        with self._lock:
            if self._agent is None:
                if self._engines is None:
                    # Stage modules print at import; keep stdout clean for the MCP stdio transport.
                    with contextlib.redirect_stdout(sys.stderr):
                        self._engines, self.stage_status = AGENT.load_stage_engines()
                registry = AGENT.build_registry(self.kb, self._engines, ledger_path=self.ledger_path,
                                                memory_path=self.memory_path)
                self._agent = AGENT.CoordinationAgent(registry)
            return self._agent

    # ---- health --------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Knowledge base, workflow graph and a model-free probe run of the full loop."""
        try:
            kb = self.kb
            problems = WORKFLOW.WorkflowGraph().validate(KNOWLEDGE.TOOL_NAMES)
            hits = kb.search_sop("search and rescue evacuation during floods", hazard="Flood", top_k=1)
            probe_agent = AGENT.CoordinationAgent(AGENT.build_registry(kb, {}))
            probe = probe_agent.run(PROBE_INCIDENT, observable=True).to_dict()
            probe_ok = (not probe["commits"] and not probe["workflow_violations"]
                        and any(e["category"] == "verification" for e in probe["escalations"]))
        except Exception as exc:
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        status = "healthy" if (not problems and hits and probe_ok) else "degraded"
        report = self.latest_report()
        summary = None
        if report and report.get("suites"):
            first = next(iter(report["suites"].values()))
            summary = {"generated_at": report.get("generated_at"),
                       "scenario_goal_success_rate": first.get("scenario_goal_success_rate"),
                       "unsafe_commits": first.get("safety", {}).get("unsafe_commits")}
        return {
            "status": status,
            "tools": len(KNOWLEDGE.TOOL_NAMES),
            "workflow_problems": problems,
            "sop_passages": len(kb.records),
            "district_profiles": len(kb.districts),
            "probe_run": {"ok": probe_ok, "steps": len(probe["trajectory"]), "commits": len(probe["commits"])},
            "models_loaded": self._agent is not None,
            "stage_status": self.stage_status,
            "latest_evaluation": summary,
            "error": None if status == "healthy" else "health probe did not behave as expected",
        }

    # ---- scenarios -----------------------------------------------------------

    @staticmethod
    def list_scenarios(suite: str = "cvae") -> list[dict[str, Any]]:
        if suite not in {"cvae", "slm", "llm"}:
            raise ValueError("suite must be 'cvae', 'slm' or 'llm'")
        try:
            scenarios = AGENT.load_stage05_scenarios(suite)
        except FileNotFoundError:
            return []
        return [{"scenario_id": s["scenario_id"], "name": s["name"], "archetype": s.get("archetype"),
                 "prompt": s.get("prompt"), "zones": len(s["zones"]), "resources": s.get("resources")}
                for s in scenarios]

    @staticmethod
    def build_custom_incident(incident: Any) -> dict[str, Any]:
        """Validate a user-supplied incident into the agent's observable format."""
        if not isinstance(incident, dict):
            raise ValueError("'incident' must be a JSON object")
        zones = incident.get("zones")
        if not isinstance(zones, list) or not 1 <= len(zones) <= MAX_CUSTOM_ZONES:
            raise ValueError(f"'incident.zones' must be a list of 1-{MAX_CUSTOM_ZONES} zones")
        resources = incident.get("resources") or {}
        if not isinstance(resources, dict):
            raise ValueError("'incident.resources' must be an object")
        clean_resources = {}
        for key, value in resources.items():
            if key not in KNOWLEDGE.RESOURCE_TYPES:
                raise ValueError(f"unknown resource type {key!r}")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_INVENTORY_UNITS:
                raise ValueError(f"resource {key} must be an integer 0-{MAX_INVENTORY_UNITS}")
            clean_resources[key] = value
        clean_zones = []
        for i, zone in enumerate(zones, start=1):
            if not isinstance(zone, dict):
                raise ValueError("each zone must be an object")
            inputs = zone.get("inputs") or {}
            if not isinstance(inputs, dict):
                raise ValueError("zone 'inputs' must be an object")
            unknown = set(inputs) - set(AGENT.OBSERVABLE_INPUT_KEYS)
            if unknown:
                raise ValueError(f"unknown zone inputs: {sorted(unknown)}")
            for key in ("text", "incident_log", "image_path"):
                if key in inputs and inputs[key] is not None and not isinstance(inputs[key], str):
                    raise ValueError(f"'{key}' must be a string")
            if "sensors" in inputs and inputs["sensors"] is not None and not isinstance(inputs["sensors"], dict):
                raise ValueError("'sensors' must be an object")
            if "water_levels" in inputs and inputs["water_levels"] is not None and not isinstance(inputs["water_levels"], list):
                raise ValueError("'water_levels' must be a list of numbers")
            clean_zones.append({"zone_id": str(zone.get("zone_id") or f"ZONE-{i}")[:40],
                                "label": str(zone.get("label") or f"Zone {i}")[:80],
                                "state": str(zone.get("state") or "")[:60] or None,
                                "district": str(zone.get("district") or "")[:60] or None,
                                "location": None, "inputs": inputs})
        return AGENT.observable_scenario({"scenario_id": "CUSTOM", "name": str(incident.get("name") or "Custom incident")[:120],
                                          "resources": clean_resources, "zones": clean_zones})

    # ---- orchestration -------------------------------------------------------

    def orchestrate(self, scenario_id: str | None = None, suite: str = "cvae",
                    incident: dict[str, Any] | None = None) -> dict[str, Any]:
        if incident is not None:
            scenario = self.build_custom_incident(incident)
        else:
            if not isinstance(scenario_id, str):
                raise ValueError("Provide 'scenario_id' (with optional 'suite') or 'incident'")
            if suite not in {"cvae", "slm", "llm"}:
                raise ValueError("suite must be 'cvae', 'slm' or 'llm'")
            try:
                scenarios = AGENT.load_stage05_scenarios(suite)
            except FileNotFoundError as exc:
                raise ValueError(f"No Stage 05 scenarios for suite {suite!r}") from exc
            scenario = next((s for s in scenarios if s["scenario_id"] == scenario_id), None)
            if scenario is None:
                raise ValueError(f"Unknown scenario_id {scenario_id!r}")
            scenario = AGENT.observable_scenario(scenario)
        with self._lock:
            agent_run = self.agent.run(scenario, observable=True)
            self.runs[agent_run.run_id] = agent_run
            while len(self.runs) > MAX_RUNS:
                self.runs.popitem(last=False)
            return agent_run.to_dict()

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            return self.runs[run_id].to_dict()

    def decide(self, run_id: str, action_id: str, decision: str, operator: str = "operator",
               note: str = "") -> dict[str, Any]:
        with self._lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            if not isinstance(operator, str) or not operator.strip():
                raise ValueError("'operator' must name the human making the decision")
            result = self.agent.decide(self.runs[run_id], action_id, decision, operator.strip()[:80],
                                       str(note or "")[:500])
            result["run"] = self.runs[run_id].to_dict()
            return result

    def override(self, run_id: str, operator: str, reason: str, action_id: str | None = None) -> dict[str, Any]:
        """Emergency override: recall one action, or halt the whole run."""
        with self._lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            if action_id is not None and not isinstance(action_id, str):
                raise ValueError("'action_id' must be a string")
            result = self.agent.override(self.runs[run_id], operator, reason, action_id)
            result["run"] = self.runs[run_id].to_dict()
            return result

    @staticmethod
    def latest_report() -> dict[str, Any] | None:
        if not EVAL_REPORT_JSON.exists():
            return None
        return json.loads(EVAL_REPORT_JSON.read_text(encoding="utf-8"))

    @staticmethod
    def workflow() -> dict[str, Any]:
        graph = WORKFLOW.WorkflowGraph()
        return {"goal": WORKFLOW.GOAL, "agents": WORKFLOW.AGENTS, "nodes": graph.nodes, "edges": graph.edges,
                "mermaid": graph.to_mermaid()}

    # ---- MCP -----------------------------------------------------------------

    def mcp_handle(self, message: Any) -> dict[str, Any] | None:
        """One JSON-RPC 2.0 message in, one response (or None for a notification) out."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _rpc_error(message.get("id") if isinstance(message, dict) else None, -32600, "Invalid Request")
        method, params, msg_id = message["method"], message.get("params") or {}, message.get("id")
        if "id" not in message:
            return None  # notification, e.g. notifications/initialized
        if not isinstance(params, dict):
            return _rpc_error(msg_id, -32602, "params must be an object")
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0]
            return _rpc_result(msg_id, {
                "protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": ("Flood-response coordination tools backed by the Stage 01-04 models, the fusion "
                                 "layer and an SOP knowledge base. Run-state tools act on the run named in "
                                 "params._meta.run_id (default: the latest run). commit_rescue_dispatch refuses "
                                 "held actions without a human approval token.")})
        if method == "ping":
            return _rpc_result(msg_id, {})
        if method == "tools/list":
            return _rpc_result(msg_id, {"tools": self.agent.registry.list_tools()})
        if method == "tools/call":
            name, arguments = params.get("name"), params.get("arguments") or {}
            if not isinstance(name, str):
                return _rpc_error(msg_id, -32602, "tools/call needs a string 'name'")
            if name not in self.agent.registry.definitions:
                return _rpc_error(msg_id, -32602, f"Unknown tool: {name}")
            meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
            with self._lock:
                run_id = meta.get("run_id") or (next(reversed(self.runs)) if self.runs else None)
                state = self.runs[run_id].state if run_id in self.runs else None
                outcome = self.agent.registry.call(name, arguments, state)
            payload = outcome["data"] if outcome["ok"] else {"error": outcome["error"], "error_type": outcome["error_type"]}
            result = {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": not outcome["ok"]}
            if isinstance(payload, dict):
                result["structuredContent"] = payload
            return _rpc_result(msg_id, result)
        return _rpc_error(msg_id, -32601, f"Method not found: {method}")

    def serve_mcp_stdio(self, stdin=None, stdout=None) -> None:
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response = _rpc_error(None, -32700, "Parse error")
            else:
                try:
                    response = self.mcp_handle(message)
                except Exception as exc:  # never let one bad call kill the server
                    logger.exception("MCP request failed")
                    response = _rpc_error(message.get("id") if isinstance(message, dict) else None,
                                          -32603, f"Internal error: {type(exc).__name__}")
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()

    # ---- UI ------------------------------------------------------------------

    @staticmethod
    def render_console() -> str:
        return CONSOLE_HTML


def _rpc_result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


agent_integration_engine = AgentIntegrationEngine()


# ===========================================================================
# Flask blueprint
# ===========================================================================

def create_blueprint(engine: AgentIntegrationEngine | None = None):
    from flask import Blueprint, Response, jsonify, request

    engine = engine or agent_integration_engine
    blueprint = Blueprint("stage06_agentic", __name__)

    def fail(message: str, status_code: int = 400, exc: Exception | None = None):
        if exc is not None:
            logger.exception("Stage 06 request failed: %s", message)
        return jsonify({"error": message}), status_code

    def body() -> dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    @blueprint.route("/agent")
    def console():
        return Response(engine.render_console(), mimetype="text/html")

    @blueprint.route("/api/agent/health")
    def health():
        return jsonify(engine.health_check())

    @blueprint.route("/api/agent/scenarios")
    def scenarios():
        try:
            return jsonify(engine.list_scenarios(request.args.get("suite", "cvae")))
        except ValueError as exc:
            return fail(str(exc), 400)

    @blueprint.route("/api/agent/tools")
    def tools():
        return jsonify({"tools": KNOWLEDGE.TOOL_DEFINITIONS})

    @blueprint.route("/api/agent/workflow")
    def workflow():
        return jsonify(engine.workflow())

    @blueprint.route("/api/agent/report")
    def report():
        latest = engine.latest_report()
        if latest is None:
            return fail("No agent evaluation yet. Run 04_evaluation_engineer.py.", 404)
        return jsonify(latest)

    @blueprint.route("/api/agent/orchestrate", methods=["POST"])
    def orchestrate():
        try:
            payload = body()
            return jsonify(engine.orchestrate(scenario_id=payload.get("scenario_id"),
                                              suite=payload.get("suite", "cvae"),
                                              incident=payload.get("incident")))
        except ValueError as exc:
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Orchestration failed. See server logs for details.", 500, exc)

    @blueprint.route("/api/agent/runs/<run_id>")
    def run(run_id):
        try:
            return jsonify(engine.get_run(run_id))
        except KeyError:
            return fail("Unknown run_id", 404)

    @blueprint.route("/api/agent/approve", methods=["POST"])
    def approve():
        try:
            payload = body()
            return jsonify(engine.decide(str(payload.get("run_id")), str(payload.get("action_id")),
                                         payload.get("decision", "approve"), payload.get("operator", ""),
                                         payload.get("note", "")))
        except KeyError:
            return fail("Unknown run_id", 404)
        except ValueError as exc:
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Decision failed. See server logs for details.", 500, exc)

    @blueprint.route("/api/agent/override", methods=["POST"])
    def override():
        try:
            payload = body()
            return jsonify(engine.override(str(payload.get("run_id")), payload.get("operator", ""),
                                           payload.get("reason", ""), payload.get("action_id")))
        except KeyError:
            return fail("Unknown run_id", 404)
        except ValueError as exc:
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Override failed. See server logs for details.", 500, exc)

    @blueprint.route("/api/agent/mcp", methods=["POST"])
    def mcp():
        try:
            response = engine.mcp_handle(request.get_json(silent=True))
        except Exception as exc:
            logger.exception("MCP over HTTP failed")
            return jsonify(_rpc_error(None, -32603, f"Internal error: {type(exc).__name__}"))
        if response is None:
            return Response(status=202)
        return jsonify(response)

    return blueprint


# ===========================================================================
# Orchestration console
# ===========================================================================

CONSOLE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 06 Agent Console</title>
<style>
:root{--bg:#0F172A;--panel:#1E293B;--line:#334155;--text:#F8FAFC;--muted:#94A3B8;--blue:#3B82F6;--green:#10B981;
--amber:#F59E0B;--orange:#F97316;--red:#EF4444;--purple:#8B5CF6;--pink:#EC4899}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0}header .sub{color:var(--muted);font-size:13px}
.chips{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}.chip{padding:3px 10px;border-radius:999px;background:var(--panel);border:1px solid var(--line);font-size:12px;color:var(--muted)}
.chip.ok{color:var(--green);border-color:rgba(16,185,129,.4)}.chip.bad{color:var(--red);border-color:rgba(239,68,68,.4)}
main{display:grid;grid-template-columns:320px 1fr;gap:16px;padding:16px 24px}@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px;min-width:0}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
label{display:block;font-size:12px;color:var(--muted);margin:8px 0 4px}
select,input,textarea{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 8px;font:inherit}
textarea{min-height:90px;resize:vertical}
button{background:var(--blue);color:#fff;border:0;border-radius:6px;padding:8px 12px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}button.ghost{background:transparent;border:1px solid var(--line);color:var(--text)}
button.approve{background:var(--green)}button.reject{background:var(--red)}
.full{width:100%;margin-top:10px}.row{display:flex;gap:8px}.row>*{flex:1}
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}.tabs button{background:var(--panel);border:1px solid var(--line);color:var(--muted)}
.tabs button.active{color:var(--text);border-color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.card .v{font-size:22px;font-weight:700}.card .k{font-size:12px;color:var(--muted)}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:700;color:#0F172A}
.note{color:var(--muted);font-size:12px}.warn{color:var(--amber)}.err{color:var(--red)}
.step{display:grid;grid-template-columns:44px 120px 1fr;gap:8px;padding:7px 0;border-bottom:1px solid var(--line)}
.step .n{color:var(--muted);font-variant-numeric:tabular-nums}.node{font-size:11px;font-weight:700;letter-spacing:.03em}
.tool{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--purple)}.obs{font-size:12px;color:var(--muted)}
.approval{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.empty{color:var(--muted);padding:24px;text-align:center}
details summary{cursor:pointer;color:var(--blue);font-size:12px}
.disclaimer{font-size:12px;color:var(--amber);margin-bottom:12px}
.cc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:10px}
.cc-card{border:1px solid var(--line);border-radius:8px;padding:12px;background:var(--bg)}
.cc-head{font-weight:700;margin-bottom:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cc-conf{font-size:13px;margin-bottom:6px}.cc-list{margin:4px 0 0 18px;font-size:13px}
.thought{margin:6px 0;font-size:13px}
</style></head><body>
<header><div><h1>Stage 06 · Agentic Coordination Console</h1>
<div class="sub">Tactical Dispatcher · Resource Allocator · Safety Auditor, orchestrated over the Stage 01–04 models, fusion, and an SOP knowledge base</div></div>
<div class="chips" id="chips"><span class="chip">checking health…</span></div></header>
<main>
<aside>
  <div class="panel"><h2>Run a Stage 05 incident</h2>
    <label for="suite">Generator suite</label>
    <select id="suite"><option value="cvae">CVAE suite</option><option value="slm">Domain SLM suite</option><option value="llm">LLM suite</option></select>
    <label for="scenario">Scenario</label><select id="scenario"></select>
    <div class="note" id="scenario-note" style="margin-top:6px"></div>
    <button class="full" id="run-btn">Run agent</button>
  </div>
  <div class="panel"><h2>Or a custom incident</h2>
    <div class="row"><div><label for="c-state">State</label><input id="c-state" value="Assam"></div>
    <div><label for="c-district">District</label><input id="c-district" value="Nagaon"></div></div>
    <label for="c-text">Emergency report</label>
    <textarea id="c-text">Water entering houses near the river bank in Nagaon. 40 people on rooftops, two injured, please send boats and ambulance.</textarea>
    <div class="row"><div><label for="c-boats">Boats</label><input id="c-boats" type="number" min="0" value="2"></div>
    <div><label for="c-amb">Ambulances</label><input id="c-amb" type="number" min="0" value="2"></div>
    <div><label for="c-beds">Beds</label><input id="c-beds" type="number" min="0" value="100"></div></div>
    <button class="full ghost" id="custom-btn">Run on custom incident</button>
  </div>
  <div class="panel"><h2>Commander</h2><label for="operator">Name recorded on approvals and overrides</label><input id="operator" value="Duty officer">
    <label for="override-reason">Override reason</label><input id="override-reason" value="Commander override from the command center"></div>
</aside>
<section style="min-width:0">
  <div class="tabs" id="tabs">
    <button data-tab="overview" class="active">Overview</button><button data-tab="approvals">Approvals</button>
    <button data-tab="reasoning">Reasoning</button><button data-tab="trajectory">Trajectory</button><button data-tab="evaluation">Evaluation</button><button data-tab="tools">Tools (MCP)</button>
  </div>
  <div id="error" class="panel err" hidden></div>
  <div data-view="overview"><div id="overview"><div class="empty">Run the agent on an incident to see its plan.</div></div></div>
  <div data-view="approvals" hidden><div id="approvals"><div class="empty">No run yet.</div></div></div>
  <div data-view="reasoning" hidden><div id="reasoning"><div class="empty">No run yet.</div></div></div>
  <div data-view="trajectory" hidden><div class="panel"><div id="trajectory"><div class="empty">No run yet.</div></div></div></div>
  <div data-view="evaluation" hidden><div id="evaluation"><div class="empty">Loading…</div></div></div>
  <div data-view="tools" hidden><div class="panel"><div id="tools"><div class="empty">Loading…</div></div></div></div>
</section>
</main>
<script>
const COLORS={ROUTINE:'#10B981',ELEVATED:'#F59E0B',URGENT:'#F97316',CRITICAL:'#EF4444'};
const LOOP={perceive:'#3B82F6',reason:'#8B5CF6',plan:'#F59E0B',act:'#EC4899',reflect:'#10B981','-':'#94A3B8'};
let RUN=null;
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge=p=>p?`<span class="badge" style="background:${COLORS[p]||'#94A3B8'}">${esc(p)}</span>`:'<span class="badge" style="background:#94A3B8">UNSCORED</span>';
const fmt=v=>v===null||v===undefined?'n/a':(typeof v==='number'&&!Number.isInteger(v)?v.toFixed(3):esc(v));
async function api(path,opts){const r=await fetch(path,opts);const d=await r.json().catch(()=>({error:'Invalid response'}));if(!r.ok)throw new Error(d.error||r.statusText);return d;}
function showError(msg){const e=$('error');e.hidden=!msg;e.textContent=msg||'';}

document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('active',x===b));
  document.querySelectorAll('[data-view]').forEach(v=>v.hidden=v.dataset.view!==b.dataset.tab);
});

async function loadHealth(){
  try{const h=await api('/api/agent/health');
    const chips=[`<span class="chip ${h.status==='healthy'?'ok':'bad'}">agent ${esc(h.status)}</span>`,
      `<span class="chip">${esc(h.tools)} tools</span>`,`<span class="chip">${esc(h.sop_passages)} SOP passages</span>`,
      `<span class="chip">${h.models_loaded?'models loaded':'models load on first run'}</span>`];
    $('chips').innerHTML=chips.join('');
  }catch(e){$('chips').innerHTML=`<span class="chip bad">health: ${esc(e.message)}</span>`;}
}
async function loadScenarios(){
  const list=await api('/api/agent/scenarios?suite='+encodeURIComponent($('suite').value)).catch(()=>[]);
  $('scenario').innerHTML=list.map(s=>`<option value="${esc(s.scenario_id)}">${esc(s.scenario_id)} · ${esc(s.name)} (${s.zones} zones)</option>`).join('');
  const note=()=>{const s=list.find(x=>x.scenario_id===$('scenario').value);$('scenario-note').textContent=s?(s.prompt||''):'No scenarios for this suite.';};
  $('scenario').onchange=note;note();
}
$('suite').onchange=loadScenarios;

async function run(payload,btn){
  btn.disabled=true;const label=btn.textContent;btn.textContent='Agent working…';showError('');
  try{RUN=await api('/api/agent/orchestrate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});render();loadHealth();}
  catch(e){showError(e.message);}finally{btn.disabled=false;btn.textContent=label;}
}
$('run-btn').onclick=()=>run({suite:$('suite').value,scenario_id:$('scenario').value},$('run-btn'));
$('custom-btn').onclick=()=>run({incident:{name:'Custom incident',
  resources:{rescue_boats:+$('c-boats').value||0,ambulances:+$('c-amb').value||0,shelter_beds:+$('c-beds').value||0},
  zones:[{zone_id:'ZONE-1',label:'Reported zone',state:$('c-state').value,district:$('c-district').value,inputs:{text:$('c-text').value}}]}},$('custom-btn'));

function render(){renderOverview();renderApprovals();renderReasoning();renderTrajectory();}

function renderOverview(){
  const r=RUN,pending=r.actions.filter(a=>a.status==='pending_approval').length;
  const cards=[['Zones',r.zones.length],['Awaiting approval',pending],['Committed',r.commits.length],['Mutual-aid requests',r.mutual_aid.length],
    ['Tool calls',r.tool_stats.calls],['Workflow violations',r.workflow_violations.length],['Latency',Math.round(r.latency_ms)+' ms'],
    ['Confidence threshold',Math.round(r.config.confidence_threshold*100)+'%']];
  const inv=Object.entries(r.inventory_after).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v.total}</td><td>${v.reserved}</td><td>${v.remaining}</td></tr>`).join('');
  const zones=r.zones.map(z=>{
    const alloc=Object.entries(z.allocated).map(([k,v])=>`${v} ${esc(k)}`).join(', ')||'—';
    const needs=Object.entries(z.needs||{}).map(([k,v])=>`${v} ${esc(k)}`).join(', ')||'—';
    const conflicts=[...(z.fusion_conflicts||[]),...(z.agent_conflicts||[])];
    const sop=(z.sop||[]).map(s=>`<div class="note">📘 <b>${esc(s.source)}</b> [${esc(s.action_type)}] ${esc(s.excerpt)}</div>`).join('');
    const brief=z.briefing?`<div class="note">📋 Stage 04: ${esc(z.briefing.priority)} — ${esc(z.briefing.situation)}</div>`:'';
    return `<tr><td><b>${esc(z.zone_id)}</b><div class="note">${esc(z.label)} · ${esc(z.district||'')}${z.state?', '+esc(z.state):''}</div></td>
      <td>${badge(z.priority)}<div class="note">${esc(z.agreement||z.status)}</div></td>
      <td>${z.headcount_estimate||0}<div class="note">${esc(z.headcount_source||'')}</div></td>
      <td>${needs}</td><td>${alloc}</td>
      <td>${conflicts.map(c=>`<div class="warn">⚠ ${esc(c)}</div>`).join('')}${(z.evidence_lost||[]).length?`<div class="err">lost: ${esc(z.evidence_lost.join(', '))}</div>`:''}
      ${(sop||brief)?`<details><summary>SOPs & briefing</summary>${brief}${sop}</details>`:''}</td></tr>`;}).join('');
  const unserved=r.unserved_high_priority.length?`<div class="err">Unserved URGENT+ zones: ${esc(r.unserved_high_priority.join(', '))}</div>`:'';
  $('overview').innerHTML=`<div class="disclaimer">${esc(r.disclaimer)}</div>${commandCenter(r)}
    <div class="cards">${cards.map(([k,v])=>`<div class="card"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join('')}</div>
    <div class="panel"><h2>${esc(r.scenario.scenario_id)} · ${esc(r.scenario.name)}</h2>${unserved}
    <div class="note">Ranking: ${esc(r.ranking.join(' › ')||'none')} · planner ${esc(r.config.planner)} · allocation ${esc(r.config.allocation_policy)}</div>
    <div class="scroll"><table><thead><tr><th>Zone</th><th>Priority</th><th>Headcount</th><th>Needs</th><th>Allocated</th><th>Notes</th></tr></thead><tbody>${zones}</tbody></table></div></div>
    <div class="panel"><h2>Inventory</h2><div class="scroll"><table><thead><tr><th>Resource</th><th>Total</th><th>Reserved</th><th>Remaining</th></tr></thead><tbody>${inv}</tbody></table></div></div>`;
  bindDecisionButtons();
}

function renderApprovals(){
  const r=RUN;
  const actions=r.actions.map(a=>{
    const res=Object.entries(a.resources||{}).map(([k,v])=>`${v} ${esc(k)}`).join(', ');
    const buttons=RUN.halted?'':(a.status==='pending_approval'?`<div class="row" style="margin-top:8px;max-width:320px"><button class="approve" data-act="${esc(a.action_id)}" data-d="approve">Approve</button><button class="reject" data-act="${esc(a.action_id)}" data-d="reject">Reject</button></div>`
      :(a.status==='committed'&&a.type==='dispatch'?`<div class="row" style="margin-top:8px;max-width:200px"><button class="reject" data-act="${esc(a.action_id)}" data-d="recall">Recall</button></div>`:''));
    return `<div class="approval"><b>${esc(a.action_id)}</b> · ${esc(a.type)} · ${esc(a.zone_id)} · <b>${esc(a.status)}</b>
      <div>${esc(a.summary)}</div>${res?`<div class="note">Reserved: ${res}</div>`:''}
      <div class="note">Auditor: ${esc(a.audit.verdict)} — ${esc(a.audit.reasons.join('; '))}</div>
      ${a.approved_by?`<div class="note">Approved by ${esc(a.approved_by)}</div>`:''}${buttons}</div>`;}).join('')||'<div class="empty">No actions proposed.</div>';
  const list=(items,f)=>items.length?items.map(f).join(''):'<div class="note">None</div>';
  $('approvals').innerHTML=`<div class="panel"><h2>Actions & sign-off</h2>${actions}</div>
    <div class="panel"><h2>Escalations</h2>${list(r.escalations,e=>`<div class="approval"><b>${esc(e.escalation_id)}</b> · ${esc(e.category)} → ${esc(String(e.level||'').replace('_',' '))} · ${esc(e.zone_id)} · ${esc(e.status)}<div class="note">${esc(e.reason)}</div></div>`)}</div>
    <div class="panel"><h2>Mutual aid</h2>${list(r.mutual_aid,m=>`<div class="note">${esc(m.request_id)} · ${esc(m.zone_id)} short ${m.shortfall} ${esc(m.resource_type)} → ${esc(m.addressed_to)}</div>`)}</div>
    <div class="panel"><h2>Committed dispatches</h2>${list(r.commits,c=>`<div class="note">${esc(c.commit_id)} · ${esc(c.zone_id)} · ${esc(JSON.stringify(c.resources))} · approved by ${esc(c.approved_by||'auto')}</div>`)}</div>
    <div class="panel"><h2>Shelter notifications</h2>${list(r.notifications,n=>`<div class="note">${esc(n.notification_id)} · ${esc(n.message)}</div>`)}</div>
    <div class="panel"><h2>Overrides & recalls</h2>${list(r.recalls,x=>`<div class="note">${esc(x.recall_id)} · ${esc(x.action_id)} recalled: ${esc(x.reason)}</div>`)}</div>`;
  bindDecisionButtons();
}

async function postJson(path,payload){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});}

function bindDecisionButtons(){
  document.querySelectorAll('[data-act]').forEach(b=>b.onclick=async()=>{
    b.disabled=true;showError('');
    try{
      const out=b.dataset.d==='recall'
        ?await postJson('/api/agent/override',{run_id:RUN.run_id,action_id:b.dataset.act,operator:$('operator').value,reason:$('override-reason').value})
        :await postJson('/api/agent/approve',{run_id:RUN.run_id,action_id:b.dataset.act,decision:b.dataset.d,operator:$('operator').value});
      RUN=out.run;render();
    }catch(e){showError(e.message);b.disabled=false;}
  });
  document.querySelectorAll('[data-stop]').forEach(b=>b.onclick=async()=>{
    if(!confirm('Emergency stop: cancel every pending action and recall every committed dispatch in this run?'))return;
    b.disabled=true;showError('');
    try{const out=await postJson('/api/agent/override',{run_id:RUN.run_id,operator:$('operator').value,reason:$('override-reason').value});RUN=out.run;render();}
    catch(e){showError(e.message);b.disabled=false;}
  });
}

function commandCenter(r){
  const zones=Object.fromEntries(r.zones.map(z=>[z.zone_id,z]));
  const items=r.actions.filter(a=>a.type==='dispatch'||a.type==='verification');
  const stop=r.halted?'<span class="err" style="font-weight:700">⛔ Run halted by emergency override</span>'
    :'<button class="reject" data-stop="1">⛔ EMERGENCY STOP</button>';
  const cards=items.map(a=>{
    const z=zones[a.zone_id]||{},conf=a.decision_confidence;
    const low=conf!=null&&conf<r.config.confidence_threshold;
    const btns=r.halted?'':(a.status==='pending_approval'
      ?`<div class="row" style="margin-top:10px;max-width:340px"><button class="approve" data-act="${esc(a.action_id)}" data-d="approve">APPROVE</button><button class="reject" data-act="${esc(a.action_id)}" data-d="reject">OVERRIDE</button></div>`
      :(a.status==='committed'&&a.type==='dispatch'?`<div class="row" style="margin-top:10px;max-width:200px"><button class="reject" data-act="${esc(a.action_id)}" data-d="recall">RECALL</button></div>`:''));
    return `<div class="cc-card"><div class="cc-head">${esc(a.zone_id)} ${badge(z.priority)} <span class="note">${esc(String(a.status).replace('_',' '))}</span></div>
      <div class="cc-conf">Agent reasoning · confidence <b>${conf==null?'n/a':Math.round(conf*100)+'%'}</b>${low?' <span class="warn">below threshold, escalated</span>':''}</div>
      <div class="note">Recommended action:</div><ul class="cc-list">${(a.recommended_actions||[a.summary]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>${btns}</div>`;
  }).join('')||'<div class="note">No dispatch decisions in this run.</div>';
  return `<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap"><h2 style="margin:0">🖥️ Command Center</h2>${stop}</div><div class="cc-grid">${cards}</div></div>`;
}

function renderReasoning(){
  const r=RUN,by=n=>r.trajectory.filter(s=>s.node===n);
  const thoughts=(n,empty)=>by(n).map(s=>`<p class="thought">${esc(s.thought)}</p>`).join('')||`<div class="note">${esc(empty)}</div>`;
  const adh=r.plan_adherence||{};
  const plan=`<div class="panel"><h2>Plan-and-execute</h2><div class="note">Plan check: ${esc(r.plan_check.summary)} · adherence ${adh.rate==null?'n/a':Math.round(adh.rate*100)+'%'}${(adh.replanned||[]).length?' · replanned: '+esc(adh.replanned.join('; ')):''}</div>
    <ol class="cc-list">${r.plan.map(p=>`<li><b>${esc(p.node)}</b> ${esc(p.subtask)}${p.zone_id?' · '+esc(p.zone_id):''}${p.tool?' · <span class="tool">'+esc(p.tool)+'</span>':''}${p.conditional?' <span class="note">(if needed)</span>':''}</li>`).join('')}</ol></div>`;
  const tree=r.tree_of_thoughts||{};
  const tot=`<div class="panel"><h2>Tree of thoughts</h2><div class="scroll"><table><thead><tr><th>Branch</th><th>Utility</th><th>Shortfalls</th></tr></thead><tbody>${(tree.branches||[]).map(b=>`<tr><td>${esc(b.branch)}${b.branch===tree.chosen?' ✅':''}</td><td>${esc(b.utility)}</td><td>${esc(b.shortfalls)}</td></tr>`).join('')||'<tr><td colspan="3" class="note">Not used (first-come-first-served ablation)</td></tr>'}</tbody></table></div></div>`;
  const precedents=r.zones.filter(z=>(z.precedents||[]).length).map(z=>`<div class="note">${esc(z.zone_id)}: ${z.precedents.length} precedent(s) recalled for ${esc(z.district)}</div>`).join('')||'<div class="note">No precedents in memory for these districts yet.</div>';
  const board=`<div class="panel"><h2>Blackboard (shared workspace)</h2><div class="scroll"><table><thead><tr><th>#</th><th>Agent</th><th>Topic</th><th>Content</th></tr></thead><tbody>${(r.blackboard||[]).map(b=>`<tr><td>${b.seq}</td><td>${esc(b.agent)}</td><td>${esc(b.topic)}</td><td class="note">${esc(JSON.stringify(b.content).slice(0,300))}</td></tr>`).join('')}</tbody></table></div></div>`;
  $('reasoning').innerHTML=plan
    +`<div class="panel"><h2>Trade-off reasoning (ReAct)</h2>${thoughts('RANK','No ranking step.')}</div>`
    +tot
    +`<div class="panel"><h2>Debate & negotiation</h2>${thoughts('DEBATE','No debate in this run.')}</div>`
    +`<div class="panel"><h2>Reflexion</h2>${thoughts('REFLECT','No reflection step.')}<div class="note">Revision ${r.reflexion&&r.reflexion.triggered?'applied':'not needed'}.</div></div>`
    +`<div class="panel"><h2>Memory</h2>${precedents}</div>`+board;
}

function renderTrajectory(){
  const steps=[...RUN.trajectory,...RUN.human_events.flatMap(e=>e.steps.map(s=>({...s,human:true})))];
  $('trajectory').innerHTML=(RUN.workflow_violations.length?`<div class="err">Workflow violations: ${esc(RUN.workflow_violations.join('; '))}</div>`:'<div class="note" style="margin-bottom:8px">Every step conformed to the workflow graph.</div>')+
    steps.map(s=>`<div class="step"><div class="n">${s.human?'👤':s.step}</div>
      <div><div class="node" style="color:${LOOP[s.loop]||'#94A3B8'}">${esc(s.node)}</div><div class="note">${esc(s.agent)}</div></div>
      <div><div>${esc(s.thought)}</div>${s.tool?`<div class="tool">${esc(s.tool)}(${esc(JSON.stringify(s.arguments))})</div>
      <div class="obs ${s.ok?'':'err'}">→ ${esc(s.observation)} <span class="note">${s.latency_ms} ms</span></div>`:''}</div></div>`).join('');
}

async function loadEvaluation(){
  try{const r=await api('/api/agent/report');
    const suites=Object.keys(r.suites);const variants={agent:r.suites[suites[0]],...Object.fromEntries(Object.entries(r.ablations).filter(([,v])=>!v.skipped)),...r.faults};
    const rows=[['Scenario goal success',s=>s.scenario_goal_success_rate],['Zone goal success',s=>s.zone_goal_success_rate],
      ['Priority exact accuracy',s=>s.reasoning.priority_exact_accuracy],['Critical misses',s=>s.reasoning.critical_misses],
      ['Tool precision',s=>s.tool_use.precision],['Tool recall',s=>s.tool_use.recall],['Workflow violations',s=>s.tool_use.workflow_violations],
      ['Unsafe commits',s=>s.safety.unsafe_commits],['High-severity need coverage',s=>s.allocation.high_severity_need_coverage],['Priority inversions',s=>s.allocation.priority_inversions]];
    const table=(cols)=>`<div class="scroll"><table><thead><tr><th>Metric</th>${Object.keys(cols).map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>
      ${rows.map(([n,f])=>`<tr><td>${esc(n)}</td>${Object.values(cols).map(v=>`<td>${fmt(f(v))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    const hitl=r.suites[suites[0]].hitl_probe||{};
    $('evaluation').innerHTML=`<div class="panel"><h2>By generator suite</h2><div class="note">Generated ${esc(r.generated_at)}</div>${table(r.suites)}</div>
      <div class="panel"><h2>Ablations & fault injection (${esc(suites[0])})</h2>${table(variants)}</div>
      <div class="panel"><h2>Human-in-the-loop probe</h2><div>${hitl.bypass_blocked} of ${hitl.bypass_attempts} token-less or forged commit attempts blocked; ${hitl.approved_and_committed} of ${hitl.held_dispatches} held dispatches committed after simulated approval.</div></div>`;
  }catch(e){$('evaluation').innerHTML=`<div class="empty">${esc(e.message)}</div>`;}
}
async function loadTools(){
  try{const t=await api('/api/agent/tools');
    $('tools').innerHTML=`<div class="note" style="margin-bottom:8px">Also served over MCP: <code>POST /api/agent/mcp</code> (JSON-RPC) or <code>python Stage06_AgenticAI/05_integration_engineer.py --mcp</code> (stdio).</div>
      <div class="scroll"><table><thead><tr><th>Tool</th><th>Owner</th><th>Backed by</th><th>Description</th></tr></thead><tbody>
      ${t.tools.map(x=>`<tr><td class="tool">${esc(x.name)}${x._meta.requires_human_approval?' 🔒':''}</td><td>${esc(x._meta.agent)}</td><td>${esc(x._meta.backed_by)}</td><td>${esc(x.description)}</td></tr>`).join('')}</tbody></table></div>`;
  }catch(e){$('tools').innerHTML=`<div class="empty">${esc(e.message)}</div>`;}
}
loadHealth();loadScenarios();loadEvaluation();loadTools();
</script></body></html>
"""


def serve(port: int) -> None:
    from flask import Flask, redirect

    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

    @app.route("/")
    def index():
        return redirect("/agent")

    print(f"Stage 06 agent console on http://127.0.0.1:{port}/agent")
    app.run(port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 06 integration: console, REST API, MCP server")
    parser.add_argument("--serve", action="store_true", help="run the orchestration console")
    parser.add_argument("--mcp", action="store_true", help="run as an MCP server over stdio")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    if args.mcp:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        agent_integration_engine.serve_mcp_stdio()
        return
    if args.serve:
        serve(args.port)
        return

    print("=" * 60)
    print("Stage 06 Agent Integration Engine -- Self-Test")
    print("=" * 60)
    health = agent_integration_engine.health_check()
    for key, value in health.items():
        print(f"  {key:<20}: {value}")
    run = agent_integration_engine.orchestrate(scenario_id="S01", suite="cvae")
    print(f"\n  S01: {len(run['zones'])} zones, {run['tool_stats']['calls']} tool calls, "
          f"{run['pending_approvals']} awaiting approval, violations {len(run['workflow_violations'])}")
    held = next((a for a in run["actions"] if a["status"] == "pending_approval" and a["type"] == "dispatch"), None)
    if held:
        bypass = agent_integration_engine.mcp_handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "commit_rescue_dispatch", "arguments": {"zone_id": held["zone_id"], "action_id": held["action_id"]},
                       "_meta": {"run_id": run["run_id"]}}})
        print(f"  MCP commit without approval -> isError {bypass['result']['isError']}: "
              f"{bypass['result']['structuredContent']['error']}")
        decided = agent_integration_engine.decide(run["run_id"], held["action_id"], "approve", operator="self-test")
        print(f"  human approval -> action {decided['action']['status']}, commit {decided['event'].get('commit', {}).get('commit_id')}")
    listed = agent_integration_engine.mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    print(f"  MCP tools/list -> {len(listed['result']['tools'])} tools")


if __name__ == "__main__":
    main()
