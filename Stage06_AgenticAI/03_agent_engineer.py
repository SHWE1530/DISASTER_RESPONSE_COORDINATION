"""Stage 06 Agentic AI -- Agent Engineer.

A multi-agent coordinator that runs a perceive -> reason -> plan -> act ->
reflect loop over a multi-zone flood incident, calling the Stage 01-04 models,
the fusion layer and the Stage 06 knowledge base as tools.

  ToolRegistry         the 15 tools from 01_knowledge_engineer.py, each call
                       schema-validated, timed, and returned as {ok, data, error}
  RulePlanner          chooses the perception tools a zone's evidence supports
  GeminiPlanner        optional LLM planner for the same decision; any invalid
                       output falls back to RulePlanner (never used silently)
  TacticalDispatcher   perceives each zone, fuses the evidence, briefs, cites SOPs
  ResourceAllocator    tree of thoughts over three allocation plans, best utility kept
  SafetyAuditor        ten safety rules; routes consequential or low-confidence actions to a human
  CoordinationAgent    owns the goal: plan-and-execute, ReAct perception, demand-aware
                       trade-off ranking, debate and negotiation over contested units,
                       reflexion, precedent memory, a shared blackboard, hierarchical
                       escalation; applies human approve / reject decisions and the
                       emergency override after the run

What the agent is NOT given: a zone's true severity, hazards, headcount,
expectations or provenance. observable_scenario() strips all ground truth
before the agent sees a Stage 05 scenario, so 04_evaluation_engineer.py can
score it honestly.

Human-in-the-loop is enforced in the tool, not in the prompt: the dispatch
tool refuses any held action without the one-time token that only
CoordinationAgent.decide() issues.

Usage:
  python Stage06_AgenticAI/03_agent_engineer.py               # live models, wildcard scenario
  python Stage06_AgenticAI/03_agent_engineer.py --no-models   # knowledge tools only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.decision_engine import FORECAST_RISE_THRESHOLD_M, PRIORITY_LEVELS, DecisionEngine  # noqa: E402

OUTPUT_DIR = BASE_DIR / "data" / "outputs"
DEMO_RUN_JSON = OUTPUT_DIR / "demo_run.json"
LEDGER_JSONL = OUTPUT_DIR / "dispatch_ledger.jsonl"
MEMORY_JSONL = OUTPUT_DIR / "agent_memory.jsonl"
STAGE05_DIR = REPO_ROOT / "Stage05_GenAI"
STAGE05_SCENARIOS = STAGE05_DIR / "data" / "outputs" / "scenarios"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KNOWLEDGE = _load_module("stage06_knowledge_engineer", BASE_DIR / "01_knowledge_engineer.py")
WORKFLOW = _load_module("stage06_workflow_engineer", BASE_DIR / "02_workflow_engineer.py")

OBSERVABLE_ZONE_KEYS = ("zone_id", "label", "state", "district", "location")
OBSERVABLE_INPUT_KEYS = ("sensors", "text", "image_path", "water_levels", "incident_log")
BRIEF_TO_LEVEL = {"ROUTINE": 0, "ELEVATED": 1, "URGENT": 2, "IMMEDIATE": 3, "CRITICAL": 3}
PERCEPTION_TOOLS = {
    "sensors": "query_sensor_risk_score",
    "text": "parse_emergency_text",
    "image_path": "predict_visual_flood",
    "water_levels": "forecast_water_level",
}
TOOL_INPUT = {tool: key for key, tool in PERCEPTION_TOOLS.items()}
TOOL_ARGUMENT = {"query_sensor_risk_score": "sensors", "parse_emergency_text": "raw_text",
                 "predict_visual_flood": "image_path", "forecast_water_level": "water_levels"}

URGENT_INDEX = PRIORITY_LEVELS.index("URGENT")
BRIEFING_DISAGREEMENT_GAP = 2   # same gap the fusion layer uses for a conflict
PERCEPTION_RETRIES = 1          # one retry on a failed stage call, then proceed without it
SOP_TOP_K = 2
PRECEDENTS_TOP_K = 3

# Decision confidence (named factors, not a learned calibration).
CONFIDENCE_THRESHOLD = 0.60     # below this an URGENT+ dispatch is escalated as low-confidence
AGREEMENT_FACTOR = {"unanimous": 1.0, "broad_agreement": 0.9, "single_source": 0.75, "disputed": 0.6}
LOST_EVIDENCE_PENALTY = 0.10
UNCORROBORATED_FACTOR = 0.7
BRIEFING_CONFLICT_FACTOR = 0.85
SEVERE_SENSOR_CONFIDENCE = 0.80  # sensor Severe at or above this, river over its danger mark = physically severe

DEFAULT_ESCALATION_LEVEL = {"approval": "district_eoc", "verification": "field_supervisor",
                            "conflict": "district_eoc", "shortage": "state_eoc", "low_confidence": "district_eoc"}

HEADCOUNT_PATTERN = re.compile(
    r"(\d{1,5})\s+(?:of us|\w*(?:person|resi|people|peopel|individ|famil|villager|patient|dead|injur)\w*)",
    re.IGNORECASE)
MEDICAL_PATTERN = re.compile(r"injur|dead|medical|hospital|unconscious|bleed|ambulance|paramedic",
                             re.IGNORECASE)
FLOOD_HAZARDS = {"Flood", "Rescue Emergency"}
HEAVY_ASSETS = {"rescue_boats", "ambulances"}
# Evidence basis of a zone's priority. Zones whose URGENT label rests on a single
# kind of signal the other sources contradict are ranked after corroborated ones.
UNCORROBORATED = {"forecast_only", "text_only", "disputed"}

DISCLAIMER = ("Agent output is decision support. Every URGENT or CRITICAL dispatch and every "
              "evacuation is held for a human decision, and nothing is committed without one.")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def observable_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """What a real coordinator would have: zones, raw inputs, inventory. No ground truth."""
    zones = []
    for zone in scenario["zones"]:
        inputs = zone.get("inputs", {})
        zones.append({
            **{key: zone.get(key) for key in OBSERVABLE_ZONE_KEYS},
            "inputs": {key: copy.deepcopy(inputs.get(key)) for key in OBSERVABLE_INPUT_KEYS
                       if inputs.get(key) not in (None, "", [])},
        })
    return {"scenario_id": scenario.get("scenario_id", "CUSTOM"), "name": scenario.get("name", "Incident"),
            "start_time": scenario.get("start_time"), "resources": dict(scenario.get("resources") or {}),
            "zones": zones}


# ===========================================================================
# Argument validation (the JSON-Schema subset the tool definitions use)
# ===========================================================================

_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _check_value(name: str, spec: dict[str, Any], value: Any) -> list[str]:
    expected = spec.get("type")
    if expected and not _TYPES[expected](value):
        return [f"'{name}' must be of type {expected}"]
    errors = []
    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"'{name}' must be one of {spec['enum']}")
    if isinstance(value, str):
        if len(value.strip()) < spec.get("minLength", 0):
            errors.append(f"'{name}' is shorter than {spec['minLength']} characters")
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            errors.append(f"'{name}' is longer than {spec['maxLength']} characters")
    if _TYPES["number"](value):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"'{name}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"'{name}' must be <= {spec['maximum']}")
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"'{name}' must be finite")
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            errors.append(f"'{name}' needs at least {spec['minItems']} items")
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            errors.append(f"'{name}' allows at most {spec['maxItems']} items")
        item_spec = spec.get("items")
        if item_spec:
            for i, item in enumerate(value):
                item_errors = _check_value(f"{name}[{i}]", item_spec, item)
                if item_errors:
                    errors.extend(item_errors[:1])
                    break
    return errors


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]
    properties = schema.get("properties", {})
    errors = [f"missing required argument '{key}'" for key in schema.get("required", [])
              if arguments.get(key) is None]
    if schema.get("additionalProperties") is False:
        errors += [f"unexpected argument '{key}'" for key in arguments if key not in properties]
    for key, value in arguments.items():
        if key in properties and value is not None:
            errors += _check_value(key, properties[key], value)
    return errors


# ===========================================================================
# Observation cache: the fusion layer calls the same stage models the
# perception tools just called; the cache makes that free and keeps the two
# views of the evidence identical.
# ===========================================================================

class _Memo:
    def __init__(self, store: dict | None = None) -> None:
        self.store = store if store is not None else {}
        self.hits = 0
        self.misses = 0

    def call(self, namespace: str, fn: Callable, args: tuple, kwargs: dict) -> Any:
        key = namespace + ":" + hashlib.sha1(
            json.dumps([args, kwargs], sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if key in self.store:
            self.hits += 1
            return copy.deepcopy(self.store[key])
        self.misses += 1
        value = fn(*args, **kwargs)  # failures are not cached, so a retry really retries
        self.store[key] = copy.deepcopy(value)
        return value


class CachedEngine:
    METHODS = {"predict", "analyze", "predict_image", "forecast_water_levels"}

    def __init__(self, engine: Any, memo: _Memo, name: str) -> None:
        self._engine, self._memo, self._name = engine, memo, name

    def __getattr__(self, item: str) -> Any:
        attribute = getattr(self._engine, item)
        if item not in self.METHODS or not callable(attribute):
            return attribute

        def wrapped(*args, **kwargs):
            return self._memo.call(f"{self._name}.{item}", attribute, args, kwargs)

        return wrapped


# ===========================================================================
# Run state and tools
# ===========================================================================

class ToolError(Exception):
    """A tool refused the call for a reason the caller should see (not a crash)."""


class ToolUnavailable(Exception):
    """The stage behind a tool is not loaded."""


class RunState:
    """Everything one coordination run can change: inventory, actions, escalations."""

    def __init__(self, run_id: str, inventory: dict[str, int]) -> None:
        self.run_id = run_id
        self.inventory = {k: int(v) for k, v in inventory.items()}
        self.reserved: dict[str, dict[str, int]] = {}
        self.actions: dict[str, dict[str, Any]] = {}
        self.escalations: list[dict[str, Any]] = []
        self.mutual_aid: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.commits: list[dict[str, Any]] = []
        self.approval_tokens: dict[str, str] = {}
        self.override_tokens: dict[str, str] = {}
        self.human_events: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.recalls: list[dict[str, Any]] = []
        self.blackboard: list[dict[str, Any]] = []  # one shared workspace every agent reads and writes
        self.halted = False

    def post(self, agent: str, topic: str, content: Any) -> None:
        self.blackboard.append({"seq": len(self.blackboard) + 1, "agent": agent, "topic": topic,
                                "content": jsonable(content)})

    def reserved_total(self, resource_type: str) -> int:
        return sum(zone.get(resource_type, 0) for zone in self.reserved.values())

    def remaining(self, resource_type: str) -> int:
        return self.inventory.get(resource_type, 0) - self.reserved_total(resource_type)

    def inventory_view(self) -> dict[str, dict[str, int]]:
        return {k: {"total": v, "reserved": self.reserved_total(k), "remaining": self.remaining(k)}
                for k, v in self.inventory.items()}


def _summarise_levels(levels: Any) -> Any:
    if isinstance(levels, list) and len(levels) > 6:
        return f"[{len(levels)} values, last {levels[-1]}]"
    return levels


class ToolRegistry:
    """Executes the tools defined in 01_knowledge_engineer.py."""

    def __init__(self, kb, ml=None, dl=None, nlp=None, briefer: Callable[[str], dict] | None = None,
                 memo_store: dict | None = None, ledger_path: Path | None = None,
                 memory_path: Path | None = None) -> None:
        self.kb = kb
        self.memory_path = Path(memory_path) if memory_path is not None else None
        self.memo = _Memo(memo_store)
        self.ml = CachedEngine(ml, self.memo, "ml") if ml is not None else None
        self.dl = CachedEngine(dl, self.memo, "dl") if dl is not None else None
        self.nlp = CachedEngine(nlp, self.memo, "nlp") if nlp is not None else None
        self.briefer = briefer
        self.decision = DecisionEngine(ml_engine=self.ml, dl_engine=self.dl, nlp_engine=self.nlp)
        self.ledger_path = ledger_path
        self.definitions = {tool["name"]: tool for tool in KNOWLEDGE.TOOL_DEFINITIONS}

    # -- protocol surface ----------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self.definitions.values()))

    def availability(self) -> dict[str, bool]:
        return {"ml": self.ml is not None, "dl": self.dl is not None, "nlp": self.nlp is not None,
                "briefer": self.briefer is not None}

    def call(self, name: str, arguments: dict[str, Any] | None, run: RunState | None = None) -> dict[str, Any]:
        started = time.perf_counter()

        def result(ok: bool, data: Any = None, error: str | None = None, error_type: str | None = None):
            return {"ok": ok, "data": jsonable(data), "error": error, "error_type": error_type,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

        if name not in self.definitions:
            return result(False, error=f"unknown tool '{name}'", error_type="unknown_tool")
        arguments = {} if arguments is None else arguments
        errors = validate_arguments(self.definitions[name]["inputSchema"], arguments)
        if errors:
            return result(False, error="; ".join(errors), error_type="invalid_arguments")
        needs_run = self.definitions[name]["_meta"]["backed_by"] == "run_state"
        if needs_run and run is None:
            return result(False, error="this tool needs an active coordination run", error_type="no_run")
        clean = {k: v for k, v in arguments.items() if v is not None}
        try:
            return result(True, getattr(self, f"_tool_{name}")(run, **clean))
        except ToolError as exc:
            return result(False, error=str(exc), error_type="rejected")
        except ToolUnavailable as exc:
            return result(False, error=str(exc), error_type="unavailable")
        except Exception as exc:  # a stage failure is an observation, not a crash
            return result(False, error=f"{type(exc).__name__}: {exc}", error_type="tool_failure")

    # -- Stage 01-04 and fusion ---------------------------------------------

    def _tool_query_sensor_risk_score(self, run, sensors):
        if self.ml is None:
            raise ToolUnavailable("Stage 01 sensor model is not loaded")
        out = self.ml.predict(dict(sensors))
        return {"risk_category": out.get("risk_category"), "confidence": out.get("confidence"),
                "top_factors": out.get("top_factors"),
                "river_above_threshold": _river_above(sensors)}

    def _tool_predict_visual_flood(self, run, image_path):
        if self.dl is None:
            raise ToolUnavailable("Stage 02 image model is not loaded")
        out = self.dl.predict_image(image_path)
        return {"label": out.get("label"), "confidence": out.get("confidence"),
                "flooded_probability": out.get("flooded_probability")}

    def _tool_forecast_water_level(self, run, water_levels):
        if self.dl is None:
            raise ToolUnavailable("Stage 02 forecasting model is not loaded")
        out = self.dl.forecast_water_levels(list(water_levels), horizon=6)
        forecast = list(out.get("forecast_water_levels") or [])
        return {"forecast": forecast, "in_distribution": out.get("in_distribution"),
                "expected_mae_at_horizon": out.get("expected_mae_at_horizon"),
                "rise_m": round(float(forecast[-1]) - float(water_levels[-1]), 3) if forecast else None,
                "warning": out.get("warning")}

    def _tool_parse_emergency_text(self, run, raw_text):
        if self.nlp is None:
            raise ToolUnavailable("Stage 03 NLP model is not loaded")
        out = self.nlp.analyze(raw_text)
        if out.get("status") != "ok":
            raise ToolError(out.get("message") or "Stage 03 could not analyse the text")
        return {key: out.get(key) for key in ("urgency", "confidence", "hazard_type", "hazard_confidence",
                                              "location", "resource_needed", "headcount")}

    def _tool_assess_zone(self, run, sensors=None, text=None, image_path=None, water_levels=None):
        return self.decision.assess(sensors=sensors, text=text, image_path=image_path,
                                    water_levels=water_levels)

    def _tool_generate_tactical_briefing(self, run, incident_log):
        if self.briefer is None:
            raise ToolUnavailable("Stage 04 briefing model is not loaded")
        out = self.briefer(incident_log) or {}
        return {"priority": out.get("priority"), "situation": out.get("situation"),
                "risk": out.get("risk"), "actions": list(out.get("actions") or [])[:5]}

    # -- knowledge base ------------------------------------------------------

    def _tool_get_district_profile(self, run, state, district):
        profile = self.kb.district_profile(state, district)
        return {"found": profile is not None, "profile": profile}

    def _tool_search_sop(self, run, query, hazard=None, phase=None, top_k=3):
        return {"results": self.kb.search_sop(query, hazard=hazard, phase=phase, top_k=top_k)}

    # -- memory ----------------------------------------------------------------

    def _precedent_rows(self, state: str, district: str) -> list[dict[str, Any]]:
        if self.memory_path is None or not self.memory_path.exists():
            return []
        rows = []
        for line in self.memory_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (str(row.get("state", "")).lower() == str(state).lower()
                    and str(row.get("district", "")).lower() == str(district).lower()):
                rows.append(row)
        return rows

    def has_precedents(self, state: str | None, district: str | None) -> bool:
        return bool(state and district and self._precedent_rows(state, district))

    def remember(self, rows: list[dict[str, Any]]) -> None:
        if self.memory_path is None or not rows:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(jsonable(row)) + "\n")

    def _tool_recall_precedents(self, run, state, district, top_k=PRECEDENTS_TOP_K):
        rows = self._precedent_rows(state, district)
        return {"count": len(rows), "precedents": rows[-top_k:][::-1]}

    def _append_ledger(self, entry: dict[str, Any]) -> None:
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    # -- run state -------------------------------------------------------------

    def _tool_check_resource_inventory(self, run):
        return run.inventory_view()

    def _tool_reserve_resources(self, run, zone_id, resource_type, quantity):
        remaining = run.remaining(resource_type)
        if quantity > remaining:
            raise ToolError(f"cannot reserve {quantity} {resource_type}: only {remaining} remaining")
        zone = run.reserved.setdefault(zone_id, {})
        zone[resource_type] = zone.get(resource_type, 0) + quantity
        return {"zone_id": zone_id, "resource_type": resource_type, "reserved": zone[resource_type],
                "remaining": run.remaining(resource_type)}

    def _tool_release_resources(self, run, zone_id, resource_type, quantity):
        held = run.reserved.get(zone_id, {}).get(resource_type, 0)
        if quantity > held:
            raise ToolError(f"zone {zone_id} holds only {held} {resource_type}")
        run.reserved[zone_id][resource_type] = held - quantity
        return {"zone_id": zone_id, "resource_type": resource_type, "released": quantity,
                "remaining": run.remaining(resource_type)}

    def _tool_escalate_to_human(self, run, zone_id, category, reason, action_id=None, level=None):
        if action_id is not None and action_id not in run.actions:
            raise ToolError(f"unknown action_id {action_id}")
        escalation = {"escalation_id": f"ESC-{len(run.escalations) + 1:03d}", "zone_id": zone_id,
                      "category": category, "level": level or DEFAULT_ESCALATION_LEVEL[category],
                      "reason": reason, "action_id": action_id, "status": "open", "raised_at": _now()}
        run.escalations.append(escalation)
        return escalation

    def _tool_request_mutual_aid(self, run, zone_id, resource_type, shortfall, state=None):
        request = {"request_id": f"AID-{len(run.mutual_aid) + 1:03d}", "zone_id": zone_id, "state": state,
                   "resource_type": resource_type, "shortfall": shortfall,
                   "addressed_to": KNOWLEDGE.SECONDARY_RESPONDERS.get(resource_type, "State EOC"),
                   "reroute": True, "status": "requested", "raised_at": _now()}
        run.mutual_aid.append(request)
        return request

    def _tool_issue_alert(self, run, zone_id, level, message):
        alert = {"alert_id": f"ALR-{len(run.alerts) + 1:03d}", "zone_id": zone_id, "level": level,
                 "message": message, "issued_at": _now()}
        run.alerts.append(alert)
        return alert

    def _tool_notify_shelter(self, run, zone_id, evacuees, beds_reserved, shelter=None):
        note = {"notification_id": f"SHL-{len(run.notifications) + 1:03d}", "zone_id": zone_id,
                "shelter": shelter or KNOWLEDGE.SECONDARY_RESPONDERS["shelter_beds"], "evacuees": evacuees,
                "beds_reserved": beds_reserved,
                "message": f"Expect ~{evacuees} evacuees from {zone_id}; {beds_reserved} beds reserved in this incident.",
                "sent_at": _now()}
        run.notifications.append(note)
        return note

    def _tool_recall_dispatch(self, run, zone_id, action_id, override_token, reason):
        action = run.actions.get(action_id)
        if action is None:
            raise ToolError(f"unknown action_id {action_id}")
        if action["zone_id"] != zone_id:
            raise ToolError(f"action {action_id} belongs to {action['zone_id']}, not {zone_id}")
        expected = run.override_tokens.get(action_id)
        if not expected or not secrets.compare_digest(override_token, expected):
            raise ToolError("a recall needs the one-time token issued by a commander's override")
        if action["type"] != "dispatch" or action["status"] != "committed":
            raise ToolError(f"action {action_id} is {action['status']}; only committed dispatches can be recalled")
        held = run.reserved.setdefault(zone_id, {})
        for resource, quantity in action.get("resources", {}).items():
            held[resource] = max(0, held.get(resource, 0) - quantity)
        action["status"] = "recalled"
        run.override_tokens.pop(action_id, None)
        recall = {"recall_id": f"RCL-{len(run.recalls) + 1:03d}", "run_id": run.run_id, "zone_id": zone_id,
                  "action_id": action_id, "released": dict(action.get("resources", {})), "reason": reason,
                  "recalled_at": _now()}
        run.recalls.append(recall)
        self._append_ledger({"type": "recall", **recall})
        return recall

    def _tool_commit_rescue_dispatch(self, run, zone_id, action_id, approval_token=None):
        if run.halted:
            raise ToolError("run halted by an emergency override; nothing further can be committed")
        action = run.actions.get(action_id)
        if action is None:
            raise ToolError(f"unknown action_id {action_id}")
        if action["zone_id"] != zone_id:
            raise ToolError(f"action {action_id} belongs to {action['zone_id']}, not {zone_id}")
        if action["status"] in {"committed", "rejected", "released"}:
            raise ToolError(f"action {action_id} is already {action['status']}")
        verdict = action["audit"]["verdict"]
        if verdict == "blocked":
            raise ToolError(f"blocked by safety policy: {'; '.join(action['audit']['reasons'])}")
        if verdict == "requires_human_approval":
            expected = run.approval_tokens.get(action_id)
            if not approval_token or not expected or not secrets.compare_digest(approval_token, expected):
                raise ToolError("held for human approval: a valid approval token is required")
        action["status"] = "committed"
        commit = {"commit_id": f"DSP-{len(run.commits) + 1:03d}", "run_id": run.run_id, "zone_id": zone_id,
                  "action_id": action_id, "resources": dict(action.get("resources", {})),
                  "evacuation": action.get("evacuation", False),
                  "approved_by": action.get("approved_by"), "committed_at": _now()}
        run.commits.append(commit)
        self._append_ledger({"type": "commit", **commit})
        return commit


def _river_above(sensors: dict[str, Any] | None) -> bool | None:
    try:
        return float(sensors["river_level_m"]) > float(sensors["river_level_threshold_m"])
    except (KeyError, TypeError, ValueError):
        return None


# ===========================================================================
# Planners
# ===========================================================================

class RulePlanner:
    """One perception tool per evidence stream the zone actually has."""

    name = "rule_based"

    def perception_plan(self, zone: dict[str, Any]) -> list[str]:
        return [tool for key, tool in PERCEPTION_TOOLS.items() if zone["inputs"].get(key)]


class GeminiQuotaExhausted(RuntimeError):
    """The API key's daily request quota is used up."""


class GeminiPlanner:
    """LLM choice of perception tools, validated against the evidence.

    `generate` is any callable prompt -> text, so the validation and fallback
    logic can be tested without an API key. Output that is not a JSON list of
    known, applicable perception tools falls back to RulePlanner, and the
    fallback is counted.
    """

    name = "gemini"

    MAX_RATE_LIMIT_RETRIES = 6
    MAX_BACKOFF_S = 65.0
    REQUEST_TIMEOUT_MS = 60_000

    def __init__(self, generate: Callable[[str], str], max_calls: int | None = None) -> None:
        self.generate = generate
        self.fallback = RulePlanner()
        self.max_calls = max_calls
        self.fallbacks = 0
        self.api_errors = 0       # the model could not be reached (quota, network)
        self.invalid_outputs = 0  # the model answered, but not with a usable plan
        self.not_called = 0       # skipped: call cap reached or daily quota exhausted
        self.quota_exhausted = False
        self.calls = 0

    @classmethod
    def from_environment(cls) -> "GeminiPlanner":
        from google import genai
        from google.genai import types

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # Without a timeout a single stalled request hangs the whole agent run.
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=cls.REQUEST_TIMEOUT_MS))
        model = os.environ.get("STAGE06_GEMINI_MODEL", "gemini-3.6-flash")

        def generate(prompt: str) -> str:
            # Free-tier keys are rate limited per minute; wait out a 429 using the
            # server's suggested delay instead of silently falling back.
            for attempt in range(cls.MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    return client.models.generate_content(model=model, contents=prompt).text
                except Exception as exc:
                    message = str(exc)
                    if "PerDay" in message:
                        # A per-day quota does not recover by waiting minutes.
                        raise GeminiQuotaExhausted(message) from exc
                    if attempt == cls.MAX_RATE_LIMIT_RETRIES or not (
                            "429" in message or "RESOURCE_EXHAUSTED" in message):
                        raise
                    delay = re.search(r"retry(?:Delay)?[^0-9]{0,20}(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
                    time.sleep(min(cls.MAX_BACKOFF_S, float(delay.group(1)) + 1 if delay else 15.0 * (attempt + 1)))
            raise RuntimeError("unreachable")

        cap = os.environ.get("STAGE06_GEMINI_MAX_CALLS")
        return cls(generate, max_calls=int(cap) if cap else None)

    def perception_plan(self, zone: dict[str, Any]) -> list[str]:
        self.calls += 1
        present = {key: bool(zone["inputs"].get(key)) for key in PERCEPTION_TOOLS}
        prompt = (
            "You coordinate flood response. Choose which perception tools to call for this zone.\n"
            f"Tools: {json.dumps(PERCEPTION_TOOLS)} (key = evidence the tool needs).\n"
            f"Evidence present: {json.dumps(present)}.\n"
            f"Text report (may be empty): {json.dumps((zone['inputs'].get('text') or '')[:400])}\n"
            "Call every tool whose evidence is present unless it cannot help. "
            "Reply with ONLY a JSON array of tool names."
        )
        if self.quota_exhausted or (self.max_calls is not None and self.calls > self.max_calls):
            self.not_called += 1
            self.fallbacks += 1
            return self.fallback.perception_plan(zone)
        try:
            raw = self.generate(prompt) or ""
        except Exception as exc:
            if isinstance(exc, GeminiQuotaExhausted):
                self.quota_exhausted = True
            self.api_errors += 1
            self.fallbacks += 1
            return self.fallback.perception_plan(zone)
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            chosen = json.loads(match.group(0)) if match else None
            valid = (isinstance(chosen, list) and chosen
                     and all(isinstance(t, str) and t in TOOL_INPUT and present[TOOL_INPUT[t]] for t in chosen))
            if valid:
                return list(dict.fromkeys(chosen))
        except (ValueError, TypeError):
            pass
        self.invalid_outputs += 1
        self.fallbacks += 1
        return self.fallback.perception_plan(zone)


# ===========================================================================
# Run context: every step goes through here, so every step is on the record
# ===========================================================================

class RunContext:
    def __init__(self, registry: ToolRegistry, run: RunState, graph) -> None:
        self.registry, self.run, self.graph = registry, run, graph
        self.steps: list[dict[str, Any]] = []
        self.memory: dict[tuple, Any] = {}  # within-run reuse of knowledge lookups

    def _record(self, node, thought, zone_id=None, tool=None, arguments=None, outcome=None, **extra):
        step = {
            "step": len(self.steps) + 1, "node": node, "agent": self.graph.agent_for(node),
            "loop": self.graph.nodes[node]["loop"], "zone_id": zone_id, "thought": thought,
            "tool": tool,
        }
        step.update(extra)
        if tool:
            step["arguments_sha"] = hashlib.sha1(
                json.dumps(arguments or {}, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
            step["arguments"] = {k: _summarise_levels(v) if k == "water_levels" else
                                 (f"{{{len(v)} fields}}" if k == "sensors" and isinstance(v, dict) else
                                  (v[:160] + "..." if isinstance(v, str) and len(v) > 160 else v))
                                 for k, v in (arguments or {}).items()}
            step.update({"ok": outcome["ok"], "error": outcome["error"], "error_type": outcome["error_type"],
                         "latency_ms": outcome["latency_ms"], "observation": summarise_observation(tool, outcome)})
        self.steps.append(step)
        return step

    def think(self, node: str, thought: str, zone_id: str | None = None, **extra) -> None:
        self._record(node, thought, zone_id, **extra)

    def call(self, node: str, tool: str, arguments: dict[str, Any], thought: str,
             zone_id: str | None = None) -> dict[str, Any]:
        outcome = self.registry.call(tool, arguments, self.run)
        self._record(node, thought, zone_id, tool, arguments, outcome)
        return outcome


def summarise_observation(tool: str, outcome: dict[str, Any]) -> str:
    if not outcome["ok"]:
        return f"{outcome['error_type']}: {outcome['error']}"
    data = outcome["data"] or {}
    if tool == "query_sensor_risk_score":
        return f"risk {data.get('risk_category')} (conf {data.get('confidence')}), river above threshold: {data.get('river_above_threshold')}"
    if tool == "parse_emergency_text":
        return (f"urgency {data.get('urgency')}, hazard {data.get('hazard_type')}, headcount {data.get('headcount')}, "
                f"resources {data.get('resource_needed')}")
    if tool == "predict_visual_flood":
        return f"image {data.get('label')} (conf {data.get('confidence')})"
    if tool == "forecast_water_level":
        return f"6 h rise {data.get('rise_m')} m, in distribution: {data.get('in_distribution')}"
    if tool == "assess_zone":
        if data.get("status") != "ok":
            return f"fusion: {data.get('status')}"
        return (f"fusion {data.get('priority')} (score {data.get('score')}, {data.get('agreement')}), "
                f"conflicts {len(data.get('conflicts', []))}, human review {data.get('human_review_required')}")
    if tool == "generate_tactical_briefing":
        return f"briefing priority {data.get('priority')}"
    if tool == "get_district_profile":
        profile = data.get("profile") or {}
        return (f"severe rate {profile.get('severe_rate')}, median threshold {profile.get('median_danger_threshold_m')} m"
                if data.get("found") else "no historical profile")
    if tool == "search_sop":
        return "; ".join(f"{r['source']} [{r['action_type']}]" for r in data.get("results", [])) or "no passages"
    if tool == "check_resource_inventory":
        return ", ".join(f"{k} {v['remaining']}/{v['total']}" for k, v in data.items())
    if tool in {"reserve_resources", "release_resources"}:
        return f"{data.get('resource_type')} remaining {data.get('remaining')}"
    if tool == "recall_precedents":
        return f"{data.get('count')} precedent(s) for the district"
    if tool == "escalate_to_human":
        return f"{data.get('escalation_id')} raised to {str(data.get('level')).replace('_', ' ')}"
    if tool == "request_mutual_aid":
        return f"{data.get('request_id')} rerouted to {data.get('addressed_to')}"
    for key in ("notification_id", "recall_id", "alert_id", "commit_id"):
        if key in data:
            return f"{data[key]} recorded"
    return "ok"


# ===========================================================================
# Agents
# ===========================================================================

class TacticalDispatcher:
    """Perceive and reason about one zone."""

    def __init__(self, planner) -> None:
        self.planner = planner

    def investigate(self, zone: dict[str, Any], ctx: RunContext, kb, plan: list[str] | None = None) -> dict[str, Any]:
        zid, inputs = zone["zone_id"], zone["inputs"]
        where = ", ".join(x for x in (zone.get("district"), zone.get("state")) if x) or "unknown location"
        plan = self.planner.perception_plan(zone) if plan is None else plan
        observations: dict[str, dict[str, Any]] = {}
        evidence_lost: list[str] = []

        if not plan:
            ctx.think("PERCEIVE", f"{zid} ({where}) has no sensor, text, image or gauge evidence. "
                                  "There is nothing to perceive, so I must not guess.", zid)
        for tool in plan:
            outcome = ctx.call("PERCEIVE", tool, {TOOL_ARGUMENT[tool]: inputs[TOOL_INPUT[tool]]},
                               f"{zid}: the zone has {TOOL_INPUT[tool].replace('_', ' ')}; reading it with {tool}.", zid)
            for _ in range(PERCEPTION_RETRIES):
                if outcome["ok"] or outcome["error_type"] not in {"tool_failure"}:
                    break
                outcome = ctx.call("PERCEIVE", tool, {TOOL_ARGUMENT[tool]: inputs[TOOL_INPUT[tool]]},
                                   f"{zid}: {tool} failed ({outcome['error']}); retrying once.", zid)
            observations[tool] = outcome
            if not outcome["ok"]:
                evidence_lost.append(TOOL_INPUT[tool])

        usable = {TOOL_INPUT[t]: inputs[TOOL_INPUT[t]] for t, o in observations.items() if o["ok"]}
        base = {"zone_id": zid, "label": zone.get("label"), "state": zone.get("state"),
                "district": zone.get("district"), "evidence_lost": evidence_lost,
                "tools_planned": plan, "observations": {t: o["data"] for t, o in observations.items() if o["ok"]}}

        if not usable:
            reason = ("no evidence stream exists" if not plan
                      else f"every perception tool failed ({', '.join(evidence_lost)})")
            return self._insufficient(zone, ctx, base, reason)

        profile, precedents = None, []
        if ctx.registry.has_precedents(zone.get("state"), zone.get("district")):
            outcome = ctx.call("CONTEXTUALISE", "recall_precedents",
                               {"state": zone["state"], "district": zone["district"], "top_k": PRECEDENTS_TOP_K},
                               f"{zid}: earlier runs made decisions for {where}; recalling them as precedent.", zid)
            precedents = outcome["data"]["precedents"] if outcome["ok"] else []
        if zone.get("state") and zone.get("district"):
            key = ("get_district_profile", zone["state"].lower(), zone["district"].lower())
            if key in ctx.memory:
                profile, source = ctx.memory[key]
                ctx.think("CONTEXTUALISE", f"{zid}: {where}'s flood history was retrieved for {source}; reusing it.",
                          zid, reused_tool="get_district_profile")
            else:
                outcome = ctx.call("CONTEXTUALISE", "get_district_profile",
                                   {"state": zone["state"], "district": zone["district"]},
                                   f"{zid}: pulling {where}'s flood history before judging severity.", zid)
                profile = (outcome["data"] or {}).get("profile") if outcome["ok"] else None
                if outcome["ok"]:
                    ctx.memory[key] = (profile, zid)

        if evidence_lost:
            thought = (f"{zid}: fusing the evidence that survived ({', '.join(usable)}); "
                       f"{', '.join(evidence_lost)} is lost and will be reported, not imputed.")
        else:
            thought = f"{zid}: fusing {', '.join(usable)} into one priority."
        fused_outcome = ctx.call("ASSESS", "assess_zone", usable, thought, zid)
        if not fused_outcome["ok"] or fused_outcome["data"].get("status") != "ok":
            reason = fused_outcome["error"] if not fused_outcome["ok"] else "fusion found no usable evidence"
            return self._insufficient(zone, ctx, base, reason)
        fused = fused_outcome["data"]
        priority_index = int(fused["priority_index"])

        nlp = base["observations"].get("parse_emergency_text") or {}
        headcount, headcount_source = estimate_headcount(inputs.get("incident_log"), nlp.get("headcount"))

        briefing, agent_conflicts = None, []
        if inputs.get("incident_log"):
            outcome = ctx.call("BRIEF", "generate_tactical_briefing", {"incident_log": inputs["incident_log"]},
                               f"{zid}: condensing the {inputs['incident_log'].count(chr(10))}-entry incident log "
                               "and cross-checking its priority against fusion.", zid)
            if outcome["ok"]:
                briefing = outcome["data"]
                brief_level = BRIEF_TO_LEVEL.get(str(briefing.get("priority")).upper())
                if brief_level is not None and abs(brief_level - priority_index) >= BRIEFING_DISAGREEMENT_GAP:
                    agent_conflicts.append(
                        f"Stage 04 briefing says {briefing['priority']} but fusion says {fused['priority']}; "
                        "not auto-resolved, surfaced for a human.")

        signals = derive_signals(inputs, base["observations"], kb)
        basis = escalation_basis(fused)
        needs = derive_needs(priority_index, headcount, signals, basis)
        if basis == "forecast_only":
            agent_conflicts.append("Priority rests on the forecast alone while every present-tense source is "
                                   "below URGENT; assets wait for a corroborating gauge reading.")
        elif basis == "text_only":
            agent_conflicts.append("Priority rests on a CRITICAL human report that the other sources contradict; "
                                   "one unit per asset type and no evacuation until a human adjudicates.")

        confidence = decision_confidence(fused, evidence_lost, basis,
                                         briefing_conflict=any(c.startswith("Stage 04 briefing") for c in agent_conflicts))
        sensor_obs = base["observations"].get("query_sensor_risk_score") or {}
        image_obs = base["observations"].get("predict_visual_flood") or {}
        severe_confirmed = bool(
            (sensor_obs.get("risk_category") == "Severe" and (sensor_obs.get("confidence") or 0) >= SEVERE_SENSOR_CONFIDENCE
             and sensor_obs.get("river_above_threshold")) or image_obs.get("label") == "flooded")
        tier = 3 if (priority_index >= URGENT_INDEX and severe_confirmed) else priority_index
        review_extra = ([f"Decision confidence {confidence:.0%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold."]
                        if confidence < CONFIDENCE_THRESHOLD and priority_index >= URGENT_INDEX else [])

        sop = []
        if priority_index >= 1:
            hazard = nlp.get("hazard_type") or ("Flood" if signals["flood"] else None)
            query = sop_query(hazard, needs, signals)
            key = ("search_sop", query, hazard)
            if key in ctx.memory:
                sop, source = ctx.memory[key]
                ctx.think("RETRIEVE_SOP", f"{zid}: the SOP passages for '{query}' were retrieved for {source}; "
                          "reusing them.", zid, reused_tool="search_sop")
            else:
                outcome = ctx.call("RETRIEVE_SOP", "search_sop",
                                   {"query": query, "hazard": hazard, "phase": "Response", "top_k": SOP_TOP_K},
                                   f"{zid}: {fused['priority']} zone; grounding the response in SOPs for '{query}'.", zid)
                if outcome["ok"]:
                    sop = outcome["data"]["results"]
                    ctx.memory[key] = (sop, zid)

        ctx.run.post("dispatcher", f"assessment:{zid}", {
            "priority": fused["priority"], "basis": basis, "confidence": confidence,
            "emergency_calls": emergency_calls(inputs), "people": headcount, "needs": needs})
        return {**base, "status": "ok", "priority": fused["priority"], "priority_index": priority_index, "basis": basis,
                "score": fused.get("score"), "agreement": fused.get("agreement"),
                "decision_confidence": confidence, "severe_confirmed": severe_confirmed, "tier": tier,
                "emergency_calls": emergency_calls(inputs), "precedents": precedents,
                "fusion_conflicts": [c.get("description") for c in fused.get("conflicts", [])],
                "agent_conflicts": agent_conflicts, "escalations": fused.get("escalations", []),
                "human_review_required": bool(fused.get("human_review_required")) or bool(agent_conflicts) or bool(review_extra),
                "human_review_reasons": fused.get("human_review_reasons", []) + agent_conflicts + review_extra,
                "fusion_actions": fused.get("recommended_actions", []), "fusion": fused,
                "headcount_estimate": headcount, "headcount_source": headcount_source,
                "signals": signals, "needs": needs, "briefing": briefing, "district_profile": profile,
                "sop": sop}

    @staticmethod
    def _insufficient(zone, ctx, base, reason):
        zid = zone["zone_id"]
        ctx.call("ESCALATE_EVIDENCE", "escalate_to_human",
                 {"zone_id": zid, "category": "verification",
                  "reason": f"No automated assessment possible: {reason}. Send a field team to verify."},
                 f"{zid}: {reason}. Silence is not safety -- requesting field verification and dispatching nothing.",
                 zid)
        return {**base, "status": "insufficient_evidence", "priority": None, "priority_index": None, "basis": None,
                "decision_confidence": None, "severe_confirmed": False, "tier": -1,
                "emergency_calls": emergency_calls(zone["inputs"]), "precedents": [],
                "score": None, "agreement": None, "fusion_conflicts": [], "agent_conflicts": [],
                "escalations": [], "human_review_required": True,
                "human_review_reasons": [f"No automated assessment possible: {reason}."],
                "fusion_actions": [], "fusion": None, "headcount_estimate": 0, "headcount_source": None,
                "signals": {}, "needs": {}, "briefing": None, "district_profile": None, "sop": []}


def estimate_headcount(incident_log: str | None, nlp_headcount: Any) -> tuple[int, str | None]:
    """Sum of per-entry headcounts in the log (distinct incidents), else Stage 03's."""
    total = 0
    if incident_log:
        for line in incident_log.splitlines()[1:]:
            match = HEADCOUNT_PATTERN.search(line)
            if match:
                total += int(match.group(1))
    if total:
        return total, "incident_log"
    try:
        value = int(nlp_headcount)
        return (value, "stage03_text") if value > 0 else (0, None)
    except (TypeError, ValueError):
        return 0, None


def derive_signals(inputs: dict[str, Any], observations: dict[str, Any], kb) -> dict[str, Any]:
    sensor = observations.get("query_sensor_risk_score") or {}
    nlp = observations.get("parse_emergency_text") or {}
    image = observations.get("predict_visual_flood") or {}
    forecast = observations.get("forecast_water_level") or {}
    requested = sorted({kb.map_resource(str(p)) for p in (nlp.get("resource_needed") or [])} - {None})
    text = " ".join(filter(None, [inputs.get("text"), inputs.get("incident_log")]))
    rise = forecast.get("rise_m")
    # A projected rise only counts when it clears the rise threshold PLUS the
    # model's own expected error at that horizon (0.73 m for the Stage 02 LSTM).
    rise_bar = FORECAST_RISE_THRESHOLD_M + float(forecast.get("expected_mae_at_horizon") or 0)
    physical, reported = [], []
    if sensor.get("river_above_threshold") and sensor.get("risk_category") == "Severe":
        physical.append("river above danger threshold with Severe sensor risk")
    if image.get("label") == "flooded":
        physical.append("imagery shows flooding")
    if nlp.get("hazard_type") in FLOOD_HAZARDS:
        reported.append(f"report hazard is {nlp.get('hazard_type')}")
    if rise is not None and forecast.get("in_distribution") and rise >= rise_bar:
        reported.append(f"forecast rise {rise} m clears threshold + model error ({rise_bar:.2f} m)")
    medical = (nlp.get("hazard_type") == "Medical Emergency" or "ambulances" in requested
               or bool(MEDICAL_PATTERN.search(text)))
    return {"flood": bool(physical or reported), "flood_corroborated": bool(physical) or len(reported) >= 2,
            "flood_reasons": physical + reported, "medical": medical, "requested": requested}


def escalation_basis(fused: dict[str, Any]) -> str:
    """What the fused priority actually rests on, read from the evidence levels.

    forecast_only  URGENT+ only because the forecast escalated; no present-tense
                   source is at URGENT
    text_only      URGENT+ only because a human report said CRITICAL; every other
                   present-tense source is below URGENT
    """
    evidence = {e["source"]: e for e in fused.get("evidence", []) if e.get("available")}
    present = {k: (e.get("level") or 0) for k, e in evidence.items() if k != "forecast_trend"}
    index = fused.get("priority_index") or 0
    if index >= URGENT_INDEX:
        if (evidence.get("forecast_trend", {}).get("level") == 3
                and max(present.values(), default=-1) < URGENT_INDEX):
            return "forecast_only"
        others = [level for source, level in present.items() if source != "text_urgency"]
        if present.get("text_urgency") == 3 and others and max(others) < URGENT_INDEX:
            return "text_only"
    if fused.get("agreement") == "single_source":
        return "single_source"
    if fused.get("agreement") == "disputed":
        return "disputed"
    return "corroborated"


def decision_confidence(fused: dict[str, Any], evidence_lost: list[str], basis: str, briefing_conflict: bool) -> float:
    """How far the agent's decision for a zone can be trusted, 0-1.

    Mean confidence of the present-tense sources that answered, discounted when
    they disagree, when evidence was lost, when the priority rests on one kind of
    signal, and when the Stage 04 briefing contradicts fusion. Named factors,
    not a learned calibration.
    """
    confidences = [float(e["confidence"]) for e in fused.get("evidence", [])
                   if e.get("available") and e.get("source") != "forecast_trend" and e.get("confidence") is not None]
    value = float(np.mean(confidences)) if confidences else 0.5
    value *= AGREEMENT_FACTOR.get(fused.get("agreement"), AGREEMENT_FACTOR["single_source"])
    value -= LOST_EVIDENCE_PENALTY * len(evidence_lost)
    if basis in {"forecast_only", "text_only"}:
        value *= UNCORROBORATED_FACTOR
    if briefing_conflict:
        value *= BRIEFING_CONFLICT_FACTOR
    return round(min(0.99, max(0.05, value)), 3)


def emergency_calls(inputs: dict[str, Any]) -> int:
    try:
        return max(0, int(float((inputs.get("sensors") or {}).get("emergency_calls") or 0)))
    except (TypeError, ValueError):
        return 0


def rank_key(a: dict[str, Any]) -> tuple:
    """Severity tier, then corroborated before uncorroborated, then field demand.

    Two zones physically confirmed as severe share the top tier even if fusion
    split them URGENT / CRITICAL on report wording, so the zone with more
    emergency calls and more people is served first.
    """
    return (-a["tier"], a["basis"] in UNCORROBORATED, -a["emergency_calls"], -a["headcount_estimate"],
            -(a["decision_confidence"] or 0), -(a["score"] or 0))


def tradeoff_thoughts(ordered: list[dict[str, Any]]) -> list[str]:
    """Write out the reasoning behind every adjacent pair the ranking had to decide."""
    thoughts = []
    for high, low in zip(ordered, ordered[1:]):
        same_corroboration = (high["basis"] in UNCORROBORATED) == (low["basis"] in UNCORROBORATED)
        if high["tier"] != low["tier"] or not same_corroboration:
            continue
        if high["priority_index"] == low["priority_index"] and high["emergency_calls"] == low["emergency_calls"] \
                and high["headcount_estimate"] == low["headcount_estimate"]:
            continue
        if high["tier"] == 3 and high["severe_confirmed"] and low["severe_confirmed"]:
            both = "physically confirmed severe (sensors or imagery)"
        else:
            both = f"in the {PRIORITY_LEVELS[max(0, high['tier'])]} tier"
        split = ("" if high["priority_index"] == low["priority_index"] else
                 f"; fusion labels them {high['priority']} and {low['priority']}, but at equal physical severity "
                 "field demand decides")
        thoughts.append(
            f"Trade-off {high['zone_id']} vs {low['zone_id']}: both {both}{split}. Demand: "
            f"{high['emergency_calls']} vs {low['emergency_calls']} emergency calls, "
            f"{high['headcount_estimate']} vs {low['headcount_estimate']} people -> {high['zone_id']} first; "
            f"{low['zone_id']} is served next or rerouted to a secondary responder.")
    return thoughts


def build_plan(incident: dict[str, Any], perception: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Plan-and-execute: the full plan, written before the first tool call."""
    steps: list[dict[str, Any]] = []

    def add(subtask, node, tool=None, zone_id=None, conditional=False):
        steps.append({"id": f"P{len(steps) + 1:02d}", "subtask": subtask, "node": node, "tool": tool,
                      "zone_id": zone_id, "conditional": conditional})

    for zone in incident["zones"]:
        zid, tools = zone["zone_id"], perception[zone["zone_id"]]
        if not tools:
            add("request field verification (no evidence)", "ESCALATE_EVIDENCE", "escalate_to_human", zid)
            continue
        for tool in tools:
            add(f"perceive {TOOL_INPUT[tool].replace('_', ' ')}", "PERCEIVE", tool, zid)
        if zone.get("state") and zone.get("district"):
            add("recall district history", "CONTEXTUALISE", "get_district_profile", zid, conditional=True)
        add("fuse evidence into a priority", "ASSESS", "assess_zone", zid)
        if zone["inputs"].get("incident_log"):
            add("brief the incident log", "BRIEF", "generate_tactical_briefing", zid)
        add("cite SOPs if ELEVATED or higher", "RETRIEVE_SOP", "search_sop", zid, conditional=True)
    for subtask, node in [("rank zones by severity, corroboration and demand", "RANK"),
                          ("explore allocation plans (tree of thoughts)", "ALLOCATE"),
                          ("debate and negotiate contested units", "DEBATE"),
                          ("audit every proposed action", "AUDIT"),
                          ("hold consequential actions for sign-off", "AWAIT_APPROVAL"),
                          ("reroute shortfalls to secondary responders", "MUTUAL_AID"),
                          ("reflect on the goal and revise", "REFLECT")]:
        add(subtask, node)
    return steps


def plan_adherence(plan: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    executed = {(s.get("zone_id"), s.get("tool") or s.get("reused_tool")) for s in steps
                if s.get("tool") or s.get("reused_tool")}
    required = [p for p in plan if p["tool"] and not p["conditional"]]
    done = [p for p in required if (p["zone_id"], p["tool"]) in executed]
    return {"planned_tool_steps": len(required), "executed": len(done),
            "rate": round(len(done) / len(required), 4) if required else None,
            "replanned": [p["id"] + " " + p["subtask"] + " " + str(p["zone_id"]) for p in required if p not in done]}


def derive_needs(priority_index: int, headcount: int, signals: dict[str, Any],
                 basis: str = "corroborated") -> dict[str, int]:
    """Asset needs for URGENT+ zones only; ELEVATED zones get standby, not assets.

    A forecast-only URGENT zone gets no assets until a gauge reading corroborates
    it. A text-only URGENT zone gets one unit per heavy type and no evacuation
    beds until a human adjudicates the disagreement.
    """
    if priority_index < URGENT_INDEX or basis == "forecast_only":
        return {}
    needs: dict[str, int] = {}
    if signals["flood"] or "rescue_boats" in signals["requested"]:
        needs["rescue_boats"] = max(1, math.ceil(headcount / KNOWLEDGE.PEOPLE_PER_BOAT))
    if signals["medical"]:
        needs["ambulances"] = max(1, math.ceil(headcount / KNOWLEDGE.PEOPLE_PER_AMBULANCE))
    if signals["flood_corroborated"] and headcount > 0:
        needs["shelter_beds"] = headcount
    if not needs:
        # URGENT with no specific signal: one crew to verify and stabilise.
        needs["rescue_boats"] = 1
    if basis == "text_only":
        needs = {r: 1 for r in needs if r in HEAVY_ASSETS} or {"rescue_boats": 1}
    return needs


def sop_query(hazard: str | None, needs: dict[str, int], signals: dict[str, Any]) -> str:
    parts = [hazard or "disaster", "response"]
    if "rescue_boats" in needs:
        parts.append("search and rescue evacuation boats")
    if "ambulances" in needs:
        parts.append("medical care injured")
    if "shelter_beds" in needs:
        parts.append("relief camps shelter")
    if not needs:
        parts.append("alert standby preparedness")
    return " ".join(parts)


class ResourceAllocator:
    """Turn ranked needs into reservations without over-committing.

    With the priority policy the allocator explores several allocation plans
    (tree of thoughts) and keeps the one with the highest utility; `fcfs` is the
    arrival-order baseline used as an ablation.
    """

    BRANCHES = ("one_each_then_rank", "rank_fill", "weighted_round_robin")

    def __init__(self, policy: str = "priority") -> None:
        if policy not in {"priority", "fcfs"}:
            raise ValueError("policy must be 'priority' or 'fcfs'")
        self.policy = policy

    @staticmethod
    def weight(a: dict[str, Any]) -> float:
        return (max(0, a["tier"]) + 1) ** 2 * (0.5 + (a.get("decision_confidence") or 0.5))

    @staticmethod
    def shortfalls(ordered: list[dict[str, Any]], grants: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        eligible = [a for a in ordered if a["status"] == "ok" and a["needs"]]
        return [{"zone_id": a["zone_id"], "state": a.get("state"), "resource_type": r,
                 "need": n, "granted": grants[a["zone_id"]].get(r, 0),
                 "shortfall": n - grants[a["zone_id"]].get(r, 0)}
                for a in eligible for r, n in a["needs"].items() if n > grants[a["zone_id"]].get(r, 0)]

    def plan(self, ordered: list[dict[str, Any]], inventory: dict[str, int], branch: str | None = None) -> tuple[dict, list]:
        branch = branch or ("one_each_then_rank" if self.policy == "priority" else "fcfs")
        remaining = dict(inventory)
        grants: dict[str, dict[str, int]] = {a["zone_id"]: {} for a in ordered}
        eligible = [a for a in ordered if a["status"] == "ok" and a["needs"]]

        def grant(zone, resource, quantity):
            quantity = min(quantity, remaining.get(resource, 0))
            if quantity > 0:
                grants[zone][resource] = grants[zone].get(resource, 0) + quantity
                remaining[resource] -= quantity

        if branch == "one_each_then_rank":
            # Every URGENT+ zone gets one unit of each type it needs before any zone gets a second.
            for a in eligible:
                for resource in a["needs"]:
                    grant(a["zone_id"], resource, 1)
        if branch == "weighted_round_robin":
            while True:
                best, best_value = None, 0.0
                for a in eligible:
                    for resource, need in a["needs"].items():
                        held = grants[a["zone_id"]].get(resource, 0)
                        if remaining.get(resource, 0) > 0 and held < need:
                            value = self.weight(a) * (need - held) / need
                            if value > best_value:
                                best, best_value = (a["zone_id"], resource), value
                if best is None:
                    break
                grant(best[0], best[1], 1)
        for a in eligible:
            for resource, need in a["needs"].items():
                grant(a["zone_id"], resource, need - grants[a["zone_id"]].get(resource, 0))
        return grants, self.shortfalls(ordered, grants)

    def utility(self, ordered: list[dict[str, Any]], grants: dict[str, dict[str, int]]) -> float:
        """Severity- and confidence-weighted value of a plan: breadth of first units and depth of coverage."""
        total = 0.0
        for a in ordered:
            if a["status"] != "ok" or not a["needs"]:
                continue
            held = grants.get(a["zone_id"], {})
            coverage = np.mean([min(held.get(r, 0), n) / n for r, n in a["needs"].items()])
            first_units = np.mean([held.get(r, 0) >= 1 for r in a["needs"]])
            total += self.weight(a) * (0.6 * coverage + 0.4 * first_units)
        return round(float(total), 4)

    def explore(self, ordered: list[dict[str, Any]], inventory: dict[str, int]) -> list[dict[str, Any]]:
        branches = []
        for branch in self.BRANCHES:
            grants, shortfalls = self.plan(ordered, inventory, branch)
            branches.append({"branch": branch, "grants": grants, "shortfalls": shortfalls,
                             "utility": self.utility(ordered, grants)})
        return branches


class SafetyAuditor:
    """Rules that decide what may happen without a human."""

    RULES = {
        "R1": "URGENT or CRITICAL dispatch requires human sign-off",
        "R2": "no dispatch to a zone without an automated assessment",
        "R3": "fusion flagged the zone for human review",
        "R4": "sources disagree; a human adjudicates before commitment",
        "R5": "evacuation orders always require human sign-off",
        "R6": "evacuation needs a corroborated flood signal",
        "R7": "no heavy assets to a ROUTINE zone",
        "R8": "reservations may never exceed the incident inventory",
        "R9": "forecast-only escalation: assets wait for a corroborating reading",
        "R10": "decision confidence below threshold: a human must confirm",
        "R0": "low-consequence, reversible standby alert",
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def audit(self, action: dict[str, Any], assessment: dict[str, Any], run: RunState) -> dict[str, Any]:
        if not self.enabled:
            return {"verdict": "auto_approved", "rules": ["auditor_disabled"], "reasons": ["auditor disabled (ablation)"]}
        blocked, held = [], []
        if action["type"] == "verification":
            held.append("R2")
        elif action["type"] == "standby_alert":
            if assessment["status"] != "ok":
                blocked.append("R2")
        elif action["type"] == "dispatch":
            if assessment["status"] != "ok":
                blocked.append("R2")
            else:
                if assessment["priority_index"] == 0 and action["resources"]:
                    blocked.append("R7")
                if action.get("evacuation") and not assessment["signals"].get("flood"):
                    blocked.append("R6")
                if assessment["priority_index"] >= URGENT_INDEX:
                    held.append("R1")
                if assessment["fusion"] and assessment["fusion"].get("human_review_required"):
                    held.append("R3")
                if assessment["fusion_conflicts"] or assessment["agent_conflicts"]:
                    held.append("R4")
                if action.get("evacuation"):
                    held.append("R5")
                if (assessment.get("decision_confidence") or 1.0) < CONFIDENCE_THRESHOLD:
                    held.append("R10")
            if any(run.reserved_total(r) > run.inventory.get(r, 0) for r in run.inventory):
                blocked.append("R8")
        rules = blocked or held or ["R0"]
        if action["type"] == "standby_alert" and assessment.get("basis") == "forecast_only" and not blocked:
            rules = rules + ["R9"]
        verdict = "blocked" if blocked else ("requires_human_approval" if held else "auto_approved")
        return {"verdict": verdict, "rules": rules, "reasons": [self.RULES[r] for r in rules]}


# ===========================================================================
# Coordinator
# ===========================================================================

class AgentRun:
    """One coordination run: the record plus the live state approvals act on."""

    def __init__(self, record: dict[str, Any], state: RunState) -> None:
        self.record, self.state = record, state

    @property
    def run_id(self) -> str:
        return self.state.run_id

    def to_dict(self) -> dict[str, Any]:
        record = dict(self.record)
        record["actions"] = list(self.state.actions.values())
        record["escalations"] = self.state.escalations
        record["mutual_aid"] = self.state.mutual_aid
        record["alerts"] = self.state.alerts
        record["commits"] = self.state.commits
        record["notifications"] = self.state.notifications
        record["recalls"] = self.state.recalls
        record["blackboard"] = self.state.blackboard
        record["halted"] = self.state.halted
        record["inventory_after"] = self.state.inventory_view()
        record["human_events"] = self.state.human_events
        record["pending_approvals"] = sum(a["status"] == "pending_approval" for a in self.state.actions.values())
        return jsonable(record)


RESOURCE_LABELS = {"rescue_boats": "rescue boat", "ambulances": "ambulance", "shelter_beds": "shelter bed"}


def _units(quantity: int, resource: str) -> str:
    label = RESOURCE_LABELS.get(resource, resource)
    return f"{quantity} {label}{'' if quantity == 1 else 's'}"


class CoordinationAgent:
    def __init__(self, registry: ToolRegistry, planner=None, graph=None, allocation_policy: str = "priority",
                 auditor_enabled: bool = True) -> None:
        self.registry = registry
        self.kb = registry.kb
        self.planner = planner or RulePlanner()
        self.graph = graph or WORKFLOW.WorkflowGraph()
        self.dispatcher = TacticalDispatcher(self.planner)
        self.allocator = ResourceAllocator(allocation_policy)
        self.auditor = SafetyAuditor(auditor_enabled)
        self.config = {"planner": self.planner.name, "allocation_policy": allocation_policy,
                       "auditor_enabled": auditor_enabled, "confidence_threshold": CONFIDENCE_THRESHOLD,
                       "memory": registry.memory_path is not None}

    def run(self, scenario: dict[str, Any], observable: bool = False) -> AgentRun:
        """Coordinate one incident. Ground truth is stripped unless `observable` says it already is."""
        started = time.perf_counter()
        incident = scenario if observable else observable_scenario(scenario)
        inventory = {k: int(incident["resources"].get(k, 0)) for k in self.kb.resource_types} \
            if incident["resources"] else self.kb.default_inventory
        state = RunState(f"RUN-{uuid.uuid4().hex[:10]}", inventory)
        ctx = RunContext(self.registry, state, self.graph)
        priority_policy = self.allocator.policy == "priority"

        # ---- plan-and-execute: the whole plan before the first tool call ------
        perception = {z["zone_id"]: self.planner.perception_plan(z) for z in incident["zones"]}
        plan = build_plan(incident, perception)
        plan_check = self.graph.check_plan(plan, required_nodes=("RANK", "ALLOCATE", "AUDIT", "REFLECT"))
        ctx.think("PLAN", f"Plan-and-execute: {len(plan)} steps for {len(incident['zones'])} zone(s) -- "
                  f"{sum(p['node'] == 'PERCEIVE' for p in plan)} perception calls, then rank, explore allocations, "
                  f"debate, audit, act and reflect. Plan check: {plan_check['summary']}.")
        state.post("coordinator", "plan", {"steps": len(plan), "check": plan_check["summary"]})

        # ---- perceive + reason, zone by zone --------------------------------
        assessments = [self.dispatcher.investigate(zone, ctx, self.kb, perception[zone["zone_id"]])
                       for zone in incident["zones"]]
        by_zone = {a["zone_id"]: a for a in assessments}

        # ---- plan: rank, with the trade-offs written out --------------------
        scored = [a for a in assessments if a["status"] == "ok"]
        if priority_policy:
            ordered = sorted(scored, key=rank_key)
            ctx.think("RANK", "Ranking by severity tier (zones physically confirmed severe share the top tier), "
                      "then corroborated before uncorroborated evidence, then emergency calls, people and "
                      "decision confidence: " + " > ".join(
                          f"{a['zone_id']} {a['priority']}" + (f" ({a['basis']})" if a["basis"] in UNCORROBORATED else "")
                          for a in ordered)
                      + (f". Unscored: {', '.join(a['zone_id'] for a in assessments if a['status'] != 'ok')}."
                         if len(scored) < len(assessments) else "."))
            for thought in tradeoff_thoughts(ordered):
                ctx.think("RANK", thought)
        else:
            ordered = scored
            ctx.think("RANK", "Allocation policy is first-come-first-served (ablation): zones keep arrival order.")
        state.post("coordinator", "ranking", [a["zone_id"] for a in ordered])

        # ---- plan: tree of thoughts over allocation plans -------------------
        ctx.call("ALLOCATE", "check_resource_inventory", {}, f"Inventory before allocation: {inventory}.")
        tree = {"branches": [], "chosen": "fcfs"}
        if priority_policy:
            branches = self.allocator.explore(ordered, inventory)
            for b in branches:
                first = {r: [z for z, g in b["grants"].items() if g.get(r)] for r in inventory}
                ctx.think("ALLOCATE", f"Tree of thoughts, branch '{b['branch']}': utility {b['utility']:.3f}; units go to "
                          + ", ".join(f"{r}: {', '.join(z) or 'none'}" for r, z in first.items())
                          + f"; {len(b['shortfalls'])} shortfall(s).")
            chosen = max(branches, key=lambda b: b["utility"])
            grants = {z: dict(g) for z, g in chosen["grants"].items()}
            ctx.think("ALLOCATE", f"Keeping branch '{chosen['branch']}': highest utility ({chosen['utility']:.3f}).")
            tree = {"branches": [{"branch": b["branch"], "utility": b["utility"], "shortfalls": len(b["shortfalls"])}
                                 for b in branches], "chosen": chosen["branch"]}
            debate = self._debate(ordered, grants, inventory, ctx)
        else:
            grants, _ = self.allocator.plan(ordered, inventory, "fcfs")
            debate = {"negotiations": [], "objections": []}
        shortfalls = self.allocator.shortfalls(ordered, grants)
        state.post("allocator", "allocation", {"branch": tree["chosen"],
                                               "grants": {z: g for z, g in grants.items() if any(g.values())},
                                               "shortfalls": len(shortfalls)})
        for a in ordered:
            for resource, quantity in grants[a["zone_id"]].items():
                if quantity:
                    ctx.call("ALLOCATE", "reserve_resources",
                             {"zone_id": a["zone_id"], "resource_type": resource, "quantity": quantity},
                             f"{a['zone_id']} ({a['priority']}) needs {a['needs'][resource]} {resource}; reserving {quantity}.",
                             a["zone_id"])
        if not any(any(g.values()) for g in grants.values()):
            ctx.think("ALLOCATE", "No zone reached URGENT with an asset need in stock; nothing to reserve.")

        # ---- plan: audit -----------------------------------------------------
        for a in assessments:
            action = self._propose(a, state, shortfalls)
            if action is None:
                continue
            action["audit"] = self.auditor.audit(action, a, state)
            action["status"] = {"auto_approved": "approved", "requires_human_approval": "pending_approval",
                                "blocked": "blocked"}[action["audit"]["verdict"]]
            state.actions[action["action_id"]] = action
        ctx.think("AUDIT", "Audit verdicts: " + ("; ".join(
            f"{x['action_id']} {x['type']} {x['zone_id']} -> {x['audit']['verdict']} ({','.join(x['audit']['rules'])})"
            for x in state.actions.values()) or "no actions proposed."))
        state.post("auditor", "verdicts", {x["action_id"]: x["audit"]["verdict"] for x in state.actions.values()})

        # ---- act -------------------------------------------------------------
        for action in state.actions.values():
            if action["status"] == "pending_approval" and action["type"] == "dispatch":
                a = by_zone[action["zone_id"]]
                if a["fusion_conflicts"] or a["agent_conflicts"]:
                    category = "conflict"
                elif "R10" in action["audit"]["rules"]:
                    category = "low_confidence"
                else:
                    category = "approval"
                level = "state_eoc" if a["priority_index"] == 3 else "district_eoc"
                resources = ", ".join(_units(q, r) for r, q in action["resources"].items()) or "no assets in stock"
                ctx.call("AWAIT_APPROVAL", "escalate_to_human",
                         {"zone_id": action["zone_id"], "category": category, "action_id": action["action_id"],
                          "level": level,
                          "reason": f"{a['priority']} dispatch ({resources}"
                                    f"{'; evacuation of ' + str(a['headcount_estimate']) if action['evacuation'] else ''}), "
                                    f"decision confidence {a['decision_confidence']:.0%}: {'; '.join(action['audit']['reasons'])}."},
                         f"{action['action_id']} is consequential (confidence {a['decision_confidence']:.0%}); "
                         f"holding it for the {level.replace('_', ' ')} instead of committing.", action["zone_id"])
        for a in scored:
            if a["basis"] == "forecast_only":
                ctx.call("AWAIT_APPROVAL", "escalate_to_human",
                         {"zone_id": a["zone_id"], "category": "verification", "level": "field_supervisor",
                          "reason": "URGENT on a forecast rise alone; confirm the gauge reading before committing assets."},
                         f"{a['zone_id']}: the forecast is the only URGENT signal and the model's error is large; "
                         "asking for a gauge check instead of sending assets.", a["zone_id"])
        for shortfall in shortfalls:
            responder = KNOWLEDGE.SECONDARY_RESPONDERS.get(shortfall["resource_type"], "State EOC")
            ctx.call("MUTUAL_AID", "request_mutual_aid",
                     {"zone_id": shortfall["zone_id"], "state": shortfall["state"],
                      "resource_type": shortfall["resource_type"], "shortfall": shortfall["shortfall"]},
                     f"{shortfall['zone_id']} is short {_units(shortfall['shortfall'], shortfall['resource_type'])} "
                     f"(needs {shortfall['need']}, got {shortfall['granted']}); rerouting to {responder}.",
                     shortfall["zone_id"])
        short_zones = sorted({s["zone_id"] for s in shortfalls})
        for zone_id in short_zones:
            items = ", ".join(_units(s["shortfall"], s["resource_type"]) for s in shortfalls if s["zone_id"] == zone_id)
            ctx.call("MUTUAL_AID", "escalate_to_human",
                     {"zone_id": zone_id, "category": "shortage", "level": "state_eoc",
                      "reason": f"Unmet need after allocation, rerouted to secondary responders: {items}."},
                     f"{zone_id}: the inventory cannot cover {items}; the state EOC must know before relying on this plan.",
                     zone_id)
        for action in state.actions.values():
            if action["type"] == "standby_alert" and action["status"] == "approved":
                ctx.call("ALERT", "issue_alert",
                         {"zone_id": action["zone_id"], "level": action["level"], "message": action["summary"]},
                         f"{action['zone_id']} is {action['level']}: standby alert is low-consequence and cleared.",
                         action["zone_id"])
                action["status"] = "committed"
        for action in state.actions.values():
            if action["type"] != "dispatch":
                continue
            if action["status"] == "approved":
                outcome = ctx.call("COMMIT", "commit_rescue_dispatch",
                                   {"zone_id": action["zone_id"], "action_id": action["action_id"]},
                                   f"{action['action_id']} was auto-approved by the auditor; committing.", action["zone_id"])
                if outcome["ok"]:
                    self._notify_shelter(action, by_zone[action["zone_id"]], ctx)
            elif action["status"] == "blocked":
                for resource, quantity in action["resources"].items():
                    if quantity:
                        ctx.call("COMMIT", "release_resources",
                                 {"zone_id": action["zone_id"], "resource_type": resource, "quantity": quantity},
                                 f"{action['action_id']} is blocked ({', '.join(action['audit']['rules'])}); "
                                 f"returning {quantity} {resource}.", action["zone_id"])
                action["status"] = "released"

        # ---- reflect, and revise once (reflexion) ------------------------------
        high = [a for a in scored if a["priority_index"] >= URGENT_INDEX]
        unserved = [a["zone_id"] for a in high if a["basis"] != "forecast_only"
                    and not any(grants[a["zone_id"]].values()) and a["zone_id"] not in short_zones]
        pending = sum(x["status"] == "pending_approval" for x in state.actions.values())
        verify = [a["zone_id"] for a in assessments if a["status"] != "ok"]
        low_confidence = [a["zone_id"] for a in high if (a["decision_confidence"] or 0) < CONFIDENCE_THRESHOLD]
        ctx.think("REFLECT", (
            f"{len(high)} URGENT+ zone(s); {len(high) - len(unserved)} have assets or a rerouted shortage"
            + (f", UNSERVED: {', '.join(unserved)}" if unserved else "")
            + f". {pending} action(s) await human approval; {len(verify)} zone(s) sent for verification"
            + (f" ({', '.join(verify)})" if verify else "")
            + (f"; low-confidence decisions escalated: {', '.join(low_confidence)}" if low_confidence else "") + "."))
        revisions = self._reflexion(shortfalls, state, ctx)
        ctx.think("END", "Run complete. Nothing consequential has been committed without a human.")

        if self.registry.memory_path is not None:
            self.registry.remember([{
                "type": "agent_decision", "run_id": state.run_id, "created_at": _now(),
                "scenario": incident.get("name"), "state": a.get("state"), "district": a.get("district"),
                "zone_id": a["zone_id"], "priority": a["priority"], "decision_confidence": a["decision_confidence"],
                "emergency_calls": a["emergency_calls"], "headcount": a["headcount_estimate"],
                "allocated": {r: q for r, q in grants.get(a["zone_id"], {}).items() if q},
            } for a in assessments if a.get("state") and a.get("district")])

        record = {
            "run_id": state.run_id, "created_at": _now(), "config": self.config,
            "scenario": {k: incident.get(k) for k in ("scenario_id", "name", "start_time")},
            "inventory_before": inventory, "ranking": [a["zone_id"] for a in ordered],
            "zones": [self._zone_view(a, grants.get(a["zone_id"], {}), state) for a in assessments],
            "shortfalls": shortfalls, "unserved_high_priority": unserved,
            "plan": plan, "plan_check": plan_check, "plan_adherence": plan_adherence(plan, ctx.steps),
            "tree_of_thoughts": tree, "debate": debate, "reflexion": {"triggered": bool(revisions), "revisions": revisions},
            "trajectory": ctx.steps, "workflow_violations": self.graph.conformance(ctx.steps),
            "tool_stats": tool_stats(ctx.steps),
            "cache": {"hits": self.registry.memo.hits, "misses": self.registry.memo.misses},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "disclaimer": DISCLAIMER,
        }
        return AgentRun(record, state)

    # ---- multi-agent debate and negotiation ------------------------------------

    def _debate(self, ordered, grants, inventory, ctx) -> dict[str, Any]:
        position = {a["zone_id"]: i for i, a in enumerate(ordered)}
        eligible = [a for a in ordered if a["needs"]]
        negotiations, objections = [], []
        for resource, supply in inventory.items():
            claimants = [a for a in eligible if a["needs"].get(resource, 0) > 0]
            if not claimants:
                continue
            if 0 < supply < len(claimants):
                winners = [a["zone_id"] for a in claimants[:supply]]
                rerouted = [a["zone_id"] for a in claimants[supply:]]
                bids = [f"{a['zone_id']} ({PRIORITY_LEVELS[max(0, a['tier'])]} tier, {a['emergency_calls']} calls, "
                        f"{a['headcount_estimate']} people, confidence {a['decision_confidence']:.0%}"
                        + (", uncorroborated" if a["basis"] in UNCORROBORATED else "") + ")" for a in claimants]
                negotiations.append({"resource_type": resource, "supply": supply, "bids": bids,
                                     "agreement": winners, "rerouted": rerouted})
                ctx.think("DEBATE", f"Negotiation for {_units(supply, resource)} with {len(claimants)} claimant zones. "
                          f"Bids: {'; '.join(bids)}. Agreement: {', '.join(winners) or 'nobody'} first "
                          f"(severity, corroboration, then demand); {', '.join(rerouted)} rerouted to "
                          f"{KNOWLEDGE.SECONDARY_RESPONDERS.get(resource, 'the state EOC')}.")
            # The Safety Auditor challenges any unit held by a lower-ranked zone while a
            # higher-ranked zone that needs the same resource has none.
            for claimant in claimants:
                if grants[claimant["zone_id"]].get(resource, 0) > 0:
                    continue
                donors = [h for h in claimants if grants[h["zone_id"]].get(resource, 0) > 0
                          and position[h["zone_id"]] > position[claimant["zone_id"]]]
                if not donors:
                    continue
                donor = max(donors, key=lambda h: position[h["zone_id"]])
                grants[donor["zone_id"]][resource] -= 1
                grants[claimant["zone_id"]][resource] = 1
                objections.append({"resource_type": resource, "from": donor["zone_id"], "to": claimant["zone_id"]})
                ctx.think("DEBATE", f"Safety Auditor objects: {donor['zone_id']} (ranked lower) holds {resource} while "
                          f"{claimant['zone_id']} (ranked higher) has none. Allocator concedes and moves one unit.")
        if not objections:
            ctx.think("DEBATE", "Safety Auditor reviewed the allocation: no zone is served ahead of a higher-ranked "
                      "zone that needs the same resource. Consensus reached.")
        return {"negotiations": negotiations, "objections": objections}

    # ---- reflexion ------------------------------------------------------------------

    def _reflexion(self, shortfalls, state, ctx) -> list[dict[str, Any]]:
        revisions = []
        for s in shortfalls:
            free = state.remaining(s["resource_type"])
            if free <= 0 or s["shortfall"] <= 0:
                continue
            action = next((x for x in state.actions.values() if x["zone_id"] == s["zone_id"]
                           and x["type"] == "dispatch" and x["status"] == "pending_approval"), None)
            if action is None:
                continue
            if not revisions:
                ctx.think("REFLECT", "Reflexion: units were freed after the first allocation while zones are still "
                          "short; revising the plan once.")
            quantity = min(free, s["shortfall"])
            outcome = ctx.call("ALLOCATE", "reserve_resources",
                               {"zone_id": s["zone_id"], "resource_type": s["resource_type"], "quantity": quantity},
                               f"Reflexion revision: {_units(quantity, s['resource_type'])} freed; assigning to {s['zone_id']}.",
                               s["zone_id"])
            if outcome["ok"]:
                action["resources"][s["resource_type"]] = action["resources"].get(s["resource_type"], 0) + quantity
                s["granted"] += quantity
                s["shortfall"] -= quantity
                revisions.append({"zone_id": s["zone_id"], "resource_type": s["resource_type"], "quantity": quantity})
        if revisions:
            ctx.think("REFLECT", f"Revision applied: {len(revisions)} change(s); the plan now uses every free unit.")
        return revisions

    # ---- proposals --------------------------------------------------------------------

    @staticmethod
    def _propose(a: dict[str, Any], state: RunState, shortfalls: list[dict[str, Any]]) -> dict[str, Any] | None:
        action_id = f"ACT-{len(state.actions) + 1:03d}"
        base = {"action_id": action_id, "zone_id": a["zone_id"], "decision_confidence": a.get("decision_confidence")}
        if a["status"] != "ok":
            return {**base, "type": "verification", "resources": {}, "evacuation": False,
                    "summary": "Field verification: no automated assessment possible.",
                    "recommended_actions": ["Send a field team to verify conditions"]}
        if a["priority_index"] >= URGENT_INDEX and a["basis"] == "forecast_only":
            return {**base, "type": "standby_alert", "level": a["priority"], "resources": {}, "evacuation": False,
                    "summary": "Forecast-driven URGENT: stand teams by and verify the gauge; no assets until corroborated.",
                    "recommended_actions": ["Stand field teams by", "Verify the river gauge before committing assets"]}
        if a["priority_index"] >= URGENT_INDEX:
            resources = {r: q for r, q in state.reserved.get(a["zone_id"], {}).items() if q}
            evacuation = bool(a["needs"].get("shelter_beds"))
            parts = [_units(q, r) for r, q in resources.items()] or ["no assets in stock"]
            recommended = [f"Dispatch {', '.join(parts)} to {a['zone_id']} now"] if resources else \
                [f"No unit in stock for {a['zone_id']}: rely on secondary responders"]
            recommended += [f"Reroute {_units(s['shortfall'], s['resource_type'])} for {a['zone_id']} to "
                            f"{KNOWLEDGE.SECONDARY_RESPONDERS.get(s['resource_type'], 'the state EOC')}"
                            for s in shortfalls if s["zone_id"] == a["zone_id"]]
            if evacuation:
                recommended.append(f"Notify shelter of ~{a['headcount_estimate']} incoming evacuees")
            return {**base, "type": "dispatch", "level": a["priority"], "resources": resources, "evacuation": evacuation,
                    "summary": f"{a['priority']} dispatch to {a['zone_id']}: {', '.join(parts)}"
                               + (f"; evacuate ~{a['headcount_estimate']} people" if evacuation else ""),
                    "recommended_actions": recommended}
        if a["priority_index"] == 1:
            return {**base, "type": "standby_alert", "level": "ELEVATED", "resources": {}, "evacuation": False,
                    "summary": "Standby: alert field teams and pre-position pumps and barriers.",
                    "recommended_actions": ["Alert field teams", "Pre-position pumps and barriers"]}
        return None

    @staticmethod
    def _zone_view(a: dict[str, Any], grant: dict[str, int], state: RunState) -> dict[str, Any]:
        view = {k: v for k, v in a.items() if k not in {"fusion", "observations"}}
        view["allocated"] = {r: q for r, q in grant.items() if q}
        view["perception"] = a["observations"]
        view["action_ids"] = [x["action_id"] for x in state.actions.values() if x["zone_id"] == a["zone_id"]]
        return view

    def _notify_shelter(self, action, assessment, ctx) -> None:
        if not action.get("evacuation") or not assessment.get("headcount_estimate"):
            return
        ctx.call("COMMIT", "notify_shelter",
                 {"zone_id": action["zone_id"], "evacuees": int(assessment["headcount_estimate"]),
                  "beds_reserved": int(action["resources"].get("shelter_beds", 0))},
                 f"{action['action_id']} includes an evacuation; telling the shelter to expect "
                 f"~{assessment['headcount_estimate']} people.", action["zone_id"])

    # ---- human in the loop ---------------------------------------------------

    def decide(self, agent_run: AgentRun, action_id: str, decision: str, operator: str = "operator",
               note: str = "") -> dict[str, Any]:
        """Apply a human approve / reject decision to a held action."""
        state = agent_run.state
        if state.halted:
            raise ValueError("This run was halted by an emergency override; no further decisions apply")
        action = state.actions.get(action_id)
        if action is None:
            raise ValueError(f"Unknown action_id {action_id!r}")
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")
        if action["status"] != "pending_approval":
            raise ValueError(f"Action {action_id} is {action['status']}, not awaiting approval")
        ctx = RunContext(self.registry, state, self.graph)
        event = {"type": "decision", "action_id": action_id, "decision": decision, "operator": operator,
                 "note": note, "decided_at": _now()}
        if decision == "approve":
            action["approved_by"] = operator
            if action["type"] == "verification":
                action["status"] = "committed"
                ctx.think("COMMIT", f"{operator} confirmed field verification for {action['zone_id']}.",
                          action["zone_id"])
            else:
                token = secrets.token_hex(16)
                state.approval_tokens[action_id] = token
                outcome = ctx.call("COMMIT", "commit_rescue_dispatch",
                                   {"zone_id": action["zone_id"], "action_id": action_id, "approval_token": token},
                                   f"{operator} approved {action_id}; committing with the issued token.",
                                   action["zone_id"])
                event["commit"] = outcome["data"] if outcome["ok"] else None
                if outcome["ok"]:
                    zone = next((z for z in agent_run.record.get("zones", []) if z["zone_id"] == action["zone_id"]), {})
                    self._notify_shelter(action, zone, ctx)
                else:
                    event["error"] = outcome["error"]
        else:
            for resource, quantity in action.get("resources", {}).items():
                if quantity:
                    ctx.call("COMMIT", "release_resources",
                             {"zone_id": action["zone_id"], "resource_type": resource, "quantity": quantity},
                             f"{operator} rejected {action_id}; releasing {quantity} {resource}.", action["zone_id"])
            action["status"] = "rejected"
            action["rejected_by"] = operator
        for escalation in state.escalations:
            if escalation.get("action_id") == action_id:
                escalation["status"] = f"resolved:{decision}"
        event["steps"] = ctx.steps
        state.human_events.append(event)
        state.post("commander", f"decision:{action_id}", {"decision": decision, "operator": operator})
        if self.registry.memory_path is not None:
            self.registry.remember([{"type": "commander_decision", "run_id": state.run_id, "created_at": _now(),
                                     "state": next((z.get("state") for z in agent_run.record.get("zones", [])
                                                    if z["zone_id"] == action["zone_id"]), None),
                                     "district": next((z.get("district") for z in agent_run.record.get("zones", [])
                                                       if z["zone_id"] == action["zone_id"]), None),
                                     "zone_id": action["zone_id"], "action_id": action_id, "decision": decision,
                                     "operator": operator}])
        return jsonable({"action": action, "event": event, "inventory": state.inventory_view()})

    def override(self, agent_run: AgentRun, operator: str, reason: str, action_id: str | None = None) -> dict[str, Any]:
        """Emergency override: recall one action, or halt the whole run.

        Pending actions are cancelled and their units released, committed dispatches
        are recalled through the override-token tool, cleared standby alerts are
        stood down. A halted run refuses every further commit.
        """
        state = agent_run.state
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("'operator' must name the commander issuing the override")
        if not isinstance(reason, str) or len(reason.strip()) < 3:
            raise ValueError("'reason' must explain the override")
        if action_id is not None and action_id not in state.actions:
            raise ValueError(f"Unknown action_id {action_id!r}")
        operator, reason = operator.strip()[:80], reason.strip()[:300]
        ctx = RunContext(self.registry, state, self.graph)
        targets = [state.actions[action_id]] if action_id else list(state.actions.values())
        affected = []
        for action in targets:
            before, zid, aid = action["status"], action["zone_id"], action["action_id"]
            if before in {"pending_approval", "approved"}:
                for resource, quantity in action.get("resources", {}).items():
                    if quantity:
                        ctx.call("OVERRIDE", "release_resources",
                                 {"zone_id": zid, "resource_type": resource, "quantity": quantity},
                                 f"{operator} overrode {aid}; releasing {quantity} {resource}.", zid)
                action["status"] = "cancelled"
            elif before == "committed" and action["type"] == "dispatch":
                token = secrets.token_hex(16)
                state.override_tokens[aid] = token
                ctx.call("OVERRIDE", "recall_dispatch",
                         {"zone_id": zid, "action_id": aid, "override_token": token, "reason": reason},
                         f"{operator} recalled committed {aid}: {reason}.", zid)
            elif before == "committed" and action["type"] == "standby_alert":
                ctx.call("OVERRIDE", "issue_alert",
                         {"zone_id": zid, "level": "ROUTINE", "message": f"Stand down: overridden by {operator} ({reason})"},
                         f"{operator} stood down the alert for {zid}.", zid)
                action["status"] = "stood_down"
            else:
                continue
            action["overridden_by"] = operator
            affected.append({"action_id": aid, "from": before, "to": action["status"]})
        if action_id is None:
            state.halted = True
        for escalation in state.escalations:
            if escalation.get("action_id") in {x["action_id"] for x in affected}:
                escalation["status"] = "resolved:override"
        event = {"type": "override", "operator": operator, "reason": reason, "action_id": action_id,
                 "halted": state.halted, "affected": affected, "decided_at": _now(), "steps": ctx.steps}
        state.human_events.append(event)
        state.post("commander", "override", {"operator": operator, "halted": state.halted, "affected": len(affected)})
        return jsonable({"event": event, "halted": state.halted, "inventory": state.inventory_view()})


def tool_stats(steps: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [s for s in steps if s.get("tool")]
    seen, redundant = set(), 0
    for s in calls:
        key = (s["tool"], s.get("arguments_sha"), s.get("ok"))
        if key in seen and s["tool"] not in {"escalate_to_human", "reserve_resources", "release_resources",
                                             "request_mutual_aid", "notify_shelter", "issue_alert"}:
            redundant += 1
        seen.add(key)
    by_error: dict[str, int] = {}
    for s in calls:
        if not s["ok"]:
            by_error[s["error_type"]] = by_error.get(s["error_type"], 0) + 1
    by_tool: dict[str, int] = {}
    for s in calls:
        by_tool[s["tool"]] = by_tool.get(s["tool"], 0) + 1
    return {"calls": len(calls), "ok": sum(bool(s["ok"]) for s in calls), "errors": by_error,
            "invalid_arguments": by_error.get("invalid_arguments", 0), "redundant": redundant,
            "by_tool": by_tool, "steps": len(steps)}


# ===========================================================================
# Wiring to the real stage models
# ===========================================================================

def load_briefer() -> Callable[[str], dict]:
    """Stage 04's baseline briefer.

    The Qwen adapter takes ~10 s per briefing on the reference GPU, which is too
    slow inside an agent loop over many zones; the baseline is what Stage 05's
    stress test uses for the same reason. Importing 03_slm_engineer installs its
    DNS override for three Hugging Face CDN hostnames in this process; nothing
    here contacts them.
    """
    module = _load_module("stage06_slm_engineer", REPO_ROOT / "Stage04_SLM" / "03_slm_engineer.py")
    return module.SLMBaseline.load(module.BASELINE_DIR).generate


def load_stage_engines() -> tuple[dict[str, Any], dict[str, str]]:
    """Stage 01-03 adapters through Stage 05's loader, which also lets Stage 02
    read Stage 05's imagery bank without modifying Stage 02."""
    stage05_eval = _load_module("stage06_stage05_eval", STAGE05_DIR / "04_evaluation_engineer.py")
    engines, status = stage05_eval.load_stage_engines("none")
    try:
        engines["briefer"], status["slm"] = load_briefer(), "healthy"
    except Exception as exc:
        engines["briefer"], status["slm"] = None, f"unavailable ({type(exc).__name__}: {exc})"
    return engines, status


def wrap_dl_for_stage05(dl_engine):
    """Give an already-loaded Stage 02 adapter access to Stage 05's imagery bank."""
    if dl_engine is None:
        return None
    stage05_eval = _load_module("stage06_stage05_eval", STAGE05_DIR / "04_evaluation_engineer.py")
    return stage05_eval.ImageryAwareDLAdapter(dl_engine)


def build_registry(kb=None, engines: dict[str, Any] | None = None, memo_store: dict | None = None,
                   ledger_path: Path | None = None, memory_path: Path | None = None) -> ToolRegistry:
    kb = kb or KNOWLEDGE.KnowledgeBase()
    engines = engines or {}
    return ToolRegistry(kb, ml=engines.get("ml"), dl=engines.get("dl"), nlp=engines.get("nlp"),
                        briefer=engines.get("briefer"), memo_store=memo_store, ledger_path=ledger_path,
                        memory_path=memory_path)


def load_stage05_scenarios(suite: str = "cvae") -> list[dict[str, Any]]:
    prefix = "" if suite == "cvae" else f"{suite}_"
    scenarios = json.loads((STAGE05_SCENARIOS / f"{prefix}scenario_suite.json").read_text(encoding="utf-8"))["scenarios"]
    wildcard_path = STAGE05_SCENARIOS / f"{prefix}wildcard_scenario.json"
    if wildcard_path.exists():
        scenarios.append(json.loads(wildcard_path.read_text(encoding="utf-8")))
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 06 agent demo run")
    parser.add_argument("--no-models", action="store_true", help="knowledge tools only, no Stage 01-04 models")
    parser.add_argument("--scenario", default="W01", help="Stage 05 scenario id (default: the wildcard)")
    parser.add_argument("--suite", default="cvae", choices=["cvae", "slm", "llm"])
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 06 Agent Engineer -- demo run")
    print("=" * 60)
    kb = KNOWLEDGE.KnowledgeBase()
    engines, status = ({}, {"models": "skipped"}) if args.no_models else load_stage_engines()
    print(f"  stage status: {status}")
    agent = CoordinationAgent(build_registry(kb, engines))
    scenario = next(s for s in load_stage05_scenarios(args.suite) if s["scenario_id"] == args.scenario)
    result = agent.run(scenario).to_dict()

    print(f"\n  {scenario['scenario_id']} {scenario['name']}  ({result['latency_ms']} ms, "
          f"{result['tool_stats']['calls']} tool calls)")
    for step in result["trajectory"]:
        tool = f" -> {step['tool']}: {step['observation']}" if step.get("tool") else ""
        print(f"  {step['step']:>3} {step['node']:<17} {step['thought'][:110]}{tool[:120]}")
    print(f"\n  workflow violations: {result['workflow_violations'] or 'none'}")
    print(f"  pending approvals: {result['pending_approvals']}, commits: {len(result['commits'])}, "
          f"mutual aid: {len(result['mutual_aid'])}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_RUN_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  -> {DEMO_RUN_JSON.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
