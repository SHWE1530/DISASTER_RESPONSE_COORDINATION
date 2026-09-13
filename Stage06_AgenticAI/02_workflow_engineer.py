"""Stage 06 Agentic AI -- Workflow Engineer.

Breaks the coordination mission into tasks, maps every task to the agent that
owns it and the tools it may call, and encodes the result as a state graph the
agent is held to at run time.

WorkflowGraph
  nodes / edges            the legal states of one coordination run
  validate(tool_names)     every node reachable, END reachable, every tool mapped
  conformance(path)        illegal transitions in an executed trajectory
  tools_for(node)          the tools a node is allowed to call

Outputs (data/outputs/workflow/)
  workflow_graph.json      nodes, edges, task decomposition, tool mapping
  tool_mapping.csv         task -> node -> agent -> tool -> trigger
  workflow.md              the same, with a Mermaid diagram

The graph is not decoration: 03_agent_engineer.py checks every step it takes
against it, and 04_evaluation_engineer.py reports the violations.

Usage:
  python Stage06_AgenticAI/02_workflow_engineer.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = BASE_DIR / "data" / "outputs" / "workflow"
GRAPH_JSON = WORKFLOW_DIR / "workflow_graph.json"
MAPPING_CSV = WORKFLOW_DIR / "tool_mapping.csv"
WORKFLOW_MD = WORKFLOW_DIR / "workflow.md"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KNOWLEDGE = _load_module("stage06_knowledge_engineer", BASE_DIR / "01_knowledge_engineer.py")

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

AGENTS = {
    "coordinator": "Coordinator: owns the goal, runs the loop, ranks zones, reflects on the run",
    "dispatcher": "Tactical Dispatcher: perceives each zone through the Stage 01-04 tools and fuses the evidence",
    "allocator": "Resource Allocator: turns ranked needs into reservations against a finite inventory",
    "auditor": "Safety Auditor: applies the safety rules and routes anything consequential to a human",
    "commander": "Human incident commander: approves, rejects, overrides or halts",
}

# ---------------------------------------------------------------------------
# Nodes. `tools` is the whitelist for that state; a call outside it is a
# workflow violation even if the tool itself succeeds.
# ---------------------------------------------------------------------------

NODES: dict[str, dict[str, Any]] = {
    "START": {"agent": "coordinator", "loop": "-", "tools": [],
              "description": "Receive the incident picture: zones, raw inputs, inventory."},
    "PLAN": {"agent": "coordinator", "loop": "plan", "tools": [],
             "description": "Plan-and-execute: write the whole task plan from the evidence first, then check it "
                            "for duplicate, missing or illegal steps."},
    "PERCEIVE": {"agent": "dispatcher", "loop": "perceive", "tools": [
        "query_sensor_risk_score", "parse_emergency_text", "predict_visual_flood", "forecast_water_level"],
        "description": "Call one perception tool per evidence stream the zone actually has."},
    "CONTEXTUALISE": {"agent": "dispatcher", "loop": "perceive", "tools": ["get_district_profile", "recall_precedents"],
                      "description": "Pull the district's flood history, and precedents from past runs (memory)."},
    "ASSESS": {"agent": "dispatcher", "loop": "reason", "tools": ["assess_zone"],
               "description": "Fuse the evidence into one priority with conflicts and a review flag."},
    "BRIEF": {"agent": "dispatcher", "loop": "reason", "tools": ["generate_tactical_briefing"],
              "description": "Condense the zone's incident log into a Stage 04 briefing."},
    "RETRIEVE_SOP": {"agent": "dispatcher", "loop": "reason", "tools": ["search_sop"],
                     "description": "Ground the response in the national SOP corpus."},
    "ESCALATE_EVIDENCE": {"agent": "auditor", "loop": "act", "tools": ["escalate_to_human"],
                          "description": "No usable evidence: request field verification, dispatch nothing."},
    "RANK": {"agent": "coordinator", "loop": "plan", "tools": [],
             "description": "Order zones by priority, fused score and estimated headcount."},
    "ALLOCATE": {"agent": "allocator", "loop": "plan", "tools": [
        "check_resource_inventory", "reserve_resources"],
        "description": "Tree of thoughts: score alternative allocation plans and keep the best; then reserve "
                       "units without over-committing."},
    "DEBATE": {"agent": "allocator", "loop": "plan", "tools": [],
               "description": "Multi-agent debate and negotiation: zones bid for contested units, the Safety "
                              "Auditor challenges the proposal, and the consensus is recorded."},
    "AUDIT": {"agent": "auditor", "loop": "plan", "tools": [],
              "description": "Apply the safety rules to every proposed action."},
    "AWAIT_APPROVAL": {"agent": "auditor", "loop": "act", "tools": ["escalate_to_human"],
                       "description": "Hold consequential actions for a human sign-off."},
    "MUTUAL_AID": {"agent": "allocator", "loop": "act", "tools": ["request_mutual_aid", "escalate_to_human"],
                   "description": "Record every shortfall as a mutual-aid request and tell a human."},
    "ALERT": {"agent": "coordinator", "loop": "act", "tools": ["issue_alert"],
              "description": "Issue low-consequence standby alerts the auditor cleared automatically."},
    "COMMIT": {"agent": "coordinator", "loop": "act",
               "tools": ["commit_rescue_dispatch", "release_resources", "notify_shelter"],
               "description": "Commit an approved dispatch and notify the shelter, or release a rejected one."},
    "REFLECT": {"agent": "coordinator", "loop": "reflect", "tools": [],
                "description": "Reflexion: check the goal, and revise the allocation once if units were freed "
                               "while zones are still short."},
    "OVERRIDE": {"agent": "commander", "loop": "act", "tools": ["recall_dispatch", "release_resources", "issue_alert"],
                 "description": "Emergency override: a human halts the run or recalls a dispatch at any time."},
    "END": {"agent": "coordinator", "loop": "-", "tools": [], "description": "Run complete."},
}

_ZONE_EXIT = ["PERCEIVE", "RANK"]  # next zone, or all zones done
_ACT = ["AWAIT_APPROVAL", "MUTUAL_AID", "ALERT", "COMMIT", "REFLECT"]

EDGES: dict[str, list[str]] = {
    "START": ["PLAN"],
    "PLAN": ["PERCEIVE", "RANK"],
    "PERCEIVE": ["PERCEIVE", "CONTEXTUALISE", "ASSESS", "ESCALATE_EVIDENCE"],
    "CONTEXTUALISE": ["ASSESS"],
    "ASSESS": ["BRIEF", "RETRIEVE_SOP", "ESCALATE_EVIDENCE", *_ZONE_EXIT],
    "BRIEF": ["RETRIEVE_SOP", *_ZONE_EXIT],
    "RETRIEVE_SOP": _ZONE_EXIT,
    "ESCALATE_EVIDENCE": _ZONE_EXIT,
    "RANK": ["ALLOCATE"],
    "ALLOCATE": ["ALLOCATE", "DEBATE", "AUDIT", "REFLECT"],
    "DEBATE": ["DEBATE", "ALLOCATE", "AUDIT"],
    "AUDIT": _ACT,
    "AWAIT_APPROVAL": ["AWAIT_APPROVAL", "MUTUAL_AID", "ALERT", "COMMIT", "REFLECT"],
    "MUTUAL_AID": ["MUTUAL_AID", "AWAIT_APPROVAL", "ALERT", "COMMIT", "REFLECT"],
    "ALERT": ["ALERT", "COMMIT", "REFLECT"],
    "COMMIT": ["COMMIT", "REFLECT"],
    "REFLECT": ["REFLECT", "ALLOCATE", "END"],
    "END": [],
}

# After a run ends, a human decision re-enters the graph here.
HUMAN_ENTRY = {"END": ["COMMIT", "OVERRIDE"], "COMMIT": ["COMMIT", "OVERRIDE", "END"], "OVERRIDE": ["OVERRIDE", "END"]}

# ---------------------------------------------------------------------------
# Task decomposition: the mission broken into subtasks with success criteria.
# ---------------------------------------------------------------------------

GOAL = ("Stabilise every zone of a multi-zone flood incident: each URGENT or CRITICAL zone "
        "receives assets or an explicit shortage escalation, every consequential action is "
        "signed off by a human, and no zone without evidence is dispatched to blind.")

TASKS: list[dict[str, Any]] = [
    {"id": "T0", "task": "Plan the whole response first", "node": "PLAN", "agent": "coordinator",
     "success": "one plan step per evidence stream and decision; no duplicate or missing steps", "tools": []},
    {"id": "T1", "task": "Perceive each zone", "node": "PERCEIVE", "agent": "dispatcher",
     "success": "every present evidence stream is read by its own stage tool",
     "tools": [("query_sensor_risk_score", "zone has sensor readings"),
               ("parse_emergency_text", "zone has a text report"),
               ("predict_visual_flood", "zone has an image"),
               ("forecast_water_level", "zone has a 72 h gauge history")]},
    {"id": "T2", "task": "Add historical context", "node": "CONTEXTUALISE", "agent": "dispatcher",
     "success": "district profile attached when the district is known; precedents recalled when they exist",
     "tools": [("get_district_profile", "state and district are known"),
               ("recall_precedents", "memory holds past decisions for the district")]},
    {"id": "T3", "task": "Fuse evidence into a priority", "node": "ASSESS", "agent": "dispatcher",
     "success": "one priority, conflicts surfaced, review flag set",
     "tools": [("assess_zone", "at least one evidence stream exists")]},
    {"id": "T4", "task": "Brief the incident log", "node": "BRIEF", "agent": "dispatcher",
     "success": "briefing attached; disagreement with fusion surfaced, never averaged",
     "tools": [("generate_tactical_briefing", "zone has a multi-entry incident log")]},
    {"id": "T5", "task": "Ground the response in SOPs", "node": "RETRIEVE_SOP", "agent": "dispatcher",
     "success": "cited SOP passages for every ELEVATED+ zone",
     "tools": [("search_sop", "fused priority is ELEVATED or higher")]},
    {"id": "T6", "task": "Handle zones with no evidence", "node": "ESCALATE_EVIDENCE", "agent": "auditor",
     "success": "verification requested, zero assets dispatched",
     "tools": [("escalate_to_human", "no evidence stream, or fusion returned insufficient_evidence")]},
    {"id": "T7", "task": "Rank zones and reason through trade-offs", "node": "RANK", "agent": "coordinator",
     "success": "severity tier first; between equally severe zones, corroboration and field demand "
                "(emergency calls, people) decide, with the reasoning written out", "tools": []},
    {"id": "T8", "task": "Explore allocation plans and reserve", "node": "ALLOCATE", "agent": "allocator",
     "success": "several allocation plans scored (tree of thoughts), the best kept; no inventory over-commit",
     "tools": [("check_resource_inventory", "start of allocation"),
               ("reserve_resources", "a ranked zone needs a resource type that is in stock")]},
    {"id": "T8b", "task": "Debate and negotiate contested units", "node": "DEBATE", "agent": "allocator",
     "success": "every contested unit has recorded bids and an agreement; ranking inversions are challenged "
                "and corrected", "tools": []},
    {"id": "T9", "task": "Audit proposed actions", "node": "AUDIT", "agent": "auditor",
     "success": "every action has a verdict and the rules that produced it", "tools": []},
    {"id": "T10", "task": "Hold consequential actions for sign-off", "node": "AWAIT_APPROVAL", "agent": "auditor",
     "success": "every URGENT+ dispatch and every evacuation waits for a human",
     "tools": [("escalate_to_human", "verdict is requires_human_approval")]},
    {"id": "T11", "task": "Close shortfalls", "node": "MUTUAL_AID", "agent": "allocator",
     "success": "each unmet need becomes a mutual-aid request and a human escalation",
     "tools": [("request_mutual_aid", "need exceeds remaining inventory"),
               ("escalate_to_human", "a shortfall exists")]},
    {"id": "T12", "task": "Issue cleared alerts", "node": "ALERT", "agent": "coordinator",
     "success": "standby alerts only for actions the auditor auto-approved",
     "tools": [("issue_alert", "verdict is auto_approved")]},
    {"id": "T13", "task": "Commit or release", "node": "COMMIT", "agent": "coordinator",
     "success": "dispatch committed only with a human approval token; shelter notified of evacuees; "
                "rejected reservations released",
     "tools": [("commit_rescue_dispatch", "a human approved the action"),
               ("notify_shelter", "an approved dispatch includes an evacuation"),
               ("release_resources", "a human rejected the action")]},
    {"id": "T14", "task": "Reflect and revise", "node": "REFLECT", "agent": "coordinator",
     "success": "unserved high-priority zones listed; the allocation revised once if units were freed", "tools": []},
    {"id": "T15", "task": "Honour an emergency override", "node": "OVERRIDE", "agent": "commander",
     "success": "a halt cancels pending actions, recalls committed dispatches and blocks further commits",
     "tools": [("recall_dispatch", "a committed dispatch is overridden"),
               ("release_resources", "a pending action is cancelled"),
               ("issue_alert", "a standby alert is stood down")]},
]


# Reasoning and coordination patterns, and where this project uses each.
PATTERNS: list[dict[str, str]] = [
    {"family": "reasoning", "pattern": "ReAct", "status": "implemented",
     "where": "every step records a thought, the tool it calls and the observation (03, RunContext)"},
    {"family": "reasoning", "pattern": "Plan-and-Execute", "status": "implemented",
     "where": "PLAN node writes the full plan first; check_plan() audits it; adherence reported (03, 04)"},
    {"family": "reasoning", "pattern": "Reflexion", "status": "implemented",
     "where": "REFLECT revises the allocation once when units were freed while zones stay short (03)"},
    {"family": "reasoning", "pattern": "Tree of Thoughts", "status": "implemented",
     "where": "ALLOCATE scores three allocation plans with an explicit utility and keeps the best (03)"},
    {"family": "reasoning", "pattern": "Function / Tool Calling", "status": "implemented",
     "where": "schema-validated tool registry, also served over MCP (01, 03, 05)"},
    {"family": "reasoning", "pattern": "Memory-Augmented Agent", "status": "implemented",
     "where": "recall_precedents reads past agent and commander decisions for the district (03)"},
    {"family": "reasoning", "pattern": "Multi-Agent Debate", "status": "implemented",
     "where": "DEBATE: the Safety Auditor challenges the allocator's proposal; consensus recorded (03)"},
    {"family": "reasoning", "pattern": "Human-in-the-Loop", "status": "implemented",
     "where": "approval tokens, confidence escalation, emergency override (03, 05)"},
    {"family": "coordination", "pattern": "Orchestrator-Worker", "status": "implemented",
     "where": "Coordinator delegates to Dispatcher, Allocator, Auditor"},
    {"family": "coordination", "pattern": "Peer-to-Peer Negotiation", "status": "implemented",
     "where": "zones bid for contested units in DEBATE; the agreement and the rerouted zones are recorded"},
    {"family": "coordination", "pattern": "Blackboard / Shared Memory", "status": "implemented",
     "where": "RunState.blackboard: every agent posts its findings to one shared workspace"},
    {"family": "coordination", "pattern": "Hierarchical Agents", "status": "implemented",
     "where": "escalations routed to field supervisor, district EOC, state EOC or incident commander"},
    {"family": "coordination", "pattern": "Swarm Coordination", "status": "not applicable",
     "where": "needs many simple autonomous agents acting locally; one incident command does not fit it"},
    {"family": "control", "pattern": "Approval Gate", "status": "implemented",
     "where": "commit_rescue_dispatch refuses held actions without a human token"},
    {"family": "control", "pattern": "Emergency Override", "status": "implemented",
     "where": "CoordinationAgent.override(): halt the run, recall committed dispatches"},
    {"family": "control", "pattern": "Confidence Threshold", "status": "implemented",
     "where": "decision confidence per zone; below the threshold a dispatch is escalated (rule R10)"},
    {"family": "control", "pattern": "Audit Trail", "status": "implemented",
     "where": "trajectory, blackboard, human events and the dispatch ledger"},
    {"family": "control", "pattern": "Guardrails / Policy Constraints", "status": "implemented",
     "where": "Safety Auditor rules R1-R10"},
]

# Example plan with a duplicated step and a missing dispatch step, used to show check_plan().
EXAMPLE_FLAWED_PLAN = [
    {"id": "S1", "subtask": "check ambulance availability", "node": "ALLOCATE", "tool": "check_resource_inventory"},
    {"id": "S2", "subtask": "check ambulance availability", "node": "ALLOCATE", "tool": "check_resource_inventory"},
    {"id": "S3", "subtask": "notify shelter of incoming evacuees", "node": "COMMIT", "tool": "notify_shelter"},
]


class WorkflowGraph:
    """The legal state machine of one coordination run."""

    def __init__(self, nodes: dict | None = None, edges: dict | None = None, human_entry: dict | None = None) -> None:
        self.nodes = nodes or NODES
        self.edges = edges or EDGES
        self.human_entry = HUMAN_ENTRY if human_entry is None else human_entry

    def _all_edges(self) -> dict[str, list[str]]:
        merged = {node: list(targets) for node, targets in self.edges.items()}
        for node, targets in self.human_entry.items():
            merged.setdefault(node, [])
            merged[node] += [t for t in targets if t not in merged[node]]
        return merged

    def tools_for(self, node: str) -> list[str]:
        return list(self.nodes[node]["tools"])

    def agent_for(self, node: str) -> str:
        return self.nodes[node]["agent"]

    def is_legal(self, previous: str, following: str) -> bool:
        return following in self.edges.get(previous, [])

    def _reachable(self, start: str, edges: dict[str, list[str]]) -> set[str]:
        seen, queue = {start}, deque([start])
        while queue:
            for nxt in edges.get(queue.popleft(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def validate(self, tool_names: Iterable[str]) -> list[str]:
        """Structural problems; an empty list means the graph is sound."""
        problems: list[str] = []
        tool_names = set(tool_names)
        edges = self._all_edges()
        for node, targets in edges.items():
            if node not in self.nodes:
                problems.append(f"edge source {node} is not a node")
            for target in targets:
                if target not in self.nodes:
                    problems.append(f"edge {node}->{target} targets an unknown node")
        unreachable = set(self.nodes) - self._reachable("START", edges)
        if unreachable:
            problems.append(f"unreachable from START: {sorted(unreachable)}")
        reverse: dict[str, list[str]] = {}
        for node, targets in edges.items():
            for target in targets:
                reverse.setdefault(target, []).append(node)
        dead_ends = set(self.nodes) - self._reachable("END", reverse)
        if dead_ends:
            problems.append(f"cannot reach END: {sorted(dead_ends)}")
        mapped = {tool for spec in self.nodes.values() for tool in spec["tools"]}
        for tool in sorted(mapped - tool_names):
            problems.append(f"node maps unknown tool {tool}")
        for tool in sorted(tool_names - mapped):
            problems.append(f"tool {tool} is not reachable from any node")
        return problems

    def conformance(self, steps: list[dict[str, Any]]) -> list[str]:
        """Illegal transitions and out-of-whitelist tool calls in a trajectory."""
        violations: list[str] = []
        previous = "START"
        for step in steps:
            node = step["node"]
            if node not in self.nodes:
                violations.append(f"step {step.get('step')}: unknown node {node}")
                continue
            if node != previous and not self.is_legal(previous, node):
                violations.append(f"step {step.get('step')}: illegal transition {previous} -> {node}")
            tool = step.get("tool")
            if tool and tool not in self.nodes[node]["tools"]:
                violations.append(f"step {step.get('step')}: tool {tool} not allowed in {node}")
            previous = node
        return violations

    def check_plan(self, plan: list[dict[str, Any]], required_nodes: Iterable[str] = (),
                   required_tools: Iterable[str] = ()) -> dict[str, Any]:
        """Duplicate, missing and illegal steps in a plan, before anything is executed."""
        seen: dict[tuple, str] = {}
        duplicates, illegal = [], []
        for step in plan:
            key = (step.get("zone_id"), step.get("tool") or step.get("subtask"))
            if key in seen:
                duplicates.append(f"{step.get('id')} repeats {seen[key]} ({step.get('subtask')})")
            else:
                seen[key] = step.get("id")
            node = step.get("node")
            if node not in self.nodes:
                illegal.append(f"{step.get('id')}: unknown node {node}")
            elif step.get("tool") and step["tool"] not in self.nodes[node]["tools"]:
                illegal.append(f"{step.get('id')}: {step['tool']} is not allowed in {node}")
        nodes = {step.get("node") for step in plan}
        tools = {step.get("tool") for step in plan}
        missing = [f"no {n} step" for n in required_nodes if n not in nodes]
        missing += [f"no step calls {t}" for t in required_tools if t not in tools]
        problems = duplicates + missing + illegal
        return {"ok": not problems, "duplicates": duplicates, "missing": missing, "illegal": illegal,
                "summary": "sound" if not problems else "; ".join(problems)}

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for node, spec in self.nodes.items():
            tools = "<br/>".join(spec["tools"]) if spec["tools"] else ""
            label = f"{node}<br/><i>{spec['agent']}</i>" + (f"<br/><small>{tools}</small>" if tools else "")
            lines.append(f'    {node}["{label}"]')
        for node, targets in self.edges.items():
            for target in targets:
                if target != node:
                    lines.append(f"    {node} --> {target}")
        return "\n".join(lines)


def tool_mapping_rows() -> list[dict[str, str]]:
    rows = []
    for task in TASKS:
        for tool, trigger in task["tools"] or [("-", "no tool: internal reasoning")]:
            rows.append({"task_id": task["id"], "task": task["task"], "node": task["node"],
                         "agent": task["agent"], "tool": tool, "trigger": trigger})
    return rows


def build_outputs(graph: WorkflowGraph | None = None) -> dict[str, Any]:
    graph = graph or WorkflowGraph()
    tool_names = [tool["name"] for tool in KNOWLEDGE.TOOL_DEFINITIONS]
    problems = graph.validate(tool_names)
    if problems:
        raise ValueError("Workflow graph is invalid:\n  " + "\n  ".join(problems))

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "goal": GOAL,
        "agents": AGENTS,
        "nodes": graph.nodes,
        "edges": graph.edges,
        "human_entry": HUMAN_ENTRY,
        "patterns": PATTERNS,
        "plan_check_example": {"plan": EXAMPLE_FLAWED_PLAN,
                               "result": graph.check_plan(EXAMPLE_FLAWED_PLAN, required_nodes=("ALLOCATE", "COMMIT"),
                                                          required_tools=("commit_rescue_dispatch",))},
        "tasks": [{**task, "tools": [{"tool": t, "trigger": c} for t, c in task["tools"]]} for task in TASKS],
        "validation": {"problems": problems, "tools": len(tool_names), "nodes": len(graph.nodes),
                       "edges": sum(len(v) for v in graph.edges.values())},
    }
    GRAPH_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = tool_mapping_rows()
    with MAPPING_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Stage 06 workflow", "", f"**Goal.** {GOAL}", "",
        "## Agents", "", *[f"- **{k}** — {v}" for k, v in AGENTS.items()], "",
        "## State graph", "", "```mermaid", graph.to_mermaid(), "```", "",
        "## Task decomposition and tool mapping", "",
        "| Task | Node | Agent | Tool | Trigger |", "| --- | --- | --- | --- | --- |",
        *[f"| {r['task_id']} {r['task']} | {r['node']} | {r['agent']} | `{r['tool']}` | {r['trigger']} |"
          for r in rows],
        "", "## Plan check example", "",
        "Goal: evacuate a zone. Plan: " + " → ".join(s["subtask"] for s in EXAMPLE_FLAWED_PLAN) + ".", "",
        f"`check_plan()` → {payload['plan_check_example']['result']['summary']}", "",
        "## Reasoning, coordination and control patterns", "",
        "| Family | Pattern | Status | Where |", "| --- | --- | --- | --- |",
        *[f"| {x['family']} | {x['pattern']} | {x['status']} | {x['where']} |" for x in PATTERNS],
        "",
    ]
    WORKFLOW_MD.write_text("\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    print("=" * 60)
    print("Stage 06 Workflow Engineer")
    print("=" * 60)
    payload = build_outputs()
    v = payload["validation"]
    print(f"  nodes {v['nodes']}, edges {v['edges']}, tools mapped {v['tools']}, problems {len(v['problems'])}")
    print(f"  tasks {len(TASKS)}")
    print(f"  plan check example -> {payload['plan_check_example']['result']['summary']}")
    print(f"  patterns: {sum(x['status'] == 'implemented' for x in PATTERNS)} implemented, "
          f"{sum(x['status'] != 'implemented' for x in PATTERNS)} not applicable")
    for path in (GRAPH_JSON, MAPPING_CSV, WORKFLOW_MD):
        print(f"  -> {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
