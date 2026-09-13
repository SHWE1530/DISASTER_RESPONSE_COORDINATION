# Stage 06 — Agentic AI

**Status: implemented. Evaluated end to end against the real Stage 01–04 models on both Stage 05 scenario suites.**

| Script | Role | Description |
| --- | --- | --- |
| `01_knowledge_engineer.py` | Data / Knowledge Engineer | SOP retrieval index, district flood profiles, resource catalog, secondary responders, glossary, and the 18 tools as MCP tool definitions |
| `02_workflow_engineer.py` | Workflow Engineer | Task decomposition, task → agent → tool mapping, the state graph the agent is held to, a plan checker, and the reasoning-pattern catalogue |
| `03_agent_engineer.py` | Agent Engineer | Coordinator + Tactical Dispatcher + Resource Allocator + Safety Auditor: plan-and-execute, ReAct perception, demand-aware trade-offs, tree of thoughts, debate and negotiation, reflexion, memory, confidence escalation, approval gate and emergency override |
| `04_evaluation_engineer.py` | Evaluation Engineer | Task success, reasoning accuracy, bad trade-off risk, confidence, tool precision/recall, safety, allocation quality, seen vs unseen split, decision probes, ablations, fault injection, failure analysis |
| `05_integration_engineer.py` | Integration Engineer | Command Center console (`/agent`: approve, override, recall, emergency stop, reasoning view), REST API (`/api/agent/*`), and an MCP server (stdio and HTTP) |

---

## 0. Requirements, and where each one lives

Roles come from the *CAT Phase II Session Plan* (Agentic AI section); learning outcomes from Day 9 of the *GenAI & Data Science Specialization Framework*.

| Requirement | Where it is |
| --- | --- |
| **Data/Knowledge Engineer**: build knowledge base, set up tools/APIs | `01_knowledge_engineer.py`: 1,038-passage SOP index, 50 district profiles, resource catalog, 18 tool definitions |
| **Workflow Engineer**: break down tasks, design workflow & tool-mapping | `02_workflow_engineer.py`: 17 tasks, 19-node state graph, plan checker, `tool_mapping.csv`, `workflow.md` (Mermaid) |
| **Agent Engineer**: architecture, reasoning/planning, tool-calling logic | `03_agent_engineer.py` (§3) |
| **Evaluation Engineer**: task success rate, reasoning accuracy, failure analysis | `04_evaluation_engineer.py` (§5) |
| **Integration Engineer**: deploy the agent, orchestration UI | `05_integration_engineer.py`, the *Agent Coordination (Stage 06)* tab in `app.py` |
| Autonomous perception, reasoning, planning and action loop | every trajectory step is tagged `perceive` / `reason` / `plan` / `act` / `reflect` |
| Agent with tool use and **MCP** | `--mcp` stdio server and `POST /api/agent/mcp`: `initialize`, `tools/list`, `tools/call`, `ping` (§4) |
| Multiple tools and APIs in one coherent workflow | Stage 01 XGBoost, Stage 02 CNN + LSTM, Stage 03 NLP, fusion, Stage 04 briefer, SOP retrieval, run-state tools |
| Safety considerations and human-in-the-loop | ten audit rules; approval gate, confidence threshold, emergency override (§3.4) |
| Deliberative agent that reasons through a trade-off, not a fixed trigger | demand-aware ranking with the reasoning written out; the one-ambulance case goes to the zone with more calls and people (§3.5) |
| Reasoning patterns: ReAct, plan-and-execute, reflexion, tree of thoughts, tool calling, memory, debate, human-in-the-loop | all implemented; see the catalogue in `data/outputs/workflow/workflow.md` (§3.6) |
| Coordination patterns: orchestrator-worker, negotiation, blackboard, hierarchy | implemented; swarm coordination is documented as not applicable (§3.6) |
| Evaluation on unseen incidents and "confident but wrong" risk | seen vs unseen split, decision probes, bad trade-off risk (§5) |

**Scope guarantee.** Stage 06 only reads earlier stages' data and calls their public adapters. The one shared file it touches is `app.py`, to register the blueprint and add the tab. It follows the same pattern Stage 05 uses.

---

## 1. Mission

Stages 01–05 each answer one question about one zone. A flood is many zones at once, sharing too few boats. Stage 06 turns the pipeline into an **agent that coordinates the whole incident**. It reads every zone through the stage models and ranks the zones. It splits a finite inventory across them and cites the SOPs that apply. Anything consequential is put in front of a human, and nothing is committed without one.

---

## 2. Architecture

```
 Stage 05 scenario (ground truth stripped by observable_scenario())
          │
          ▼
 ┌──────────────────────── CoordinationAgent (03) ───────────────────────────┐
 │  PERCEIVE ─► CONTEXTUALISE ─► ASSESS ─► BRIEF ─► RETRIEVE_SOP   per zone  │
 │     │ Tactical Dispatcher                    └─► ESCALATE_EVIDENCE        │
 │     ▼                                                                     │
 │  PLAN (plan-and-execute, checked) runs first                              │
 │  RANK ─► ALLOCATE (tree of thoughts) ─► DEBATE ─► AUDIT ─► AWAIT_APPROVAL │
 │  ─► MUTUAL_AID (reroute) ─► ALERT ─► COMMIT (+ notify shelter) ─► REFLECT │
 │  human: APPROVE / OVERRIDE / RECALL / EMERGENCY STOP  ─►  OVERRIDE node   │
 │                                                                           │
 │  every step checked against WorkflowGraph (02); every call through ToolRegistry
 └───────────────┬───────────────────────────────────────────────────────────┘
                 │ schema-validated tool calls
   ┌─────────────┼───────────────┬──────────────┬─────────────┬───────────────┐
   ▼             ▼               ▼              ▼             ▼               ▼
 Stage 01     Stage 02        Stage 03       fusion/       Stage 04       Knowledge base (01)
 XGBoost      CNN + LSTM      NLP            decision_     briefer        SOPs · districts ·
                                             engine                       inventory · run state
                 │
   ┌─────────────┴───────────────────────────────┐
   ▼                                             ▼
 /agent console + /api/agent/*  (05)        MCP server: stdio (--mcp) or POST /api/agent/mcp
```

A shared observation cache sits between the tools and the models. Fusion re-reads the same Stage 01–03 outputs the perception tools just produced, so it costs no second inference and both views of the evidence stay identical.

---

## 3. The agent (`03_agent_engineer.py`)

### 3.1 Perceive and reason, per zone

1. **Plan perception.** `RulePlanner` calls one tool per evidence stream the zone actually has. A failed stage call is retried once, then the stream is marked *lost*, not imputed.
2. **Contextualise** with the district's flood history. **Assess** through the fusion layer. **Brief** the incident log with Stage 04; if the briefing and fusion differ by two or more levels, the gap is surfaced to a human, never averaged.
3. **Read the evidence basis** of the priority. This matters because fusion can raise a zone to URGENT on a single signal:

| Basis | Meaning | What the agent does |
| --- | --- | --- |
| `corroborated` / `single_source` | a present-tense source supports URGENT+ | normal allocation |
| `forecast_only` | URGENT only because the LSTM forecast escalated | standby alert + gauge verification; **no assets** until corroborated |
| `text_only` | a CRITICAL report that every other source contradicts | at most 1 unit per asset type, no evacuation, ranked after corroborated zones |

4. **Derive needs** with named planning ratios: 1 boat per 50 people, 1 ambulance per 40 when injuries are signalled, 1 bed per evacuee. A projected rise counts as a flood signal only when it clears the rise threshold **plus the LSTM's own 0.73 m error**. Evacuation needs a physical flood signal (sensors or imagery), or two independent reported ones.
5. **Retrieve SOPs** for every ELEVATED+ zone. Repeated lookups within a run are reused and logged as reuse.

### 3.2 Plan

* **Plan first:** `PLAN` writes and checks the full plan before any tool call (§3.6).
* **Rank:** severity tier (physically confirmed severe zones share the top tier), then corroborated before uncorroborated, then emergency calls, people and decision confidence, with each trade-off written out (§3.5).
* **Allocate (two-pass):** every URGENT+ zone gets one unit of each asset type it needs before any zone gets a second. Remaining needs are then filled in rank order. `reserve_resources` refuses to over-commit.

### 3.3 Act and reflect

Held dispatches go to `escalate_to_human`. Every shortfall becomes `request_mutual_aid` plus a shortage escalation. Standby alerts the auditor cleared are issued. **Reflect** lists any URGENT+ zone left with neither assets nor an escalated shortage.

### 3.4 Safety Auditor and human-in-the-loop

| Rule | Effect |
| --- | --- |
| R1 URGENT/CRITICAL dispatch | held for sign-off |
| R2 no automated assessment | dispatch blocked; verification requested |
| R3 fusion flagged human review | held |
| R4 sources disagree | held as a *conflict* escalation |
| R5 evacuation order | held |
| R6 evacuation without a corroborated flood signal | blocked |
| R7 heavy assets to a ROUTINE zone | blocked |
| R8 reservations exceed inventory | blocked |
| R9 forecast-only escalation | assets wait for a corroborating reading |
| R10 decision confidence below 60% | held and escalated as low-confidence |

**The gate is in the tool, not the prompt.** `commit_rescue_dispatch` refuses a held action unless it is given the one-time token that `CoordinationAgent.decide()` issues on a human approval. The same check applies from the console, the REST API and MCP. A rejection releases the reservations. Every commit is appended to `data/outputs/dispatch_ledger.jsonl` with the approving operator.

### 3.5 Deliberation: one ambulance, two severe zones

Fusion can split two zones that are equally severe on the ground. In the case below, Stage 03 reads a panicked message as HIGH and a calm one as CRITICAL, so fusion labels them URGENT and CRITICAL. The agent does not take that label as the final word:

* A zone whose sensors read Severe (confidence at least 0.8, river over its danger mark), or whose imagery shows flooding, is **physically confirmed severe**. All such URGENT+ zones share the top ranking tier.
* Within a tier, **corroborated** zones come before uncorroborated ones. After that, **field demand** decides: emergency calls, then people, then decision confidence.
* The reasoning is written into the trajectory, and the losing zone's unmet need is **rerouted to a secondary responder** (SDRF/NDRF, district EMS, DDMA relief camp).

```
RANK   Trade-off ZONE-7 vs ZONE-12: both physically confirmed severe (sensors or imagery); fusion labels them
       URGENT and CRITICAL, but at equal physical severity field demand decides. Demand: 38 vs 15 emergency
       calls, 40 vs 15 people -> ZONE-7 first; ZONE-12 is served next or rerouted to a secondary responder.
DEBATE Negotiation for 1 ambulance with 2 claimant zones ... Agreement: ZONE-7 first; ZONE-12 rerouted to
       District EMS (108) / nearest hospital ambulance.
```

Command Center card: **Dispatch 1 ambulance to ZONE-7 now · Reroute 1 ambulance for ZONE-12 to District EMS · Notify shelter of ~40 incoming evacuees**, with APPROVE and OVERRIDE buttons.

### 3.6 Reasoning, coordination and control patterns

| Pattern | How it is implemented |
| --- | --- |
| ReAct | every step records a thought, the tool it calls and the observation |
| Plan-and-Execute | `PLAN` writes the whole plan before the first tool call; `check_plan()` flags duplicate, missing and illegal steps; adherence is measured |
| Tree of Thoughts | three allocation plans (`one_each_then_rank`, `rank_fill`, `weighted_round_robin`) are scored with a severity- and confidence-weighted utility; the best is kept |
| Multi-Agent Debate | the Safety Auditor challenges any unit held by a lower-ranked zone while a higher-ranked zone has none; the allocator concedes and moves it |
| Peer-to-Peer Negotiation | zones bid for contested units (tier, calls, people, confidence); the agreement and the rerouted zones are recorded |
| Reflexion | `REFLECT` revises the allocation once if units were freed while zones are still short |
| Memory-Augmented | `recall_precedents` reads past agent and commander decisions for the same district (`data/outputs/agent_memory.jsonl` in the dashboard; disabled during evaluation so runs stay independent) |
| Blackboard | every agent posts its findings to `RunState.blackboard`, shown in the console |
| Hierarchical escalation | escalations go to the field supervisor, district EOC, state EOC or incident commander |
| Human-in-the-loop | approval tokens, confidence threshold (R10), emergency override |
| Swarm coordination | **not applicable**: it needs many simple local agents, and one incident command does not fit that model |

Every new step type is a node in the workflow graph (`PLAN`, `DEBATE`, `OVERRIDE`), so conformance checking still covers the whole run.

### 3.7 Decision confidence and the emergency override

* **Decision confidence** is computed per zone from named factors, not a learned calibration: the mean confidence of the answering sources, discounted for disagreement, lost evidence, one-signal priority and a contradicting briefing. Below **60%**, an URGENT+ dispatch is held under rule **R10** and escalated as `low_confidence`. The evaluation checks the threshold means something: exact accuracy is 0.857 on confident zones against 0.421 below the threshold (CVAE).
* **Emergency override.** `CoordinationAgent.override()` either halts the whole run or recalls one action. Pending actions are cancelled and their units released. Committed dispatches are recalled through `recall_dispatch`, which, like commits, needs a one-time token that only the override issues. A halted run refuses every further commit. The console has **EMERGENCY STOP**, **OVERRIDE** and **RECALL**; the API route is `POST /api/agent/override`.
* **Shelter notification.** An approved dispatch that includes an evacuation calls `notify_shelter`.

### 3.8 Planners: one running, one withheld

| Planner | Status |
| --- | --- |
| `RulePlanner` | ✅ used for every number in this README |
| `GeminiPlanner` (LLM chooses the perception tools) | implemented and run live against `gemini-3.6-flash`; **not fully evaluated** (see below) |

**What was measured with a real key.**
* Every Gemini reply that came back was a valid plan: 8 of 8 during the evaluation run and 6 of 6 in a separate probe. Each reply was a JSON list of exactly the tools the zone's evidence supported.
* A full ablation needs 55 planner calls. The key is on the free tier, capped at **20 requests per day** for this model, so every other call came back `429 RESOURCE_EXHAUSTED`.
* That first run fell back to the rule planner 47 times, which made its "Gemini" scores really the rule planner's. Those numbers are not reported.

**How the planner handles this now.**
* It retries per-minute 429s using the server's suggested delay.
* It stops calling for the rest of the run once the per-day quota is exhausted.
* Every request times out after 60 s.
* The report counts accepted plans, API errors, invalid outputs and skipped calls separately.

A key with more quota runs the ablation automatically; `STAGE06_GEMINI_MAX_CALLS` caps spend. The model can be overridden with `STAGE06_GEMINI_MODEL`.

The reasoning shown in trajectories is **templated from observed values**, not generated text. It is a faithful record of why each step was taken, not LLM chain-of-thought.

---

## 4. Tools and MCP

| Tool | Owner | Backed by |
| --- | --- | --- |
| `query_sensor_risk_score` | dispatcher | Stage 01 |
| `predict_visual_flood`, `forecast_water_level` | dispatcher | Stage 02 |
| `parse_emergency_text` | dispatcher | Stage 03 |
| `assess_zone` | dispatcher | fusion |
| `generate_tactical_briefing` | dispatcher | Stage 04 (baseline) |
| `get_district_profile`, `search_sop` | dispatcher | knowledge base |
| `check_resource_inventory`, `reserve_resources`, `request_mutual_aid` (reroute to a secondary responder) | allocator | run state |
| `escalate_to_human` (with an escalation level) | auditor | run state |
| `issue_alert`, `release_resources`, `notify_shelter`, `commit_rescue_dispatch` 🔒 | coordinator | run state |
| `recall_precedents` | dispatcher | agent memory |
| `recall_dispatch` 🔒 | commander (human override) | run state |

Every call is validated against its `inputSchema` before it reaches a model. Validation caught a real bug during development: the first live run passed `text` where the tool takes `raw_text`.

```bash
python Stage06_AgenticAI/05_integration_engineer.py --mcp
```

This is newline-delimited JSON-RPC 2.0 over stdio. Stage modules print while loading, so stdout is redirected to stderr during model load to keep the protocol stream clean. Run-state tools act on the run named in `params._meta.run_id`, or the latest run if none is named.

---

## 5. Evaluation (`data/outputs/evaluation/agent_eval_report.md`)

The agent never sees a zone's true severity, hazards, headcount or expectations. `observable_scenario()` strips them, and a test asserts it. Scores are computed afterwards against Stage 05's ground truth.

**Zone goal.**
* No-evidence zone: verified, not dispatched.
* True CRITICAL: assets, or an escalated shortage.
* True URGENT: at least some response.
* True ROUTINE: no heavy assets.
* Every zone: Stage 05's expectations met, and nothing committed without a human.

A **scenario** passes when every zone passes, with no workflow violation and no over-commit.

### 5.0 Decision probes and seen vs unseen

Rules were tuned while looking at failures on the **CVAE suite**, so that suite is labelled *development*. The **SLM suite** was never used to design a rule, so it is *held-out*. The **decision probes** are hand-written one-ambulance trade-offs with a known right answer. Each runs through the real models, is approved, then overridden.

| Probe | Expected | Result | Rerouted | Approved | Override recalled |
| --- | --- | --- | --- | --- | --- |
| P1 Two severe zones, 38 vs 15 emergency calls (the panicked message misread as HIGH) | Zone 7 | **PASS** | yes | yes | yes |
| P2 Same case, zones listed in the opposite order | Zone 7 | **PASS** | yes | yes | yes |
| P3 Equal call volume, 60 vs 12 people waiting | Zone B | **PASS** | yes | yes | yes |
| P4 Sensor-confirmed zone vs a dramatic report the sensors contradict | Zone C | **PASS** | yes | yes | yes |

Before §3.5, the agent sent P1's ambulance to Zone 12, the wrong zone.

### 5.1 Agent, by generator suite (21 incidents each: 20 scenarios + wildcard, 55 zones)

| Metric | CVAE suite | Domain-SLM suite |
| --- | ---: | ---: |
| Scenario goal success | **0.857** | 0.762 |
| Zone goal success | **0.927** | 0.855 |
| Critical misses | **0** | **0** |
| Priority exact / within-one accuracy | 0.704 / 0.907 | 0.611 / 0.852 |
| Human-review recall | 1.000 | 1.000 |
| Tool precision / recall | 0.987 / 1.000 | 0.982 / 1.000 |
| Argument validity · tool error rate | 1.000 · 0.000 | 1.000 · 0.000 |
| Workflow violations · redundant calls | 0 · 0 | 0 · 0 |
| **Unsafe commits** | **0** | **0** |
| High-severity need coverage | 0.910 | 0.910 |
| Priority inversions | 0 | 1 |
| **Bad trade-off risk** (confident and wrong) | **0.000** | 0.018 |
| Exact accuracy: confident / not confident | 0.857 / 0.421 | 0.824 / 0.250 |
| Low-confidence URGENT+ dispatches escalated | 1.000 | 1.000 |
| Plan adherence · plan-check failures | 1.000 · 0 | 1.000 · 0 |
| Tree of thoughts kept `one_each_then_rank` / `weighted_round_robin` | 17 / 4 | 19 / 2 |
| Debate objections that moved a unit · negotiations | 2 · 6 | 0 · 5 |
| Run latency p50 | 186 ms | 155 ms |

**Human-in-the-loop probe (CVAE).** 39 dispatches were held. All 78 commit attempts without a token or with a forged one were blocked. All 39 committed after simulated approval. An emergency override then recalled all 39, and all 39 commit attempts after the halt were blocked.

On the held-out SLM suite, one allocation was confident and wrong: a unit went to a less severe zone while a more severe one waited. Its bad trade-off risk is 1 of 54 decisions (0.018).

### 5.2 Ablations and fault injection (CVAE suite)

| | Agent | FCFS allocation | No Safety Auditor | Stage 01 outage | Stage 03 outage | Flaky Stage 02 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scenario goal success | 0.857 | 0.857 | 0.048 | 0.619 | 0.762 | 0.857 |
| High-severity need coverage | **0.910** | 0.640 | 0.910 | 0.930 | 0.916 | 0.910 |
| Unsafe commits | 0 | 0 | **39** | 0 | 0 | 0 |
| Critical misses | 0 | 0 | 0 | 2 | 0 | 0 |
| Under-triaged high-severity zones | 1 | 1 | 1 | 11 | 0 | 1 |
| Workflow violations | 0 | 0 | 0 | 0 | 0 | 0 |

* **Two-pass priority allocation** meets 91% of high-severity needs. Arrival-order allocation meets 64%. Goal success ties only because the goal counts an escalated shortage as served; coverage is the metric that separates the two.
* **Removing the Safety Auditor** does not change a single priority, but 39 URGENT/CRITICAL dispatches then commit with no human. The auditor is what makes the agent safe.
* **Losing Stage 01** is the most damaging outage: 11 high-severity zones are under-triaged and 2 become critical misses. Sensors carry most of the severity signal. Every outage run still completes with zero workflow violations and zero unsafe commits.
* **A flaky Stage 02** costs nothing: the single retry recovers every transient failure.

### 5.3 The evidence-basis fix, found by this evaluation

The first evaluation run sent heavy assets to 5 truly ROUTINE zones, and showed one priority inversion in *People versus sensors*: a false CRITICAL report took the only boat from a real CRITICAL zone. Reading the trajectories showed the cause. Fusion had raised calm zones to URGENT on LSTM "steep rises" of 1–2.7 m, which sit inside the LSTM's documented absolute-scale error, and on single CRITICAL reports that every other source contradicted. §3.1's evidence-basis rules came out of that:

| CVAE suite | before | after |
| --- | ---: | ---: |
| Scenario goal success | 0.810 | **0.857** |
| Heavy assets to ROUTINE zones | 5 | **2** |
| Priority inversions | 1 | **0** |
| High-severity need coverage | 0.905 | **0.910** |
| Domain-SLM suite scenario goal success | 0.667 | **0.762** |

The redundant-call count was also wrong in the first run. Trajectories summarise sensor arguments as `{12 fields}`, so every sensor call looked identical. Calls are now compared by a hash of their full arguments.

### 5.4 What still fails, and why

| Failure (CVAE) | Cause |
| --- | --- |
| S05 ZONE-1, S09 ZONE-2 get one boat as ROUTINE zones | a text-only CRITICAL report. In S09 the *other* zone's text-only report is true, and the agent cannot tell them apart without ground truth. The cap limits the damage to 1 unit and the conflict goes to a human. |
| S18 ZONE-2/3 scored URGENT ("All quiet") | fusion over-triage. The agent now sends them no assets, only standby alerts and gauge checks, but their priority label still breaks Stage 05's maximum. |

On the SLM suite, 3 missed conflicts and 1 under-triaged CRITICAL zone (S07) come from the fusion layer's output on those scenarios. The agent passes both through unchanged; it does not create them.

---

## 6. Honest limitations

1. **The evaluated planner is rule-based.** The Gemini planner produced valid plans on every call that went through, but a free-tier quota of 20 requests per day prevented a full 55-call ablation (§3.8).
2. **Ground truth is synthetic.** Scenarios and expectations come from Stage 05's generators, so these numbers measure behaviour on generated incidents, not real floods.
3. **The tool-precision reference is derived** from the evidence each zone had. A plan that skips a tool deliberately scores as a miss.
4. **Planning ratios, confidence factors and the 60% threshold are policy constants**, not learned from deployment data. Inventory comes from each Stage 05 scenario, or from the suite median for custom incidents.
5. **Nothing leaves the process.** Alerts, mutual-aid requests and dispatches are recorded in run state and the ledger; no external system is contacted.
6. **Stage 04 runs its TF-IDF baseline**, not Qwen. Qwen takes about 10 s per briefing, too slow inside an agent loop; Stage 05's stress test uses the baseline for the same reason.
7. **The HITL probe approves with a simulated human.** It proves the gate works, not that approvals are good decisions.

---

## 7. API

| Method | Path | Body / purpose |
| --- | --- | --- |
| GET | `/agent` | orchestration console |
| GET | `/api/agent/health` | knowledge base, workflow graph and a model-free probe run |
| GET | `/api/agent/scenarios?suite=cvae` | Stage 05 incidents |
| POST | `/api/agent/orchestrate` | `{"suite": "cvae", "scenario_id": "S01"}` or `{"incident": {"resources": {...}, "zones": [{"state", "district", "inputs": {"text", "sensors", "water_levels", "image_path", "incident_log"}}]}}` |
| GET | `/api/agent/runs/<run_id>` | a stored run |
| POST | `/api/agent/approve` | `{"run_id", "action_id", "decision": "approve" \| "reject", "operator", "note"}` |
| POST | `/api/agent/override` | `{"run_id", "operator", "reason", "action_id"?}`: without `action_id` it halts the whole run |
| GET | `/api/agent/tools` · `/api/agent/workflow` · `/api/agent/report` | tool definitions · state graph · latest evaluation |
| POST | `/api/agent/mcp` | one JSON-RPC 2.0 message |

---

## 8. Execution order

```bash
python Stage06_AgenticAI/01_knowledge_engineer.py    # knowledge base + tool registry
python Stage06_AgenticAI/02_workflow_engineer.py     # state graph, task & tool mapping
python Stage06_AgenticAI/03_agent_engineer.py        # demo run on the wildcard incident
python Stage06_AgenticAI/04_evaluation_engineer.py   # full evaluation (~30 s after model load)
python Stage06_AgenticAI/05_integration_engineer.py  # self-test: health, run, MCP bypass attempt, approval
python Stage06_AgenticAI/05_integration_engineer.py --serve   # console on :5006
python app.py                                        # or the full dashboard: "Agent Coordination (Stage 06)"
pytest Stage06_AgenticAI/test/ -v
```

Requires Stage 05's scenario files and the Stage 01–04 model artifacts. No new dependencies: MCP is implemented directly on JSON-RPC, and `google-genai` (already pinned) is needed only for the optional Gemini planner.
