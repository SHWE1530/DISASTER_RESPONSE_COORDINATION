"""Stage 05 GenAI -- Integration Engineer.

Feeds synthetic scenario output back into a testing dashboard for continuous
evaluation.

GenAIIntegrationEngine
  health_check()              proves the generator can realise a probe scenario
  list_prompts()              the scenario prompt library (+ wildcard)
  generate(prompt_id | spec)  -> a generated multi-zone scenario
  stress_test(scenario)       -> scored through the real Stage 01-04 pipeline
  latest_report() / history() -> results written by 04_evaluation_engineer.py
  render_dashboard(live)      -> self-contained HTML
  build_dashboard()           -> data/outputs/stress_test_dashboard.html

create_blueprint(engine)      -> Flask Blueprint (/genai, /api/genai/*).
  Ready to be registered by app.py with two lines; this stage deliberately does
  NOT modify app.py. Until then it runs standalone with --serve.

Usage:
  python Stage05_GenAI/05_integration_engineer.py           # self-test + static dashboard
  python Stage05_GenAI/05_integration_engineer.py --serve   # live dashboard, :5005
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
DASHBOARD_HTML = OUTPUT_DIR / "stress_test_dashboard.html"

logger = logging.getLogger("disaster_response.stage05")

SEED = 42
MAX_SEED = 2**31 - 1

# Smallest scenario that exercises the whole generator: CVAE sample, gauge
# history, dispatcher text and expectations.
PROBE_SPEC = {
    "id": "PROBE", "name": "Health probe", "archetype": "probe",
    "state": "Assam", "start_time": "14-08-2026 02:00", "global_modifiers": [],
    "zones": [{"label": "Probe zone", "true_severity": "URGENT", "hazards": ["Flood"],
               "modifiers": ["river_overflow"],
               "evidence": {"sensors": True, "text": "dispatcher", "image": None,
                            "water_history": "steady_rise"}}],
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_module("stage05_evaluation_engineer", BASE_DIR / "04_evaluation_engineer.py")
GENAI = EVAL.GENAI


class GenAIIntegrationEngine:
    """App-facing wrapper around the Stage 05 generator and stress tester.

    Everything heavy is lazy: the generator loads on first use and the Stage
    01-04 models only when a live stress test is actually requested.
    """

    def __init__(self) -> None:
        self._generator = None
        self._tester = None
        self.stage_status: dict[str, str] = {}
        self.load_error: str | None = None
        self._lock = threading.Lock()

    # ---- generator -----------------------------------------------------------

    @property
    def generator(self):
        if self._generator is None:
            try:
                self._generator = GENAI.ScenarioGenerator()
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(self.load_error) from exc
        return self._generator

    def health_check(self) -> dict[str, Any]:
        """Report readiness by generating a real probe scenario."""
        try:
            generator = self.generator
        except RuntimeError:
            return {"status": "unavailable", "generator_ok": False, "error": self.load_error}
        try:
            scenario = generator.generate(PROBE_SPEC, seed=0)
            zone = scenario["zones"][0]
            if not (zone["inputs"]["sensors"] and zone["inputs"]["text"]
                    and len(zone["inputs"]["water_levels"]) == GENAI.LOOKBACK_HOURS):
                raise RuntimeError("probe scenario is missing sensors, text or gauge history")
        except Exception as exc:
            return {"status": "degraded", "generator_ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        report = self.latest_report()
        bundle = Path(generator.sampler.bundle_path).resolve()
        try:
            bundle_label = str(bundle.relative_to(BASE_DIR)).replace("\\", "/")
        except ValueError:
            bundle_label = bundle.name  # a bundle outside the stage (e.g. tests) -- show no host path
        return {
            "status": "healthy",
            "generator_ok": True,
            "cvae_bundle": bundle_label,
            "prompts": len(self.list_prompts()),
            "latest_report_at": report.get("generated_at") if report else None,
            "latest_scenario_pass_rate": report["summary"]["scenario_pass_rate"] if report else None,
            "error": None,
        }

    def list_prompts(self) -> list[dict[str, Any]]:
        library = GENAI.load_prompts()
        return [{"id": p["id"], "name": p["name"], "archetype": p["archetype"],
                 "prompt": p["prompt"], "blind_spots": p["blind_spots"], "zones": len(p["zones"])}
                for p in library["prompts"] + [library["wildcard"]]]

    def _prompt_spec(self, prompt_id: str) -> dict[str, Any]:
        library = GENAI.load_prompts()
        for prompt in library["prompts"] + [library["wildcard"]]:
            if prompt["id"] == prompt_id:
                return prompt
        raise ValueError(f"Unknown prompt_id {prompt_id!r}")

    def generate(self, prompt_id: str | None = None, spec: dict | None = None,
                 seed: int = SEED) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer between 0 and {MAX_SEED}")
        if spec is None:
            if not isinstance(prompt_id, str):
                raise ValueError("Provide either 'prompt_id' or 'spec'")
            spec = self._prompt_spec(prompt_id)
        return self.generator.generate(spec, seed=seed)

    # ---- stress test ---------------------------------------------------------

    def _stress_tester(self):
        if self._tester is None:
            engines, self.stage_status = EVAL.load_stage_engines("baseline")
            self._tester = EVAL.StressTester(engines.get("ml"), engines.get("dl"),
                                             engines.get("nlp"), engines.get("slm"))
        return self._tester

    def stress_test(self, scenario: dict[str, Any]) -> dict[str, Any]:
        # The stage models are not documented as thread-safe; serialise runs.
        with self._lock:
            result = self._stress_tester().run_scenario(scenario)
        return EVAL._jsonable(result)

    # ---- reports & dashboard -------------------------------------------------

    @staticmethod
    def latest_report() -> dict[str, Any] | None:
        if not EVAL.REPORT_JSON.exists():
            return None
        return json.loads(EVAL.REPORT_JSON.read_text(encoding="utf-8"))

    @staticmethod
    def history() -> list[dict[str, Any]]:
        if not EVAL.HISTORY_CSV.exists():
            return []
        frame = pd.read_csv(EVAL.HISTORY_CSV)
        return json.loads(frame.to_json(orient="records"))

    def render_dashboard(self, live: bool = False, report: dict | None = None) -> str:
        report = report if report is not None else self.latest_report()
        try:
            prompts = self.list_prompts()
        except FileNotFoundError:
            prompts = []
        payload = {"report": report, "history": self.history(), "prompts": prompts, "live": bool(live)}
        # "</" is escaped so no string inside the data can close the <script> tag.
        data = json.dumps(payload).replace("</", "<\\/")
        return DASHBOARD_TEMPLATE.replace("__DASHBOARD_DATA__", data)

    def build_dashboard(self) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DASHBOARD_HTML.write_text(self.render_dashboard(live=False), encoding="utf-8")
        return DASHBOARD_HTML


genai_integration_engine = GenAIIntegrationEngine()


# ===========================================================================
# Flask blueprint
# ===========================================================================

def create_blueprint(engine: GenAIIntegrationEngine | None = None):
    """Routes for the live dashboard. Register with app.register_blueprint(...)."""
    from flask import Blueprint, Response, jsonify, request

    engine = engine or genai_integration_engine
    blueprint = Blueprint("stage05_genai", __name__)

    def fail(message: str, status_code: int = 400, exc: Exception | None = None):
        if exc is not None:
            logger.exception("Stage 05 request failed: %s", message)
        return jsonify({"error": message}), status_code

    def parse(payload: Any) -> tuple[str | None, dict | None, int]:
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        seed = payload.get("seed", SEED)
        spec = payload.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("'spec' must be a JSON object")
        return payload.get("prompt_id"), spec, seed

    @blueprint.route("/genai")
    def dashboard():
        return Response(engine.render_dashboard(live=True), mimetype="text/html")

    @blueprint.route("/api/genai/health")
    def health():
        return jsonify(engine.health_check())

    @blueprint.route("/api/genai/prompts")
    def prompts():
        return jsonify(engine.list_prompts())

    @blueprint.route("/api/genai/report")
    def report():
        latest = engine.latest_report()
        if latest is None:
            return fail("No stress-test report yet. Run 04_evaluation_engineer.py.", 404)
        return jsonify(latest)

    @blueprint.route("/api/genai/scenario", methods=["POST"])
    def scenario():
        try:
            prompt_id, spec, seed = parse(request.get_json(silent=True))
            return jsonify(engine.generate(prompt_id=prompt_id, spec=spec, seed=seed))
        except ValueError as exc:
            # Validation messages are written for the caller and safe to show.
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Scenario generation failed. See server logs for details.", 500, exc)

    @blueprint.route("/api/genai/stress-test", methods=["POST"])
    def stress_test():
        try:
            prompt_id, spec, seed = parse(request.get_json(silent=True))
            generated = engine.generate(prompt_id=prompt_id, spec=spec, seed=seed)
        except ValueError as exc:
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Scenario generation failed. See server logs for details.", 500, exc)
        try:
            return jsonify(engine.stress_test(generated))
        except Exception as exc:
            return fail("Stress test failed. See server logs for details.", 500, exc)

    return blueprint


def serve(port: int) -> None:
    from flask import Flask, redirect

    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

    @app.route("/")
    def index():
        return redirect("/genai")

    print(f"Stage 05 stress-test dashboard on http://127.0.0.1:{port}/genai")
    app.run(port=port, debug=False)


# ===========================================================================
# Dashboard template (self-contained: no external scripts or fonts)
# ===========================================================================

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage 05 Stress Test</title>
<style>
  :root { --bg:#0B1120; --panel:#0F172A; --card:rgba(30,41,59,.7); --line:#1E293B; --text:#F8FAFC;
          --muted:#94A3B8; --blue:#3B82F6; --green:#10B981; --amber:#F59E0B; --orange:#F97316; --red:#EF4444; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:24px 32px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:22px; } header p { color:var(--muted); font-size:13px; margin-top:4px; }
  main { padding:24px 32px; max-width:1400px; margin:0 auto; }
  h2 { font-size:16px; margin:28px 0 12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr)); gap:14px; }
  .kpi, .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
  .kpi .label { color:var(--muted); font-size:12px; } .kpi .value { font-size:24px; font-weight:700; margin-top:4px; }
  .kpi .hint { color:var(--muted); font-size:11px; margin-top:2px; }
  .table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; background:rgba(15,23,42,.6); }
  tr.clickable { cursor:pointer; } tr.clickable:hover { background:rgba(59,130,246,.08); }
  tr.detail td { background:rgba(2,6,23,.5); }
  .badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; border:1px solid; }
  .ROUTINE { color:var(--green); } .ELEVATED { color:var(--amber); } .URGENT { color:var(--orange); } .CRITICAL { color:var(--red); }
  .pass { color:var(--green); } .fail { color:var(--red); } .muted { color:var(--muted); }
  .chips span { display:inline-block; margin:2px 4px 2px 0; padding:1px 8px; border-radius:10px; background:#1E293B; font-size:11px; color:var(--muted); }
  .zone-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr)); gap:14px; }
  @media (max-width:600px) { header, main { padding-left:16px; padding-right:16px; } }
  .zone-cards ul { margin:6px 0 0 18px; color:var(--muted); font-size:12px; }
  .live { display:flex; gap:10px; flex-wrap:wrap; align-items:end; }
  .live label { display:flex; flex-direction:column; font-size:12px; color:var(--muted); gap:4px; }
  select, input { background:#1E293B; border:1px solid #334155; color:var(--text); padding:8px 10px; border-radius:6px; }
  button { background:var(--blue); color:white; border:0; padding:9px 18px; border-radius:6px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  svg text { fill:var(--muted); font-size:10px; }
  .empty { color:var(--muted); padding:16px; }
</style>
</head>
<body>
<header>
  <h1>Stage 05 &middot; Generative Scenario Stress Test</h1>
  <p id="meta"></p>
</header>
<main>
  <div id="live-section" hidden>
    <h2>Run a scenario live</h2>
    <div class="card">
      <div class="live">
        <label>Scenario prompt <select id="prompt-select"></select></label>
        <label>Seed <input id="seed-input" type="number" value="42" min="0" style="width:110px"></label>
        <button id="run-btn">Generate &amp; stress-test</button>
      </div>
      <p id="prompt-text" class="muted" style="margin-top:10px;font-size:12px"></p>
      <div id="live-result" style="margin-top:14px"></div>
    </div>
  </div>

  <h2>Headline results</h2>
  <div class="grid" id="kpis"></div>

  <h2>Run history</h2>
  <div class="card" id="history"></div>

  <h2>Scenarios <span class="muted" style="font-weight:400;font-size:12px">(click a row for zone detail)</span></h2>
  <div class="table-wrap"><table id="scenario-table"></table></div>

  <h2 id="wildcard-title">Wildcard</h2>
  <p id="wildcard-prompt" class="muted" style="margin-bottom:10px"></p>
  <div class="zone-cards" id="wildcard"></div>

  <h2>Realism audit</h2>
  <div class="table-wrap"><table id="realism"></table></div>

  <h2>Stage 02 forecast probe <span class="muted" style="font-weight:400;font-size:12px">(perfectly flat 72 h river)</span></h2>
  <p id="probe-note" class="muted" style="margin-bottom:10px;font-size:12px"></p>
  <div class="table-wrap"><table id="probe"></table></div>

  <h2>Failure log</h2>
  <div class="table-wrap"><table id="failures"></table></div>
</main>
<script>
const DATA = __DASHBOARD_DATA__;
const LEVELS = ['ROUTINE', 'ELEVATED', 'URGENT', 'CRITICAL'];

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children || [])) {
    if (child == null) continue;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}
const pct = v => v == null ? 'n/a' : (v * 100).toFixed(1) + '%';
const priorityBadge = p => p && LEVELS.includes(p) ? el('span', {class: 'badge ' + p, text: p}) : el('span', {class: 'muted', text: p || '-'});
const passBadge = ok => el('span', {class: ok ? 'pass' : 'fail', text: ok ? 'PASS' : 'FAIL'});

function table(target, headers, rows) {
  const t = document.getElementById(target);
  t.replaceChildren(el('tr', {}, headers.map(h => el('th', {text: h}))));
  if (!rows.length) { t.appendChild(el('tr', {}, el('td', {class: 'empty', colspan: String(headers.length), text: 'Nothing to show.'}))); return t; }
  rows.forEach(r => t.appendChild(el('tr', {}, r.map(c => el('td', {}, c instanceof Node ? c : String(c ?? '-'))))));
  return t;
}

function zoneTable(zones) {
  const t = el('table');
  t.appendChild(el('tr', {}, ['Zone', 'True', 'Called', 'Review', 'Conflicts', 'Sources used', 'Evidence lost', 'Result'].map(h => el('th', {text: h}))));
  zones.forEach(z => t.appendChild(el('tr', {}, [
    el('td', {text: z.zone_id + ' ' + z.label}), el('td', {}, priorityBadge(z.true_severity)),
    el('td', {}, z.status === 'ok' ? priorityBadge(z.priority) : el('span', {class: 'muted', text: z.status})),
    el('td', {text: z.human_review ? 'yes' : 'no'}), el('td', {text: String(z.n_conflicts)}),
    el('td', {class: 'muted', text: z.sources_used || 'none'}), el('td', {class: 'muted', text: z.evidence_lost || '-'}),
    el('td', {}, z.passed ? passBadge(true) : el('span', {class: 'fail', text: 'FAIL: ' + z.failures})),
  ])));
  return t;
}

function renderKpis(s, realism) {
  const cards = [
    ['Scenarios passed', s.scenarios_passed + '/' + s.scenarios, pct(s.scenario_pass_rate)],
    ['Zones passed', s.zones_passed + '/' + s.zones, pct(s.zone_pass_rate)],
    ['Critical-miss rate', pct(s.critical_miss_rate), s.critical_misses + ' of ' + s.high_risk_zones_scored + ' high-risk zones'],
    ['Pipeline crashes', String(s.crashes), 'exceptions during assessment'],
    ['Conflicts surfaced', pct(s.conflict_detection_rate), 'where evidence disagrees'],
    ['No-evidence refused', pct(s.insufficient_evidence_handled), 'silence never read as safe'],
    ['Zone ranking tau', s.mean_ranking_tau ?? 'n/a', 'Kendall, ' + s.ranking_scenarios + ' scenarios'],
    ['C2ST ROC-AUC', realism ? realism.c2st.cvae.auc_mean : 'n/a', realism ? 'naive baseline ' + realism.c2st.naive_baseline.auc_mean : 'realism audit not run'],
  ];
  document.getElementById('kpis').replaceChildren(...cards.map(([label, value, hint]) =>
    el('div', {class: 'kpi'}, [el('div', {class: 'label', text: label}), el('div', {class: 'value', text: String(value)}), el('div', {class: 'hint', text: hint})])));
}

function renderHistory(history) {
  const box = document.getElementById('history');
  if (history.length < 2) { box.replaceChildren(el('p', {class: 'muted', text: history.length + ' run(s) recorded. The trend appears after the second run of 04_evaluation_engineer.py.'})); return; }
  const w = 640, h = 140, pad = 28, ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', '100%'); svg.setAttribute('style', 'max-width:720px');
  const x = i => pad + i * (w - 2 * pad) / (history.length - 1), y = v => h - pad - v * (h - 2 * pad);
  [['scenario_pass_rate', '#10B981'], ['critical_miss_rate', '#EF4444']].forEach(([key, color]) => {
    const pts = history.map((r, i) => r[key] == null ? null : `${x(i)},${y(r[key])}`).filter(Boolean).join(' ');
    const line = document.createElementNS(ns, 'polyline');
    line.setAttribute('points', pts); line.setAttribute('fill', 'none'); line.setAttribute('stroke', color); line.setAttribute('stroke-width', '2');
    svg.appendChild(line);
  });
  [0, 0.5, 1].forEach(v => { const t = document.createElementNS(ns, 'text'); t.setAttribute('x', '0'); t.setAttribute('y', String(y(v) + 3)); t.textContent = (v * 100) + '%'; svg.appendChild(t); });
  box.replaceChildren(svg, el('p', {class: 'muted', style: 'font-size:12px', text: 'Green: scenario pass rate. Red: critical-miss rate. ' + history.length + ' runs.'}));
}

function renderScenarios(scenarios) {
  const t = document.getElementById('scenario-table');
  t.replaceChildren(el('tr', {}, ['ID', 'Scenario', 'Blind spots', 'Zones', 'Result', 'Ranking tau'].map(h => el('th', {text: h}))));
  scenarios.forEach(sc => {
    const passed = sc.zones.filter(z => z.passed).length;
    const row = el('tr', {class: 'clickable'}, [
      el('td', {text: sc.scenario_id}), el('td', {text: sc.name}),
      el('td', {class: 'chips'}, sc.blind_spots.map(b => el('span', {text: b}))),
      el('td', {text: String(sc.zones.length)}),
      el('td', {}, sc.passed ? passBadge(true) : el('span', {class: 'fail', text: passed + '/' + sc.zones.length})),
      el('td', {text: sc.ranking_tau == null ? '-' : String(sc.ranking_tau)}),
    ]);
    const detail = el('tr', {class: 'detail', hidden: ''}, el('td', {colspan: '6'}, [el('p', {class: 'muted', style: 'margin-bottom:8px', text: sc.prompt}), zoneTable(sc.zones)]));
    row.addEventListener('click', () => { detail.hidden = !detail.hidden; });
    t.append(row, detail);
  });
}

function renderWildcard(w) {
  if (!w) { document.getElementById('wildcard').replaceChildren(el('p', {class: 'empty', text: 'No wildcard result.'})); return; }
  document.getElementById('wildcard-title').textContent = 'Wildcard: ' + w.name;
  document.getElementById('wildcard-prompt').textContent = w.prompt;
  document.getElementById('wildcard').replaceChildren(...w.details.map(d => {
    const dec = d.decision || {};
    const kept = Object.entries(d.evidence_available).filter(([, v]) => v).map(([k]) => k);
    const reasons = (dec.escalations || []).concat(dec.conflicts || [], dec.human_review_reasons || []);
    return el('div', {class: 'card'}, [
      el('div', {style: 'display:flex;justify-content:space-between;gap:8px'}, [el('strong', {text: d.label}), d.passed ? passBadge(true) : passBadge(false)]),
      el('p', {class: 'muted', style: 'font-size:12px;margin-top:6px'}, ['Truth ', priorityBadge(d.true_severity), '  Called ', dec.status === 'ok' ? priorityBadge(dec.priority) : el('span', {class: 'fail', text: dec.status})]),
      el('p', {style: 'font-size:12px;margin-top:6px', text: 'Evidence left: ' + (kept.join(', ') || 'none') + (d.evidence_lost.length ? '  |  lost: ' + d.evidence_lost.join(', ') : '')}),
      el('ul', {}, (reasons.length ? reasons : [dec.message || 'No escalations or review reasons.']).map(r => el('li', {text: r}))),
      d.passed ? null : el('p', {class: 'fail', style: 'font-size:12px;margin-top:6px', text: d.failures}),
    ]);
  }));
}

function renderRealism(r) {
  if (!r) { table('realism', ['Check', 'Result'], []); return; }
  const rows = [
    ['C2ST ROC-AUC (0.5 = indistinguishable)', r.c2st.cvae.auc_mean, r.c2st.naive_baseline.auc_mean],
    ['Rainfall-calls correlation (real ' + r.correlation.rainfall_calls_real + ')', r.correlation.rainfall_calls_synthetic, r.correlation.rainfall_calls_naive_baseline],
    ['Mean KS statistic across features', r.mean_ks_overall, '-'],
  ];
  if (r.label_fidelity) rows.push(['Stage 01 recovers conditioning class (macro F1)', r.label_fidelity.cvae.macro_f1, r.label_fidelity.naive_baseline.macro_f1]);
  if (r.text_fidelity) {
    const tf = r.text_fidelity;
    rows.push(['Stage 03 hazard accuracy: clean / degraded text', tf.clean.hazard_accuracy + ' / ' + tf.degraded_comms_noise.hazard_accuracy, '-']);
    rows.push(['Stage 03 urgency macro F1: clean / degraded text', tf.clean.urgency_macro_f1 + ' / ' + tf.degraded_comms_noise.urgency_macro_f1, '-']);
    rows.push(['Implicit-urgency messages scored HIGH+', pct(tf.implicit_urgency_recall), '-']);
  }
  table('realism', ['Check', 'CVAE', 'Naive baseline'], rows);
}

function renderProbe(probe) {
  if (!probe) { table('probe', ['Flat level (m)', 'Projected 6 h change (m)', 'Fusion reads it as'], []); return; }
  document.getElementById('probe-note').textContent = 'Real gauges rise more than ' + probe.real_gauge_rise_6h_q99_m +
    ' m in 6 h in only 1% of windows; fusion escalates to URGENT at a projected rise of ' + probe.steep_rise_threshold_m + ' m.';
  table('probe', ['Flat level (m)', 'Forecast 6 h peak (m)', 'Projected change (m)', 'Fusion reads it as'], probe.rows.map(r => [
    r.flat_level_m.toFixed(1), r.peak_6h_m.toFixed(2), (r.change_6h_m >= 0 ? '+' : '') + r.change_6h_m.toFixed(2),
    el('span', {class: r.fusion_reading.startsWith('steep') ? 'fail' : 'muted', text: r.fusion_reading})]));
}

function renderFailures(failures) {
  table('failures', ['Scenario', 'Zone', 'True', 'Called', 'Why'], failures.map(f =>
    [f.scenario_id, f.zone_id + ' ' + f.label, priorityBadge(f.true_severity), f.priority ? priorityBadge(f.priority) : (f.status || '-'), f.failures]));
}

function setupLive() {
  if (!DATA.live) return;
  document.getElementById('live-section').hidden = false;
  const select = document.getElementById('prompt-select');
  DATA.prompts.forEach(p => select.appendChild(el('option', {value: p.id, text: p.id + ' - ' + p.name})));
  const describe = () => { const p = DATA.prompts.find(q => q.id === select.value); document.getElementById('prompt-text').textContent = p ? p.prompt : ''; };
  select.addEventListener('change', describe); describe();
  const btn = document.getElementById('run-btn'), out = document.getElementById('live-result');
  btn.addEventListener('click', async () => {
    btn.disabled = true; btn.textContent = 'Running (first run loads the Stage 01-04 models)...';
    try {
      const res = await fetch('/api/genai/stress-test', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt_id: select.value, seed: parseInt(document.getElementById('seed-input').value, 10)})});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'request failed');
      const passed = data.zones.filter(z => z.passed).length;
      out.replaceChildren(el('p', {style: 'margin-bottom:8px'}, [el('strong', {text: data.name + ': '}), data.passed ? passBadge(true) : el('span', {class: 'fail', text: passed + '/' + data.zones.length + ' zones passed'}), ' ranking tau ' + (data.ranking_tau ?? '-')]), el('div', {class: 'table-wrap'}, zoneTable(data.zones)));
    } catch (err) { out.replaceChildren(el('p', {class: 'fail', text: 'Error: ' + err.message})); }
    btn.disabled = false; btn.textContent = 'Generate & stress-test';
  });
}

const report = DATA.report;
setupLive();
if (!report) {
  document.getElementById('meta').textContent = 'No stress-test report yet. Run python Stage05_GenAI/04_evaluation_engineer.py.';
} else {
  const status = Object.entries(report.stage_status || {}).map(([k, v]) => k + ': ' + v).join('  |  ');
  document.getElementById('meta').textContent = 'Report ' + report.generated_at + '  |  commit ' + (report.git_commit || 'n/a') + '  |  ' + status;
  renderKpis(report.summary, report.realism);
  renderScenarios(report.scenarios || []);
  renderWildcard(report.wildcard);
  renderRealism(report.realism);
  renderProbe(report.forecast_probe);
  renderFailures(report.failures || []);
}
renderHistory(DATA.history || []);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 integration: dashboard + API")
    parser.add_argument("--serve", action="store_true", help="Run the live dashboard")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    if args.serve:
        serve(args.port)
        return

    print("=" * 60)
    print("Stage 05 GenAI Integration Engine -- Self-Test")
    print("=" * 60)
    health = genai_integration_engine.health_check()
    for key, value in health.items():
        print(f"  {key:<26}: {value}")
    if health["status"] == "healthy":
        scenario = genai_integration_engine.generate(prompt_id="W01", seed=SEED + 100)
        print(f"\n  generated '{scenario['name']}' with {len(scenario['zones'])} zones; "
              f"shelter gap {scenario['demand']['shelter_gap']} people")
    path = genai_integration_engine.build_dashboard()
    print(f"\n  static dashboard -> {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
