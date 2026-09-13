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
import os
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


LLM = _load_module("stage05_llm_generator", BASE_DIR / "03b_llm_scenario_generator.py")
SLM = _load_module("stage05_slm_generator", BASE_DIR / "03c_slm_sequence_generator.py")


class GenAIIntegrationEngine:
    """App-facing wrapper around the Stage 05 generator and stress tester.

    Everything heavy is lazy: the generator loads on first use and the Stage
    01-04 models only when a live stress test is actually requested.
    """

    def __init__(self) -> None:
        self._generator = None
        self._llm_generator = None
        self._slm_generator = None
        self.llm_error: str | None = None
        self.slm_error: str | None = None
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

    @property
    def llm_generator(self):
        """The LLM path, loaded on first request.

        Qwen2.5-3B takes tens of seconds to load and holds GPU memory, so the
        dashboard only pays that cost if someone actually asks for an LLM
        scenario. A load failure is recorded and the caller falls back to the
        CVAE path rather than losing the request.

        Which backend serves the request is an environment decision, not a code
        one: STAGE05_LLM_BACKEND=gemini (with GEMINI_API_KEY set) routes the
        dashboard at the hosted model instead, which is how this runs on a box
        with no GPU and no room for a 6 GB checkpoint. The default stays local.
        """
        if self._llm_generator is None:
            backend = None
            kind = os.environ.get("STAGE05_LLM_BACKEND", "qwen").strip().lower()
            try:
                backend = LLM.build_backend(
                    kind if kind in {"qwen", "gemini"} else "qwen",
                    gemini_model=os.environ.get("STAGE05_GEMINI_MODEL",
                                                LLM.DEFAULT_GEMINI_MODEL))
            except Exception as exc:
                self.llm_error = f"{type(exc).__name__}: {exc}"
            self._llm_generator = LLM.LLMScenarioGenerator(backend)
        return self._llm_generator

    @property
    def slm_generator(self):
        """The domain SLM path: 3.6M params, loads in well under a second."""
        if self._slm_generator is None:
            sampler = None
            try:
                if SLM.SLM_PATH.is_file():
                    sampler = SLM.SLMSampler()
                else:
                    self.slm_error = f"{SLM.SLM_PATH.name} not found (run 03c --train)"
            except Exception as exc:
                self.slm_error = f"{type(exc).__name__}: {exc}"
            self._slm_generator = SLM.SLMScenarioGenerator(sampler)
        return self._slm_generator

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
            "generators": {
                "llm": {"available": LLM.QWEN_BASE_DIR.is_dir(),
                        "model": "Qwen2.5-3B", "techniques": LLM.TECHNIQUES,
                        "loaded": self._llm_generator is not None,
                        "error": self.llm_error},
                "slm": {"available": SLM.SLM_PATH.is_file(),
                        "model": "ScenarioSLM (trained on this project's corpus)",
                        "techniques": SLM.TECHNIQUES,
                        "techniques_not_claimed": SLM.TECHNIQUES_NOT_CLAIMED,
                        "loaded": self._slm_generator is not None,
                        "error": self.slm_error},
                "cvae": {"available": True, "model": "SensorCVAE",
                         "role": "fallback for any zone a sequence model cannot produce validly"},
            },
            "prompts": len(self.list_prompts()),
            "latest_report_at": report.get("generated_at") if report else None,
            "latest_scenario_pass_rate": report["summary"]["scenario_pass_rate"] if report else None,
            "error": None,
        }

    def list_prompts(self) -> list[dict[str, Any]]:
        library = GENAI.load_prompts()
        return [{"id": p["id"], "name": p["name"], "archetype": p["archetype"],
                 "prompt": p["prompt"], "blind_spots": p["blind_spots"], "zones": len(p["zones"]),
                 # Seed conditions shown in the dashboard's first panel.
                 "state": p.get("state"),
                 "severities": [z.get("true_severity") for z in p["zones"]],
                 "hazards": sorted({h for z in p["zones"] for h in z.get("hazards", [])})}
                for p in library["prompts"] + [library["wildcard"]]]

    def _prompt_spec(self, prompt_id: str) -> dict[str, Any]:
        library = GENAI.load_prompts()
        for prompt in library["prompts"] + [library["wildcard"]]:
            if prompt["id"] == prompt_id:
                return prompt
        raise ValueError(f"Unknown prompt_id {prompt_id!r}")

    def generate(self, prompt_id: str | None = None, spec: dict | None = None,
                 seed: int = SEED, generator: str = "cvae") -> dict[str, Any]:
        """Realise one scenario.

        `generator="llm"` is the project's headline path -- sensors and
        narrative decoded together by Qwen2.5-3B. It costs seconds per zone, so
        the default stays on the CVAE path for interactive use and for the API
        contract that existed before.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer between 0 and {MAX_SEED}")
        if generator not in {"cvae", "llm", "slm"}:
            raise ValueError("generator must be 'cvae', 'llm' or 'slm'")
        if spec is None:
            if not isinstance(prompt_id, str):
                raise ValueError("Provide either 'prompt_id' or 'spec'")
            spec = self._prompt_spec(prompt_id)
        if generator == "llm":
            return self.llm_generator.generate(spec, seed=seed)
        if generator == "slm":
            return self.slm_generator.generate(spec, seed=seed)
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

    def parse(payload: Any) -> tuple[str | None, dict | None, int, str]:
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        seed = payload.get("seed", SEED)
        spec = payload.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("'spec' must be a JSON object")
        generator = payload.get("generator", "cvae")
        if generator not in {"cvae", "llm", "slm"}:
            raise ValueError("'generator' must be 'cvae', 'llm' or 'slm'")
        return payload.get("prompt_id"), spec, seed, generator

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
            prompt_id, spec, seed, family = parse(request.get_json(silent=True))
            return jsonify(engine.generate(prompt_id=prompt_id, spec=spec, seed=seed,
                                           generator=family))
        except ValueError as exc:
            # Validation messages are written for the caller and safe to show.
            return fail(str(exc), 400)
        except Exception as exc:
            return fail("Scenario generation failed. See server logs for details.", 500, exc)

    @blueprint.route("/api/genai/stress-test", methods=["POST"])
    def stress_test():
        try:
            prompt_id, spec, seed, family = parse(request.get_json(silent=True))
            generated = engine.generate(prompt_id=prompt_id, spec=spec, seed=seed,
                                        generator=family)
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
<title>Stage 05 GenAI Scenario Studio</title>
<style>
  /* Same palette as the main dashboard (app.py). */
  :root { --bg:#0B1120; --card:rgba(30,41,59,.7); --line:#1E293B; --soft:#0F172A; --text:#F8FAFC; --muted:#94A3B8;
          --chip:#1E293B; --chip-text:#CBD5E1; --blue:#3B82F6; --blue-soft:rgba(59,130,246,.12); --green:#10B981;
          --green-soft:rgba(16,185,129,.12); --amber:#F59E0B; --amber-soft:rgba(245,158,11,.12); --red:#EF4444;
          --tint-blue:rgba(59,130,246,.06); --tint-green:rgba(16,185,129,.05); --hover:rgba(59,130,246,.08); --shadow:none; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  .topbar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 24px 0; }
  .topbar .title { display:flex; flex-wrap:wrap; align-items:center; gap:8px 14px; }
  .topbar h1 { font-size:18px; }
  .pills { display:flex; flex-wrap:wrap; gap:6px; }
  .pill { font-size:11px; color:var(--muted); background:var(--card); border:1px solid var(--line); border-radius:999px; padding:2px 10px; }
  .pill.ok { color:var(--green); }
  .tabs { display:flex; gap:4px; margin-top:10px; overflow-x:auto; }
  .tabs button { background:none; border:0; border-bottom:2px solid transparent; color:var(--muted); font:inherit; font-weight:600; padding:8px 14px; cursor:pointer; white-space:nowrap; }
  .tabs button.active { color:var(--text); border-bottom-color:var(--blue); }
  main { padding:18px 24px 40px; max-width:1360px; margin:0 auto; }
  .row2 { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:16px; margin-bottom:16px; }
  .row-gen { display:grid; grid-template-columns:1fr; gap:16px; margin-bottom:16px; }  /* scenario on top, validation below */
  @media (max-width:980px) { .row2, .row-gen { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:var(--shadow); min-width:0; }
  .card.tint { background:linear-gradient(180deg,var(--tint-blue),var(--card) 60%); }
  .card.tint-green { background:linear-gradient(180deg,var(--tint-green),var(--card) 60%); }
  .head { display:flex; gap:12px; align-items:center; margin-bottom:14px; }
  .step, .icon-circle { width:34px; height:34px; border-radius:50%; background:var(--blue); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:16px; flex:none; }
  .icon-circle.green { background:var(--green); }
  .head h2 { font-size:14px; letter-spacing:.05em; text-transform:uppercase; }
  .head p { color:var(--muted); font-size:12px; }
  .form-grid { display:grid; grid-template-columns:minmax(0,1fr) 120px; gap:12px; }
  .form-grid .wide { grid-column:1 / -1; }
  .f { display:flex; flex-direction:column; gap:5px; font-size:12px; font-weight:600; color:var(--muted); min-width:0; }
  select, input { width:100%; min-width:0; background:var(--soft); border:1px solid var(--line); color:var(--text); padding:9px 10px; border-radius:8px; font:inherit; font-size:13px; }
  .seed-summary { display:grid; grid-template-columns:max-content minmax(0,1fr); gap:8px 14px; margin-top:14px; padding:12px; background:var(--soft); border:1px solid var(--line); border-radius:10px; font-size:12px; align-items:center; }
  .seed-summary dt { color:var(--muted); font-weight:600; white-space:nowrap; }
  .chips { display:flex; flex-wrap:wrap; gap:4px; }
  .chips span { padding:1px 9px; border-radius:10px; background:var(--chip); color:var(--chip-text); font-size:11px; display:inline-flex; align-items:center; gap:5px; }
  .chips span i { width:7px; height:7px; border-radius:50%; display:inline-block; }
  .prompt-text { margin-top:10px; font-size:12px; color:var(--muted); display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
  .gen-card { display:flex; flex-direction:column; }
  .btn { display:flex; gap:10px; align-items:center; justify-content:center; width:100%; border:0; border-radius:10px; font:inherit; font-weight:700; cursor:pointer; }
  .btn.primary { background:var(--blue); color:#fff; font-size:17px; padding:15px; box-shadow:0 6px 16px rgba(29,111,216,.25); margin:6px 0 10px; }
  .btn.secondary { background:transparent; color:var(--blue); border:1.5px solid var(--blue); font-size:13px; padding:10px; }
  .btn:disabled { opacity:.55; cursor:not-allowed; }
  .hint { display:flex; gap:10px; background:var(--blue-soft); border-radius:10px; padding:10px 12px; margin-top:auto; font-size:12px; color:var(--muted); }
  .status { font-size:12px; margin:8px 0 12px; min-height:18px; }
  .empty-card { text-align:center; padding:30px 18px; color:var(--muted); margin-bottom:16px; border-style:dashed; }
  .empty-card b { color:var(--text); }
  .kv { display:grid; grid-template-columns:120px minmax(0,1fr); gap:5px 12px; font-size:13px; }
  .kv dt { color:var(--muted); font-weight:600; } .kv dd { min-width:0; overflow-wrap:anywhere; }
  .gen-body { display:flex; flex-direction:column; gap:12px; }
  @media (max-width:640px) { .gen-body { grid-template-columns:1fr; } .form-grid { grid-template-columns:1fr; } main, .topbar { padding-left:14px; padding-right:14px; } }
  .badges { display:flex; flex-wrap:wrap; gap:6px; }
  .badges span { font-size:11px; color:var(--chip-text); background:var(--chip); border-radius:999px; padding:2px 10px; }
  .empty { color:var(--muted); padding:12px 0; font-size:13px; }
  .vrow { display:grid; grid-template-columns:minmax(120px,160px) minmax(0,1fr) 44px 22px; align-items:center; gap:10px; padding:6px 0; }
  .vrow .name { font-weight:600; font-size:13px; line-height:1.25; } .vrow .sub { color:var(--muted); font-size:11px; font-weight:400; }
  .bar { height:9px; background:var(--chip); border-radius:6px; overflow:hidden; } .bar i { display:block; height:100%; border-radius:6px; }
  .vrow .val { font-weight:700; font-size:13px; text-align:right; }
  .tick { width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:11px; font-weight:700; }
  .verdict { margin-top:12px; border-radius:10px; padding:12px; display:flex; gap:10px; align-items:center; font-weight:800; letter-spacing:.04em; }
  .verdict.ok { background:var(--green-soft); color:var(--green); } .verdict.warn { background:var(--amber-soft); color:var(--amber); }
  .verdict small { display:block; font-weight:500; letter-spacing:0; font-size:11px; color:var(--muted); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .stat { background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:12px; }
  .stat .top { display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:12px; }
  .stat .num { font-size:24px; font-weight:800; margin:4px 0 0; } .stat .note { color:var(--muted); font-size:11px; }
  .donut-card { grid-column:span 2; display:flex; gap:16px; align-items:center; background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:12px; }
  @media (max-width:520px) { .donut-card { grid-column:auto; flex-direction:column; } }
  .donut-card h3 { font-size:13px; margin-bottom:6px; }
  .legend { font-size:12px; display:grid; grid-template-columns:12px auto auto auto; gap:4px 10px; align-items:center; }
  .legend i { width:10px; height:10px; border-radius:50%; display:block; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr)); gap:12px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; min-width:0; }
  .kpi .label { color:var(--muted); font-size:12px; } .kpi .value { font-size:22px; font-weight:800; margin-top:2px; } .kpi .hint { color:var(--muted); font-size:11px; }
  .table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; background:var(--soft); white-space:nowrap; }
  tr.clickable { cursor:pointer; } tr.clickable:hover, tr.detail td { background:var(--hover); }
  .badge { display:inline-block; padding:1px 9px; border-radius:20px; font-size:11px; font-weight:700; color:#fff; white-space:nowrap; }
  .pass { color:var(--green); font-weight:700; } .fail { color:var(--red); font-weight:700; } .muted { color:var(--muted); }
  .zone-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr)); gap:12px; }
  .zone-cards ul { margin:6px 0 0 18px; color:var(--muted); font-size:12px; }
  h3.sub { font-size:14px; margin:20px 0 8px; } h3.sub:first-child { margin-top:0; }
  svg text { fill:var(--muted); font-size:10px; }
</style>
</head>
<body>
<header class="topbar">
  <div class="title"><h1>&#10024; GenAI Scenario Studio</h1><div class="pills" id="meta"></div></div>
  <nav class="tabs" id="tabs">
    <button data-tab="studio" class="active">Studio</button>
    <button data-tab="results">Stress-test results</button>
    <button data-tab="audit">Audits</button>
    <button data-tab="history">History</button>
  </nav>
</header>
<main>
  <section data-view="studio">
    <div class="row2">
      <div class="card">
        <div class="head"><div class="step">1</div><div><h2>Seed conditions</h2><p>Choose the scenario prompt and how it is generated</p></div></div>
        <div class="form-grid">
          <label class="f wide">&#128203; Scenario prompt<select id="prompt-select"></select></label>
          <label class="f">&#129504; Generator
            <select id="generator-select">
              <option value="slm">Domain SLM (sensors + narrative together)</option>
              <option value="cvae">CVAE + phrase bank (statistical)</option>
              <option value="llm">LLM (Qwen2.5-3B or Gemini backend)</option>
            </select></label>
          <label class="f">&#127922; Seed<input id="seed-input" type="number" value="42" min="0"></label>
        </div>
        <dl class="seed-summary">
          <dt>&#128205; Region</dt><dd id="seed-state">-</dd>
          <dt>&#9888;&#65039; Severity mix</dt><dd id="seed-severity" class="chips"></dd>
          <dt>&#127754; Hazards</dt><dd id="seed-hazards" class="chips"></dd>
          <dt>&#127919; Blind spots</dt><dd id="seed-blind" class="chips"></dd>
        </dl>
        <p class="prompt-text" id="prompt-text"></p>
      </div>
      <div class="card gen-card">
        <div class="head"><div class="step">2</div><div><h2>Generate</h2><p>Create a synthetic multi-zone disaster scenario</p></div></div>
        <button class="btn primary" id="generate-btn">&#10022; Generate Scenario</button>
        <button class="btn secondary" id="stress-btn">&#129514; Generate &amp; stress-test through Stages 01&ndash;04</button>
        <p class="status" id="gen-status"></p>
        <div class="hint"><span>&#128161;</span><span>Generates sensor readings, a 72&nbsp;h gauge history, dispatcher messages and an incident log for every zone. The stress test then runs the scenario through the real Stage 01&ndash;04 models and the fusion layer (the first run loads the models, about 20&nbsp;s).</span></div>
      </div>
    </div>

    <div class="card empty-card" id="empty-state"><b>No scenario generated yet.</b><br>Pick seed conditions and press <b>Generate Scenario</b> to see the scenario profile and its validation here.</div>
    <div class="row-gen" id="result-row" hidden>
      <div class="card tint">
        <div class="head"><div class="icon-circle">&#128101;</div><div><h2>Generated synthetic scenario</h2><p>Scenario profile produced from your seed conditions</p></div></div>
        <div id="generated"></div>
      </div>
      <div class="card tint-green">
        <div class="head"><div class="icon-circle green">&#10004;</div><div><h2>Validation</h2><p>Checks for realism, consistency and quality</p></div></div>
        <div id="validation-body"></div>
      </div>
    </div>

    <div class="card">
      <div class="head"><div class="icon-circle">&#128202;</div><div><h2>Case analysis</h2><p>The generated scenario suite and the data behind it</p></div></div>
      <div id="case-analysis"></div>
    </div>
  </section>

  <section data-view="results" hidden>
    <h3 class="sub">Headline results</h3>
    <div class="grid" id="kpis"></div>
    <h3 class="sub">Scenarios <span class="muted" style="font-weight:400;font-size:12px">(click a row for zone detail)</span></h3>
    <div class="table-wrap"><table id="scenario-table"></table></div>
    <h3 class="sub" id="wildcard-title">Wildcard</h3>
    <p id="wildcard-prompt" class="muted" style="margin-bottom:10px;font-size:12px"></p>
    <div class="zone-cards" id="wildcard"></div>
    <h3 class="sub">Failure log</h3>
    <div class="table-wrap"><table id="failures"></table></div>
  </section>

  <section data-view="audit" hidden>
    <h3 class="sub">Scenario audit <span class="muted" style="font-weight:400;font-size:12px">(physical coherence, per zone)</span></h3>
    <div class="grid" id="audit-kpis"></div>
    <p id="audit-note" class="muted" style="margin:10px 0;font-size:12px"></p>
    <div class="table-wrap"><table id="overconfident"></table></div>
    <h3 class="sub">Realism audit <span class="muted" style="font-weight:400;font-size:12px">(generator vs the real record)</span></h3>
    <div class="table-wrap"><table id="realism"></table></div>
    <h3 class="sub">Stage 02 forecast probe <span class="muted" style="font-weight:400;font-size:12px">(perfectly flat 72 h river)</span></h3>
    <p id="probe-note" class="muted" style="margin-bottom:10px;font-size:12px"></p>
    <div class="table-wrap"><table id="probe"></table></div>
  </section>

  <section data-view="history" hidden>
    <div class="card" id="history-chart"></div>
  </section>
</main>
<script>
const DATA = __DASHBOARD_DATA__;
const LEVELS = ['ROUTINE', 'ELEVATED', 'URGENT', 'CRITICAL'];
const SEV_COLORS = {ROUTINE: '#16A34A', ELEVATED: '#F59E0B', URGENT: '#F97316', CRITICAL: '#DC2626'};
const BAR_COLORS = ['#16A34A', '#0EA5E9', '#8B5CF6', '#14B8A6', '#F59E0B', '#3B82F6'];
const ACCEPT_THRESHOLD = 0.8;
const SENSOR_FIELDS = ['timestamp', 'state', 'district', 'rainfall_mm', 'river_level_m', 'river_level_threshold_m',
  'emergency_calls', 'road_closures', 'bridge_closures', 'flood_history_count', 'population_affected', 'water_level_change_m'];
const LOOKBACK = 72;
const EVIDENCE_ICONS = {sensors: '\u{1F4DF}', text: '\u{1F4AC}', image: '\u{1F4F7}', water_history: '\u{1F4C8}'};
const report = DATA.report;
const audit = report ? report.scenario_audit : null;
const realism = report ? report.realism : null;
let currentTest = null;

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
const $ = id => document.getElementById(id);
const pct = v => v == null ? 'n/a' : (v * 100).toFixed(1) + '%';
const pct0 = v => v == null ? 'n/a' : Math.round(v * 100) + '%';
const num = v => v == null ? 'n/a' : Number(v).toLocaleString();
const priorityBadge = p => p && LEVELS.includes(p) ? el('span', {class: 'badge', style: 'background:' + SEV_COLORS[p], text: p}) : el('span', {class: 'muted', text: p || '-'});
const passBadge = ok => el('span', {class: ok ? 'pass' : 'fail', text: ok ? 'PASS' : 'FAIL'});
const countBy = items => items.reduce((acc, k) => (acc[k] = (acc[k] || 0) + 1, acc), {});
const chips = (target, items) => $(target).replaceChildren(...(items.length ? items : [el('span', {text: '-'})]).map(t => typeof t === 'string' ? el('span', {text: t}) : t));

function table(target, headers, rows) {
  const t = $(target);
  t.replaceChildren(el('tr', {}, headers.map(h => el('th', {text: h}))));
  if (!rows.length) { t.appendChild(el('tr', {}, el('td', {class: 'empty', colspan: String(headers.length), text: 'Nothing to show.'}))); return t; }
  rows.forEach(r => t.appendChild(el('tr', {}, r.map(c => el('td', {}, c instanceof Node ? c : String(c ?? '-'))))));
  return t;
}

// ---------------------------------------------------------------- tabs + header
function setupTabs() {
  document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('active', x === b));
    document.querySelectorAll('[data-view]').forEach(v => { v.hidden = v.dataset.view !== b.dataset.tab; });
    window.scrollTo(0, 0);
  }));
}

function renderMeta() {
  const box = $('meta');
  if (!report) { box.replaceChildren(el('span', {class: 'pill', text: 'No stress-test report yet: run 04_evaluation_engineer.py'})); return; }
  const stages = Object.values(report.stage_status || {});
  const healthy = stages.filter(v => v === 'healthy').length;
  box.replaceChildren(
    el('span', {class: 'pill', text: 'Report ' + String(report.generated_at || '').replace('T', ' ').slice(0, 16)}),
    el('span', {class: 'pill', text: 'commit ' + (report.git_commit || 'n/a')}),
    el('span', {class: 'pill' + (healthy === stages.length ? ' ok' : ''), title: Object.entries(report.stage_status || {}).map(([k, v]) => k + ': ' + v).join(', '),
      text: healthy + '/' + stages.length + ' stages healthy'}),
  );
}

// ---------------------------------------------------------------- seed + generate
function selectedPrompt() { return (DATA.prompts || []).find(p => p.id === $('prompt-select').value); }

function describePrompt() {
  const p = selectedPrompt();
  $('prompt-text').textContent = p ? p.prompt : 'No prompt library found. Run 02_eda_engineer.py.';
  $('prompt-text').title = p ? p.prompt : '';
  $('seed-state').textContent = p ? (p.state || 'multi-state') + ' · ' + p.zones + ' zones' : '-';
  const mix = p && p.severities ? countBy(p.severities) : {};
  chips('seed-severity', LEVELS.filter(l => mix[l]).map(l => el('span', {}, [el('i', {style: 'background:' + SEV_COLORS[l]}), mix[l] + ' ' + l])));
  chips('seed-hazards', p && p.hazards ? p.hazards : []);
  chips('seed-blind', p ? p.blind_spots : []);
}

function setupSeed() {
  const select = $('prompt-select');
  (DATA.prompts || []).forEach(p => select.appendChild(el('option', {value: p.id, text: p.id + ' · ' + p.name})));
  select.addEventListener('change', describePrompt);
  describePrompt();
  if (!DATA.live) {
    [$('generate-btn'), $('stress-btn')].forEach(b => { b.disabled = true; });
    $('gen-status').textContent = 'Static snapshot: open /genai on the running dashboard to generate live.';
    return;
  }
  $('generate-btn').addEventListener('click', () => run(false));
  $('stress-btn').addEventListener('click', () => run(true));
}

async function post(path, body) {
  let res;
  try {
    res = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  } catch (err) {
    throw new Error('the dashboard server is not reachable. Is python app.py still running?');
  }
  const data = await res.json().catch(() => ({error: 'invalid response'}));
  if (!res.ok) throw new Error(data.error || 'request failed');
  return data;
}

async function run(withStress) {
  const body = {prompt_id: $('prompt-select').value, seed: parseInt($('seed-input').value, 10), generator: $('generator-select').value};
  const buttons = [$('generate-btn'), $('stress-btn')];
  buttons.forEach(b => b.disabled = true);
  const status = $('gen-status');
  status.className = 'status muted';
  status.textContent = withStress ? 'Generating and stress-testing…' : 'Generating…';
  try {
    const scenario = await post('/api/genai/scenario', body);
    currentTest = withStress ? await post('/api/genai/stress-test', body) : null;
    renderGenerated(scenario);
    renderValidation(scenario);
    $('empty-state').hidden = true;
    $('result-row').hidden = false;
    status.className = 'status pass';
    status.textContent = '✓ Generated ' + scenario.scenario_id + ' (seed ' + body.seed + ')' + (currentTest ? ' and stress-tested.' : '.');
    $('result-row').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (err) {
    status.className = 'status fail';
    status.textContent = 'Error: ' + err.message;
  }
  buttons.forEach(b => b.disabled = false);
}

// ---------------------------------------------------------------- generated scenario
function generatorLabel(sc) {
  const g = sc.generator;
  if (g && typeof g === 'object') return [g.family, g.model].filter(Boolean).join(' · ');
  if (typeof g === 'string') return g;
  return {slm: 'Domain SLM', cvae: 'CVAE + phrase bank', llm: 'LLM'}[$('generator-select').value];
}

function renderGenerated(sc) {
  const zones = sc.zones || [];
  const mix = countBy(zones.map(z => z.true_severity));
  const hazards = [...new Set(zones.flatMap(z => z.hazards || []))];
  const modifiers = [...new Set((sc.global_modifiers || []).concat(zones.flatMap(z => z.modifiers || [])))];
  const states = [...new Set(zones.map(z => z.state).filter(Boolean))];
  const demand = sc.demand || {}, res = sc.resources || {};
  const rows = [
    ['Scenario', sc.scenario_id + ' · ' + sc.name], ['Region', states.join(', ')], ['Start time', sc.start_time],
    ['Zones', zones.length + ' (' + LEVELS.filter(l => mix[l]).map(l => mix[l] + ' ' + l).join(', ') + ')'],
    ['Hazards', hazards.join(', ') || '-'], ['Modifiers', modifiers.join(', ').replace(/_/g, ' ') || 'none'],
    ['Blind spots', (sc.blind_spots || []).join(', ') || '-'],
    ['People reported', num(demand.total_headcount_reported) + ' (shelter gap ' + num(demand.shelter_gap) + ')'],
    ['Resources', res.rescue_boats + ' boats · ' + res.ambulances + ' ambulances · ' + res.shelter_beds + ' beds'],
    ['Generator', generatorLabel(sc)],
  ];
  const kv = el('dl', {class: 'kv'}, rows.flatMap(([k, v]) => [el('dt', {text: k}), el('dd', {text: v == null ? '-' : String(v)})]));
  const lost = zones.filter(z => ((z.provenance || {}).evidence_lost || []).length).length;
  const beyond = zones.filter(z => ((z.provenance || {}).beyond_record_fields || []).length).length;
  const badges = el('div', {class: 'badges'}, [
    el('span', {text: '\u{1F5FA} ' + zones.length + '-zone incident'}),
    el('span', {text: '\u{1F3AF} ' + (sc.blind_spots || []).length + ' blind spot(s) targeted'}),
    el('span', {text: '⚡ ' + (beyond ? beyond + ' zone(s) beyond the record' : 'within recorded range')}),
    el('span', {text: '\u{1F4E1} ' + (lost ? lost + ' zone(s) lost evidence' : 'all evidence present')}),
    el('span', {text: '✨ Generated by GenAI'}),
  ]);
  const zt = el('table', {id: 'generated-zones'});
  $('generated').replaceChildren(el('div', {class: 'gen-body'}, [kv, badges]), el('div', {class: 'table-wrap', style: 'margin-top:14px'}, zt));
  table('generated-zones', ['Zone', 'District', 'Truth', 'River / danger', 'Evidence', 'People'], zones.map(z => {
    const s = (z.inputs || {}).sensors, ev = (z.provenance || {}).evidence_available || {};
    return [z.zone_id + ' ' + (z.label || ''), z.district || '-', priorityBadge(z.true_severity),
      s ? Number(s.river_level_m).toFixed(2) + ' / ' + Number(s.river_level_threshold_m).toFixed(2) + ' m' : 'no sensors',
      Object.entries(ev).filter(([, v]) => v).map(([k]) => EVIDENCE_ICONS[k] || k).join(' ') || 'none', z.headcount_reported ?? '-'];
  }));
}

// ---------------------------------------------------------------- validation
function structuralCompliance(sc) {
  const zones = sc.zones || [];
  let ok = 0;
  zones.forEach(z => {
    const inputs = z.inputs || {}, ev = (z.provenance || {}).evidence_available || {};
    const sensorsOk = !inputs.sensors || SENSOR_FIELDS.every(f => inputs.sensors[f] !== undefined && inputs.sensors[f] !== null && inputs.sensors[f] !== '');
    const levelsOk = !inputs.water_levels || (inputs.water_levels.length === LOOKBACK && inputs.water_levels.every(Number.isFinite));
    const textOk = !ev.text || (typeof inputs.text === 'string' && inputs.text.trim().length > 0);
    if (sensorsOk && levelsOk && textOk && z.expected && LEVELS.includes(z.true_severity)) ok += 1;
  });
  return {rate: zones.length ? ok / zones.length : null, detail: ok + '/' + zones.length + ' zones meet the input contract'};
}

function reportScenario(id) {
  if (!report) return null;
  return (report.scenarios || []).concat(report.wildcard ? [report.wildcard] : []).find(s => s.scenario_id === id) || null;
}

function validationChecks(sc) {
  const checks = [];
  if (realism) checks.push({name: 'Statistical', sub: '1 − mean KS vs real rows', value: 1 - realism.mean_ks_overall});
  const ps = audit ? (audit.per_scenario || []).find(p => p.scenario_id === sc.scenario_id) : null;
  if (ps && ps.realism_score != null) checks.push({name: 'Physical plausibility', sub: 'scenario audit', value: ps.realism_score});
  const sc2 = structuralCompliance(sc);
  checks.push({name: 'Constraint compliance', sub: sc2.detail, value: sc2.rate, required: true});
  if (audit) checks.push({name: 'Diversity', sub: 'blind spots covered', value: audit.diversity_coverage.blind_spot_coverage});
  if (audit) checks.push({name: 'Severity calibration', sub: '1 − overconfidence', value: 1 - audit.overconfidence.rate});
  const test = currentTest || reportScenario(sc.scenario_id);
  if (test) {
    const passed = test.zones.filter(z => z.passed).length;
    checks.push({name: 'Pipeline stress test', sub: passed + '/' + test.zones.length + ' zones' + (currentTest ? ' (this run)' : ' (last evaluation)'), value: passed / test.zones.length});
  }
  return checks;
}

function renderValidation(sc) {
  const checks = validationChecks(sc);
  const meets = c => c.value != null && (c.required ? c.value === 1 : c.value >= ACCEPT_THRESHOLD);
  const rows = checks.map((c, i) => el('div', {class: 'vrow'}, [
    el('div', {class: 'name'}, [c.name, el('div', {class: 'sub', text: c.sub})]),
    el('div', {class: 'bar'}, el('i', {style: 'width:' + Math.max(0, Math.min(100, (c.value || 0) * 100)) + '%;background:' + BAR_COLORS[i % BAR_COLORS.length]})),
    el('div', {class: 'val', text: pct0(c.value)}),
    el('div', {class: 'tick', style: 'background:' + (meets(c) ? 'var(--green)' : 'var(--amber)'), text: meets(c) ? '✓' : '!'}),
  ]));
  const weak = checks.filter(c => !meets(c)).map(c => c.name);
  const misses = currentTest ? currentTest.zones.filter(z => z.critical_miss).length : 0;
  if (misses) weak.push(misses + ' critical miss(es)');
  const accepted = weak.length === 0;
  const verdict = el('div', {class: 'verdict ' + (accepted ? 'ok' : 'warn')}, [
    el('span', {style: 'font-size:18px', text: accepted ? '✅' : '⚠️'}),
    el('div', {}, [accepted ? 'SYNTHETIC SCENARIO ACCEPTED' : 'NEEDS REVIEW',
      el('small', {text: accepted ? 'All checks ≥ ' + pct0(ACCEPT_THRESHOLD) + ', input contract 100%.' : 'Below threshold: ' + weak.join(', ')})]),
  ]);
  const out = [...rows, verdict];
  if (currentTest) out.push(el('div', {class: 'table-wrap', style: 'margin-top:12px'}, zoneTable(currentTest.zones, true)));
  $('validation-body').replaceChildren(...out);
}

// ---------------------------------------------------------------- case analysis
function donut(counts) {
  const ns = 'http://www.w3.org/2000/svg', size = 110, r = 40, c = 2 * Math.PI * r;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`); svg.setAttribute('width', '110'); svg.setAttribute('height', '110');
  let offset = 0;
  LEVELS.filter(l => counts[l]).forEach(l => {
    const seg = document.createElementNS(ns, 'circle'), len = c * counts[l] / total;
    [['cx', size / 2], ['cy', size / 2], ['r', r], ['fill', 'none'], ['stroke', SEV_COLORS[l]], ['stroke-width', '15'],
     ['stroke-dasharray', `${len} ${c - len}`], ['stroke-dashoffset', String(-offset)], ['transform', `rotate(-90 ${size / 2} ${size / 2})`]]
      .forEach(([k, v]) => seg.setAttribute(k, v));
    svg.appendChild(seg); offset += len;
  });
  [[String(total), size / 2 + 4, 17, 800], ['zones', size / 2 + 17, 10, 400]].forEach(([t, y, fs, fw]) => {
    const txt = document.createElementNS(ns, 'text');
    txt.setAttribute('x', size / 2); txt.setAttribute('y', y); txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('style', `fill:var(--text);font-size:${fs}px;font-weight:${fw}`); txt.textContent = t; svg.appendChild(txt);
  });
  const legend = el('div', {class: 'legend'}, LEVELS.filter(l => counts[l]).flatMap(l => [
    el('i', {style: 'background:' + SEV_COLORS[l]}), el('span', {text: l}), el('span', {class: 'muted', text: String(counts[l])}),
    el('b', {text: Math.round(100 * counts[l] / total) + '%'})]));
  return el('div', {class: 'donut-card'}, [svg, el('div', {}, [el('h3', {text: 'Severity distribution'}), legend])]);
}

function renderCaseAnalysis() {
  const box = $('case-analysis');
  if (!report) { box.replaceChildren(el('p', {class: 'empty', text: 'No stress-test report yet. Run python Stage05_GenAI/04_evaluation_engineer.py.'})); return; }
  const scenarios = (report.scenarios || []).concat(report.wildcard ? [report.wildcard] : []);
  const zones = scenarios.flatMap(sc => sc.zones || []);
  const dv = audit ? audit.diversity_coverage : null;
  const failed = zones.filter(z => !z.passed).length;
  const stat = (icon, label, value, note, color) => el('div', {class: 'stat'}, [
    el('div', {class: 'top'}, [el('span', {text: label}), el('span', {text: icon})]),
    el('div', {class: 'num', style: color ? 'color:' + color : '', text: value}), el('div', {class: 'note', text: note})]);
  box.replaceChildren(el('div', {class: 'stats'}, [
    stat('\u{1F5C4}', 'Real records', num(realism ? realism.n_real : null), 'historical sensor rows', 'var(--blue)'),
    stat('\u{1F4C4}', 'Synthetic samples', num(realism ? realism.n_synthetic : null), 'generated for the audit', 'var(--blue)'),
    stat('\u{1F9E9}', 'Scenarios', String(scenarios.length), zones.length + ' zones incl. wildcard', '#8B5CF6'),
    stat('⭐', 'Rare conditions', dv ? String(dv.blind_spots_rare_or_absent) : 'n/a', 'rare / absent blind spots', 'var(--amber)'),
    stat('⚠️', 'Stress failures', String(failed), 'zones failing expectations', failed ? 'var(--red)' : 'var(--green)'),
    donut(countBy(zones.map(z => z.true_severity))),
  ]));
}

// ---------------------------------------------------------------- results + audits
function zoneTable(zones, compact) {
  const t = el('table');
  const headers = compact ? ['Zone', 'True', 'Called', 'Result'] : ['Zone', 'True', 'Called', 'Review', 'Conflicts', 'Sources used', 'Evidence lost', 'Result'];
  t.appendChild(el('tr', {}, headers.map(h => el('th', {text: h}))));
  zones.forEach(z => {
    const called = el('td', {}, z.status === 'ok' ? priorityBadge(z.priority) : el('span', {class: 'muted', text: z.status}));
    const result = el('td', {}, z.passed ? passBadge(true) : el('span', {class: 'fail', text: 'FAIL: ' + z.failures}));
    t.appendChild(el('tr', {}, compact
      ? [el('td', {text: z.zone_id}), el('td', {}, priorityBadge(z.true_severity)), called, result]
      : [el('td', {text: z.zone_id + ' ' + z.label}), el('td', {}, priorityBadge(z.true_severity)), called,
         el('td', {text: z.human_review ? 'yes' : 'no'}), el('td', {text: String(z.n_conflicts)}),
         el('td', {class: 'muted', text: z.sources_used || 'none'}), el('td', {class: 'muted', text: z.evidence_lost || '-'}), result]));
  });
  return t;
}

function kpiCards(target, cards) {
  $(target).replaceChildren(...cards.map(([label, value, hint]) =>
    el('div', {class: 'kpi'}, [el('div', {class: 'label', text: label}), el('div', {class: 'value', text: String(value)}), el('div', {class: 'hint', text: hint})])));
}

function renderKpis(s, r) {
  kpiCards('kpis', [
    ['Scenarios passed', s.scenarios_passed + '/' + s.scenarios, pct(s.scenario_pass_rate)],
    ['Zones passed', s.zones_passed + '/' + s.zones, pct(s.zone_pass_rate)],
    ['Critical-miss rate', pct(s.critical_miss_rate), s.critical_misses + ' of ' + s.high_risk_zones_scored + ' high-risk zones'],
    ['Pipeline crashes', String(s.crashes), 'exceptions during assessment'],
    ['Conflicts surfaced', pct(s.conflict_detection_rate), 'where evidence disagrees'],
    ['No-evidence refused', pct(s.insufficient_evidence_handled), 'silence never read as safe'],
    ['Zone ranking tau', s.mean_ranking_tau ?? 'n/a', 'Kendall, ' + s.ranking_scenarios + ' scenarios'],
    ['C2ST ROC-AUC', r ? r.c2st.cvae.auc_mean : 'n/a', r ? 'naive baseline ' + r.c2st.naive_baseline.auc_mean : 'realism audit not run'],
  ]);
}

function renderAudit(a) {
  const note = $('audit-note');
  if (!a) { $('audit-kpis').replaceChildren(); note.textContent = 'Scenario audit not present in this report.'; $('overconfident').replaceChildren(); return; }
  const ra = a.realism_score, oc = a.overconfidence, dv = a.diversity_coverage;
  kpiCards('audit-kpis', [
    ['Realism score', ra.mean ?? 'n/a', 'mean over ' + ra.zones_scored + ' scored zones, min ' + (ra.min ?? 'n/a')],
    ['Fully plausible', pct(ra.fully_plausible_rate), ra.fully_plausible_zones + ' of ' + ra.zones_scored + ' zones break no rule'],
    ['Overconfidence', pct(oc.rate), oc.zones + ' zones: severe label, benign physics'],
    ['Blind-spot coverage', pct(dv.blind_spot_coverage), (dv.blind_spots_covered ?? 0) + ' of ' + dv.blind_spots_rare_or_absent + ' rare/absent spots'],
    ['Failure modes', String(dv.distinct_archetypes), 'distinct archetypes in the suite'],
    ['Condition coverage', dv.sensor_grid_cells_occupied + '/' + dv.sensor_grid_cells_total, pct(dv.sensor_grid_coverage) + ' of the rainfall x river grid'],
  ]);
  const rules = Object.entries(ra.violations_by_rule || {});
  note.textContent = rules.length ? 'Plausibility rules violated: ' + rules.map(([k, v]) => k + ' x' + v).join(', ') + '.' : 'No plausibility rule was violated anywhere in the suite.';
  table('overconfident', ['Scenario', 'Zone', 'Label', 'Truth', 'Rules broken'],
    (oc.detail || []).map(d => [d.scenario_id, d.zone_id, d.label, d.true_severity, (d.violations || []).join(', ')]));
}

function renderHistory(history) {
  const box = $('history-chart');
  if (history.length < 2) { box.replaceChildren(el('p', {class: 'muted', text: history.length + ' run(s) recorded. The trend appears after the second run of 04_evaluation_engineer.py.'})); return; }
  const w = 640, h = 140, pad = 28, ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', '100%'); svg.setAttribute('style', 'max-width:720px');
  const x = i => pad + i * (w - 2 * pad) / (history.length - 1), y = v => h - pad - v * (h - 2 * pad);
  [['scenario_pass_rate', '#16A34A'], ['critical_miss_rate', '#DC2626']].forEach(([key, color]) => {
    const pts = history.map((r, i) => r[key] == null ? null : `${x(i)},${y(r[key])}`).filter(Boolean).join(' ');
    const line = document.createElementNS(ns, 'polyline');
    line.setAttribute('points', pts); line.setAttribute('fill', 'none'); line.setAttribute('stroke', color); line.setAttribute('stroke-width', '2');
    svg.appendChild(line);
  });
  [0, 0.5, 1].forEach(v => { const t = document.createElementNS(ns, 'text'); t.setAttribute('x', '0'); t.setAttribute('y', String(y(v) + 3)); t.textContent = (v * 100) + '%'; svg.appendChild(t); });
  box.replaceChildren(el('h3', {class: 'sub', text: 'Run history'}), svg,
    el('p', {class: 'muted', style: 'font-size:12px', text: 'Green: scenario pass rate. Red: critical-miss rate. ' + history.length + ' runs.'}));
}

function renderScenarios(scenarios) {
  const t = $('scenario-table');
  t.replaceChildren(el('tr', {}, ['ID', 'Scenario', 'Blind spots', 'Zones', 'Result', 'Ranking tau'].map(h => el('th', {text: h}))));
  scenarios.forEach(sc => {
    const passed = sc.zones.filter(z => z.passed).length;
    const row = el('tr', {class: 'clickable'}, [
      el('td', {text: sc.scenario_id}), el('td', {text: sc.name}),
      el('td', {}, el('div', {class: 'chips'}, sc.blind_spots.map(b => el('span', {text: b})))),
      el('td', {text: String(sc.zones.length)}),
      el('td', {}, sc.passed ? passBadge(true) : el('span', {class: 'fail', text: passed + '/' + sc.zones.length})),
      el('td', {text: sc.ranking_tau == null ? '-' : String(sc.ranking_tau)}),
    ]);
    const detail = el('tr', {class: 'detail', hidden: ''}, el('td', {colspan: '6'}, [el('p', {class: 'muted', style: 'margin-bottom:8px;font-size:12px', text: sc.prompt}), zoneTable(sc.zones)]));
    row.addEventListener('click', () => { detail.hidden = !detail.hidden; });
    t.append(row, detail);
  });
}

function renderWildcard(w) {
  if (!w) { $('wildcard').replaceChildren(el('p', {class: 'empty', text: 'No wildcard result.'})); return; }
  $('wildcard-title').textContent = 'Wildcard: ' + w.name;
  $('wildcard-prompt').textContent = w.prompt;
  $('wildcard').replaceChildren(...w.details.map(d => {
    const dec = d.decision || {};
    const kept = Object.entries(d.evidence_available).filter(([, v]) => v).map(([k]) => k);
    const reasons = (dec.escalations || []).concat(dec.conflicts || [], dec.human_review_reasons || []);
    return el('div', {class: 'kpi'}, [
      el('div', {style: 'display:flex;justify-content:space-between;gap:8px'}, [el('strong', {text: d.label}), passBadge(d.passed)]),
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
  $('probe-note').textContent = 'Real gauges rise more than ' + probe.real_gauge_rise_6h_q99_m +
    ' m in 6 h in only 1% of windows; fusion escalates to URGENT at a projected rise of ' + probe.steep_rise_threshold_m + ' m.';
  table('probe', ['Flat level (m)', 'Forecast 6 h peak (m)', 'Projected change (m)', 'Fusion reads it as'], probe.rows.map(r => [
    r.flat_level_m.toFixed(1), r.peak_6h_m.toFixed(2), (r.change_6h_m >= 0 ? '+' : '') + r.change_6h_m.toFixed(2),
    el('span', {class: r.fusion_reading.startsWith('steep') ? 'fail' : 'muted', text: r.fusion_reading})]));
}

function renderFailures(failures) {
  table('failures', ['Scenario', 'Zone', 'True', 'Called', 'Why'], failures.map(f =>
    [f.scenario_id, f.zone_id + ' ' + f.label, priorityBadge(f.true_severity), f.priority ? priorityBadge(f.priority) : (f.status || '-'), f.failures]));
}

// ---------------------------------------------------------------- boot
setupTabs();
renderMeta();
setupSeed();
renderCaseAnalysis();
if (report) {
  renderKpis(report.summary, report.realism);
  renderScenarios(report.scenarios || []);
  renderWildcard(report.wildcard);
  renderAudit(report.scenario_audit);
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
