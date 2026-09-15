import logging
import os
import sys
import uuid
import importlib.util
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("disaster_response.app")

# Uploads are bounded so a single request cannot exhaust memory or disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage(label, engine_path, attribute):
    """Load one stage adapter and record its REAL readiness.

    The status shown on the dashboard now comes from the stage's health_check(),
    which scores a reference record. Previously any stage whose module merely
    imported was labelled "Online" -- which is how the NLP tab advertised itself
    as online while failing every request.
    """
    try:
        module = load_module(f"{label.lower()}_api", engine_path)
        engine = getattr(module, attribute)
    except Exception as exc:
        logger.exception("Failed to load %s adapter", label)
        return None, "Offline", str(exc)

    try:
        health = engine.health_check()
    except Exception as exc:
        logger.exception("%s health check raised", label)
        return engine, "Degraded", str(exc)

    status_map = {"healthy": "Online", "degraded": "Degraded", "unavailable": "Offline"}
    status = status_map.get(health.get("status"), "Degraded")
    if status != "Online":
        logger.warning("%s reported %s: %s", label, status, health.get("error"))
    return engine, status, health.get("error")


stage01_engine, stage01_status, stage01_error = load_stage(
    "Stage01", BASE_DIR / "Stage01_ML" / "05_integration_engineer.py", "integration_engine"
)
stage02_engine, stage02_status, stage02_error = load_stage(
    "Stage02", BASE_DIR / "Stage02_DL" / "05_integration_engineer.py", "dl_integration_engine"
)
stage03_engine, stage03_status, stage03_error = load_stage(
    "Stage03", BASE_DIR / "Stage03_NLP" / "05_integration_engineer.py", "nlp_integration_engine"
)
stage04_engine, stage04_status, stage04_error = load_stage(
    "Stage04", BASE_DIR / "Stage04_SLM" / "05_integration_engineer.py", "slm_integration_engine"
)
stage05_engine, stage05_status, stage05_error = load_stage(
    "Stage05", BASE_DIR / "Stage05_GenAI" / "05_integration_engineer.py", "genai_integration_engine"
)
stage06_engine, stage06_status, stage06_error = load_stage(
    "Stage06", BASE_DIR / "Stage06_AgenticAI" / "05_integration_engineer.py", "agent_integration_engine"
)

# Cross-stage fusion. Degrades gracefully: it uses whichever stages loaded.
try:
    from fusion.decision_engine import DecisionEngine

    decision_engine = DecisionEngine(
        ml_engine=stage01_engine, dl_engine=stage02_engine, nlp_engine=stage03_engine
    )
    fusion_status = "Online" if any(
        (stage01_engine, stage02_engine, stage03_engine)
    ) else "Offline"
except Exception as exc:
    logger.exception("Failed to load the fusion decision engine")
    decision_engine = None
    fusion_status = "Offline"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = BASE_DIR / "Stage02_DL" / "data" / "raw"
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

if stage05_engine is not None:
    try:
        stage05_module = load_module("stage05_api", BASE_DIR / "Stage05_GenAI" / "05_integration_engineer.py")
        app.register_blueprint(stage05_module.create_blueprint(stage05_engine))
        logger.info("Registered Stage 05 blueprint under /genai")
    except Exception as exc:
        logger.exception("Failed to register Stage 05 blueprint")

if stage06_engine is not None:
    try:
        # Reuse the Stage 01-03 adapters already loaded above instead of loading the models twice.
        stage06_engine.attach_engines(ml=stage01_engine, dl=stage02_engine, nlp=stage03_engine)
        stage06_module = load_module("stage06_api", BASE_DIR / "Stage06_AgenticAI" / "05_integration_engineer.py")
        app.register_blueprint(stage06_module.create_blueprint(stage06_engine))
        logger.info("Registered Stage 06 blueprint under /agent")
    except Exception as exc:
        logger.exception("Failed to register Stage 06 blueprint")


def fail(message, status_code=400, exc=None):
    """Return a safe client-facing error and log the detail server-side.

    Handlers used to return str(exc) straight to the browser, leaking absolute
    filesystem paths, internal class names and library internals to any caller.
    """
    if exc is not None:
        logger.exception("Request failed: %s", message)
    return jsonify({"error": message}), status_code

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disaster Response Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-main: #F3F6FB;
            --bg-sidebar: #FFFFFF;
            --bg-card: #FFFFFF;
            --border-color: #E2E8F0;
            --text-main: #0F172A;
            --text-muted: #5B6B82;
            --accent-blue: #2563EB;
            --accent-purple: #7C3AED;
            --accent-green: #16A34A;
            --accent-amber: #D97706;
            --accent-red: #DC2626;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { background: radial-gradient(circle at 85% -10%, rgba(239, 68, 68, 0.08), transparent 45%), radial-gradient(circle at 0% 100%, rgba(37, 99, 235, 0.07), transparent 40%), var(--bg-main); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

        .sidebar { width: 270px; flex-shrink: 0; background-color: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 20px 14px; overflow-y: auto; }
        .sidebar-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; padding: 4px 10px 18px; border-bottom: 1px solid var(--border-color); }
        .brand-mark { width: 42px; height: 42px; flex-shrink: 0; border-radius: 10px; background: linear-gradient(135deg, #DC2626, #F97316); display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 0 18px rgba(239, 68, 68, 0.35); }
        .sidebar-brand h1 { font-size: 16px; font-weight: 700; line-height: 1.25; }
        .sidebar-brand p { font-size: 11px; color: var(--text-muted); margin-top: 3px; letter-spacing: 0.3px; }

        .nav-section { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #94A3B8; margin: 18px 12px 6px; }
        .nav-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; color: var(--text-muted); text-decoration: none; border-radius: 8px; border-left: 3px solid transparent; margin-bottom: 2px; transition: all 0.2s; font-weight: 500; font-size: 13.5px; cursor: pointer; }
        .nav-item:hover { background-color: #F1F5F9; color: var(--text-main); }
        .nav-item.active { background-color: #FEF2F2; color: #DC2626; font-weight: 600; border-left-color: var(--accent-red); }
        .nav-icon { width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; }
        .nav-tag { margin-left: auto; font-size: 10px; font-weight: 600; padding: 2px 6px; border-radius: 4px; background: #F1F5F9; color: #64748B; border: 1px solid var(--border-color); }
        .sidebar-footer { margin-top: auto; padding: 14px 12px 4px; border-top: 1px solid var(--border-color); font-size: 11px; color: var(--text-muted); line-height: 1.8; }
        .sidebar-footer strong { color: #DC2626; }

        .main-content { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
        .hazard-stripe { height: 4px; flex-shrink: 0; background: repeating-linear-gradient(135deg, #D97706 0 12px, #111827 12px 24px); }
        .header { min-height: 64px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 10px 32px; border-bottom: 1px solid var(--border-color); flex-shrink: 0; background: rgba(255, 255, 255, 0.92); box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
        .eoc-title { display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 600; }
        .eoc-title span.sub { color: var(--text-muted); font-weight: 500; }
        .live-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-red); animation: pulse 1.8s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
        .header-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
        .pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid var(--border-color); background: #FFFFFF; color: var(--text-main); }
        .pill-ok { border-color: #86EFAC; background: #F0FDF4; color: #15803D; }
        .pill-warn { border-color: #FCD34D; background: #FFFBEB; color: #B45309; }

        .view-section { padding: 32px; display: none; }
        .view-section.active { display: block; }

        .page-header { margin-bottom: 24px; }
        .page-header h2 { font-size: 24px; margin-bottom: 4px; }
        .page-header p { color: var(--text-muted); font-size: 14px; line-height: 1.6; }

        .chart-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .chart-header h3 { font-size: 16px; font-weight: 600; }

        .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }
        .input-group { display: flex; flex-direction: column; }
        .input-group label { font-size: 12px; color: var(--text-muted); margin-bottom: 5px; }
        .input-group input { background: #FFFFFF; border: 1px solid #CBD5E1; color: #0F172A; padding: 10px; border-radius: 6px; outline: none; }
        .input-group input:focus, textarea:focus { border-color: var(--accent-amber); }
        .btn { background: #2563EB; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; transition: background 0.2s; }
        .btn:hover { background: #1D4ED8; }
        .btn-green { background: #16A34A; }
        .btn-green:hover { background: #15803D; }
        .btn-alert { background: linear-gradient(135deg, #DC2626, #F97316); padding: 12px 22px; font-size: 15px; box-shadow: 0 6px 20px rgba(239, 68, 68, 0.3); }
        .btn-alert:hover { filter: brightness(1.08); }

        .hero { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; padding: 24px; margin-bottom: 24px; border-radius: 14px; border: 1px solid #FECACA; background: linear-gradient(120deg, #FEE2E2, #FFF7ED 55%, #EFF6FF); box-shadow: 0 2px 10px rgba(220, 38, 38, 0.08); }
        .hero h2 { font-size: 26px; margin-bottom: 6px; }
        .hero p { color: var(--text-muted); font-size: 14px; max-width: 660px; line-height: 1.6; }
        .section-title { font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--text-muted); margin: 8px 0 12px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 24px; }
        .status-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; display: flex; gap: 12px; align-items: flex-start; cursor: pointer; transition: all 0.2s; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }
        .status-card:hover { border-color: #F87171; box-shadow: 0 4px 14px rgba(220, 38, 38, 0.10); }
        .status-card .icon { font-size: 22px; width: 40px; height: 40px; flex-shrink: 0; border-radius: 10px; background: #FEF2F2; display: flex; align-items: center; justify-content: center; }
        .status-card h4 { font-size: 14px; margin-bottom: 2px; }
        .status-card p { font-size: 12px; color: var(--text-muted); line-height: 1.5; }
        .status { display: inline-block; margin-top: 8px; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
        .status-Online { color: #15803D; background: #DCFCE7; }
        .status-Degraded { color: #B45309; background: #FEF3C7; }
        .status-Offline { color: #B91C1C; background: #FEE2E2; }
        .pipeline { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 24px; }
        .pipe-step { background: var(--bg-card); border: 1px solid var(--border-color); border-top: 4px solid var(--c); border-radius: 10px; padding: 14px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06); }
        .pipe-step .n { font-size: 11px; font-weight: 700; letter-spacing: 1px; color: var(--c); }
        .pipe-step h4 { font-size: 14px; margin: 4px 0; }
        .pipe-step p { font-size: 12px; color: var(--text-muted); line-height: 1.5; }
        .protocol { border-left: 3px solid var(--accent-amber); background: #FFFBEB; padding: 14px 16px; border-radius: 0 10px 10px 0; font-size: 13px; color: var(--text-muted); line-height: 1.7; }
        .protocol strong { color: var(--text-main); }

        .result-box { background: #F8FAFC; border: 1px solid var(--border-color); padding: 15px; border-radius: 8px; margin-top: 15px; font-family: monospace; }
        .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 12px; }
        .badge-Severe { background: #FEE2E2; color: #B91C1C; border: 1px solid #F87171; }
        .badge-Moderate { background: #FEF3C7; color: #B45309; border: 1px solid #FBBF24; }
        .badge-Low { background: #DCFCE7; color: #15803D; border: 1px solid #4ADE80; }

        .grid-half { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        textarea { width: 100%; min-height: 120px; resize: vertical; border-radius: 8px; border: 1px solid #CBD5E1; background: #FFFFFF; outline: none; color: #0F172A; padding: 12px; }
        .nlp-output { margin-top: 15px; background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-brand">
            <div class="brand-mark">🚨</div>
            <div><h1>Disaster Response Command</h1><p>Flood Emergency Coordination</p></div>
        </div>

        <div class="nav-section">Command</div>
        <a class="nav-item active" data-target="view-dashboard"><div class="nav-icon">🛰️</div> Command Overview</a>
        <a class="nav-item" data-target="view-flow"><div class="nav-icon">🚨</div> Incident Response Flow</a>
        <a class="nav-item" data-target="view-fusion"><div class="nav-icon">🎯</div> Threat Assessment</a>

        <div class="nav-section">Situational Intelligence</div>
        <a class="nav-item" data-target="view-ml"><div class="nav-icon">📡</div> Sensor Risk <span class="nav-tag">ML</span></a>
        <a class="nav-item" data-target="view-dl"><div class="nav-icon">🌊</div> Imagery &amp; Forecast <span class="nav-tag">DL</span></a>
        <a class="nav-item" data-target="view-nlp"><div class="nav-icon">📞</div> Field Reports <span class="nav-tag">NLP</span></a>
        <a class="nav-item" data-target="view-slm"><div class="nav-icon">📋</div> Tactical Briefing <span class="nav-tag">SLM</span></a>

        <div class="nav-section">Preparedness &amp; Dispatch</div>
        <a class="nav-item" data-target="view-genai"><div class="nav-icon">🧪</div> Scenario Drills <span class="nav-tag">GenAI</span></a>
        <a class="nav-item" data-target="view-agent"><div class="nav-icon">🚑</div> Resource Dispatch <span class="nav-tag">Agent</span></a>

        <div class="sidebar-footer">
            Emergency: <strong>112</strong> (ERSS)<br>
            NDMA helpline: <strong>1078</strong><br>
            Decision support only · human sign-off required
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <div class="hazard-stripe"></div>
        {% set online = [stage01, stage02, stage03, stage04, stage05, stage06] | select('equalto', 'Online') | list | length %}
        <div class="header">
            <div class="eoc-title"><span class="live-dot"></span> Emergency Operations Center <span class="sub">· Flood Response</span></div>
            <div class="header-actions">
                <span class="pill {{ 'pill-ok' if online == 6 else 'pill-warn' }}">● {{ online }}/6 models ready</span>
                <span class="pill" id="eoc-clock"></span>
                <span class="pill">🎖 Duty Officer</span>
            </div>
            <script>
                (function () {
                    const clock = document.getElementById('eoc-clock');
                    const tick = () => { clock.innerText = '🕒 ' + new Date().toLocaleString('en-IN', {day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false}); };
                    tick();
                    setInterval(tick, 1000);
                })();
            </script>
        </div>

        <!-- VIEW: COMMAND OVERVIEW -->
        <div id="view-dashboard" class="view-section active">
            <div class="hero">
                <div>
                    <h2>Command Overview</h2>
                    <p>Welcome to the Disaster Response AI Interface. Detect rising flood risk, read incoming field reports, brief commanders and dispatch rescue resources from one operations console.</p>
                </div>
                <button class="btn btn-alert" onclick="document.querySelector('.nav-item[data-target=view-flow]').click()">🚨 Launch Incident Response</button>
            </div>

            <div class="section-title">Model Readiness</div>
            <div class="status-grid">
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-ml]').click()"><div class="icon">📡</div><div><h4>Sensor Risk</h4><p>Rainfall, river gauge and call-volume severity scoring (ML)</p><span class="status status-{{ stage01 }}">{{ stage01 }}</span></div></div>
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-dl]').click()"><div class="icon">🌊</div><div><h4>Imagery &amp; River Forecast</h4><p>Drone flood detection and 6-hour water-level forecast (DL)</p><span class="status status-{{ stage02 }}">{{ stage02 }}</span></div></div>
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-nlp]').click()"><div class="icon">📞</div><div><h4>Field Report Analysis</h4><p>Urgency, hazard, location and headcount from messages (NLP)</p><span class="status status-{{ stage03 }}">{{ stage03 }}</span></div></div>
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-slm]').click()"><div class="icon">📋</div><div><h4>Tactical Briefing</h4><p>Situation, risk and action briefings from incident logs (SLM)</p><span class="status status-{{ stage04 }}">{{ stage04 }}</span></div></div>
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-genai]').click()"><div class="icon">🧪</div><div><h4>Scenario Drills</h4><p>Synthetic disaster scenarios that stress-test the pipeline (GenAI)</p><span class="status status-{{ stage05 }}">{{ stage05 }}</span></div></div>
                <div class="status-card" onclick="document.querySelector('.nav-item[data-target=view-agent]').click()"><div class="icon">🚑</div><div><h4>Resource Dispatch</h4><p>Zone triage, resource allocation and approval workflow (Agent)</p><span class="status status-{{ stage06 }}">{{ stage06 }}</span></div></div>
            </div>

            <div class="section-title">Response Pipeline</div>
            <div class="pipeline">
                <div class="pipe-step" style="--c: #0284C7"><div class="n">01 · DETECT</div><h4>Sensors &amp; imagery</h4><p>Score gauge telemetry, confirm flooding from drone images, forecast the river.</p></div>
                <div class="pipe-step" style="--c: #7C3AED"><div class="n">02 · UNDERSTAND</div><h4>Field reports</h4><p>Extract urgency, trapped headcount, location and resources needed.</p></div>
                <div class="pipe-step" style="--c: #D97706"><div class="n">03 · DECIDE</div><h4>Threat assessment</h4><p>Fuse every signal into one priority, flag conflicts for review.</p></div>
                <div class="pipe-step" style="--c: #F97316"><div class="n">04 · BRIEF</div><h4>Tactical briefing</h4><p>Condense the incident log into situation, risk and actions.</p></div>
                <div class="pipe-step" style="--c: #EC4899"><div class="n">05 · REHEARSE</div><h4>Scenario drills</h4><p>Test decisions against generated multi-zone emergencies.</p></div>
                <div class="pipe-step" style="--c: #DC2626"><div class="n">06 · DISPATCH</div><h4>Resource dispatch</h4><p>Rank zones, allocate boats, ambulances and shelter, await approval.</p></div>
            </div>

            <div class="protocol"><strong>Operating protocol:</strong> every output here is decision support for trained responders. URGENT and CRITICAL dispatches, evacuations and low-confidence calls are held for human sign-off before any resource is committed.</div>
        </div>

        <!-- VIEW: UNIFIED DECISION (FUSION) -->
        <div id="view-fusion" class="view-section">
            <div class="page-header">
                <h2>Threat Assessment</h2>
                <p>Combines sensor risk (ML), visual confirmation and forecast (DL), and the text report (NLP) into one prioritised decision.</p>
            </div>

            <div class="chart-card">
                <div class="chart-header"><h3>Incident Evidence</h3></div>
                <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 15px;">
                    Supply any subset. Missing evidence is reported, never assumed benign.
                </p>
                <form id="fusionForm">
                    <div class="form-grid">
                        <div class="input-group"><label>Rainfall (mm)</label><input type="number" step="0.1" id="fu-rainfall" value="118.0"></div>
                        <div class="input-group"><label>River Level (m)</label><input type="number" step="0.1" id="fu-river" value="5.4"></div>
                        <div class="input-group"><label>River Threshold (m)</label><input type="number" step="0.1" id="fu-thresh" value="5.0"></div>
                        <div class="input-group"><label>Emergency Calls</label><input type="number" id="fu-calls" value="140"></div>
                        <div class="input-group"><label>District</label><input type="text" id="fu-district" value="Pune"></div>
                        <div class="input-group"><label>State</label><input type="text" id="fu-state" value="Maharashtra"></div>
                    </div>
                    <div class="input-group" style="margin-bottom: 15px;">
                        <label>Field report / emergency message (optional)</label>
                        <textarea id="fu-text" style="min-height: 80px;">Water is rising fast near the railway bridge, three people are trapped and need immediate rescue.</textarea>
                    </div>
                    <label style="font-size: 12px; color: var(--text-muted);">
                        <input type="checkbox" id="fu-forecast" checked> Include 6-hour water-level forecast
                    </label>
                    <div style="margin-top: 15px;">
                        <button type="submit" class="btn btn-green">Run Unified Assessment</button>
                    </div>
                </form>
            </div>

            <div class="chart-card" id="fusion-result" style="display: none;">
                <div class="chart-header">
                    <h3>Decision</h3>
                    <span id="fu-badge" class="badge"></span>
                </div>
                <div style="line-height: 1.9;">
                    <div><span style="color: var(--text-muted);">Fused score (0-3):</span> <strong id="fu-score"></strong></div>
                    <div><span style="color: var(--text-muted);">Source agreement:</span> <strong id="fu-agree"></strong></div>
                    <div><span style="color: var(--text-muted);">Human review required:</span> <strong id="fu-review"></strong></div>
                </div>

                <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px;">Evidence</h4>
                <div id="fu-evidence" style="font-size: 13px;"></div>

                <div id="fu-conflict-wrap" style="display:none;">
                    <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px; color: #DC2626;">Conflicts</h4>
                    <div id="fu-conflicts" style="font-size: 13px;"></div>
                </div>

                <div id="fu-escalation-wrap" style="display:none;">
                    <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px; color: var(--accent-amber);">Escalations applied</h4>
                    <div id="fu-escalations" style="font-size: 13px;"></div>
                </div>

                <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px;">Recommended actions</h4>
                <div id="fu-actions" style="font-size: 13px;"></div>

                <div id="fu-warn-wrap" style="display:none;">
                    <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px; color: var(--accent-amber);">Caveats</h4>
                    <div id="fu-warnings" style="font-size: 13px;"></div>
                </div>

                <p id="fu-disclaimer" style="margin-top: 20px; padding: 12px; border-left: 3px solid var(--accent-amber); background: rgba(245,158,11,0.08); font-size: 12px; color: var(--text-muted); line-height: 1.6;"></p>
            </div>
        </div>

        <!-- VIEW: END-TO-END FLOW (all stages chained on one incident) -->
        <div id="view-flow" class="view-section">
            <style>
                .flow-strip { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 20px; }
                .flow-pill { padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #CBD5E1; background: #FFFFFF; color: var(--text-muted); }
                .flow-arrow { color: #94A3B8; }
                .flow-pill[data-state="running"], .flow-step[data-state="running"] { border-color: var(--accent-blue); color: var(--accent-blue); }
                .flow-pill[data-state="done"], .flow-step[data-state="done"] { border-color: var(--accent-green); color: var(--accent-green); }
                .flow-pill[data-state="error"], .flow-step[data-state="error"] { border-color: #DC2626; color: #DC2626; }
                .flow-pill[data-state="skipped"], .flow-step[data-state="skipped"] { border-color: #CBD5E1; color: #94A3B8; }
                .flow-step { border-left: 4px solid #CBD5E1; }
                .flow-step .chart-header h3 { color: var(--text-main); }
                .flow-status { font-size: 12px; font-weight: 600; text-transform: uppercase; }
                .flow-body { font-size: 13px; line-height: 1.8; color: var(--text-main); }
                .flow-body .muted { color: var(--text-muted); }
                .flow-body table { width: 100%; border-collapse: collapse; margin-top: 8px; }
                .flow-body th, .flow-body td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border-color); vertical-align: top; }
                .flow-body th { color: var(--text-muted); font-weight: 500; }
                .flow-section-title { font-size: 12px; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin: 18px 0 8px; }
                .input-group select { background: #FFFFFF; border: 1px solid #CBD5E1; color: #0F172A; padding: 10px; border-radius: 6px; outline: none; }
                .btn-sm { padding: 4px 10px; font-size: 12px; margin-left: 6px; }
                .btn-red { background: #DC2626; }
                .btn-red:hover { background: #DC2626; }
            </style>

            <div class="page-header">
                <h2>Incident Response Flow</h2>
                <p>One incident travels through every model in order: sensor risk → imagery &amp; forecast → report analysis → fused decision → tactical briefing → generated stress scenario → agent coordination with human approval.</p>
            </div>

            <div class="flow-strip" id="flow-strip">
                <span class="flow-pill" data-step="1">1 · ML Sensor Risk</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="2">2 · DL Imagery &amp; Forecast</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="3">3 · NLP Report</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="fusion">Fused Decision</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="4">4 · SLM Briefing</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="5">5 · GenAI Scenario</span><span class="flow-arrow">→</span>
                <span class="flow-pill" data-step="6">6 · Agent Coordination</span>
            </div>

            <div class="chart-card">
                <div class="chart-header"><h3>Incident Intake</h3></div>
                <form id="flowForm">
                    <div class="form-grid">
                        <div class="input-group"><label>State</label><input type="text" id="fl-state" value="Maharashtra"></div>
                        <div class="input-group"><label>District</label><input type="text" id="fl-district" value="Pune"></div>
                        <div class="input-group"><label>Population Affected</label><input type="number" id="fl-pop" value="2500"></div>
                        <div class="input-group"><label>Rainfall (mm)</label><input type="number" step="0.1" id="fl-rainfall" value="118.0"></div>
                        <div class="input-group"><label>River Level (m)</label><input type="number" step="0.1" id="fl-river" value="5.4"></div>
                        <div class="input-group"><label>River Threshold (m)</label><input type="number" step="0.1" id="fl-thresh" value="5.0"></div>
                        <div class="input-group"><label>Water Level Change (m)</label><input type="number" step="0.1" id="fl-change" value="0.4"></div>
                        <div class="input-group"><label>Emergency Calls</label><input type="number" id="fl-calls" value="140"></div>
                        <div class="input-group"><label>Road Closures</label><input type="number" id="fl-roads" value="1"></div>
                    </div>
                    <div class="input-group" style="margin-bottom: 15px;">
                        <label>Field report / emergency message</label>
                        <textarea id="fl-text" style="min-height: 80px;">Water is rising fast near the railway bridge, three people are trapped and need immediate rescue. Ambulance needed for two injured residents.</textarea>
                    </div>
                    <div class="form-grid">
                        <div class="input-group"><label>Camera / drone image (optional)</label><input type="file" id="fl-image" accept="image/jpeg, image/png"></div>
                        <div class="input-group"><label>Stress-test scenario (GenAI)</label><select id="fl-prompt"><option value="">None: only this incident</option></select></div>
                        <div class="input-group"><label>Duty officer (for approvals)</label><input type="text" id="fl-operator" value="Duty officer"></div>
                        <div class="input-group"><label>Rescue boats available</label><input type="number" id="fl-boats" value="3"></div>
                        <div class="input-group"><label>Ambulances available</label><input type="number" id="fl-amb" value="4"></div>
                        <div class="input-group"><label>Shelter beds available</label><input type="number" id="fl-beds" value="200"></div>
                    </div>
                    <label style="font-size: 12px; color: var(--text-muted);">
                        <input type="checkbox" id="fl-forecast" checked> Include 72 h river history for the 6-hour forecast
                    </label>
                    <div style="margin-top: 15px;">
                        <button type="submit" class="btn btn-green" id="fl-run">Run Complete Flow</button>
                    </div>
                </form>
            </div>

            <div id="flow-steps" style="display: none;">
                <div class="chart-card flow-step" id="flow-step-1"><div class="chart-header"><h3>1 · ML Sensor Risk</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
                <div class="chart-card flow-step" id="flow-step-2"><div class="chart-header"><h3>2 · DL Imagery &amp; Water-Level Forecast</h3><span class="flow-status"></span></div><div class="flow-body"></div>
                    <div style="height: 180px; width: 100%; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; margin-top: 10px;"><canvas id="flowLstmChart"></canvas></div></div>
                <div class="chart-card flow-step" id="flow-step-3"><div class="chart-header"><h3>3 · NLP Report Analysis</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
                <div class="chart-card flow-step" id="flow-step-fusion"><div class="chart-header"><h3>Fused Decision (ML + DL + NLP)</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
                <div class="chart-card flow-step" id="flow-step-4"><div class="chart-header"><h3>4 · SLM Tactical Briefing</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
                <div class="chart-card flow-step" id="flow-step-5"><div class="chart-header"><h3>5 · GenAI Scenario &amp; Stress Test</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
                <div class="chart-card flow-step" id="flow-step-6"><div class="chart-header"><h3>6 · Agent Coordination &amp; Approval</h3><span class="flow-status"></span></div><div class="flow-body"></div></div>
            </div>

            <script>
            (function () {
                const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
                const pct = v => (v == null || isNaN(v)) ? 'N/A' : (v * 100).toFixed(1) + '%';
                const COLORS = { ROUTINE: '#059669', ELEVATED: '#D97706', URGENT: '#F97316', CRITICAL: '#DC2626', IMMEDIATE: '#DC2626',
                                 Low: '#059669', Moderate: '#D97706', Severe: '#DC2626' };
                const tag = v => v ? `<strong style="color:${COLORS[v] || '#0F172A'}">${esc(v)}</strong>` : '<span class="muted">N/A</span>';
                const STATUS_TEXT = { running: 'Running…', done: 'Done', error: 'Failed', skipped: 'Skipped', pending: 'Waiting' };
                let flowChart = null;
                let agentRun = null;

                async function request(url, options, allowStatus) {
                    const res = await fetch(url, options);
                    let data = null;
                    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
                    if ((!res.ok && res.status !== allowStatus) || !data || data.error) {
                        throw new Error((data && data.error) || `HTTP ${res.status}`);
                    }
                    return data;
                }
                const post = (url, body, allowStatus) => request(url, {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
                }, allowStatus);

                function setStep(step, state, html) {
                    const card = document.getElementById('flow-step-' + step);
                    card.dataset.state = state;
                    card.querySelector('.flow-status').innerText = STATUS_TEXT[state] || '';
                    if (html !== undefined) card.querySelector('.flow-body').innerHTML = html;
                    const pill = document.querySelector(`#flow-strip .flow-pill[data-step="${step}"]`);
                    if (pill) pill.dataset.state = state;
                }

                // Each step reports its own failure and the flow continues with whatever evidence remains.
                async function runStep(step, fn) {
                    setStep(step, 'running', '<span class="muted">Working…</span>');
                    try {
                        return await fn();
                    } catch (err) {
                        setStep(step, 'error', `<span style="color:#DC2626">${esc(err.message)}</span><br><span class="muted">Later steps continue without this output.</span>`);
                        return null;
                    }
                }

                function stamp(date) {
                    const p = n => String(n).padStart(2, '0');
                    return `${p(date.getDate())}-${p(date.getMonth() + 1)}-${date.getFullYear()} ${p(date.getHours())}:${p(date.getMinutes())}`;
                }

                const num = id => parseFloat(document.getElementById(id).value);
                const int = id => parseInt(document.getElementById(id).value, 10) || 0;
                const val = id => document.getElementById(id).value.trim();

                // Populate the GenAI scenario library.
                request('/api/genai/prompts').then(prompts => {
                    const select = document.getElementById('fl-prompt');
                    prompts.forEach(p => select.add(new Option(`${p.id} · ${p.name} (${p.zones} zones)`, p.id)));
                    const baseline = prompts.find(p => p.id === 'S01');
                    if (baseline) select.value = baseline.id;
                }).catch(() => { /* GenAI offline: the flow runs with the operator's incident only */ });

                function renderAgent(run) {
                    agentRun = run;
                    const zones = run.zones || [];
                    const zoneRows = zones.map(z => {
                        const alloc = Object.entries(z.allocated || {}).filter(([, q]) => q).map(([r, q]) => `${q} ${r.replace('_', ' ')}`).join(', ');
                        return `<tr><td>${esc(z.label)}<br><span class="muted">${esc(z.zone_id)}</span></td>
                                <td>${tag(z.priority)}<br><span class="muted">${esc(z.status)}</span></td>
                                <td>${pct(z.decision_confidence)}</td>
                                <td>${esc(alloc) || '<span class="muted">none</span>'}</td>
                                <td>${z.human_review_required ? '<span style="color:#D97706">Yes</span>' : 'No'}</td></tr>`;
                    }).join('');
                    const actions = (run.actions || []).map(a => {
                        const res = Object.entries(a.resources || {}).filter(([, q]) => q).map(([r, q]) => `${q} ${r.replace('_', ' ')}`).join(', ');
                        const buttons = (a.status === 'pending_approval' && !run.halted)
                            ? `<button class="btn btn-green btn-sm" data-action="${esc(a.action_id)}" data-decision="approve">Approve</button>
                               <button class="btn btn-red btn-sm" data-action="${esc(a.action_id)}" data-decision="reject">Reject</button>` : '';
                        const reasons = (a.audit && a.audit.reasons || []).length ? `<br><span class="muted">Audit: ${esc(a.audit.reasons.join('; '))}</span>` : '';
                        return `<tr><td>${esc(a.action_id)}<br><span class="muted">${esc(a.zone_id)}</span></td>
                                <td>${esc(a.type)}</td>
                                <td>${esc(a.summary)}${res ? `<br><span class="muted">Resources: ${esc(res)}</span>` : ''}${reasons}</td>
                                <td>${esc(a.status)}${buttons}</td></tr>`;
                    }).join('');
                    const shortfalls = (run.shortfalls || []).length
                        ? `<div class="flow-section-title" style="color:#DC2626">Shortfalls</div>${run.shortfalls.map(s => `<div>⚠ ${typeof s === 'string' ? esc(s) : `<strong>${esc(s.zone_id)}</strong>: ${esc(String(s.resource_type || '').replace('_', ' '))} short by ${esc(s.shortfall)} <span class="muted">(needs ${esc(s.need)}, granted ${esc(s.granted)})</span>`}</div>`).join('')}` : '';
                    setStep(6, 'done', `
                        <div><span class="muted">Run:</span> <strong>${esc(run.run_id)}</strong> · <span class="muted">Triage order:</span> ${esc((run.ranking || []).join(' → '))}
                            · <span class="muted">Pending approvals:</span> <strong style="color:${run.pending_approvals ? '#D97706' : '#059669'}">${run.pending_approvals}</strong>
                            · <span class="muted">Latency:</span> ${esc(run.latency_ms)} ms</div>
                        <div class="flow-section-title">Zone decisions</div>
                        <div style="overflow-x:auto"><table><tr><th>Zone</th><th>Priority</th><th>Confidence</th><th>Allocated</th><th>Human review</th></tr>${zoneRows}</table></div>
                        ${shortfalls}
                        <div class="flow-section-title">Proposed actions</div>
                        <div style="overflow-x:auto"><table><tr><th>Action</th><th>Type</th><th>Summary</th><th>Status</th></tr>${actions || '<tr><td colspan="4" class="muted">No actions proposed</td></tr>'}</table></div>
                        <p style="margin-top:14px; padding:10px; border-left:3px solid var(--accent-amber); background:rgba(245,158,11,0.08); font-size:12px; color:var(--text-muted);">${esc(run.disclaimer)}</p>`);
                }

                document.getElementById('flow-step-6').addEventListener('click', async (e) => {
                    const btn = e.target.closest('button[data-action]');
                    if (!btn || !agentRun) return;
                    btn.disabled = true;
                    try {
                        const result = await post('/api/agent/approve', {
                            run_id: agentRun.run_id, action_id: btn.dataset.action,
                            decision: btn.dataset.decision, operator: val('fl-operator') || 'Duty officer'
                        });
                        renderAgent(result.run);
                    } catch (err) {
                        alert('Decision failed: ' + err.message);
                        btn.disabled = false;
                    }
                });

                document.getElementById('flowForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const runBtn = document.getElementById('fl-run');
                    runBtn.disabled = true;
                    runBtn.innerText = 'Running flow…';
                    document.getElementById('flow-steps').style.display = 'block';
                    ['1', '2', '3', 'fusion', '4', '5', '6'].forEach(s => setStep(s, 'pending', '<span class="muted">Waiting for earlier steps…</span>'));

                    const now = new Date();
                    const state = val('fl-state'), district = val('fl-district');
                    const text = val('fl-text');
                    const river = num('fl-river');
                    const sensors = {
                        timestamp: stamp(now), state, district,
                        rainfall_mm: num('fl-rainfall'), river_level_m: river,
                        river_level_threshold_m: num('fl-thresh'), emergency_calls: int('fl-calls'),
                        water_level_change_m: num('fl-change'), road_closures: int('fl-roads'),
                        bridge_closures: 0, flood_history_count: 5, population_affected: int('fl-pop')
                    };
                    let waterLevels = null;
                    if (document.getElementById('fl-forecast').checked) {
                        // 72 h of history ending at the entered river level, within the trained range.
                        const start = Math.max(2.4, river - 2.5);
                        waterLevels = Array.from({length: 72}, (_, i) => +(start + (river - start) * (i / 71)).toFixed(3));
                    }

                    // ---- 1. ML sensor risk ----
                    const ml = await runStep(1, async () => {
                        const d = await post('/api/predict/ml', sensors);
                        setStep(1, 'done', `<div><span class="muted">Risk category:</span> ${tag(d.risk_category)} <span class="muted">(model score ${pct(d.confidence)})</span></div>
                            <div><span class="muted">Top drivers for this input:</span> ${esc((d.top_factors || []).join(', ')) || 'N/A'}</div>
                            <div class="muted">↓ Risk category feeds the fused decision and the incident log for the briefing.</div>`);
                        return d;
                    });

                    // ---- 2. DL imagery + forecast ----
                    const dl = await runStep(2, async () => {
                        const out = {};
                        const lines = [];
                        const file = document.getElementById('fl-image').files[0];
                        if (file) {
                            const form = new FormData();
                            form.append('file', file);
                            try {
                                out.image = await request('/api/predict/dl/image', {method: 'POST', body: form});
                                lines.push(`<div><span class="muted">Imagery (CNN):</span> <strong style="color:${out.image.label === 'flooded' ? '#DC2626' : '#059669'}">${esc(out.image.label)}</strong> <span class="muted">(${pct(out.image.confidence)})</span></div>`);
                            } catch (err) {
                                lines.push(`<div><span class="muted">Imagery (CNN):</span> <span style="color:#DC2626">${esc(err.message)}</span></div>`);
                            }
                        } else {
                            lines.push('<div><span class="muted">Imagery (CNN):</span> no image supplied</div>');
                        }
                        if (waterLevels) {
                            out.forecast = await post('/api/predict/dl/lstm', {sequence: waterLevels});
                            const f = out.forecast.forecast_water_levels;
                            lines.push(`<div><span class="muted">+6 h river forecast (LSTM):</span> <strong style="color:var(--accent-amber)">${f[f.length - 1].toFixed(2)} m</strong>
                                ${out.forecast.expected_mae_at_horizon != null ? `<span class="muted">(±${out.forecast.expected_mae_at_horizon.toFixed(3)} m measured MAE)</span>` : ''}</div>`);
                            if (out.forecast.warning) lines.push(`<div style="color:#D97706">⚠ ${esc(out.forecast.warning)}</div>`);
                            if (flowChart) flowChart.destroy();
                            flowChart = new Chart(document.getElementById('flowLstmChart').getContext('2d'), {
                                type: 'line',
                                data: {
                                    labels: Array.from({length: 78}, (_, i) => i < 72 ? `T-${72 - i}` : `T+${i - 71}`),
                                    datasets: [
                                        { label: 'Historical', data: [...waterLevels, ...Array(6).fill(null)], borderColor: '#2563EB', pointRadius: 0, tension: 0.2 },
                                        { label: 'Forecast', data: [...Array(71).fill(null), waterLevels[71], ...f], borderColor: '#D97706', borderDash: [5, 5], pointRadius: 0, tension: 0.2 }
                                    ]
                                },
                                options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: '#E2E8F0' } }, x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } } } }
                            });
                        } else {
                            lines.push('<div><span class="muted">Forecast (LSTM):</span> no river history supplied</div>');
                        }
                        lines.push('<div class="muted">↓ Visual confirmation and forecast trend feed the fused decision.</div>');
                        setStep(2, (out.image || out.forecast) ? 'done' : 'skipped', lines.join(''));
                        return out;
                    });

                    // ---- 3. NLP report analysis ----
                    let nlp = null;
                    if (text) {
                        nlp = await runStep(3, async () => {
                            const d = await post('/api/predict/nlp', {text});
                            const list = v => Array.isArray(v) ? v.join(', ') : (v || 'N/A');
                            setStep(3, 'done', `<div><span class="muted">Urgency:</span> ${tag(d.urgency)} <span class="muted">(${pct(d.confidence)})</span> · <span class="muted">Hazard:</span> <strong>${esc(d.hazard_type)}</strong> <span class="muted">(${pct(d.hazard_confidence)})</span></div>
                                <div><span class="muted">Location:</span> ${esc(list(d.location))} · <span class="muted">Resources needed:</span> ${esc(list(d.resource_needed))} · <span class="muted">Headcount:</span> ${esc(d.headcount ?? 'N/A')}</div>
                                <div class="muted">↓ Urgency and entities feed the fused decision and the incident log.</div>`);
                            return d;
                        });
                    } else {
                        setStep(3, 'skipped', '<span class="muted">No field report supplied.</span>');
                    }

                    // ---- Fused decision ----
                    const fused = await runStep('fusion', async () => {
                        const body = {sensors, text: text || null};
                        if (waterLevels) body.water_levels = waterLevels;
                        const d = await post('/api/assess', body, 422);
                        const evidence = (d.evidence || []).map(ev => `<div>${ev.available ? '🟢' : '⚪'} <strong>${esc(ev.source)}</strong> → ${ev.available ? tag(ev.level_name) : '<span class="muted">not available</span>'} <span class="muted">${esc(ev.detail)}</span></div>`).join('');
                        const actions = (d.recommended_actions || []).map(a => `<div>▸ ${esc(a)}</div>`).join('');
                        const conflicts = (d.conflicts || []).map(c => `<div style="color:#DC2626">⚠ ${esc(c.description)}</div>`).join('');
                        setStep('fusion', 'done', `<div><span class="muted">Priority:</span> ${tag(d.priority || 'NO ASSESSMENT')} · <span class="muted">Score:</span> ${esc(d.score ?? 'N/A')} · <span class="muted">Agreement:</span> ${esc(d.agreement || 'N/A')} · <span class="muted">Human review:</span> ${d.human_review_required ? '<span style="color:#D97706">YES</span>' : 'No'}</div>
                            <div class="flow-section-title">Evidence</div>${evidence}${conflicts}
                            <div class="flow-section-title">Recommended actions</div>${actions || '<span class="muted">None</span>'}
                            <div class="muted" style="margin-top:8px">↓ Fused priority is written into the incident log for the briefing.</div>`);
                        return d;
                    });

                    // ---- 4. SLM tactical briefing (incident log built from steps 1-3) ----
                    const entries = [];
                    const at = mins => stamp(new Date(now.getTime() - mins * 60000));
                    if (text) entries.push(`${text} Rescue Emergency.`);
                    entries.push(`Rainfall ${sensors.rainfall_mm} mm, river at ${river} m against danger mark ${sensors.river_level_threshold_m} m, rising ${sensors.water_level_change_m} m. ${sensors.emergency_calls} emergency calls, ${sensors.population_affected} people affected.${ml ? ` Sensor risk ${ml.risk_category}.` : ''}`);
                    if (dl && dl.forecast) {
                        const f = dl.forecast.forecast_water_levels;
                        entries.push(`Water levels rising. River projected to reach ${f[f.length - 1].toFixed(2)} m within 6 hours.`);
                    }
                    if (dl && dl.image) entries.push(`Drone imagery shows area ${dl.image.label}.`);
                    if (nlp) entries.push(`${nlp.hazard_type || 'Incident'} report. ${nlp.urgency || ''} severity.${nlp.headcount ? ` ${nlp.headcount} people affected.` : ''}${(nlp.resource_needed || []).length ? ` Needs: ${[].concat(nlp.resource_needed).join(', ')}.` : ''}`);
                    if (fused && fused.priority) entries.push(`Unified assessment ${fused.priority}. ${(fused.recommended_actions || []).slice(0, 2).join(' ')}`);
                    const incidentLog = `INCIDENT LOG | ${state} / ${district} / ZONE-1 | ${entries.length} entries\\n`
                        + entries.map((entry, i) => `[${at((entries.length - 1 - i) * 30)}] ERSS-${String(900001 + i).padStart(6, '0')} | ${entry}`).join('\\n');

                    await runStep(4, async () => {
                        const d = await post('/api/slm/summarize', {report: incidentLog});
                        const actions = Array.isArray(d.actions) ? d.actions : [d.actions].filter(Boolean);
                        setStep(4, 'done', `<div><span class="muted">Priority:</span> ${tag(d.priority)} · <span class="muted">Model:</span> ${esc(d.model)} · <span class="muted">Latency:</span> ${esc(d.latency_ms)} ms</div>
                            <div class="flow-section-title">Situation</div><div>${esc(d.situation)}</div>
                            <div class="flow-section-title">Risk</div><div>${esc(d.risk)}</div>
                            <div class="flow-section-title">Actions</div><ol style="padding-left:20px">${actions.map(a => `<li>${esc(a)}</li>`).join('')}</ol>
                            <details style="margin-top:8px"><summary class="muted" style="cursor:pointer">Incident log generated from steps 1-3</summary><pre style="white-space:pre-wrap; font-size:12px; margin-top:6px">${esc(incidentLog)}</pre></details>
                            <div class="muted">↓ The same incident log is handed to the agent for this zone.</div>`);
                        return d;
                    });

                    // ---- 5. GenAI scenario + stress test ----
                    const promptId = document.getElementById('fl-prompt').value;
                    let simulatedZones = [];
                    if (promptId) {
                        await runStep(5, async () => {
                            const scenario = await post('/api/genai/scenario', {prompt_id: promptId});
                            setStep(5, 'running', `<span class="muted">Generated “${esc(scenario.name)}” (${scenario.zones.length} zones). Stress-testing the pipeline on it…</span>`);
                            simulatedZones = scenario.zones.slice(0, 9).map((z, i) => ({
                                zone_id: `SIM-${i + 1}`, label: `Simulated: ${z.label}`, state: z.state, district: z.district,
                                inputs: Object.fromEntries(['sensors', 'text', 'water_levels', 'incident_log']
                                    .filter(k => z.inputs && z.inputs[k] != null).map(k => [k, z.inputs[k]]))
                            }));
                            let testHtml;
                            try {
                                const t = await post('/api/genai/stress-test', {prompt_id: promptId});
                                const rows = (t.zones || []).map(r => `<tr><td>${esc(r.label)}</td><td>${tag(r.true_severity)}</td><td>${tag(r.priority)}</td>
                                    <td>${r.passed ? '<span style="color:#059669">Pass</span>' : `<span style="color:#DC2626">Fail</span><br><span class="muted">${esc(r.failures)}</span>`}</td></tr>`).join('');
                                testHtml = `<div><span class="muted">Stress test:</span> ${t.passed ? '<strong style="color:#059669">PASSED</strong>' : '<strong style="color:#DC2626">FAILED</strong>'}
                                    · <span class="muted">Ranking τ:</span> ${esc(t.ranking_tau ?? 'N/A')}</div>
                                    <div style="overflow-x:auto"><table><tr><th>Zone</th><th>True severity</th><th>System priority</th><th>Result</th></tr>${rows}</table></div>`;
                            } catch (err) {
                                testHtml = `<div style="color:#DC2626">Stress test failed: ${esc(err.message)}</div>`;
                            }
                            setStep(5, 'done', `<div><span class="muted">Scenario:</span> <strong>${esc(scenario.name)}</strong> <span class="muted">${esc(scenario.archetype || '')}</span></div>
                                <div class="muted">${esc(scenario.prompt || '')}</div>${testHtml}
                                <div class="muted" style="margin-top:8px">↓ ${simulatedZones.length} generated zones join your incident so the agent must triage competing demand.</div>`);
                        });
                    } else {
                        setStep(5, 'skipped', '<span class="muted">No stress-test scenario selected; the agent coordinates this incident alone.</span>');
                    }

                    // ---- 6. Agent coordination ----
                    await runStep(6, async () => {
                        const inputs = {sensors, incident_log: incidentLog};
                        if (text) inputs.text = text;
                        if (waterLevels) inputs.water_levels = waterLevels;
                        const incident = {
                            name: `${district}, ${state} incident`,
                            resources: {rescue_boats: int('fl-boats'), ambulances: int('fl-amb'), shelter_beds: int('fl-beds')},
                            zones: [{zone_id: 'ZONE-1', label: `${district} (reported incident)`, state, district, inputs}, ...simulatedZones]
                        };
                        renderAgent(await post('/api/agent/orchestrate', {incident}));
                    });

                    runBtn.disabled = false;
                    runBtn.innerText = 'Run Complete Flow';
                });
            })();
            </script>
        </div>

        <!-- VIEW: NLP (STAGE 03) -->
        <div id="view-nlp" class="view-section">
            <div class="page-header">
                <h2>Field Report Analysis</h2>
                <p>Classify emergency text, detect hazard type, and extract entities.</p>
            </div>

            <div class="chart-card">
                <div class="chart-header"><h3>Emergency Message Input</h3></div>
                <form id="nlpForm">
                    <div class="input-group" style="margin-bottom: 15px;">
                        <label>Emergency report / message</label>
                        <textarea id="nlp-text" placeholder="Example: Three people are trapped in a flooded building near the railway bridge and need immediate rescue."></textarea>
                    </div>
                    <button type="submit" class="btn btn-green">Analyze Message</button>
                </form>

                <div class="nlp-output" id="nlp-result" style="display: none;">
                    <strong>NLP Result</strong>
                    <div style="margin-top: 10px; line-height: 1.8;">
                        <div><span style="color: var(--text-muted);">Urgency:</span> <strong id="nlp-urgency"></strong></div>
                        <div><span style="color: var(--text-muted);">Hazard:</span> <strong id="nlp-hazard"></strong></div>
                        <div><span style="color: var(--text-muted);">Confidence:</span> <span id="nlp-confidence"></span></div>
                        <div><span style="color: var(--text-muted);">Location:</span> <span id="nlp-location"></span></div>
                        <div><span style="color: var(--text-muted);">Resource:</span> <span id="nlp-resource"></span></div>
                        <div><span style="color: var(--text-muted);">Headcount:</span> <span id="nlp-headcount"></span></div>
                        <small id="nlp-meta" style="color: var(--text-muted); display: block; margin-top: 10px;"></small>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW: SLM TACTICAL BRIEFING (STAGE 04) -->
        <div id="view-slm" class="view-section">
            <div class="page-header">
                <h2>Tactical Briefing</h2>
                <p>Paste a multi-entry incident log and generate a structured SITUATION / RISK / ACTIONS briefing using the trained Small Language Model.</p>
            </div>

            <div class="chart-card">
                <div class="chart-header"><h3>Incident Log Input</h3></div>
                <form id="slmForm">
                    <div class="input-group" style="margin-bottom: 15px;">
                        <label>Paste full incident log (multi-entry)</label>
                        <textarea id="slm-report" style="min-height: 180px;" placeholder="INCIDENT LOG | Maharashtra / Pune / ZONE-3 | 6 entries&#10;[01-09-2026 08:15] ERSS-004201 | Severe flooding reported near the main bridge. 45 people affected...&#10;..."></textarea>
                    </div>
                    <button type="submit" class="btn" style="background:#F97316;">Generate Tactical Briefing</button>
                </form>

                <div class="chart-card" id="slm-result" style="display: none; margin-top: 20px; border: 1px solid #F97316;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <h3 style="margin: 0;">Tactical Briefing</h3>
                        <span id="slm-priority-badge" class="badge"></span>
                    </div>

                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 12px; color: #F97316; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Situation</div>
                        <div id="slm-situation" style="line-height: 1.6;"></div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 12px; color: var(--accent-amber); text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Risk Assessment</div>
                        <div id="slm-risk" style="line-height: 1.6;"></div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 12px; color: var(--accent-green); text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Recommended Actions</div>
                        <ol id="slm-actions" style="padding-left: 20px; line-height: 1.8;"></ol>
                    </div>

                    <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); margin-bottom: 6px;">
                            <span>Time Savings vs Full Log</span>
                            <span id="slm-time-pct" style="color: var(--accent-green); font-weight: 700;"></span>
                        </div>
                        <div style="background: #E2E8F0; border-radius: 6px; height: 8px; overflow: hidden;">
                            <div id="slm-time-bar" style="background: linear-gradient(90deg, #059669, #D97706); height: 100%; width: 0%; transition: width 0.6s;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 8px;">
                            <span>Model: <span id="slm-model"></span></span>
                            <span>Latency: <span id="slm-latency"></span></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW: ML (STAGE 01) -->
        <div id="view-ml" class="view-section">
            <div class="page-header">
                <h2>Sensor Risk Assessment</h2>
                <p>Determine disaster severity via numerical sensor thresholds.</p>
            </div>
            
            <div class="chart-card">
                <div class="chart-header"><h3>Severity Prediction Form</h3></div>
                <form id="mlForm">
                    <div class="form-grid">
                        <div class="input-group"><label>Rainfall (mm)</label><input type="number" step="0.1" id="ml-rainfall" value="120.5"></div>
                        <div class="input-group"><label>River Level (m)</label><input type="number" step="0.1" id="ml-river" value="14.2"></div>
                        <div class="input-group"><label>River Threshold (m)</label><input type="number" step="0.1" id="ml-thresh" value="12.0"></div>
                        <div class="input-group"><label>Emergency Calls</label><input type="number" id="ml-calls" value="150"></div>
                        <div class="input-group"><label>Water Level Change (m)</label><input type="number" step="0.1" id="ml-change" value="0.8"></div>
                        <div class="input-group"><label>District</label><input type="text" id="ml-district" value="Pune"></div>
                    </div>
                    <button type="submit" class="btn btn-green">Run Inference</button>
                </form>

                <div class="result-box" id="ml-result" style="display: none;">
                    <strong>Prediction Result:</strong><br>
                    Risk Category: <span id="ml-badge" class="badge"></span><br>
                    Model confidence: <span id="ml-conf"></span>%<br>
                    <small id="ml-json" style="color: var(--text-muted); display: block; margin-top: 10px;"></small>
                    <small style="color: var(--text-muted); display: block; margin-top: 8px; line-height: 1.5;">
                        Confidence is the model's uncalibrated score, not a probability of being
                        correct (see <code>data/outputs/calibration_report.json</code>).
                        This is decision support for a human responder, not a verified assessment.
                    </small>
                </div>
            </div>
        </div>

        <!-- VIEW: DL (STAGE 02) -->
        <div id="view-dl" class="view-section">
            <div class="page-header">
                <h2>Flood Imagery &amp; River Forecast</h2>
                <p>CNN visual analysis and LSTM sequential forecasting.</p>
            </div>

            <div class="grid-half">
                <!-- CNN Form -->
                <div class="chart-card">
                    <div class="chart-header"><h3>Visual Recognition (CNN)</h3></div>
                    <form id="cnnForm">
                        <div class="input-group" style="margin-bottom: 15px;">
                            <label>Upload Camera/Drone Image</label>
                            <input type="file" id="cnn-image" accept="image/jpeg, image/png">
                        </div>
                        <button type="submit" class="btn">Analyze Image</button>
                    </form>
                    <div class="result-box" id="cnn-result" style="display: none;">
                        <strong>Analysis Result:</strong><br>
                        Status: <span id="cnn-status" style="color: #0F172A; font-weight: bold;"></span><br>
                        Probability: <span id="cnn-prob"></span>%<br>
                    </div>
                </div>

                <!-- LSTM Form -->
                <div class="chart-card">
                    <div class="chart-header"><h3>Sequential Forecast (LSTM)</h3></div>
                    <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 15px;">Simulate a 6-hour forecast using recent sensor trajectory.</p>
                    <button id="btn-lstm" class="btn btn-green" style="width: 100%; margin-bottom: 20px;">Generate Forecast Plot</button>
                    <div style="height: 200px; width: 100%; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <canvas id="lstmChart"></canvas>
                    </div>
                    <div class="result-box" id="lstm-result" style="display: none; margin-top: 10px;">
                        Target +6hr Forecast: <strong id="lstm-val" style="color: var(--accent-amber);"></strong> m
                        <small id="lstm-note" style="color: var(--text-muted); display: block; margin-top: 8px; line-height: 1.5;"></small>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW: GENAI STRESS TEST (STAGE 05) -->
        <div id="view-genai" class="view-section" style="padding: 0; height: calc(100vh - 72px);">
            <iframe src="/genai" style="width: 100%; height: 100%; border: none;"></iframe>
        </div>

        <!-- VIEW: AGENT COORDINATION (STAGE 06) -->
        <div id="view-agent" class="view-section" style="padding: 0; height: calc(100vh - 72px);">
            <iframe src="/agent" style="width: 100%; height: 100%; border: none;"></iframe>
        </div>

    </div>

    <!-- Interactivity Script -->
    <script>
        // Tab switching logic
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', event => {
                const targetId = item.getAttribute('data-target');
                if (!targetId) return;
                event.preventDefault();
                document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                document.querySelectorAll('.view-section').forEach(view => view.classList.remove('active'));
                
                const targetView = document.getElementById(targetId);
                if (targetView) targetView.classList.add('active');
            });
        });

        // Unified Assessment (fusion) Submission
        const PRIORITY_COLORS = { ROUTINE: '#059669', ELEVATED: '#D97706', URGENT: '#F97316', CRITICAL: '#DC2626' };

        function renderList(elementId, items, bullet) {
            document.getElementById(elementId).innerHTML = items.length
                ? items.map(t => `<div style="margin-bottom:6px;">${bullet} ${t}</div>`).join('')
                : '<div style="color: var(--text-muted);">None</div>';
        }

        document.getElementById('fusionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            btn.innerText = 'Assessing...';

            const river = parseFloat(document.getElementById('fu-river').value);
            const payload = {
                sensors: {
                    timestamp: "05-09-2026 12:00",
                    state: document.getElementById('fu-state').value,
                    district: document.getElementById('fu-district').value,
                    rainfall_mm: parseFloat(document.getElementById('fu-rainfall').value),
                    river_level_m: river,
                    river_level_threshold_m: parseFloat(document.getElementById('fu-thresh').value),
                    emergency_calls: parseInt(document.getElementById('fu-calls').value),
                    water_level_change_m: 0.4,
                    road_closures: 1, bridge_closures: 0,
                    flood_history_count: 5, population_affected: 50000
                },
                text: document.getElementById('fu-text').value
            };

            if (document.getElementById('fu-forecast').checked) {
                // 72 h of history ending at the entered river level, within the trained range.
                const start = Math.max(2.4, river - 2.5);
                payload.water_levels = Array.from({length: 72}, (_, i) =>
                    +(start + (river - start) * (i / 71)).toFixed(3));
            }

            try {
                const res = await fetch('/api/assess', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                document.getElementById('fusion-result').style.display = 'block';

                const badge = document.getElementById('fu-badge');
                badge.innerText = data.priority || 'NO ASSESSMENT';
                badge.style.color = PRIORITY_COLORS[data.priority] || '#94A3B8';
                badge.style.border = '1px solid ' + (PRIORITY_COLORS[data.priority] || '#94A3B8');
                badge.style.background = 'transparent';

                document.getElementById('fu-score').innerText = data.score ?? 'N/A';
                document.getElementById('fu-agree').innerText = data.agreement || 'N/A';
                const review = document.getElementById('fu-review');
                review.innerText = data.human_review_required ? 'YES' : 'No';
                review.style.color = data.human_review_required ? '#D97706' : '#059669';

                document.getElementById('fu-evidence').innerHTML = (data.evidence || []).map(ev => {
                    const dot = ev.available ? '🟢' : '⚪';
                    const lvl = ev.available
                        ? `<strong style="color:${PRIORITY_COLORS[ev.level_name] || '#94A3B8'}">${ev.level_name}</strong>`
                          + (ev.confidence != null ? ` <span style="color:var(--text-muted)">(conf ${(ev.confidence*100).toFixed(1)}%)</span>` : '')
                        : '<span style="color:var(--text-muted)">not available</span>';
                    return `<div style="margin-bottom:8px;">${dot} <strong>${ev.source}</strong> → ${lvl}<br>
                            <span style="color:var(--text-muted); margin-left:20px;">${ev.detail}</span></div>`;
                }).join('');

                const conflicts = data.conflicts || [];
                document.getElementById('fu-conflict-wrap').style.display = conflicts.length ? 'block' : 'none';
                renderList('fu-conflicts', conflicts.map(c => c.description + ' <em style="color:var(--text-muted)">' + c.resolution + '</em>'), '⚠');

                const escalations = data.escalations || [];
                document.getElementById('fu-escalation-wrap').style.display = escalations.length ? 'block' : 'none';
                renderList('fu-escalations', escalations, '↑');

                renderList('fu-actions', data.recommended_actions || [], '▸');

                const warnings = (data.warnings || []).concat(data.human_review_reasons || []);
                document.getElementById('fu-warn-wrap').style.display = warnings.length ? 'block' : 'none';
                renderList('fu-warnings', warnings, '•');

                document.getElementById('fu-disclaimer').innerText = data.disclaimer || '';
            } catch (err) {
                alert('Unified assessment error: ' + err.message);
            }
            btn.innerText = 'Run Unified Assessment';
        });

        // ML Form Submission
        document.getElementById('mlForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
                timestamp: "05-09-2026 12:00",
                state: "CA",
                district: document.getElementById('ml-district').value,
                rainfall_mm: parseFloat(document.getElementById('ml-rainfall').value),
                river_level_m: parseFloat(document.getElementById('ml-river').value),
                river_level_threshold_m: parseFloat(document.getElementById('ml-thresh').value),
                emergency_calls: parseInt(document.getElementById('ml-calls').value),
                water_level_change_m: parseFloat(document.getElementById('ml-change').value),
                road_closures: 0,
                bridge_closures: 0,
                flood_history_count: 5,
                population_affected: 50000
            };

            const btn = e.target.querySelector('button');
            btn.innerText = "Processing...";
            
            try {
                const res = await fetch('/api/predict/ml', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok || data.error) {
                    throw new Error(data.error || 'ML prediction failed');
                }

                const resBox = document.getElementById('ml-result');
                const badge = document.getElementById('ml-badge');

                resBox.style.display = 'block';
                badge.innerText = data.risk_category || "Error";
                badge.className = "badge badge-" + data.risk_category;

                document.getElementById('ml-conf').innerText = data.confidence ? (data.confidence * 100).toFixed(2) : "N/A";
                document.getElementById('ml-json').innerText =
                    'Top drivers for this input: ' + (data.top_factors || []).join(', ')
                    + '  |  Dataset-wide drivers: ' + (data.global_top_factors || []).join(', ');
            } catch (err) {
                alert("Error calling ML API: " + err.message);
            }
            btn.innerText = "Run Inference";
        });

        // CNN Form Submission
        document.getElementById('cnnForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('cnn-image');
            if (!fileInput.files[0]) return alert("Please select an image");

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const btn = e.target.querySelector('button');
            btn.innerText = "Analyzing...";

            try {
                const res = await fetch('/api/predict/dl/image', { method: 'POST', body: formData });
                const data = await res.json();

                if(data.error) throw new Error(data.error);

                document.getElementById('cnn-result').style.display = 'block';
                document.getElementById('cnn-status').innerText = data.label;
                document.getElementById('cnn-prob').innerText = (data.confidence * 100).toFixed(2);
                
                const c = data.label;
                document.getElementById('cnn-status').style.color = (c === 'flooded') ? '#DC2626' : '#059669';
            } catch (err) {
                alert("Error calling CNN API: " + err.message);
            }
            btn.innerText = "Analyze Image";
        });

        // NLP Form Submission
        document.getElementById('nlpForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('nlp-text').value.trim();
            if (!text) {
                alert('Please enter an emergency message.');
                return;
            }

            const btn = e.target.querySelector('button');
            btn.innerText = 'Analyzing...';

            try {
                const res = await fetch('/api/predict/nlp', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text })
                });
                const data = await res.json();

                if (!res.ok || data.error) {
                    throw new Error(data.error || 'NLP analysis failed');
                }

                const resultBox = document.getElementById('nlp-result');
                resultBox.style.display = 'block';
                document.getElementById('nlp-urgency').innerText = data.urgency || 'N/A';
                document.getElementById('nlp-hazard').innerText = data.hazard_type || 'N/A';
                document.getElementById('nlp-confidence').innerText = (data.confidence * 100).toFixed(2) + '%';
                document.getElementById('nlp-location').innerText = Array.isArray(data.location) ? data.location.join(', ') : (data.location || 'N/A');
                document.getElementById('nlp-resource').innerText = Array.isArray(data.resource_needed) ? data.resource_needed.join(', ') : (data.resource_needed || 'N/A');
                document.getElementById('nlp-headcount').innerText = data.headcount ?? 'N/A';
                document.getElementById('nlp-meta').innerText = 'Hazard confidence: ' + ((data.hazard_confidence || 0) * 100).toFixed(2) + '%';
            } catch (err) {
                alert('NLP analysis error: ' + err.message);
            }
            btn.innerText = 'Analyze Message';
        });

        // LSTM Chart Setup
        Chart.defaults.color = '#475569';
        Chart.defaults.font.family = 'Inter';
        let lstmChart = null;

        document.getElementById('btn-lstm').addEventListener('click', async (e) => {
            const btn = e.target;
            btn.innerText = "Simulating...";

            // 72 hours of water levels, IN the model's training range.
            //
            // The previous demo sequence ran 8.0 -> 15.0 m. The LSTM was trained
            // on a series with mean 3.12 m and std 1.56 m, so 15 m is roughly
            // +7.6 sigma - far outside anything the model ever saw. It responded
            // with a physically impossible forecast (an instant 4 m drop, then
            // oscillation), which is what the demo chart plotted.
            //
            // This is a realistic rising-river scenario within the trained range:
            // a slow baseline around 2.9 m climbing to ~5.4 m.
            const inputSeq = Array.from({length: 48}, (_, i) => 2.9 + (i * 0.9 / 48)).concat([
                3.85, 3.92, 4.00, 4.05, 4.12, 4.20, 4.28, 4.35,
                4.44, 4.51, 4.58, 4.66, 4.73, 4.80, 4.88, 4.95,
                5.02, 5.09, 5.16, 5.22, 5.28, 5.33, 5.37, 5.40
            ]);

            try {
                const res = await fetch('/api/predict/dl/lstm', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ sequence: inputSeq })
                });
                const data = await res.json();
                
                if (data.error) throw new Error(data.error);

                const forecasts = data.forecast_water_levels;
                document.getElementById('lstm-result').style.display = 'block';
                document.getElementById('lstm-val').innerText = forecasts[forecasts.length - 1].toFixed(2);

                // Show measured multi-step error and any out-of-distribution warning,
                // so the forecast is never presented as unqualified ground truth.
                const notes = [];
                if (data.expected_mae_at_horizon != null) {
                    notes.push('Measured MAE at +6h on the test partition: ±'
                        + data.expected_mae_at_horizon.toFixed(3) + ' m');
                }
                if (data.warning) {
                    notes.push('⚠ ' + data.warning);
                } else if (data.distribution_check) {
                    notes.push('Input is within the trained range ('
                        + data.distribution_check.max_sigma_from_training_mean
                        + 'σ from training mean).');
                }
                const noteEl = document.getElementById('lstm-note');
                noteEl.innerText = notes.join('  •  ');
                noteEl.style.color = data.warning ? '#D97706' : 'var(--text-muted)';

                const labels = Array.from({length: 78}, (_, i) => `T-${72-i}`);
                const histData = [...inputSeq, ...Array(6).fill(null)];
                const foreData = [...Array(71).fill(null), inputSeq[71], ...forecasts];

                if(lstmChart) lstmChart.destroy();

                const ctx = document.getElementById('lstmChart').getContext('2d');
                lstmChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            { label: 'Historical', data: histData, borderColor: '#2563EB', tension: 0.2 },
                            { label: 'Forecast', data: foreData, borderColor: '#D97706', borderDash: [5, 5], tension: 0.2 }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: { y: { grid: { color: '#E2E8F0' } }, x: { grid: { display: false } } }
                    }
                });

            } catch (err) {
                alert("Error calling LSTM API: " + err.message);
            }
            btn.innerText = "Generate Forecast Plot";
        });

        // --- SLM Briefing (Stage 04) ---
        const slmForm = document.getElementById('slmForm');
        if (slmForm) {
            slmForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const report = document.getElementById('slm-report').value;
                const resultEl = document.getElementById('slm-result');
                const btn = e.target.querySelector('button');
                btn.innerText = 'Generating...';
                try {
                    const res = await fetch('/api/slm/summarize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({report: report})
                    });
                    const data = await res.json();
                    if (data.error) {
                        alert(data.error);
                        btn.innerText = 'Generate Tactical Briefing';
                        return;
                    }
                    document.getElementById('slm-situation').innerText = data.situation || '';
                    document.getElementById('slm-risk').innerText = data.risk || '';
                    const actionsEl = document.getElementById('slm-actions');
                    if (Array.isArray(data.actions)) {
                        actionsEl.innerHTML = data.actions.map(a => '<li>' + a + '</li>').join('');
                    } else {
                        actionsEl.innerHTML = '<li>' + (data.actions || 'N/A') + '</li>';
                    }
                    document.getElementById('slm-priority-badge').className = 'badge badge-' + (data.priority || 'Unknown');
                    document.getElementById('slm-priority-badge').innerText = data.priority || 'Unknown';
                    document.getElementById('slm-model').innerText = data.model || '';
                    document.getElementById('slm-latency').innerText = (data.latency_ms || 0) + ' ms';

                    // Time savings estimate
                    const actionsText = Array.isArray(data.actions) ? data.actions.join(' ') : (data.actions || '');
                    const summaryText = ((data.situation || '') + ' ' + (data.risk || '') + ' ' + actionsText).trim();
                    const reportWords = report.trim().split(/\s+/).filter(Boolean).length;
                    const summaryWords = summaryText.split(/\s+/).filter(Boolean).length;
                    const pctSaving = Math.max(0, Math.min(99, Math.round((1 - summaryWords / Math.max(1, reportWords)) * 100)));
                    document.getElementById('slm-time-pct').innerText = pctSaving + '%';
                    document.getElementById('slm-time-bar').style.width = Math.min(100, Math.max(0, pctSaving)) + '%';

                    resultEl.style.display = 'block';
                } catch (err) {
                    alert('Error calling SLM API: ' + err.message);
                }
                btn.innerText = 'Generate Tactical Briefing';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(
        HTML_TEMPLATE,
        stage01=stage01_status,
        stage02=stage02_status,
        stage03=stage03_status,
        stage04=stage04_status,
        stage05=stage05_status,
        stage06=stage06_status,
    )

@app.route('/health')
def health():
    """Per-stage readiness, each verified by an actual prediction."""
    report = {}
    for name, engine in (
        ("stage01_ml", stage01_engine),
        ("stage02_dl", stage02_engine),
        ("stage03_nlp", stage03_engine),
        ("stage04_slm", stage04_engine),
        ("stage05_genai", stage05_engine),
        ("stage06_agentic", stage06_engine),
    ):
        if engine is None:
            report[name] = {"status": "unavailable", "error": "adapter failed to load"}
            continue
        try:
            report[name] = engine.health_check()
        except Exception as exc:
            logger.exception("%s health check raised", name)
            report[name] = {"status": "degraded", "error": type(exc).__name__}
    overall = "healthy" if all(
        entry.get("status") == "healthy" for entry in report.values()
    ) else "degraded"
    return jsonify({"status": overall, "stages": report})


@app.route('/api/assess', methods=['POST'])
def unified_assessment():
    """Fuse every available stage into one prioritised decision.

    Accepts any subset of: sensors, text, water_levels, image_path.
    Missing modalities are reported, not silently treated as zero evidence.
    """
    if decision_engine is None:
        return fail("Fusion decision engine is offline", 503)
    data = request.get_json(silent=True)
    if data is None:
        return fail("Request body must be JSON", 400)

    sensors = data.get("sensors")
    if sensors is not None and not isinstance(sensors, dict):
        return fail("'sensors' must be a JSON object", 400)
    water_levels = data.get("water_levels")
    if water_levels is not None and not isinstance(water_levels, list):
        return fail("'water_levels' must be a list of numbers", 400)

    try:
        result = decision_engine.assess(
            sensors=sensors,
            text=data.get("text"),
            image_path=data.get("image_path"),
            water_levels=water_levels,
        )
    except Exception as exc:
        return fail("Assessment failed. See server logs for details.", 500, exc)

    status_code = 200 if result.get("status") == "ok" else 422
    return jsonify(result), status_code


@app.route('/api/slm/summarize', methods=['POST'])
def slm_summarize():
    if not stage04_engine:
        return fail("SLM API is offline", 503)
    data = request.get_json(silent=True) or {}
    report_text = data.get("report", "")
    if not str(report_text).strip():
        return fail("Please provide a non-empty incident report.", 400)
    try:
        result = stage04_engine.summarize(str(report_text))
    except Exception as exc:
        return fail("SLM summarization failed. See server logs for details.", 500, exc)
    if result.get("status") == "error":
        return fail(result.get("message", "SLM error"), 400)
    return jsonify({
        "situation": result.get("situation"),
        "risk":      result.get("risk"),
        "actions":   result.get("actions", []),
        "priority":  result.get("priority"),
        "model":     result.get("model"),
        "latency_ms": result.get("latency_ms"),
    })


@app.route('/api/predict/nlp', methods=['POST'])
def predict_nlp():
    if not stage03_engine:
        return fail("NLP API is offline", 503)
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if text is None or not str(text).strip():
        return fail("Please enter an emergency message.", 400)
    try:
        result = stage03_engine.analyze(str(text))
    except Exception as exc:
        return fail("NLP analysis failed. See server logs for details.", 500, exc)

    if result.get("status") == "error":
        return fail(result.get("message", "Invalid NLP input"), 400)
    return jsonify({
        "urgency": result.get("urgency"),
        "hazard_type": result.get("hazard_type"),
        "confidence": result.get("confidence", 0.0),
        "hazard_confidence": result.get("hazard_confidence", 0.0),
        "location": result.get("location"),
        "resource_needed": result.get("resource_needed", []),
        "headcount": result.get("headcount"),
        "entities": result.get("entities", {})
    })


@app.route('/api/predict/ml', methods=['POST'])
def predict_ml():
    if not stage01_engine:
        return fail("ML API is offline", 503)
    data = request.get_json(silent=True)
    if data is None:
        return fail("Request body must be JSON", 400)
    try:
        return jsonify(stage01_engine.predict(data))
    except ValueError as exc:
        # Input-validation messages are written for the caller and safe to show.
        return fail(str(exc), 400)
    except Exception as exc:
        return fail("Prediction failed. See server logs for details.", 500, exc)


@app.route('/api/predict/dl/image', methods=['POST'])
def predict_dl_image():
    if not stage02_engine:
        return fail("DL API is offline", 503)
    if 'file' not in request.files:
        return fail("No file uploaded", 400)
    file = request.files['file']
    if not file.filename:
        return fail("No file selected", 400)

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return fail(
            f"Unsupported image type '{extension}'. Allowed: "
            f"{', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
            400,
        )

    # Unique per-request filename. Every upload previously overwrote the same
    # dashboard_upload.jpg, so two concurrent users raced each other and could
    # be shown a prediction for someone else's image.
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / f"upload_{uuid.uuid4().hex}{extension}"
    try:
        file.save(filepath)
        return jsonify(stage02_engine.predict_image(str(filepath)))
    except ValueError as exc:
        return fail(str(exc), 400)
    except Exception as exc:
        return fail("Image analysis failed. See server logs for details.", 500, exc)
    finally:
        # Uploads are transient; do not accumulate user images on disk.
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary upload %s", filepath)


@app.route('/api/predict/dl/lstm', methods=['POST'])
def predict_dl_lstm():
    if not stage02_engine:
        return fail("DL API is offline", 503)
    data = request.get_json(silent=True)
    if data is None:
        return fail("Request body must be JSON", 400)
    sequence = data.get("sequence", [])
    try:
        return jsonify(stage02_engine.forecast_water_levels(sequence, horizon=6))
    except ValueError as exc:
        return fail(str(exc), 400)
    except Exception as exc:
        return fail("Forecast failed. See server logs for details.", 500, exc)


@app.errorhandler(413)
def upload_too_large(_error):
    return fail(
        f"Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit", 413
    )


if __name__ == '__main__':
    # debug=True enables the Werkzeug interactive debugger, which is a remote
    # code execution console. It is opt-in via the environment now instead of
    # being hardcoded on.
    debug_enabled = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting Interactive AI Dashboard on http://127.0.0.1:{port}")
    print(f"  Stage 01 ML : {stage01_status}")
    print(f"  Stage 02 DL : {stage02_status}")
    print(f"  Stage 03 NLP: {stage03_status}")
    print(f"  Stage 04 SLM: {stage04_status}")
    print(f"  Stage 05 GenAI: {stage05_status}")
    print(f"  Stage 06 Agentic: {stage06_status}")
    if debug_enabled:
        print("  WARNING: debug mode is ON (interactive debugger enabled).")
    app.run(debug=debug_enabled, port=port)
