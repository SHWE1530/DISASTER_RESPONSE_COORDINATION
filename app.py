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
    <title>AI Platform Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-main: #0B1120;
            --bg-sidebar: #0F172A;
            --bg-card: rgba(30, 41, 59, 0.7);
            --border-color: #1E293B;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --accent-blue: #3B82F6;
            --accent-purple: #8B5CF6;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

        .sidebar { width: 260px; background-color: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 24px 16px; }
        .sidebar-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 40px; padding: 0 12px; }
        .sidebar-brand h1 { font-size: 20px; font-weight: 700; }
        .sidebar-brand p { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
        
        .nav-item { display: flex; align-items: center; gap: 12px; padding: 12px; color: var(--text-muted); text-decoration: none; border-radius: 8px; margin-bottom: 4px; transition: all 0.2s; font-weight: 500; font-size: 14px; cursor: pointer; }
        .nav-item:hover, .nav-item.active { background-color: #1E293B; color: var(--text-main); }
        .nav-item.active { background-color: rgba(59, 130, 246, 0.1); color: var(--accent-blue); }
        .nav-icon { width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; }

        .main-content { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
        .header { height: 72px; display: flex; align-items: center; justify-content: space-between; padding: 0 32px; border-bottom: 1px solid var(--border-color); flex-shrink: 0; }
        .search-bar { background-color: var(--bg-sidebar); border: 1px solid var(--border-color); border-radius: 8px; padding: 8px 16px; color: var(--text-main); width: 300px; outline: none; }
        
        .view-section { padding: 32px; display: none; }
        .view-section.active { display: block; }
        
        .page-header { margin-bottom: 24px; }
        .page-header h2 { font-size: 24px; margin-bottom: 4px; }
        .page-header p { color: var(--text-muted); font-size: 14px; }

        .chart-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 20px;}
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .chart-header h3 { font-size: 16px; font-weight: 600; }

        .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }
        .input-group { display: flex; flex-direction: column; }
        .input-group label { font-size: 12px; color: var(--text-muted); margin-bottom: 5px; }
        .input-group input { background: #1E293B; border: 1px solid #334155; color: white; padding: 10px; border-radius: 6px; outline: none; }
        .btn { background: var(--accent-blue); color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; transition: background 0.2s; }
        .btn:hover { background: #2563EB; }
        .btn-green { background: var(--accent-green); }
        .btn-green:hover { background: #059669; }

        .result-box { background: rgba(0,0,0,0.2); border: 1px solid var(--border-color); padding: 15px; border-radius: 8px; margin-top: 15px; font-family: monospace; }
        .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 12px; }
        .badge-Severe { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
        .badge-Moderate { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
        .badge-Low { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }

        .grid-half { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        textarea { width: 100%; min-height: 120px; resize: vertical; border-radius: 8px; border: 1px solid #334155; background: #1E293B; color: white; padding: 12px; }
        .nlp-output { margin-top: 15px; background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-brand">
            <div><h1>AI Platform</h1><p>Build • Train • Deploy</p></div>
        </div>

        <a class="nav-item active" data-target="view-dashboard"><div class="nav-icon">🏠</div> Dashboard</a>
        <a class="nav-item" data-target="view-fusion"><div class="nav-icon" style="color: var(--accent-amber)">🎯</div> Unified Decision</a>
        <a class="nav-item" data-target="view-ml"><div class="nav-icon" style="color: var(--accent-green)">⚙️</div> ML (Stage 01)</a>
        <a class="nav-item" data-target="view-dl"><div class="nav-icon" style="color: var(--accent-blue)">⚡</div> DL (Stage 02)</a>
        <a class="nav-item" data-target="view-nlp"><div class="nav-icon" style="color: var(--accent-purple)">💬</div> NLP (Stage 03)</a>
        <a class="nav-item" data-target="view-slm"><div class="nav-icon" style="color: #F97316">📋</div> SLM Briefing (Stage 04)</a>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <div class="header">
            <input type="text" class="search-bar" placeholder="Search projects, models, datasets...">
            <div class="header-actions">
                <span>🔔</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 32px; height: 32px; background: #334155; border-radius: 50%; display: flex; align-items: center; justify-content: center;">JD</div>
                    <span>John Doe</span>
                </div>
            </div>
        </div>

        <!-- VIEW: DASHBOARD -->
        <div id="view-dashboard" class="view-section active">
            <div class="page-header">
                <h2>Dashboard Overview</h2>
                <p>Welcome to the Disaster Response AI Interface.</p>
            </div>
            <div class="chart-card">
                <h3>System Status</h3>
                <p><strong>Stage 01 API (ML):</strong> {{ stage01 }}</p>
                <p><strong>Stage 02 API (DL):</strong> {{ stage02 }}</p>
                <p><strong>Stage 03 API (NLP):</strong> {{ stage03 }}</p>
                <p><strong>Stage 04 API (SLM):</strong> {{ stage04 }}</p>
                <p style="margin-top: 15px; color: var(--text-muted);">Please use the navigation menu on the left to access the ML prediction forms, DL image/forecasting tools, NLP emergency text analysis, and the SLM tactical briefing generator.</p>
            </div>
        </div>

        <!-- VIEW: UNIFIED DECISION (FUSION) -->
        <div id="view-fusion" class="view-section">
            <div class="page-header">
                <h2>Unified Incident Assessment</h2>
                <p>Combines sensor risk (Stage 01), visual confirmation and forecast (Stage 02), and the text report (Stage 03) into one prioritised decision.</p>
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
                    <h4 style="margin-top: 20px; margin-bottom: 8px; font-size: 14px; color: #EF4444;">Conflicts</h4>
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

        <!-- VIEW: NLP (STAGE 03) -->
        <div id="view-nlp" class="view-section">
            <div class="page-header">
                <h2>NLP Emergency Analysis (Stage 03)</h2>
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
                <h2>SLM Tactical Briefing (Stage 04)</h2>
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
                        <div style="background: #1E293B; border-radius: 6px; height: 8px; overflow: hidden;">
                            <div id="slm-time-bar" style="background: linear-gradient(90deg, #10B981, #F59E0B); height: 100%; width: 0%; transition: width 0.6s;"></div>
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
                <h2>Machine Learning Predictor (Stage 01)</h2>
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
                        <div class="input-group"><label>District</label><input type="text" id="ml-district" value="Downtown"></div>
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
                <h2>Deep Learning Predictor (Stage 02)</h2>
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
                        Status: <span id="cnn-status" style="color: white; font-weight: bold;"></span><br>
                        Probability: <span id="cnn-prob"></span>%<br>
                    </div>
                </div>

                <!-- LSTM Form -->
                <div class="chart-card">
                    <div class="chart-header"><h3>Sequential Forecast (LSTM)</h3></div>
                    <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 15px;">Simulate a 6-hour forecast using recent sensor trajectory.</p>
                    <button id="btn-lstm" class="btn btn-green" style="width: 100%; margin-bottom: 20px;">Generate Forecast Plot</button>
                    <div style="height: 200px; width: 100%; background: #0B1120; border-radius: 8px;">
                        <canvas id="lstmChart"></canvas>
                    </div>
                    <div class="result-box" id="lstm-result" style="display: none; margin-top: 10px;">
                        Target +6hr Forecast: <strong id="lstm-val" style="color: var(--accent-amber);"></strong> m
                        <small id="lstm-note" style="color: var(--text-muted); display: block; margin-top: 8px; line-height: 1.5;"></small>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- Interactivity Script -->
    <script>
        // Tab switching logic
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', event => {
                event.preventDefault();
                document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                const targetId = item.getAttribute('data-target');
                document.querySelectorAll('.view-section').forEach(view => view.classList.remove('active'));
                
                const targetView = document.getElementById(targetId);
                if (targetView) targetView.classList.add('active');
            });
        });

        // Unified Assessment (fusion) Submission
        const PRIORITY_COLORS = { ROUTINE: '#10B981', ELEVATED: '#F59E0B', URGENT: '#F97316', CRITICAL: '#EF4444' };

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
                review.style.color = data.human_review_required ? '#F59E0B' : '#10B981';

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
                document.getElementById('cnn-status').style.color = (c === 'flooded') ? '#EF4444' : '#10B981';
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
        Chart.defaults.color = '#94A3B8';
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
                noteEl.style.color = data.warning ? '#F59E0B' : 'var(--text-muted)';

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
                            { label: 'Historical', data: histData, borderColor: '#3B82F6', tension: 0.2 },
                            { label: 'Forecast', data: foreData, borderColor: '#F59E0B', borderDash: [5, 5], tension: 0.2 }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: { y: { grid: { color: '#1E293B' } }, x: { grid: { display: false } } }
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
    return render_template_string(HTML_TEMPLATE, stage01=stage01_status, stage02=stage02_status, stage03=stage03_status, stage04=stage04_status)

@app.route('/health')
def health():
    """Per-stage readiness, each verified by an actual prediction."""
    report = {}
    for name, engine in (
        ("stage01_ml", stage01_engine),
        ("stage02_dl", stage02_engine),
        ("stage03_nlp", stage03_engine),
        ("stage04_slm", stage04_engine),
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
        return fail("Stage 04 SLM API is offline", 503)
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
        return fail("Stage 03 NLP API is offline", 503)
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
        return fail("Stage 01 ML API is offline", 503)
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
        return fail("Stage 02 DL API is offline", 503)
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
        return fail("Stage 02 DL API is offline", 503)
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
    if debug_enabled:
        print("  WARNING: debug mode is ON (interactive debugger enabled).")
    app.run(debug=debug_enabled, port=port)
