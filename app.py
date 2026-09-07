from importlib import import_module
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
STAGE01_DIR = BASE_DIR / "Stage01_ML"
sys.path.insert(0, str(STAGE01_DIR))

integration_module = import_module("05_integration_engineer")
integration_engine = integration_module.integration_engine

app = Flask(__name__)

DATA_DIR = STAGE01_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
PATTERN_DIR = OUTPUT_DIR / "pattern"
SESSION_HISTORY = []


def _read_json(path, default):
	try:
		with path.open(encoding="utf-8") as file:
			return json.load(file)
	except (OSError, ValueError):
		return default


def _read_csv(path):
	try:
		return pd.read_csv(path)
	except (OSError, ValueError):
		return pd.DataFrame()


def _number(value, digits=3):
	if value is None or pd.isna(value):
		return None
	return round(float(value), digits)


def dashboard_data():
	dataset = _read_csv(PATTERN_DIR / "eda_checked_dataset.csv")
	if dataset.empty:
		dataset = _read_csv(DATA_DIR / "processed" / "Master_Dataset.csv")
	predictions = _read_csv(OUTPUT_DIR / "ml_predictions.csv")
	metrics = _read_json(OUTPUT_DIR / "eval_final_report.json", {})
	validation = _read_json(OUTPUT_DIR / "ml_evaluation_metrics.json", {})
	importance = _read_csv(OUTPUT_DIR / "feature_importance.csv")

	counts = dataset.get("zone_risk", pd.Series(dtype=str)).value_counts()
	classification = metrics.get("metrics", {}).get("classification_report", {})
	selected_model = metrics.get("model_name") or validation.get("model_selected") or "N/A"
	model_metrics = {
		"accuracy": _number(metrics.get("metrics", {}).get("accuracy")),
		"precision": _number(classification.get("weighted avg", {}).get("precision")),
		"recall": _number(classification.get("weighted avg", {}).get("recall")),
		"f1": _number(metrics.get("metrics", {}).get("macro_f1")),
		"roc_auc": None,
	}

	trend = []
	if not dataset.empty and "timestamp" in dataset and "zone_risk" in dataset:
		trend_frame = dataset.copy()
		trend_frame["parsed_time"] = pd.to_datetime(trend_frame["timestamp"], dayfirst=True, errors="coerce")
		trend_frame = trend_frame.dropna(subset=["parsed_time"])
		if not trend_frame.empty:
			grouped = trend_frame.groupby(trend_frame["parsed_time"].dt.strftime("%b %Y"))["zone_risk"].value_counts().unstack(fill_value=0)
			for period, row in grouped.tail(12).iterrows():
				trend.append({"period": period, "Low": int(row.get("Low", 0)), "Moderate": int(row.get("Moderate", 0)), "Severe": int(row.get("Severe", 0))})

	district_counts = []
	if not dataset.empty and "district" in dataset and "zone_risk" in dataset:
		for district, group in dataset.groupby("district"):
			severe = int((group["zone_risk"] == "Severe").sum())
			district_counts.append({"district": str(district), "total": int(len(group)), "severe": severe, "severe_rate": _number(severe / len(group) * 100, 1)})
		district_counts = sorted(district_counts, key=lambda item: item["severe_rate"], reverse=True)[:8]

	feature_rows = []
	if not importance.empty and {"feature", "importance"}.issubset(importance.columns):
		for _, row in importance.head(8).iterrows():
			feature_rows.append({"feature": str(row["feature"]), "importance": _number(row["importance"], 4)})

	history = []
	if not predictions.empty:
		for _, row in predictions.tail(12).iloc[::-1].iterrows():
			history.append({
				"timestamp": str(row.get("timestamp", "N/A")),
				"district": str(row.get("district", "N/A")),
				"risk": str(row.get("predicted_risk_category", row.get("zone_risk", "N/A"))),
				"confidence": _number(row.get("confidence"), 4),
				"factors": str(row.get("top_factors", "N/A")),
			})
		history.extend(SESSION_HISTORY)

	confusion = metrics.get("metrics", {}).get("confusion_matrix", [])
	return {
		"generated_at": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
		"records": int(len(dataset)),
		"risk_counts": {"Low": int(counts.get("Low", 0)), "Moderate": int(counts.get("Moderate", 0)), "Severe": int(counts.get("Severe", 0))},
		"predictions": len(predictions) + len(SESSION_HISTORY),
		"model": selected_model,
		"model_metrics": model_metrics,
		"model_candidates": validation.get("candidate_benchmarks", {}),
		"feature_count": len(importance) if not importance.empty else None,
		"confusion_matrix": confusion,
		"feature_importance": feature_rows,
		"trend": trend,
		"districts": district_counts,
		"history": history[:20],
		"model_loaded": integration_engine.model is not None,
		"load_error": integration_engine.load_error,
	}


@app.get("/api/dashboard")
def dashboard_api():
	return jsonify(dashboard_data())


def dashboard_html():
	return """
	<!doctype html>
	<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Disaster Response Intelligence Center</title>
		<style>
			:root { --ink:#13252b; --muted:#6c7e82; --paper:#edf2f0; --panel:#fffefa; --line:#d9e4e1; --navy:#102d39; --teal:#087f78; --teal-dark:#07534f; --coral:#d9684b; --low:#268763; --moderate:#c88728; --severe:#c94d47; }
			* { box-sizing:border-box; } html { scroll-behavior:smooth; }
			body { margin:0; color:var(--ink); background:var(--paper); font-family:Arial, sans-serif; }
			.shell { max-width:1440px; margin:auto; padding:24px; }
			header { padding:28px 30px; color:white; background:linear-gradient(120deg,#102d39,#07534f); border-bottom:5px solid var(--coral); }
			.eyebrow,.kicker { margin:0 0 10px; color:#f6b49e; font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }
			h1 { margin:0; max-width:800px; font:700 clamp(30px,5vw,58px)/.98 Georgia,serif; }
			.subtitle { margin:14px 0 0; max-width:700px; color:#d2e4e1; line-height:1.5; }
			.header-meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; align-items:center; }
			.status-pill,.nav a,.refresh { padding:9px 12px; border:1px solid rgba(255,255,255,.25); color:white; font-size:11px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; }
			.status-pill { border-color:rgba(94,214,164,.45); color:#94e6c1; } .status-pill::before { content:'●'; margin-right:7px; }
			.refresh { background:var(--coral); cursor:pointer; } .refresh:hover { background:#b9533c; }
			.nav { display:flex; gap:6px; flex-wrap:wrap; margin-top:18px; } .nav a { text-decoration:none; color:#cde2df; border-color:transparent; } .nav a:hover { background:rgba(255,255,255,.1); }
			.section { margin-top:22px; } .section-head { display:flex; justify-content:space-between; align-items:end; gap:16px; margin:0 0 12px; } .section-head h2 { margin:0; font:700 24px Georgia,serif; } .section-head p { margin:0; color:var(--muted); font-size:13px; }
			.kpis { display:grid; grid-template-columns:repeat(6,1fr); gap:12px; } .kpi,.panel { background:var(--panel); border:1px solid var(--line); box-shadow:0 8px 24px rgba(19,37,43,.06); }
			.kpi { padding:17px; border-top:4px solid var(--teal); } .kpi.risk-severe { border-top-color:var(--severe); } .kpi.risk-moderate { border-top-color:var(--moderate); } .kpi.risk-low { border-top-color:var(--low); }
			.kpi-label { color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; } .kpi-value { margin-top:8px; font-size:28px; font-weight:700; } .kpi-note { margin-top:5px; color:var(--muted); font-size:11px; }
			.two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; } .three-col { display:grid; grid-template-columns:1.25fr 1fr 1fr; gap:16px; }
			.panel { padding:20px; } .panel h3 { margin:0 0 5px; font-size:16px; } .panel-sub { margin:0 0 18px; color:var(--muted); font-size:12px; }
			.donut-wrap { display:flex; align-items:center; gap:24px; min-height:190px; } .donut { width:156px; height:156px; flex:none; border-radius:50%; background:conic-gradient(var(--severe) 0 70%,var(--moderate) 70% 90%,var(--low) 90% 100%); position:relative; } .donut::after { content:''; position:absolute; inset:30px; border-radius:50%; background:var(--panel); } .legend { display:grid; gap:12px; width:100%; } .legend-row { display:flex; justify-content:space-between; gap:10px; font-size:13px; } .legend-row span:first-child::before { content:'●'; margin-right:8px; color:var(--teal); } .legend-row.severe span:first-child::before { color:var(--severe); } .legend-row.moderate span:first-child::before { color:var(--moderate); } .legend-row.low span:first-child::before { color:var(--low); }
			.metric { margin:15px 0; } .metric-line { display:flex; justify-content:space-between; font-size:12px; } .track { height:8px; margin-top:7px; background:#e7eeec; overflow:hidden; } .track i { display:block; height:100%; background:var(--teal); }
			.bars { display:grid; gap:10px; } .bar-line { display:grid; grid-template-columns:125px 1fr 52px; align-items:center; gap:8px; font-size:11px; } .bar-track { height:10px; background:#e7eeec; } .bar-track i { display:block; height:100%; background:var(--coral); }
			.trend { display:grid; grid-template-columns:repeat(12,1fr); gap:6px; align-items:end; min-height:180px; padding-top:12px; } .trend-col { display:flex; flex-direction:column; justify-content:end; height:160px; gap:2px; } .trend-stack { display:flex; flex-direction:column-reverse; justify-content:flex-start; height:130px; } .trend-stack i { display:block; min-height:1px; } .trend-stack .severe { background:var(--severe); } .trend-stack .moderate { background:var(--moderate); } .trend-stack .low { background:var(--low); } .trend-label { margin-top:7px; overflow:hidden; color:var(--muted); font-size:9px; text-align:center; white-space:nowrap; }
			.model-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; } .model-stat { padding:12px; background:#f1f6f4; } .model-stat b { display:block; margin-top:5px; font-size:19px; } .model-stat span { color:var(--muted); font-size:10px; text-transform:uppercase; }
			.matrix { display:grid; grid-template-columns:repeat(4,1fr); gap:3px; margin-top:14px; font-size:12px; text-align:center; } .matrix div { padding:13px 4px; background:#edf4f2; } .matrix .head { color:var(--muted); background:transparent; font-size:10px; font-weight:700; } .matrix .diag { color:white; background:var(--teal); }
			.assessment { display:grid; grid-template-columns:1.2fr .8fr; gap:16px; } .input-panel { border-top:5px solid var(--teal); } .result-panel { color:white; background:var(--navy); border-color:#274653; } .form-section { margin-top:20px; padding-top:16px; border-top:1px solid var(--line); } .form-section:first-child { margin-top:0; padding-top:0; border-top:0; } .form-section h4 { margin:0 0 11px; color:var(--teal-dark); font-size:11px; letter-spacing:1px; text-transform:uppercase; } .form-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; } .field label { display:block; margin-bottom:5px; font-size:12px; font-weight:700; } .field small { display:block; min-height:27px; margin-top:4px; color:var(--muted); font-size:10px; line-height:1.3; } input { width:100%; padding:11px; border:1px solid var(--line); border-radius:2px; background:#fbfdfc; font-size:13px; } input:focus { outline:2px solid #9bd7ce; border-color:var(--teal); } .actions { display:flex; gap:10px; margin-top:20px; } button { padding:12px 15px; border:0; color:white; background:var(--teal); font-weight:700; cursor:pointer; } button:hover { background:var(--teal-dark); } button.secondary { color:var(--teal-dark); background:#edf4f2; }
			.result-top { display:flex; justify-content:space-between; gap:10px; color:#9fc0c4; font-size:10px; letter-spacing:1px; text-transform:uppercase; } .risk-output { margin-top:28px; padding:24px 0; border-top:1px solid #365761; border-bottom:1px solid #365761; } .risk-output h4 { margin:0 0 8px; color:#9fc0c4; font-size:10px; letter-spacing:1px; text-transform:uppercase; } .risk-output strong { font-size:48px; } .risk-output strong.low { color:#72d6a7; } .risk-output strong.moderate { color:#ffd166; } .risk-output strong.severe { color:#ff8066; } .confidence { margin-top:26px; } .confidence-line { display:flex; justify-content:space-between; color:#c8dadd; font-size:12px; } .confidence-track { height:10px; margin-top:8px; background:#29454d; } .confidence-track i { display:block; height:100%; width:0; background:#72d6a7; transition:width .3s ease; } .decision { margin-top:22px; color:#d2e2e4; font-size:13px; line-height:1.5; } .disclaimer { color:#89a8ad; font-size:10px; }
			.history { overflow:auto; } table { width:100%; border-collapse:collapse; font-size:12px; } th,td { padding:11px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; } th { color:var(--muted); font-size:10px; letter-spacing:.7px; text-transform:uppercase; } .badge { display:inline-block; padding:4px 7px; font-size:10px; font-weight:700; } .badge.Low { color:#12603f; background:#dff2e8; } .badge.Moderate { color:#7c5011; background:#fff0cf; } .badge.Severe { color:#8e2e2a; background:#ffe0dc; }
			.system-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; } .system-item { padding:13px; background:#f1f6f4; font-size:11px; } .system-item b { display:block; margin-bottom:6px; color:var(--muted); font-size:10px; letter-spacing:.5px; text-transform:uppercase; } .system-item span { color:var(--low); font-weight:700; }
			.empty { color:var(--muted); font-size:13px; } .error { color:#a53731; font-size:12px; }
			@media (max-width:1000px) { .kpis { grid-template-columns:repeat(3,1fr); } .assessment,.three-col { grid-template-columns:1fr; } .result-panel { position:static; } }
			@media (max-width:720px) { .shell { padding:12px; } .two-col { grid-template-columns:1fr; } .kpis,.system-grid { grid-template-columns:repeat(2,1fr); } .form-grid { grid-template-columns:1fr; } .header-meta { align-items:stretch; flex-direction:column; } .donut-wrap { flex-direction:column; align-items:flex-start; } }
		</style>
	</head>
	<body>
		<main class="shell">
		<header id="overview">
			<p class="eyebrow">Disaster response coordination / Stage 01</p>
			<h1>Disaster Response Intelligence Center</h1>
			<p class="subtitle">A single-page command view for flood-risk patterns, model intelligence, and zone-level assessment.</p>
			<div class="header-meta"><span class="status-pill" id="system-status">SYSTEM ONLINE</span><span class="status-pill" id="model-status">MODEL STATUS</span><span id="clock">--</span><button class="refresh" id="refresh">Refresh data</button></div>
			<nav class="nav"><a href="#overview">Overview</a><a href="#analytics">Analytics</a><a href="#prediction">Prediction</a><a href="#model">Model</a><a href="#history">History</a><a href="#system">System</a></nav>
		</header>

		<section class="section" aria-label="KPI summary"><div class="section-head"><div><p class="kicker">Current situation</p><h2>Executive summary</h2></div><p id="data-updated">Loading artifact-backed data...</p></div><div class="kpis">
			<div class="kpi"><div class="kpi-label">Total records</div><div class="kpi-value" id="kpi-total">--</div><div class="kpi-note">EDA-approved dataset</div></div>
			<div class="kpi risk-severe"><div class="kpi-label">High risk</div><div class="kpi-value" id="kpi-severe">--</div><div class="kpi-note">Severe classification</div></div>
			<div class="kpi risk-moderate"><div class="kpi-label">Medium risk</div><div class="kpi-value" id="kpi-moderate">--</div><div class="kpi-note">Moderate classification</div></div>
			<div class="kpi risk-low"><div class="kpi-label">Low risk</div><div class="kpi-value" id="kpi-low">--</div><div class="kpi-note">Low classification</div></div>
			<div class="kpi"><div class="kpi-label">Model accuracy</div><div class="kpi-value" id="kpi-accuracy">--</div><div class="kpi-note">Independent evaluation</div></div>
			<div class="kpi"><div class="kpi-label">Predictions</div><div class="kpi-value" id="kpi-predictions">--</div><div class="kpi-note">Available prediction records</div></div>
		</div></section>

		<section class="section two-col" id="analytics"><article class="panel"><h3>Risk distribution</h3><p class="panel-sub">Actual class balance in the available dataset.</p><div class="donut-wrap"><div class="donut" id="risk-donut"></div><div class="legend"><div class="legend-row severe"><span>High / Severe</span><b id="legend-severe">--</b></div><div class="legend-row moderate"><span>Medium / Moderate</span><b id="legend-moderate">--</b></div><div class="legend-row low"><span>Low</span><b id="legend-low">--</b></div></div></div></article><article class="panel" id="model"><h3>Model performance</h3><p class="panel-sub">Held-out evaluation metrics from the existing pipeline.</p><div id="metrics"></div></article></section>

		<section class="section three-col"><article class="panel"><h3>Risk trend</h3><p class="panel-sub">Monthly record distribution from the source timestamps.</p><div class="trend" id="trend"><p class="empty">Loading trend...</p></div></article><article class="panel"><h3>Risk by region</h3><p class="panel-sub">Districts ranked by observed severe-risk rate.</p><div class="bars" id="district-bars"></div></article><article class="panel"><h3>Confusion matrix</h3><p class="panel-sub">Actual rows versus predicted columns.</p><div class="matrix" id="confusion"></div></article></section>

		<section class="section two-col"><article class="panel"><h3>Top risk factors</h3><p class="panel-sub">Global feature importance from the persisted model artifact.</p><div class="bars" id="importance"></div></article><article class="panel"><h3>Model intelligence</h3><p class="panel-sub">Selected model and evaluation record.</p><div class="model-grid" id="model-info"></div><div id="model-comparison"></div></article></section>

		<section class="section" id="prediction"><div class="section-head"><div><p class="kicker">Decision support</p><h2>Live risk assessment</h2></div><p>Enter available observations to run the existing model.</p></div><div class="assessment"><article class="panel input-panel"><h3>Assessment inputs</h3><p class="panel-sub">Use the same fields consumed by the Stage 01 integration engine.</p><form id="prediction-form">
			<div class="form-section"><h4>Zone and time</h4><div class="form-grid"><div class="field"><label for="district">District</label><input id="district" name="district" value="Mysuru" required></div><div class="field"><label for="state">State</label><input id="state" name="state" value="Karnataka" required></div><div class="field"><label for="timestamp">Observation time</label><input id="timestamp" name="timestamp" value="19-10-2025 16:00" placeholder="DD-MM-YYYY HH:MM" required></div></div></div>
			<div class="form-section"><h4>Environmental conditions</h4><div class="form-grid"><div class="field"><label for="rainfall_mm">Rainfall (mm)</label><input id="rainfall_mm" name="rainfall_mm" type="number" step="any" value="120" required></div><div class="field"><label for="river_level_m">River level (m)</label><input id="river_level_m" name="river_level_m" type="number" step="any" value="4.2" required></div><div class="field"><label for="river_level_threshold_m">River threshold (m)</label><input id="river_level_threshold_m" name="river_level_threshold_m" type="number" step="any" value="3.5" required></div><div class="field"><label for="water_level_change_m">Water level change (m)</label><input id="water_level_change_m" name="water_level_change_m" type="number" step="any" value="0.8" required></div></div></div>
			<div class="form-section"><h4>Incident context</h4><div class="form-grid"><div class="field"><label for="emergency_calls">Emergency calls</label><input id="emergency_calls" name="emergency_calls" type="number" min="0" value="40" required></div><div class="field"><label for="road_closures">Road closures</label><input id="road_closures" name="road_closures" type="number" min="0" value="2" required></div><div class="field"><label for="bridge_closures">Bridge closures</label><input id="bridge_closures" name="bridge_closures" type="number" min="0" value="1" required></div><div class="field"><label for="flood_history_count">Flood history count</label><input id="flood_history_count" name="flood_history_count" type="number" min="0" value="3" required></div><div class="field"><label for="population_affected">Population affected</label><input id="population_affected" name="population_affected" type="number" min="0" value="5000" required></div></div></div>
			<div class="actions"><button type="submit">Run risk assessment</button><button type="reset" class="secondary">Clear</button></div></form></article><article class="panel result-panel"><div class="result-top"><span>Live model output</span><span id="last-updated">Waiting</span></div><div class="risk-output"><h4>Risk assessment</h4><strong id="risk-label">Awaiting input</strong><p id="risk-details">Run an assessment to receive a classification.</p></div><div class="confidence"><div class="confidence-line"><span>Confidence score</span><b id="confidence-value">N/A</b></div><div class="confidence-track"><i id="confidence-bar"></i></div></div><div class="decision"><b>Decision support</b><p id="decision-text">No guidance available until an assessment is completed.</p><p class="disclaimer">Decision-support guidance, not an emergency directive.</p></div></article></div></section>

		<section class="section" id="history"><div class="section-head"><div><p class="kicker">Audit trail</p><h2>Recent predictions</h2></div><button class="refresh" id="history-refresh">Refresh history</button></div><article class="panel history"><table><thead><tr><th>Timestamp</th><th>District</th><th>Risk</th><th>Confidence</th><th>Top factors</th></tr></thead><tbody id="history-body"><tr><td colspan="5" class="empty">Loading prediction history...</td></tr></tbody></table></article></section>

		<section class="section" id="system"><div class="section-head"><div><p class="kicker">Operations</p><h2>System status</h2></div></div><article class="panel"><div class="system-grid" id="system-grid"><div class="system-item"><b>Integration engine</b><span>Checking...</span></div></div></article></section>
		</main>
		<script>
			const $ = (id) => document.getElementById(id);
			const pct = (value) => value == null ? 'N/A' : (value * 100).toFixed(1) + '%';
			const safe = (value) => value == null ? 'N/A' : value;
			function renderMetrics(data) { const items = [['Accuracy',data.model_metrics.accuracy],['Precision',data.model_metrics.precision],['Recall',data.model_metrics.recall],['F1 score',data.model_metrics.f1],['ROC-AUC',data.model_metrics.roc_auc]]; $('metrics').innerHTML = items.map(([name,value]) => '<div class="metric"><div class="metric-line"><span>'+name+'</span><b>'+pct(value)+'</b></div><div class="track"><i style="width:'+(value == null ? 0 : value*100)+'%"></i></div></div>').join(''); }
			function renderRisk(data) { const total = data.records || 1; const severe = data.risk_counts.Severe, moderate = data.risk_counts.Moderate, low = data.risk_counts.Low; $('kpi-total').textContent=data.records.toLocaleString(); $('kpi-severe').textContent=severe.toLocaleString(); $('kpi-moderate').textContent=moderate.toLocaleString(); $('kpi-low').textContent=low.toLocaleString(); $('kpi-accuracy').textContent=pct(data.model_metrics.accuracy); $('kpi-predictions').textContent=data.predictions.toLocaleString(); $('legend-severe').textContent=severe.toLocaleString()+' ('+(severe/total*100).toFixed(1)+'%)'; $('legend-moderate').textContent=moderate.toLocaleString()+' ('+(moderate/total*100).toFixed(1)+'%)'; $('legend-low').textContent=low.toLocaleString()+' ('+(low/total*100).toFixed(1)+'%)'; $('risk-donut').style.background='conic-gradient(var(--severe) 0 '+(severe/total*100)+'%, var(--moderate) '+(severe/total*100)+'% '+((severe+moderate)/total*100)+'%, var(--low) '+((severe+moderate)/total*100)+'% 100%)'; }
			function renderTrend(data) { if (!data.trend.length) { $('trend').innerHTML='<p class="empty">No timestamp trend data available.</p>'; return; } const max=Math.max(...data.trend.map(x=>x.Low+x.Moderate+x.Severe),1); $('trend').innerHTML=data.trend.map(x=>'<div class="trend-col"><div class="trend-stack"><i class="low" style="height:'+(x.Low/max*130)+'px"></i><i class="moderate" style="height:'+(x.Moderate/max*130)+'px"></i><i class="severe" style="height:'+(x.Severe/max*130)+'px"></i></div><div class="trend-label">'+x.period+'</div></div>').join(''); }
			function renderBars(id, rows, labelKey, valueKey, suffix) { if (!rows.length) { $(id).innerHTML='<p class="empty">No data available.</p>'; return; } const max=Math.max(...rows.map(x=>x[valueKey] || 0),1); $(id).innerHTML=rows.map(x=>'<div class="bar-line"><span>'+x[labelKey]+'</span><div class="bar-track"><i style="width:'+(x[valueKey]/max*100)+'%"></i></div><b>'+safe(x[valueKey])+ (suffix || '')+'</b></div>').join(''); }
			function renderConfusion(data) { const matrix=data.confusion_matrix; if (!matrix.length) { $('confusion').innerHTML='<p class="empty">No confusion matrix available.</p>'; return; } const labels=['','Low','Moderate','Severe']; $('confusion').innerHTML=labels.map((label,ri)=>'<div class="head">'+label+'</div>'+ (ri ? matrix[ri-1].map((value,ci)=>'<div class="'+(ri-1===ci?'diag':'')+'">'+value+'</div>').join('') : '<div class="head">Actual / Predicted</div><div class="head"></div><div class="head"></div>')).join(''); }
			function renderModelInfo(data) { const metrics=data.model_metrics; const rows=[['Selected model',data.model],['Features',safe(data.feature_count)],['Accuracy',pct(metrics.accuracy)],['Precision',pct(metrics.precision)],['Recall',pct(metrics.recall)],['Macro F1',pct(metrics.f1)],['ROC-AUC','N/A']]; $('model-info').innerHTML=rows.map(x=>'<div class="model-stat"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>').join(''); }
			function renderModelComparison(data) { const candidates=data.model_candidates || {}; const names=Object.keys(candidates); if (!names.length) { $('model-comparison').innerHTML=''; return; } $('model-comparison').innerHTML='<table style="margin-top:18px"><thead><tr><th>Candidate</th><th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>'+names.map(name=>'<tr><td>'+name+(name===data.model?' *':'')+'</td><td>'+pct(candidates[name].accuracy)+'</td><td>'+pct(candidates[name].macro_f1)+'</td></tr>').join('')+'</tbody></table><p class="panel-sub">* Selected model in the persisted evaluation.</p>'; }
			function renderHistory(data) { $('history-body').innerHTML=data.history.length ? data.history.map(x=>'<tr><td>'+x.timestamp+'</td><td>'+x.district+'</td><td><span class="badge '+x.risk+'">'+x.risk+'</span></td><td>'+pct(x.confidence)+'</td><td>'+x.factors+'</td></tr>').join('') : '<tr><td colspan="5" class="empty">No prediction history available.</td></tr>'; }
			function renderSystem(data) { $('system-grid').innerHTML=[['Data pipeline',data.records?'CONNECTED':'NO DATA'],['EDA pipeline',data.records?'CONNECTED':'NO DATA'],['ML model',data.model_loaded?'LOADED':'UNAVAILABLE'],['Preprocessing',data.model_loaded?'READY':'UNAVAILABLE'],['Evaluation',data.model_metrics.accuracy == null?'N/A':'AVAILABLE'],['Integration engine','ACTIVE'],['Flask application','RUNNING'],['Last refresh',data.generated_at]].map(x=>'<div class="system-item"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join(''); }
			async function loadDashboard() { try { const response=await fetch('/api/dashboard'); const data=await response.json(); renderRisk(data); renderMetrics(data); renderModelInfo(data); renderModelComparison(data); renderTrend(data); renderBars('district-bars',data.districts,'district','severe_rate','%'); renderBars('importance',data.feature_importance,'feature','importance',''); renderConfusion(data); renderHistory(data); renderSystem(data); $('data-updated').textContent='Artifact data refreshed '+data.generated_at; $('system-status').textContent='SYSTEM ONLINE'; $('model-status').textContent=data.model_loaded?'MODEL READY':'MODEL UNAVAILABLE'; } catch (error) { $('data-updated').textContent='Dashboard data unavailable'; $('system-status').textContent='SYSTEM ERROR'; } }
			$('prediction-form').addEventListener('submit', async (event) => { event.preventDefault(); const button=event.target.querySelector('button[type=submit]'); button.disabled=true; button.textContent='Assessing...'; const data=Object.fromEntries(new FormData(event.target)); Object.keys(data).forEach(key=>{if(!['district','state','timestamp'].includes(key)) data[key]=Number(data[key]);}); try { const response=await fetch('/api/ml/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}); const payload=await response.json(); if(!response.ok || !payload.result) throw new Error(payload.error || 'Assessment failed'); const result=payload.result; $('risk-label').textContent=result.risk_category; $('risk-label').className=result.risk_category.toLowerCase(); $('confidence-value').textContent=pct(result.confidence); $('confidence-bar').style.width=(result.confidence*100)+'%'; $('risk-details').textContent='Zone: '+result.zone+' | Factors: '+result.top_factors.join(', '); $('decision-text').textContent=result.risk_category==='Severe'?'High-priority review recommended based on the model assessment.':result.risk_category==='Moderate'?'Additional monitoring and review recommended.':'Routine monitoring recommended.'; $('last-updated').textContent='Updated '+new Date().toLocaleTimeString(); loadDashboard(); } catch(error) { $('risk-label').textContent='ASSESSMENT FAILED'; $('risk-details').textContent='Unable to generate the prediction. Please verify the supplied inputs.'; } finally { button.disabled=false; button.textContent='Run risk assessment'; } });
			$('prediction-form').addEventListener('reset',()=>{setTimeout(()=>{$('risk-label').textContent='Awaiting input';$('risk-label').className='';$('confidence-value').textContent='N/A';$('confidence-bar').style.width='0';$('risk-details').textContent='Run an assessment to receive a classification.';$('last-updated').textContent='Waiting';},0);}); $('refresh').addEventListener('click',loadDashboard); $('history-refresh').addEventListener('click',loadDashboard); setInterval(()=>{$('clock').textContent=new Date().toLocaleString();},1000); loadDashboard();
		</script>
	</body>
	</html>
	"""


def compact_dashboard_html():
	return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Disaster Response Intelligence Center</title>
<style>
:root{--ink:#e8eef7;--muted:#8ea0b8;--paper:#07111f;--panel:#0d1b2d;--line:#1d3049;--navy:#0b1728;--teal:#4d83ff;--green:#28c99a;--amber:#f4b54a;--red:#ff6b68}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 75% -10%,#18345a 0,#07111f 42%);font:14px Arial,sans-serif}.shell{max-width:1220px;margin:auto;padding:24px 28px 40px}header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding:20px 0;border-bottom:1px solid var(--line)}h1{margin:0;font:700 34px Georgia,serif;letter-spacing:.2px}.eyebrow{margin:0 0 8px;color:#78a5ff;font-size:11px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase}.subtitle{margin:8px 0 0;color:var(--muted);font-size:13px}.header-actions{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px}.online{color:var(--green);font-weight:700}.online:before{content:'●';margin-right:6px}button,.details-link{border:0;padding:10px 13px;color:#fff;background:var(--teal);font-size:12px;font-weight:700;cursor:pointer;text-decoration:none;border-radius:4px}button:hover,.details-link:hover{background:#6f9cff}.refresh{background:#182b48}.section-title{display:flex;justify-content:space-between;align-items:baseline;margin:26px 0 12px}.section-title h2{margin:0;font:700 20px Georgia,serif}.section-title span{color:var(--muted);font-size:12px}.constraint-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.constraint{min-height:150px;padding:17px;background:rgba(13,27,45,.9);border:1px solid var(--line);border-top:3px solid var(--teal);border-left:0;box-shadow:0 10px 25px rgba(0,0,0,.14);border-radius:5px}.constraint.risk{border-top-color:var(--red)}.constraint.model{border-top-color:var(--amber)}.constraint.system{border-top-color:var(--green)}.constraint-name{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.7px;text-transform:uppercase}.key-value{margin:12px 0 8px;color:#fff;font-size:27px;font-weight:700}.constraint-meta{display:flex;justify-content:space-between;align-items:center;gap:8px}.status{font-size:11px;font-weight:700}.status.ready{color:var(--green)}.status.attention{color:var(--amber)}.status.alert{color:var(--red)}.status.neutral{color:var(--muted)}.details-link{padding:0;color:#8fb2ff;background:none;font-size:11px}.details-link:hover{color:#fff;background:none}.footer-row{display:flex;justify-content:space-between;align-items:center;margin:24px 0 8px;color:var(--muted);font-size:12px}.empty{padding:24px;color:var(--muted);text-align:center}.dialog{width:min(720px,calc(100% - 30px));max-height:88vh;padding:0;color:var(--ink);background:var(--panel);border:1px solid var(--line);box-shadow:0 20px 60px rgba(0,0,0,.45)}.dialog::backdrop{background:rgba(2,8,16,.72)}.dialog-head{display:flex;justify-content:space-between;align-items:center;padding:18px 20px;color:#fff;background:#142b48}.dialog-head h3{margin:0;font:700 20px Georgia,serif}.close{padding:4px 9px;background:transparent;font-size:22px}.dialog-body{padding:20px;overflow:auto}.detail-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.detail{padding:12px;background:#13253d;border:1px solid var(--line)}.detail b{display:block;margin-bottom:5px;color:var(--muted);font-size:10px;letter-spacing:.6px;text-transform:uppercase}.detail span{font-size:17px;font-weight:700}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px 7px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-size:10px;text-transform:uppercase}.badge{padding:4px 6px;font-size:10px;font-weight:700}.badge.Low{color:#12603f;background:#dff2e8}.badge.Moderate{color:#7c5011;background:#fff0cf}.badge.Severe{color:#8e2e2a;background:#ffe0dc}.form-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.field label{display:block;margin-bottom:5px;font-size:12px;font-weight:700}.field input{width:100%;padding:11px;color:var(--ink);background:#091729;border:1px solid var(--line);font-size:13px}.form-actions{display:flex;gap:10px;margin-top:18px}.result{margin-top:18px;padding:18px;color:#fff;background:#081525;border:1px solid var(--line)}.result strong{display:block;margin:6px 0;font-size:34px}.result .low{color:#72d6a7}.result .moderate{color:#ffd166}.result .severe{color:#ff8066}.result p{margin:6px 0;color:#c8dadd;font-size:12px}.loading{color:var(--muted);font-size:13px}
@media(max-width:820px){header{display:block}.header-actions{margin-top:16px}.constraint-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.shell{padding:15px}.constraint-grid,.detail-grid,.form-grid{grid-template-columns:1fr}h1{font-size:29px}.header-actions{align-items:stretch;flex-direction:column}.header-actions button{width:100%}}
</style>
</head>
<body>
<main class="shell">
<header><div><p class="eyebrow">Stage 01 / ML intelligence</p><h1>Disaster Response Coordination</h1><p class="subtitle">See what is happening, understand the risk, and check a zone when you need to.</p></div><div class="header-actions"><span class="online" id="system-state">Everything is working</span><span id="clock">--</span><button class="refresh" id="refresh">Refresh</button></div></header>
<section><div class="section-title"><h2>At a glance</h2><span id="updated">Getting the latest information...</span></div><div class="constraint-grid" id="constraints"><div class="loading">Getting things ready...</div></div></section>
<div class="footer-row"><span>Choose a card to see the details behind it.</span><button id="assess">Check a zone</button></div>
</main>
<dialog class="dialog" id="details-dialog"><div class="dialog-head"><h3 id="dialog-title">Details</h3><button class="close" data-close>&times;</button></div><div class="dialog-body" id="dialog-body"></div></dialog>
<dialog class="dialog" id="assessment-dialog"><div class="dialog-head"><h3>Live risk assessment</h3><button class="close" data-close>&times;</button></div><div class="dialog-body"><form id="prediction-form"><div class="form-grid"><div class="field"><label>District</label><input name="district" value="Mysuru" required></div><div class="field"><label>State</label><input name="state" value="Karnataka" required></div><div class="field"><label>Observation time</label><input name="timestamp" value="19-10-2025 16:00" required></div><div class="field"><label>Rainfall (mm)</label><input name="rainfall_mm" type="number" step="any" value="120" required></div><div class="field"><label>River level (m)</label><input name="river_level_m" type="number" step="any" value="4.2" required></div><div class="field"><label>River threshold (m)</label><input name="river_level_threshold_m" type="number" step="any" value="3.5" required></div><div class="field"><label>Water level change (m)</label><input name="water_level_change_m" type="number" step="any" value="0.8" required></div><div class="field"><label>Emergency calls</label><input name="emergency_calls" type="number" value="40" required></div><div class="field"><label>Road closures</label><input name="road_closures" type="number" value="2" required></div><div class="field"><label>Bridge closures</label><input name="bridge_closures" type="number" value="1" required></div><div class="field"><label>Flood history count</label><input name="flood_history_count" type="number" value="3" required></div><div class="field"><label>Population affected</label><input name="population_affected" type="number" value="5000" required></div></div><div class="form-actions"><button type="submit">Run assessment</button><button type="reset" class="refresh">Clear</button></div></form><div class="result" id="assessment-result">Enter observations and run an assessment.</div></div></dialog>
<script>
const $=id=>document.getElementById(id), details=$('details-dialog'), assessment=$('assessment-dialog'); let dashboard=null;
const pct=v=>v==null?'N/A':(v*100).toFixed(1)+'%'; const num=v=>v==null?'N/A':Number(v).toLocaleString();
function card(name,value,status,kind,key,detail){return '<article class="constraint '+kind+'"><div class="constraint-name">'+name+'</div><div class="key-value">'+value+'</div><div class="constraint-meta"><span class="status '+status.class+'">'+status.label+'</span><button class="details-link" data-detail="'+key+'">View details</button></div></article>';}
function render(data){dashboard=data; const c=data.risk_counts, total=data.records||0; $('constraints').innerHTML=[card('Information available',num(total),{label:'Ready to explore',class:'ready'},'','dataset',''),card('Places needing attention',pct(total?c.Severe/total:null),{label:c.Severe?'Worth a closer look':'Looking good',class:c.Severe?'alert':'ready'},'risk','risk',''),card('Risk model',data.model,{label:data.model_loaded?'Ready to help':'Needs attention',class:data.model_loaded?'ready':'alert'},'model','model',''),card('How accurate is it?',pct(data.model_metrics.accuracy),{label:data.model_metrics.accuracy==null?'Not available':'Checked on test data',class:data.model_metrics.accuracy==null?'attention':'ready'},'','accuracy',''),card('What matters most?',data.feature_importance[0]?.feature||'Not available',{label:data.feature_importance.length?'Main signal':'Not available',class:data.feature_importance.length?'ready':'attention'},'','factors',''),card('Checks made so far',num(data.predictions),{label:data.predictions?'Saved for review':'No checks yet',class:data.predictions?'ready':'neutral'},'system','history','')].join(''); $('updated').textContent='Last updated '+data.generated_at; document.querySelectorAll('[data-detail]').forEach(b=>b.onclick=()=>showDetails(b.dataset.detail));}
function detail(title,items,extra=''){ $('dialog-title').textContent=title; $('dialog-body').innerHTML='<div class="detail-grid">'+items.map(x=>'<div class="detail"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join('')+'</div>'+extra; details.showModal(); }
function showDetails(key){const d=dashboard;if(key==='dataset')detail('Dataset coverage',[['Records',num(d.records)],['Low risk',num(d.risk_counts.Low)],['Moderate risk',num(d.risk_counts.Moderate)],['Severe risk',num(d.risk_counts.Severe)],['Predictions',num(d.predictions)],['Last refresh',d.generated_at]]);if(key==='risk')detail('Risk exposure',[['Severe',num(d.risk_counts.Severe)],['Moderate',num(d.risk_counts.Moderate)],['Low',num(d.risk_counts.Low)],['Severe rate',pct(d.records?d.risk_counts.Severe/d.records:null)],['Top district',d.districts[0]?.district||'N/A'],['Top district rate',d.districts[0]?.severe_rate+'%'||'N/A']]);if(key==='model'||key==='accuracy')detail('Model intelligence',[['Selected model',d.model],['Accuracy',pct(d.model_metrics.accuracy)],['Precision',pct(d.model_metrics.precision)],['Recall',pct(d.model_metrics.recall)],['Macro F1',pct(d.model_metrics.f1)],['ROC-AUC','N/A']],'<p class="panel-sub">Evaluation values are loaded from the existing Stage 01 artifacts.</p>');if(key==='factors')detail('Top risk factors',d.feature_importance.slice(0,8).map(x=>[x.feature,x.importance]));if(key==='history'){const rows=d.history.map(x=>'<tr><td>'+x.timestamp+'</td><td>'+x.district+'</td><td><span class="badge '+x.risk+'">'+x.risk+'</span></td><td>'+pct(x.confidence)+'</td></tr>').join('');detail('Prediction history',[], '<div class="table-wrap"><table><thead><tr><th>Timestamp</th><th>District</th><th>Risk</th><th>Confidence</th></tr></thead><tbody>'+rows+'</tbody></table></div>');}}
async function load(){try{const r=await fetch('/api/dashboard');dashboard=await r.json();render(dashboard);$('system-state').textContent=dashboard.model_loaded?'Everything is working':'The model needs attention';}catch(e){$('constraints').innerHTML='<div class="empty">We could not load the dashboard right now.</div>';$('system-state').textContent='Something needs attention';}}
$('refresh').onclick=load;$('assess').onclick=()=>assessment.showModal();document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>b.closest('dialog').close());$('prediction-form').onsubmit=async e=>{e.preventDefault();const data=Object.fromEntries(new FormData(e.target));Object.keys(data).forEach(k=>{if(!['district','state','timestamp'].includes(k))data[k]=Number(data[k]);});const result=$('assessment-result');result.textContent='Assessing...';try{const r=await fetch('/api/ml/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const p=await r.json();if(!r.ok)throw Error();result.innerHTML='<strong class="'+p.result.risk_category.toLowerCase()+'">'+p.result.risk_category+'</strong><p>Confidence: '+pct(p.result.confidence)+' | '+p.result.top_factors.join(', ')+'</p>';load();}catch(e){result.textContent='Assessment failed. Please verify the supplied inputs.';}};setInterval(()=>{$('clock').textContent=new Date().toLocaleString();},1000);load();
</script>
</body></html>
"""


@app.get("/")
def home():
	return compact_dashboard_html()
	return """
	<!doctype html>
	<html lang="en">
	  <head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Disaster Response Coordination | Capstone Dashboard</title>
				<style>
					:root {
						--ink: #17252a;
						--muted: #617477;
						--paper: #eef4f1;
						--panel: #fffdfa;
						--line: #d7e3de;
						--teal: #087f78;
						--teal-dark: #07534f;
						--coral: #d9684b;
						--green: #238b63;
					}

					* { box-sizing: border-box; }
					body {
						margin: 0;
						color: var(--ink);
						background-color: var(--paper);
						background-image: linear-gradient(rgba(8, 127, 120, .045) 1px, transparent 1px), linear-gradient(90deg, rgba(8, 127, 120, .045) 1px, transparent 1px);
						background-size: 32px 32px;
						font-family: Georgia, "Times New Roman", serif;
					}
					.shell { max-width: 1160px; margin: 0 auto; padding: 28px 20px 56px; }
					header { padding: 18px 26px 30px; color: white; background: var(--teal-dark); border-bottom: 5px solid var(--coral); box-shadow: 0 16px 30px rgba(7, 83, 79, .18); }
					.eyebrow {
						margin: 0 0 10px;
						color: #f7b59f;
						font: 700 12px/1.2 Arial, sans-serif;
						letter-spacing: 1.5px;
						text-transform: uppercase;
					}
					h1 { margin: 0; font-size: clamp(32px, 5vw, 58px); line-height: .98; max-width: 680px; letter-spacing: .2px; }
					  .subtitle { margin: 12px 0 0; color: #d7e9e4; font: 15px/1.5 Arial, sans-serif; }
					.workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, .86fr); gap: 18px; margin-top: 24px; align-items: stretch; }
					.panel {
						grid-column: 1 / -1;
						padding: 22px;
						background: rgba(255, 253, 250, .96);
						border: 1px solid var(--line);
						box-shadow: 0 14px 35px rgba(23, 37, 42, .1);
					}
					.input-panel { grid-column: auto; border-top: 6px solid var(--teal); }
					.prediction-panel { grid-column: auto; background: #193238; border-color: #28515a; color: white; }
					.panel h2 { margin: 0 0 18px; font-size: 22px; }
					.prediction-panel h2 { color: white; font-size: 30px; }
					.prediction-panel { position: sticky; top: 20px; }
					.live-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: -4px 0 24px; padding-bottom: 14px; border-bottom: 1px solid #41636a; font: 11px Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; color: #b4cbd0; }
					.live-state { display: inline-flex; align-items: center; gap: 7px; color: var(--green); font-weight: 700; }
					.live-state::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 4px rgba(35, 139, 99, .12); }
					.label { color: var(--muted); font: 11px Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; }
					.value { margin-top: 7px; font: 700 25px Arial, sans-serif; }
					.status-row { display: flex; align-items: center; gap: 10px; }
					.dot { width: 11px; height: 11px; border-radius: 50%; background: var(--orange); }
					.dot.live { background: var(--green); }
					.meta { color: var(--muted); font: 13px/1.5 Arial, sans-serif; word-break: break-word; }
					.form-section { margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--line); }
					.form-section:first-of-type { margin-top: 0; padding-top: 0; border-top: 0; }
					.section-heading { margin: 0 0 12px; font: 700 12px Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; color: var(--teal-dark); }
					.form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
					.field label { display: block; margin-bottom: 6px; font: 700 13px Arial, sans-serif; }
					.field small { display: block; min-height: 30px; margin-top: 5px; color: var(--muted); font: 11px/1.35 Arial, sans-serif; }
					input, button {
						width: 100%;
						padding: 12px;
						border: 1px solid var(--line);
						border-radius: 2px;
						font: 14px Arial, sans-serif;
					}
					input:focus { outline: 2px solid #9bd7ce; border-color: var(--teal); }
					input::placeholder { color: #9aa8bb; }
					.actions { display: flex; gap: 12px; margin-top: 24px; }
					button { color: white; background: var(--teal); border-color: var(--teal); cursor: pointer; font-weight: 700; }
					button:hover { background: var(--teal-dark); }
					button.secondary { color: var(--teal-dark); background: transparent; border-color: var(--line); }
					button.secondary:hover { background: #eef5fb; }
					  .prediction-result { margin-top: 16px; padding: 18px; background: #14213d; color: white; min-height: 110px; }
					  .prediction-result h3 { margin: 0 0 8px; font: 12px Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; color: #9cc9e8; }
					  .risk-label { margin: 0; font: 700 34px Arial, sans-serif; }
					  .risk-label.low { color: #72d6a7; }
					  .risk-label.moderate { color: #ffd166; }
					  .risk-label.severe { color: #ff8066; }
					  .risk-details { margin: 8px 0 0; color: #d9e7f0; font: 13px/1.5 Arial, sans-serif; }
					  pre { margin: 16px 0 0; padding: 14px; overflow: auto; color: #d9f4ff; background: #14213d; font: 12px/1.5 Consolas, monospace; min-height: 76px; }
					@media (max-width: 680px) {
						.workspace { grid-template-columns: 1fr; }
						.prediction-panel { position: static; }
					}
					@media (max-width: 760px) {
						.form-grid { grid-template-columns: 1fr; }
						.actions { flex-direction: column; }
					}
				</style>
	  </head>
	  <body>
				<main class="shell">
					<header>
						<div>
							<p class="eyebrow">Emergency intelligence</p>
							<h1>Zone risk prediction</h1>
						</div>
						<p class="subtitle">Enter current conditions to predict the disaster risk category for a zone.</p>
					</header>
					<section class="workspace">
						<article class="panel input-panel">
							<h2>Input console</h2>
							<form id="prediction-form">
								<div class="form-section">
									<h3 class="section-heading">Zone and time</h3>
									<div class="form-grid">
										<div class="field"><label for="district">District</label><input id="district" name="district" placeholder="Example: Mysuru" value="Mysuru" required><small>Name of the affected district.</small></div>
										<div class="field"><label for="state">State</label><input id="state" name="state" placeholder="Example: Karnataka" value="Karnataka" required><small>State containing the district.</small></div>
										<div class="field"><label for="timestamp">Observation time</label><input id="timestamp" name="timestamp" placeholder="DD-MM-YYYY HH:MM" value="19-10-2025 16:00" required><small>Example: 19-10-2025 16:00.</small></div>
									</div>
								</div>
								<div class="form-section">
									<h3 class="section-heading">Weather and river conditions</h3>
									<div class="form-grid">
										<div class="field"><label for="rainfall_mm">Rainfall</label><input id="rainfall_mm" name="rainfall_mm" type="number" min="0" step="any" placeholder="Example: 120" value="120" required><small>Rainfall measured in millimetres.</small></div>
										<div class="field"><label for="river_level_m">River level</label><input id="river_level_m" name="river_level_m" type="number" min="0" step="any" placeholder="Example: 4.2" value="4.2" required><small>Current river level in metres.</small></div>
										<div class="field"><label for="river_level_threshold_m">River threshold</label><input id="river_level_threshold_m" name="river_level_threshold_m" type="number" min="0" step="any" placeholder="Example: 3.5" value="3.5" required><small>Warning threshold in metres.</small></div>
										<div class="field"><label for="water_level_change_m">Water level change</label><input id="water_level_change_m" name="water_level_change_m" type="number" step="any" placeholder="Example: 0.8" value="0.8" required><small>Recent change in metres; use negative values for a fall.</small></div>
									</div>
								</div>
								<div class="form-section">
									<h3 class="section-heading">Disruption and community impact</h3>
									<div class="form-grid">
										<div class="field"><label for="emergency_calls">Emergency calls</label><input id="emergency_calls" name="emergency_calls" type="number" min="0" step="1" placeholder="Example: 40" value="40" required><small>Number of emergency calls received.</small></div>
										<div class="field"><label for="road_closures">Road closures</label><input id="road_closures" name="road_closures" type="number" min="0" step="1" placeholder="Example: 2" value="2" required><small>Number of closed roads.</small></div>
										<div class="field"><label for="bridge_closures">Bridge closures</label><input id="bridge_closures" name="bridge_closures" type="number" min="0" step="1" placeholder="Example: 1" value="1" required><small>Number of closed bridges.</small></div>
										<div class="field"><label for="flood_history_count">Flood history count</label><input id="flood_history_count" name="flood_history_count" type="number" min="0" step="1" placeholder="Example: 3" value="3" required><small>Recorded flood events for the zone.</small></div>
										<div class="field"><label for="population_affected">Population affected</label><input id="population_affected" name="population_affected" type="number" min="0" step="1" placeholder="Example: 5000" value="5000" required><small>Estimated people currently affected.</small></div>
									</div>
								</div>
								<div class="actions"><button type="submit">Assess zone risk</button><button type="reset" class="secondary">Clear inputs</button></div>
							</form>
						</article>
						<article class="panel prediction-panel">
							<div class="live-bar"><span class="live-state">Live model output</span><span id="last-updated">Waiting for input</span></div>
							<h2>Live risk assessment</h2>
							<div id="prediction-result" class="prediction-result">
								<h3>Predicted risk</h3>
								<p id="risk-label" class="risk-label">Awaiting input</p>
								<p id="risk-details" class="risk-details">Submit the conditions above to receive a Low, Moderate, or Severe assessment.</p>
								document.getElementById('last-updated').textContent = 'Waiting for input';
							</div>
						</article>
					</section>
				</main>
				<script>
					const json = (value) => JSON.stringify(value, null, 2);
					document.getElementById('prediction-form').addEventListener('submit', async (event) => {
						event.preventDefault();
						const data = Object.fromEntries(new FormData(event.target));
						for (const key of Object.keys(data)) {
							if (!['district', 'state', 'timestamp'].includes(key)) data[key] = Number(data[key]);
						}
						const response = await fetch('/api/ml/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
						const payload = await response.json();
						const label = document.getElementById('risk-label');
						const details = document.getElementById('risk-details');
						if (!response.ok || !payload.result) {
							label.textContent = 'Unable to predict';
							label.className = 'risk-label';
							details.textContent = payload.error || 'The prediction request failed.';
							return;
						}
						const result = payload.result;
						label.textContent = result.risk_category;
						label.className = 'risk-label ' + result.risk_category.toLowerCase();
						details.textContent = 'Confidence: ' + (result.confidence * 100).toFixed(1) + '% | Severe risk score: ' + (result.risk_score * 100).toFixed(1) + '% | Factors: ' + result.top_factors.join(', ');
										document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
					});
					document.getElementById('prediction-form').addEventListener('reset', () => {
									const label = document.getElementById('risk-label');
									label.textContent = 'Awaiting input';
									label.className = 'risk-label';
									document.getElementById('risk-details').textContent = 'Submit the conditions above to receive a Low, Moderate, or Severe assessment.';
						document.getElementById('last-updated').textContent = 'Waiting for input';
								});
				</script>
	  </body>
	</html>
	"""


@app.get("/api/ml/health")
def health():
	return jsonify(integration_engine.health_check())


@app.get("/api/ml/model-info")
def model_info():
	try:
		return jsonify(integration_engine.get_model_info())
	except RuntimeError as exc:
		return jsonify({"success": False, "error": str(exc)}), 503


@app.post("/api/ml/predict")
def predict():
	try:
		data = request.get_json(silent=True)
		if not data:
			return jsonify({"success": False, "error": "No input data provided"}), 400
		result = integration_engine.predict(data)
		SESSION_HISTORY.insert(0, {
			"timestamp": result.get("timestamp", "N/A"),
			"district": result.get("zone", "N/A"),
			"risk": result.get("risk_category", "N/A"),
			"confidence": result.get("confidence"),
			"factors": ", ".join(result.get("top_factors", [])),
		})
		return jsonify({"success": True, "result": result})
	except ValueError as exc:
		return jsonify({"success": False, "error": str(exc)}), 400
	except RuntimeError as exc:
		return jsonify({"success": False, "error": str(exc)}), 503
	except Exception as exc:
		return jsonify({"success": False, "error": str(exc)}), 500


@app.post("/api/ml/predict-batch")
def predict_batch():
	try:
		data = request.get_json(silent=True)
		if not isinstance(data, dict) or "records" not in data:
			return jsonify({"success": False, "error": "records are required"}), 400
		return jsonify({
			"success": True,
			"result": integration_engine.predict_batch(data["records"]),
		})
	except ValueError as exc:
		return jsonify({"success": False, "error": str(exc)}), 400
	except RuntimeError as exc:
		return jsonify({"success": False, "error": str(exc)}), 503
	except Exception as exc:
		return jsonify({"success": False, "error": str(exc)}), 500


@app.get("/api/stage01/status")
def stage01_status():
	health_result = integration_engine.health_check()
	return jsonify({
		"stage": "Stage 01 - Machine Learning",
		"integration_engineer": "Active",
		"components": {
			"data_engineer": "Integrated",
			"eda_engineer": "Integrated",
			"ml_engineer": "Integrated" if integration_engine.model is not None else "Model Not Loaded",
			"evaluation_engineer": "Integrated" if integration_engine.metrics else "Metrics Not Loaded",
			"flask_application": "Running",
		},
		"health": health_result,
	})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000, debug=True)
