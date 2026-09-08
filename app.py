import os
import sys
import datetime
import importlib.util
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    stage01_api = load_module("stage01_api", BASE_DIR / "Stage01_ML" / "05_integration_engineer.py")
    stage01_engine = stage01_api.integration_engine
    stage01_status = "Online"
except Exception as e:
    stage01_engine = None
    stage01_status = "Offline"
    print(f"Failed to load ML API: {e}")

try:
    stage02_api = load_module("stage02_api", BASE_DIR / "Stage02_DL" / "05_integration_engineer.py")
    stage02_engine = stage02_api.dl_integration_engine
    stage02_status = "Online"
except Exception as e:
    stage02_engine = None
    stage02_status = "Offline"
    print(f"Failed to load DL API: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = BASE_DIR / "Stage02_DL" / "data" / "raw"

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
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-brand">
            <div><h1>AI Platform</h1><p>Build • Train • Deploy</p></div>
        </div>

        <a class="nav-item active" data-target="view-dashboard"><div class="nav-icon">🏠</div> Dashboard</a>
        <a class="nav-item" data-target="view-ml"><div class="nav-icon" style="color: var(--accent-green)">⚙️</div> ML (Stage 01)</a>
        <a class="nav-item" data-target="view-dl"><div class="nav-icon" style="color: var(--accent-blue)">⚡</div> DL (Stage 02)</a>
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
                <p style="margin-top: 15px; color: var(--text-muted);">Please use the navigation menu on the left to access the ML prediction forms and the DL image/forecasting tools.</p>
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
                    Confidence: <span id="ml-conf"></span>%<br>
                    <small id="ml-json" style="color: var(--text-muted); display: block; margin-top: 10px;"></small>
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
                
                const resBox = document.getElementById('ml-result');
                const badge = document.getElementById('ml-badge');
                
                resBox.style.display = 'block';
                badge.innerText = data.risk_category || "Error";
                badge.className = "badge badge-" + data.risk_category;
                
                document.getElementById('ml-conf').innerText = data.confidence ? (data.confidence * 100).toFixed(2) : "N/A";
                document.getElementById('ml-json').innerText = JSON.stringify(data.top_factors || []);
            } catch (err) {
                alert("Error calling ML API");
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

        // LSTM Chart Setup
        Chart.defaults.color = '#94A3B8';
        Chart.defaults.font.family = 'Inter';
        let lstmChart = null;

        document.getElementById('btn-lstm').addEventListener('click', async (e) => {
            const btn = e.target;
            btn.innerText = "Simulating...";

            // Send dummy sequence (72 hours of water levels)
            // Start around 8.0, slowly rising, then the last 24 match the old array roughly.
            const inputSeq = Array.from({length: 48}, (_, i) => 8.0 + (i * 1.5 / 48)).concat([
                9.5, 9.6, 9.7, 9.9, 10.2, 10.4, 10.8, 11.1, 11.3, 11.5, 11.7, 12.0, 
                12.4, 12.8, 13.0, 13.2, 13.5, 13.8, 14.0, 14.2, 14.4, 14.5, 14.7, 15.0
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
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, stage01=stage01_status, stage02=stage02_status)

@app.route('/api/predict/ml', methods=['POST'])
def predict_ml():
    if not stage01_engine:
        return jsonify({"error": "Stage 01 API offline"}), 500
    try:
        data = request.json
        result = stage01_engine.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/dl/image', methods=['POST'])
def predict_dl_image():
    if not stage02_engine:
        return jsonify({"error": "Stage 02 API offline"}), 500
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save securely into expected local path for Stage02
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "dashboard_upload.jpg")
        file.save(filepath)
        
        # Run inference
        result = stage02_engine.predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/dl/lstm', methods=['POST'])
def predict_dl_lstm():
    if not stage02_engine:
        return jsonify({"error": "Stage 02 API offline"}), 500
    try:
        data = request.json
        sequence = data.get("sequence", [])
        result = stage02_engine.forecast_water_levels(sequence, horizon=6)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Starting Interactive AI Dashboard on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
