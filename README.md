# Disaster Response Coordination

Stage-01 ML integration foundation for a future autonomous, multi-agent decision engine for urban crises.

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend (second terminal):

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. API docs are available at `http://localhost:8000/docs`.

This release intentionally contains no prediction implementation and no fabricated disaster data. See `docs/INTEGRATION_GUIDE.md` and `integration_contracts/` for the exact handoff requirements.
# DISASTER_RESPONSE_COORDINATION
An autonomous multi-agent AI system for urban disaster response that combines ML, Deep Learning, NLP, SLMs, Generative AI, and Agentic AI to detect hazards, analyze crisis data, predict risks, generate tactical insights, and autonomously coordinate emergency resources with human override controls.
