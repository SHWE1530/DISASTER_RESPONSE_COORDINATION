"""Stage 04 SLM — Integration Adapter.

Exposes a thin, app-friendly API so the Flask dashboard can call the Stage 04
SLM without duplicating model-loading or preprocessing logic.

Mirrors the structure of Stage03_NLP/05_integration_engineer.py.

Usage (in app.py or any Flask route)::

    from Stage04_SLM.05_integration_engineer import slm_integration_engine
    result = slm_integration_engine.summarize(report_text)

Public API
----------
SLMIntegrationEngine.health_check()     → dict
SLMIntegrationEngine.summarize(text)    → dict
SLMIntegrationEngine.batch_summarize()  → list[dict]

slm_integration_engine                  → module-level singleton
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
SLM_MODULE_PATH = BASE_DIR / "03_slm_engineer.py"

BASELINE_DIR = BASE_DIR / "data" / "models" / "slm_baseline"
QWEN_DIR = BASE_DIR / "data" / "models" / "qwen_slm_qlora"

# Fixed probe report used by health_check
_PROBE_REPORT = (
    "INCIDENT LOG | Maharashtra / Pune / ZONE-3 | 6 entries\n"
    "[10-09-2026 08:00] ERSS-000001 | Severe flooding near river bank. 50 people affected. "
    "Rescue Emergency. HIGH severity.\n"
    "[10-09-2026 08:30] ERSS-000002 | Water levels rising. Medical Emergency. 30 more affected.\n"
    "[10-09-2026 09:00] ERSS-000003 | Access road blocked. Evacuation needed. CRITICAL.\n"
    "[10-09-2026 09:30] ERSS-000004 | NDRF requested. Rescue ongoing. 80 total affected.\n"
    "[10-09-2026 10:00] ERSS-000005 | Helicopter support needed for rooftop rescue.\n"
    "[10-09-2026 10:30] ERSS-000006 | State EOC notified. Full deployment in progress.\n"
)

VALID_PRIORITIES = {"ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE"}


class SLMIntegrationEngine:
    """Thin wrapper around the Stage 04 SLM inference engine.

    Automatically selects the best available model:
      1. Qwen2.5-3B-Instruct QLoRA adapter (if present)
      2. TF-IDF baseline (always available after training)

    Both expose the same generate() interface via the SLM module.
    """

    def __init__(self,
                 module_path: Path | str = SLM_MODULE_PATH,
                 prefer_qwen: bool = True) -> None:
        self.module_path = Path(module_path)
        self.prefer_qwen = prefer_qwen
        self.load_error: str | None = None
        self._model: Any = None
        self._model_name: str = "unloaded"
        self._module: Any = None

        self._module = self._load_module()
        if self._module is not None:
            self._model, self._model_name = self._load_best_model()

    # ---- Module loading ----------------------------------------------------

    def _load_module(self) -> Any:
        try:
            spec = importlib.util.spec_from_file_location(
                "stage04_slm_engineer", self.module_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot create loader for {self.module_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return None

    # ---- Model selection ----------------------------------------------------

    def _load_best_model(self) -> tuple[Any, str]:
        """Try Qwen adapter first; fall back to baseline."""
        if self.prefer_qwen and QWEN_DIR.exists():
            manifest_path = QWEN_DIR / "training_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    base_id = manifest.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
                    model = self._module.QwenSLM(QWEN_DIR, base_id)
                    return model, "qwen2.5-3b-instruct-qlora"
                except Exception as exc:
                    self.load_error = f"Qwen load failed: {exc}; falling back to baseline."

        # Baseline
        if BASELINE_DIR.exists():
            try:
                model = self._module.SLMBaseline.load(BASELINE_DIR)
                return model, "slm_baseline"
            except Exception as exc:
                self.load_error = str(exc)
                return None, "unloaded"

        self.load_error = (
            f"No trained model found. "
            f"Run `python {self.module_path.name}` first."
        )
        return None, "unloaded"

    # ---- Public properties --------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._model is not None and self.load_error is None

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---- health_check -------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Report readiness via a real inference probe.

        Returns a dict with at minimum:
          status          "healthy" | "degraded" | "unavailable"
          model_loaded    bool
          inference_ok    bool
          model_name      str
          error           str | None
        """
        if self._module is None:
            return {
                "status": "unavailable",
                "model_loaded": False,
                "inference_ok": False,
                "model_name": self._model_name,
                "model_path": str(self.module_path),
                "error": self.load_error,
            }

        if self._model is None:
            return {
                "status": "unavailable",
                "model_loaded": False,
                "inference_ok": False,
                "model_name": self._model_name,
                "model_path": str(self.module_path),
                "error": self.load_error or "Model not loaded.",
            }

        inference_ok = True
        inference_error: str | None = None
        probe_priority: str | None = None

        try:
            probe = self._model.generate(_PROBE_REPORT)
            situation = probe.get("situation", "")
            # Validate the probe produced a recognisable priority keyword
            found = any(p in situation.upper() for p in VALID_PRIORITIES)
            if not found:
                raise RuntimeError(
                    f"Probe output missing priority keyword: {situation[:80]!r}"
                )
            probe_priority = probe.get("priority") or next(
                (p for p in VALID_PRIORITIES if p in situation.upper()), None
            )
        except Exception as exc:
            inference_ok = False
            inference_error = f"{type(exc).__name__}: {exc}"

        return {
            "status": "healthy" if inference_ok else "degraded",
            "model_loaded": True,
            "inference_ok": inference_ok,
            "model_name": self._model_name,
            "model_path": str(self.module_path),
            "probe_priority": probe_priority,
            "error": self.load_error or inference_error,
        }

    # ---- summarize ----------------------------------------------------------

    def summarize(self, report_text: str) -> dict[str, Any]:
        """Generate a 3-part structured briefing from a disaster incident report.

        Parameters
        ----------
        report_text : str
            Multi-entry incident log string, as produced by Stage 03 / Stage 04
            data pipeline.

        Returns
        -------
        dict with keys:
            status          "ok" | "error"
            message         human-readable status
            situation       str   — priority, hazard, location, scale
            risk            str   — severity assessment, escalation note
            actions         list[str]  — numbered recommended actions
            priority        str   — ROUTINE / ELEVATED / URGENT / IMMEDIATE
            model           str   — model name
            latency_ms      float — end-to-end inference latency
            raw             dict  — full model output
        """
        if self._model is None:
            return {
                "status": "error",
                "message": self.load_error or "SLM model not loaded. Run 03_slm_engineer.py first.",
                "situation": None,
                "risk": None,
                "actions": [],
                "priority": None,
                "model": self._model_name,
                "latency_ms": 0.0,
                "raw": {},
            }

        text = (report_text or "").strip()
        if not text:
            return {
                "status": "error",
                "message": "Please provide a non-empty incident report.",
                "situation": None,
                "risk": None,
                "actions": [],
                "priority": None,
                "model": self._model_name,
                "latency_ms": 0.0,
                "raw": {},
            }

        try:
            t0 = time.time()
            raw = self._model.generate(text)
            latency_ms = round((time.time() - t0) * 1000, 1)

            priority = raw.get("priority") or _extract_priority(raw.get("situation", ""))
            actions = raw.get("actions") or []

            return {
                "status": "ok",
                "message": "SLM briefing generated successfully.",
                "situation": raw.get("situation", ""),
                "risk": raw.get("risk", ""),
                "actions": actions,
                "priority": priority,
                "model": self._model_name,
                "latency_ms": latency_ms,
                "raw": raw,
            }

        except Exception as exc:
            return {
                "status": "error",
                "message": f"SLM inference failed: {type(exc).__name__}: {exc}",
                "situation": None,
                "risk": None,
                "actions": [],
                "priority": None,
                "model": self._model_name,
                "latency_ms": 0.0,
                "raw": {},
            }

    # ---- batch_summarize ----------------------------------------------------

    def batch_summarize(self, reports: list[str]) -> list[dict[str, Any]]:
        """Run summarize() over a list of report strings."""
        return [self.summarize(r) for r in reports]


# ---------------------------------------------------------------------------
# Priority extraction helper (used by summarize when raw.priority is absent)
# ---------------------------------------------------------------------------

def _extract_priority(situation: str) -> str | None:
    for p in ("IMMEDIATE", "URGENT", "ELEVATED", "ROUTINE"):
        if p in (situation or "").upper():
            return p
    return None


# ---------------------------------------------------------------------------
# Module-level singleton — import this in app.py
# ---------------------------------------------------------------------------

slm_integration_engine = SLMIntegrationEngine()


# ---------------------------------------------------------------------------
# Self-test (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 04 SLM Integration Engine — Self-Test")
    print("=" * 60)

    health = slm_integration_engine.health_check()
    print(f"\nHealth check:")
    for k, v in health.items():
        print(f"  {k:20s}: {v}")

    if health["inference_ok"]:
        print("\nRunning probe summarize()...")
        result = slm_integration_engine.summarize(_PROBE_REPORT)
        print(f"\n  status    : {result['status']}")
        print(f"  model     : {result['model']}")
        print(f"  priority  : {result['priority']}")
        print(f"  latency   : {result['latency_ms']} ms")
        print(f"\n  SITUATION : {result['situation']}")
        print(f"  RISK      : {result['risk'][:100]}..." if result["risk"] else "  RISK      : (none)")
        if result["actions"]:
            print(f"  ACTIONS   :")
            for a in result["actions"][:3]:
                print(f"    {a}")
    else:
        print(f"\n[!] Engine not ready: {health['error']}")
        print("    Run `python Stage04_SLM/03_slm_engineer.py` to train the baseline.")
