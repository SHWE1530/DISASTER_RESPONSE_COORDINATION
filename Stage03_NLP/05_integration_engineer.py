"""Integration adapter for the trained Stage03 NLP pipeline.

This module exposes a small, app-friendly API that the existing Flask dashboard can
call without duplicating the Stage03 preprocessing or model logic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
MODEL_MODULE_PATH = BASE_DIR / "03_nlp_engineer.py"


class NLPIntegrationEngine:
    """Thin wrapper around the Stage03 inference engine."""

    def __init__(self, module_path: Path | str = MODEL_MODULE_PATH) -> None:
        self.module_path = Path(module_path)
        self.module = self._load_module()
        self.load_error: str | None = None

    def _load_module(self):
        try:
            spec = importlib.util.spec_from_file_location("stage03_nlp_engineer", self.module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to create loader for {self.module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            self.load_error = str(exc)
            return None

    @property
    def ready(self) -> bool:
        return self.module is not None and self.load_error is None

    def health_check(self) -> dict[str, Any]:
        """Report readiness by running a real analysis, not just by importing.

        This used to return "healthy" whenever the module imported. That is why
        the dashboard advertised "Stage 03 API (NLP): Online" while every single
        prediction failed: importing the module says nothing about whether the
        model artifacts load and score under the installed library versions.
        """
        if not self.ready:
            return {
                "status": "unavailable",
                "model_loaded": False,
                "inference_ok": False,
                "model_path": str(self.module_path),
                "error": self.load_error,
            }

        inference_ok = True
        inference_error: str | None = None
        backend = None
        try:
            probe = self.module.analyze_text(
                "Flooding reported near the main bridge. 5 people affected."
            )
            backend = probe.get("urgency_backend")
            if probe.get("urgency_level") not in self.module.URGENCY_CLASSES:
                raise RuntimeError(
                    f"Urgency probe returned an unexpected label: {probe.get('urgency_level')!r}"
                )
        except Exception as exc:
            inference_ok = False
            inference_error = f"{type(exc).__name__}: {exc}"

        return {
            "status": "healthy" if inference_ok else "degraded",
            "model_loaded": True,
            "inference_ok": inference_ok,
            "urgency_backend": backend,
            "model_path": str(self.module_path),
            "error": self.load_error or inference_error,
        }

    def analyze(self, text: str) -> dict[str, Any]:
        if self.module is None:
            raise RuntimeError(self.load_error or "Stage03 NLP model could not be loaded.")

        if text is None:
            raise ValueError("Please enter an emergency message.")

        cleaned = str(text).strip()
        if not cleaned:
            return {
                "status": "error",
                "message": "Please enter an emergency message.",
                "urgency": None,
                "hazard_type": None,
                "confidence": 0.0,
                "hazard_confidence": 0.0,
                "location": None,
                "resource_needed": [],
                "headcount": None,
                "entities": {"location": None, "resource_needed": [], "headcount": None},
            }

        try:
            result = self.module.analyze_text(cleaned)
        except Exception as exc:
            raise RuntimeError(f"NLP inference failed: {exc}") from exc

        entities = result.get("entities", {}) or {}
        location = entities.get("location")
        if isinstance(location, list) and location:
            location = location[0]
        resource_needed = entities.get("resource_needed") or []
        headcount = entities.get("headcount")

        return {
            "status": "ok",
            "message": "NLP analysis completed.",
            "urgency": result.get("urgency_level"),
            "hazard_type": result.get("hazard_type"),
            "confidence": float(result.get("urgency_confidence") or 0.0),
            "hazard_confidence": float(result.get("hazard_confidence") or 0.0),
            "location": location,
            "resource_needed": resource_needed,
            "headcount": headcount,
            "entities": entities,
            "raw": result,
        }


nlp_integration_engine = NLPIntegrationEngine()
