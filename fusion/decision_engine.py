"""Cross-stage fusion and decision layer.

Until now the three stages were parallel and independent: Stage 01 scored sensor
telemetry, Stage 02 classified imagery and forecast water levels, Stage 03 read
free text, and *nothing combined them*. An operator had to open three tabs and
reconcile three answers in their head.

This module takes whatever evidence is available for a single incident and
produces one prioritised decision, with the provenance of every contributing
signal and an explicit conflict report.

WHAT THIS IS
------------
A **deterministic, auditable decision policy** -- not a learned model.

That is a deliberate choice, not a shortcut. Learning a fusion model would
require a labelled corpus of incidents where sensor readings, imagery, water
levels and a text report all describe the SAME event and carry a known ground
truth outcome. No such joint dataset exists here, and inventing one would put a
trained-looking number on top of fabricated supervision -- exactly the defect
this project already had to remove from Stage 03.

Every weight and escalation rule below is a documented policy constant that a
domain expert can inspect and change. In a safety-critical system that
auditability is worth more than a marginally better fitted score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Common severity scale
# ---------------------------------------------------------------------------
# Each stage speaks a different label space. They are projected onto one ordinal
# scale so they can be compared at all.
PRIORITY_LEVELS = ["ROUTINE", "ELEVATED", "URGENT", "CRITICAL"]

ML_RISK_TO_LEVEL = {"Low": 0, "Moderate": 1, "Severe": 3}
NLP_URGENCY_TO_LEVEL = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Relative influence of each evidence source on the base score.
#
# Rationale, stated so it can be challenged:
#   - Sensor telemetry (Stage 01) is the most objective signal and the only one
#     validated against held-out ground-truth labels, so it carries the most weight.
#   - Text urgency (Stage 03) is a direct human report of the situation and is
#     weighted almost as highly, but its model is trained on synthetic text, so
#     it is not allowed to dominate.
#   - Visual confirmation (Stage 02 CNN) is narrow -- it answers one binary
#     question -- but a positive is strong corroboration, so it is weighted
#     moderately and additionally drives an escalation rule below.
#   - The forecast (Stage 02 LSTM) is about the FUTURE, not the present, so it
#     informs the score least and mainly triggers a pre-positioning advisory.
EVIDENCE_WEIGHTS = {
    "sensor_risk": 0.35,
    "text_urgency": 0.30,
    "visual_flood": 0.20,
    "forecast_trend": 0.15,
}

# A stage's contribution is scaled by its own confidence, but never below this
# floor -- otherwise a low-confidence CRITICAL report would be silently ignored.
MIN_CONFIDENCE_FLOOR = 0.25

# Confidence below this is treated as weak evidence and flagged for review.
LOW_CONFIDENCE_THRESHOLD = 0.50

# Absolute rise (metres) over the forecast horizon that counts as a rising trend.
FORECAST_RISE_THRESHOLD_M = 0.30

# Disagreement of this many levels or more between two stages is a conflict.
CONFLICT_LEVEL_GAP = 2


@dataclass
class Evidence:
    """One stage's contribution, kept with its provenance."""

    source: str
    available: bool
    level: int | None = None          # 0-3 on the common scale
    confidence: float | None = None   # the stage's own confidence, 0-1
    detail: str = ""
    warnings: list[str] = field(default_factory=list)

    def weighted_level(self) -> float | None:
        if not self.available or self.level is None:
            return None
        confidence = self.confidence if self.confidence is not None else 1.0
        return self.level * max(confidence, MIN_CONFIDENCE_FLOOR)


class DecisionEngine:
    """Combine per-stage outputs into one prioritised, auditable decision."""

    def __init__(self, ml_engine=None, dl_engine=None, nlp_engine=None) -> None:
        self.ml_engine = ml_engine
        self.dl_engine = dl_engine
        self.nlp_engine = nlp_engine

    # -- evidence collection -------------------------------------------------

    def _sensor_evidence(self, sensors: dict[str, Any] | None) -> Evidence:
        if not sensors or self.ml_engine is None:
            return Evidence("sensor_risk", available=False,
                            detail="No sensor readings supplied")
        try:
            result = self.ml_engine.predict(dict(sensors))
        except Exception as exc:
            return Evidence("sensor_risk", available=False,
                            detail=f"Stage 01 unavailable: {type(exc).__name__}",
                            warnings=[f"Sensor scoring failed: {exc}"])

        category = result.get("risk_category")
        confidence = result.get("confidence")
        warnings: list[str] = []
        # Stage 01's confidence is uncalibrated and known to be overconfident in
        # the mid range; say so rather than propagating it as a probability.
        if confidence is not None and 0.55 <= confidence <= 0.85:
            warnings.append(
                "Sensor model confidence falls in its poorly-calibrated mid band; "
                "treat this level as weakly supported."
            )
        return Evidence(
            source="sensor_risk",
            available=True,
            level=ML_RISK_TO_LEVEL.get(category, 1),
            confidence=confidence,
            detail=(
                f"Zone risk '{category}' for {result.get('zone')} "
                f"(drivers: {', '.join(result.get('top_factors', [])[:3])})"
            ),
            warnings=warnings,
        )

    def _text_evidence(self, text: str | None) -> Evidence:
        if not text or not str(text).strip() or self.nlp_engine is None:
            return Evidence("text_urgency", available=False,
                            detail="No text report supplied")
        try:
            result = self.nlp_engine.analyze(str(text))
        except Exception as exc:
            return Evidence("text_urgency", available=False,
                            detail=f"Stage 03 unavailable: {type(exc).__name__}",
                            warnings=[f"Text analysis failed: {exc}"])
        if result.get("status") == "error":
            return Evidence("text_urgency", available=False,
                            detail=result.get("message", "Text could not be analysed"))

        urgency = result.get("urgency")
        confidence = float(result.get("confidence") or 0.0)
        warnings = []
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Text urgency confidence is low ({confidence:.2f}); the message "
                "may be ambiguous, very short, or unlike the training corpus."
            )
        entities = result.get("entities", {}) or {}
        return Evidence(
            source="text_urgency",
            available=True,
            level=NLP_URGENCY_TO_LEVEL.get(urgency, 1),
            confidence=confidence,
            detail=(
                f"Reported urgency '{urgency}', hazard '{result.get('hazard_type')}'"
                + (f", {entities.get('headcount')} people affected"
                   if entities.get("headcount") else "")
            ),
            warnings=warnings,
        )

    def _visual_evidence(self, image_path: str | None) -> Evidence:
        if not image_path or self.dl_engine is None:
            return Evidence("visual_flood", available=False,
                            detail="No imagery supplied")
        try:
            result = self.dl_engine.predict_image(image_path)
        except Exception as exc:
            return Evidence("visual_flood", available=False,
                            detail=f"Stage 02 CNN unavailable: {type(exc).__name__}",
                            warnings=[f"Image analysis failed: {exc}"])

        flooded = result.get("label") == "flooded"
        confidence = float(result.get("confidence") or 0.0)
        return Evidence(
            source="visual_flood",
            available=True,
            # A confirmed flooded scene is strong present-tense evidence (3);
            # a clear scene is weak counter-evidence (0), not proof of safety --
            # one camera does not observe a whole zone.
            level=3 if flooded else 0,
            confidence=confidence,
            detail=(
                f"Imagery classified '{result.get('label')}' "
                f"(P(flooded)={result.get('flooded_probability', 0):.2f})"
            ),
            warnings=(
                []
                if flooded else
                ["Imagery shows no flooding at this vantage point only; it does "
                 "not clear the wider zone."]
            ),
        )

    def _forecast_evidence(self, water_levels: list[float] | None) -> Evidence:
        if not water_levels or self.dl_engine is None:
            return Evidence("forecast_trend", available=False,
                            detail="No water-level history supplied")
        try:
            result = self.dl_engine.forecast_water_levels(list(water_levels), horizon=6)
        except Exception as exc:
            return Evidence("forecast_trend", available=False,
                            detail=f"Stage 02 LSTM unavailable: {type(exc).__name__}",
                            warnings=[f"Forecast failed: {exc}"])

        forecasts = result.get("forecast_water_levels", [])
        if not forecasts:
            return Evidence("forecast_trend", available=False, detail="Empty forecast")

        current = float(water_levels[-1])
        peak = max(forecasts)
        rise = peak - current
        warnings: list[str] = []

        # An out-of-distribution forecast is extrapolation. Do not let it move
        # the decision at all -- report it and stop.
        if not result.get("in_distribution", True):
            warnings.append(result.get("warning", "Forecast input is out of distribution."))
            return Evidence(
                "forecast_trend", available=False,
                detail=f"Forecast suppressed: input outside the model's trained range "
                       f"({result.get('distribution_check', {}).get('max_sigma_from_training_mean')}σ)",
                warnings=warnings,
            )

        expected_error = result.get("expected_mae_at_horizon")
        if expected_error is not None and abs(rise) < expected_error:
            warnings.append(
                f"Projected change ({rise:+.2f} m) is smaller than the model's own "
                f"measured error at this horizon (±{expected_error:.2f} m); treat "
                "the trend as indistinguishable from flat."
            )

        if rise >= FORECAST_RISE_THRESHOLD_M:
            level = 3 if rise >= 3 * FORECAST_RISE_THRESHOLD_M else 2
        elif rise <= -FORECAST_RISE_THRESHOLD_M:
            level = 0
        else:
            level = 1

        return Evidence(
            source="forecast_trend",
            available=True,
            level=level,
            # The forecast is a point estimate with no per-prediction confidence,
            # so it enters at full weight but with the smallest weighting overall.
            confidence=1.0,
            detail=(
                f"Water level {current:.2f} m now, peak {peak:.2f} m projected "
                f"over 6 h ({rise:+.2f} m)"
                + (f", model MAE ±{expected_error:.2f} m" if expected_error else "")
            ),
            warnings=warnings,
        )

    # -- fusion --------------------------------------------------------------

    def assess(
        self,
        sensors: dict[str, Any] | None = None,
        text: str | None = None,
        image_path: str | None = None,
        water_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Fuse all supplied evidence into one decision."""
        evidence = [
            self._sensor_evidence(sensors),
            self._text_evidence(text),
            self._visual_evidence(image_path),
            self._forecast_evidence(water_levels),
        ]
        by_source = {item.source: item for item in evidence}
        available = [item for item in evidence if item.available]

        if not available:
            return {
                "status": "insufficient_evidence",
                "priority": None,
                "score": None,
                "message": (
                    "No usable evidence was supplied or every stage failed. "
                    "Provide at least one of: sensor readings, a text report, "
                    "imagery, or a water-level history."
                ),
                "evidence": [self._serialise(item) for item in evidence],
                "human_review_required": True,
                "conflicts": [],
                "recommended_actions": ["Escalate to a human operator: no automated assessment possible."],
            }

        # Weighted mean over CONTRIBUTING sources only, renormalised so a missing
        # stage does not silently drag the score toward zero.
        #
        # The forecast is excluded from the base score unless it is actually
        # warning of a rise. It describes the FUTURE, so a flat or falling
        # projection is not evidence that the present situation is calm -- letting
        # it pull the score down would mean a stable-but-flooded zone scores lower
        # than an identical zone with no forecast data at all. When it does warn
        # of a rise it still cannot lower anything, because the escalation rules
        # below only ever raise the priority.
        contributing = [
            item for item in available
            if not (item.source == "forecast_trend" and (item.level or 0) <= 1)
        ]
        non_contributing = [item for item in available if item not in contributing]
        for item in non_contributing:
            item.warnings.append(
                "Forecast is flat or falling, so it does not reduce the present-tense "
                "assessment; it is reported as context only."
            )

        if contributing:
            total_weight = sum(EVIDENCE_WEIGHTS[item.source] for item in contributing)
            weighted_sum = sum(
                EVIDENCE_WEIGHTS[item.source] * item.weighted_level() for item in contributing
            )
            base_score = weighted_sum / total_weight  # 0-3
        else:
            # Only a flat forecast was available: no present-tense evidence at all.
            base_score = float(available[0].level or 0)

        level_index = int(round(base_score))
        escalations: list[str] = []

        # --- escalation rules -------------------------------------------------
        # Deliberately asymmetric: it is far cheaper to over-respond to a false
        # alarm than to under-respond to a real emergency. Every rule can only
        # RAISE the priority, never lower it.

        text_ev = by_source["text_urgency"]
        if text_ev.available and text_ev.level == 3:
            if level_index < 2:
                escalations.append(
                    "A human reported CRITICAL urgency; floor set to URGENT even "
                    "though other evidence is weaker."
                )
            level_index = max(level_index, 2)

        visual_ev = by_source["visual_flood"]
        sensor_ev = by_source["sensor_risk"]
        if visual_ev.available and visual_ev.level == 3 and sensor_ev.available and sensor_ev.level >= 2:
            if level_index < 3:
                escalations.append(
                    "Imagery visually confirms flooding AND sensors indicate high "
                    "risk; corroborated evidence escalates to CRITICAL."
                )
            level_index = 3

        forecast_ev = by_source["forecast_trend"]
        if forecast_ev.available and forecast_ev.level == 3 and level_index < 2:
            escalations.append(
                "Forecast shows a steep rise within 6 hours; raised to URGENT for "
                "pre-positioning."
            )
            level_index = max(level_index, 2)

        level_index = max(0, min(3, level_index))
        priority = PRIORITY_LEVELS[level_index]

        conflicts = self._detect_conflicts(available)
        warnings = [w for item in evidence for w in item.warnings]

        # Human review is mandatory when the machine is least trustworthy.
        review_reasons: list[str] = []
        if conflicts:
            review_reasons.append("Stages disagree materially about severity.")
        if len(available) == 1:
            review_reasons.append("Only one evidence source was available.")
        weak = [
            item.source for item in available
            if item.confidence is not None and item.confidence < LOW_CONFIDENCE_THRESHOLD
        ]
        if weak:
            review_reasons.append(f"Low-confidence evidence from: {', '.join(weak)}.")
        if level_index >= 2:
            review_reasons.append("Priority is URGENT or higher; confirm before committing resources.")

        return {
            "status": "ok",
            "priority": priority,
            "priority_index": level_index,
            "score": round(base_score, 3),
            "agreement": self._agreement(available),
            "evidence": [self._serialise(item) for item in evidence],
            "sources_used": [item.source for item in available],
            "sources_missing": [item.source for item in evidence if not item.available],
            "escalations": escalations,
            "conflicts": conflicts,
            "warnings": warnings,
            "human_review_required": bool(review_reasons),
            "human_review_reasons": review_reasons,
            "recommended_actions": self._actions(priority, by_source),
            "disclaimer": (
                "Automated decision support derived from models trained on limited "
                "and partly synthetic data. This is not a verified assessment of "
                "conditions on the ground and must be confirmed by a human "
                "responder before resources are committed."
            ),
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _serialise(item: Evidence) -> dict[str, Any]:
        return {
            "source": item.source,
            "available": item.available,
            "level": item.level,
            "level_name": PRIORITY_LEVELS[item.level] if item.level is not None else None,
            "confidence": round(item.confidence, 4) if item.confidence is not None else None,
            "detail": item.detail,
            "warnings": item.warnings,
        }

    @staticmethod
    def _detect_conflicts(available: list[Evidence]) -> list[dict[str, Any]]:
        """Report material disagreement instead of averaging it away.

        The forecast is left out: it describes a different point in time, so a
        flat forecast beside a severe present reading is not a contradiction.
        """
        present_tense = [item for item in available if item.source != "forecast_trend"]
        conflicts = []
        for i, first in enumerate(present_tense):
            for second in present_tense[i + 1:]:
                gap = abs((first.level or 0) - (second.level or 0))
                if gap >= CONFLICT_LEVEL_GAP:
                    higher, lower = (
                        (first, second) if (first.level or 0) > (second.level or 0)
                        else (second, first)
                    )
                    conflicts.append({
                        "between": [first.source, second.source],
                        "level_gap": gap,
                        "description": (
                            f"{higher.source} indicates "
                            f"{PRIORITY_LEVELS[higher.level or 0]} while "
                            f"{lower.source} indicates "
                            f"{PRIORITY_LEVELS[lower.level or 0]}."
                        ),
                        "resolution": (
                            "Not auto-resolved. The higher assessment drives the "
                            "priority floor and the disagreement is surfaced for a "
                            "human to adjudicate."
                        ),
                    })
        return conflicts

    @staticmethod
    def _agreement(available: list[Evidence]) -> str:
        # Present-tense sources only, for the same reason as _detect_conflicts.
        levels = [item.level or 0 for item in available if item.source != "forecast_trend"]
        if len(levels) < 2:
            return "single_source"
        spread = max(levels) - min(levels)
        if spread == 0:
            return "unanimous"
        if spread == 1:
            return "broad_agreement"
        return "disputed"

    @staticmethod
    def _actions(priority: str, by_source: dict[str, Evidence]) -> list[str]:
        actions = {
            "ROUTINE": ["Continue routine monitoring.", "No field deployment required."],
            "ELEVATED": [
                "Alert field teams and place them on standby.",
                "Pre-position pumps and barriers in historically vulnerable zones.",
            ],
            "URGENT": [
                "Deploy a field team to verify conditions on site.",
                "Prepare evacuation routes and notify local authorities.",
                "Confirm the assessment before committing major resources.",
            ],
            "CRITICAL": [
                "Immediate full-scale deployment.",
                "Initiate evacuation protocols for the affected population.",
                "Escalate to central command and coordinate road/bridge closures.",
            ],
        }[priority]

        text_ev = by_source.get("text_urgency")
        if text_ev and text_ev.available and "people affected" in text_ev.detail:
            actions.append("Dispatch medical/rescue capacity sized to the reported headcount.")

        forecast_ev = by_source.get("forecast_trend")
        if forecast_ev and forecast_ev.available and (forecast_ev.level or 0) >= 2:
            actions.append("Water level is projected to rise; stage resources ahead of the peak.")

        return actions
