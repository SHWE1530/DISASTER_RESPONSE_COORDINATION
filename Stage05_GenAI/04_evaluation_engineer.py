"""Stage 05 GenAI -- Evaluation Engineer.

Two audits.

A. REALISM AUDIT -- do generated scenarios look like reality where they should?
   * per-feature two-sample KS statistic, CVAE samples vs the real record
   * correlation structure (max |delta r|, and the rainfall-calls r = 0.82)
   * classifier two-sample test (C2ST): a random forest tries to tell real rows
     from synthetic ones. ROC-AUC 0.5 = indistinguishable, 1.0 = trivially fake.
     A naive generator (each field sampled independently per class) is scored
     the same way, so the CVAE is judged against a baseline, not in a vacuum.
   * label fidelity: the independent Stage 01 model scores CVAE samples -- does
     it agree with the class each sample was conditioned on?
   * text fidelity: the independent Stage 03 model reads generated messages --
     does it recover the intended hazard, urgency and headcount? Clean,
     degraded (comms-noise) and implicit-urgency text are scored separately.

B. STRESS TEST -- run the 20-scenario suite (and the wildcard) through the REAL
   pipeline: Stage 01 + Stage 02 (CNN, LSTM) + Stage 03, fused by
   fusion/decision_engine.py, plus a Stage 04 briefing for every incident log.
   Every zone is checked against its ground-truth expectations.

Nothing in Stages 01-04 or fusion/ is modified; their public adapters are called.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.decision_engine import (  # noqa: E402
    FORECAST_RISE_THRESHOLD_M,
    PRIORITY_LEVELS,
    DecisionEngine,
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
SCENARIO_DIR = OUTPUT_DIR / "scenarios"
IMAGERY_DIR = BASE_DIR / "data" / "raw" / "imagery"

SENSOR_REFERENCE_CSV = PROCESSED_DIR / "sensor_reference.csv"
SYNTHETIC_SAMPLES_CSV = OUTPUT_DIR / "synthetic_sensor_samples.csv"
WATER_REFERENCE_JSON = PROCESSED_DIR / "water_level_reference.json"
SUITE_JSON = SCENARIO_DIR / "scenario_suite.json"
WILDCARD_JSON = SCENARIO_DIR / "wildcard_scenario.json"

REALISM_JSON = EVAL_DIR / "realism_report.json"
ZONE_RESULTS_CSV = EVAL_DIR / "stress_test_zone_results.csv"
SCENARIO_RESULTS_CSV = EVAL_DIR / "stress_test_scenarios.csv"
REPORT_JSON = EVAL_DIR / "stress_test_report.json"
REPORT_MD = EVAL_DIR / "stress_test_report.md"
HISTORY_CSV = EVAL_DIR / "stress_test_history.csv"

SEED = 42
SLM_TO_LEVEL = {"ROUTINE": 0, "ELEVATED": 1, "URGENT": 2, "IMMEDIATE": 3}
TEXT_PROBES_PER_CELL = 10
WORDS_PER_MINUTE = 200  # reading speed used for the briefing time-saving estimate


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENAI = _load_module("stage05_genai_engineer", BASE_DIR / "03_genai_engineer.py")


# ===========================================================================
# Stage adapters
# ===========================================================================

class ImageryAwareDLAdapter:
    """Stage 02 adapter that can also read Stage 05's imagery bank.

    Stage 02's own adapter only accepts images inside Stage02_DL/data/raw -- a
    path-traversal guard for dashboard uploads. Copying the bank into Stage 02's
    tree would modify an earlier stage, so this wrapper applies the same guard
    to Stage 05's imagery directory and calls Stage 02's own model function.
    Forecasts pass straight through to the Stage 02 adapter.
    """

    def __init__(self, dl_engine) -> None:
        self.dl = dl_engine

    def predict_image(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()
        try:
            path.relative_to(IMAGERY_DIR.resolve())
        except ValueError as error:
            raise ValueError("image_path must point inside Stage05_GenAI/data/raw/imagery") from error
        if not path.is_file():
            raise ValueError(f"Image file not found: {path.name}")
        if getattr(self.dl, "module", None) is None or not getattr(self.dl, "models_ready", False):
            raise RuntimeError("Stage02 models are unavailable")
        return self.dl.module.predict_image(path)

    def forecast_water_levels(self, water_levels, horizon: int = 6) -> dict[str, Any]:
        return self.dl.forecast_water_levels(water_levels, horizon)

    def health_check(self) -> dict[str, Any]:
        return self.dl.health_check()


class Briefer:
    """Uniform wrapper over Stage 04's baseline model or its Qwen adapter."""

    def __init__(self, generate: Callable[[str], dict], name: str) -> None:
        self._generate = generate
        self.name = name

    def brief(self, incident_log: str) -> dict[str, Any]:
        started = time.perf_counter()
        output = self._generate(incident_log) or {}
        latency_ms = (time.perf_counter() - started) * 1000
        actions = output.get("actions") or []
        text = " ".join([str(output.get("situation") or ""), str(output.get("risk") or ""),
                         *[str(a) for a in actions]])
        return {"priority": output.get("priority"), "latency_ms": round(latency_ms, 1),
                "words": len(text.split()), "situation": output.get("situation")}


def _load_briefer(backend: str) -> tuple[Briefer | None, str]:
    if backend == "none":
        return None, "skipped"
    if backend == "qwen":
        module = _load_module("stage05_eval_slm_api", REPO_ROOT / "Stage04_SLM" / "05_integration_engineer.py")
        engine = module.slm_integration_engine
        return Briefer(engine.summarize, engine.model_name), engine.health_check().get("status", "unknown")
    # Baseline: load Stage 04's model code directly. Importing Stage 04's
    # integration adapter would load Qwen2.5-3B (~10 s per briefing on the
    # reference GPU), turning a 55-zone run into many minutes. Note that
    # importing 03_slm_engineer also installs its DNS override for three
    # Hugging Face CDN hostnames in this process; nothing here contacts them.
    module = _load_module("stage05_eval_slm_engineer", REPO_ROOT / "Stage04_SLM" / "03_slm_engineer.py")
    model = module.SLMBaseline.load(module.BASELINE_DIR)
    return Briefer(model.generate, "slm_baseline"), "healthy"


def load_stage_engines(slm_backend: str = "baseline") -> tuple[dict[str, Any], dict[str, str]]:
    """Load each earlier stage's public adapter; a failed stage becomes None."""
    adapters = {
        "ml": (REPO_ROOT / "Stage01_ML" / "05_integration_engineer.py", "integration_engine"),
        "dl": (REPO_ROOT / "Stage02_DL" / "05_integration_engineer.py", "dl_integration_engine"),
        "nlp": (REPO_ROOT / "Stage03_NLP" / "05_integration_engineer.py", "nlp_integration_engine"),
    }
    engines: dict[str, Any] = {}
    status: dict[str, str] = {}
    for key, (path, attribute) in adapters.items():
        try:
            engine = getattr(_load_module(f"stage05_eval_{key}_api", path), attribute)
            health = engine.health_check().get("status", "unknown")
        except Exception as exc:
            engine, health = None, f"unavailable ({type(exc).__name__}: {exc})"
        if health == "unavailable":
            engine = None
        engines[key], status[key] = engine, health
    if engines["dl"] is not None:
        engines["dl"] = ImageryAwareDLAdapter(engines["dl"])
    try:
        engines["slm"], status["slm"] = _load_briefer(slm_backend)
    except Exception as exc:
        engines["slm"], status["slm"] = None, f"unavailable ({type(exc).__name__}: {exc})"
    return engines, status


# ===========================================================================
# B. Stress test
# ===========================================================================

def check_expectations(expected: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Return every way a fusion result misses its zone's ground truth."""
    failures: list[str] = []
    status = result.get("status")
    if expected.get("status") and status != expected["status"]:
        failures.append(f"status '{status}', expected '{expected['status']}'")
        return failures
    if expected.get("human_review") is True and not result.get("human_review_required"):
        failures.append("human review not requested")
    if status != "ok":
        return failures
    index = result.get("priority_index")
    if expected.get("min_priority") and index < PRIORITY_LEVELS.index(expected["min_priority"]):
        failures.append(f"priority {result.get('priority')} below minimum {expected['min_priority']}")
    if expected.get("max_priority") and index > PRIORITY_LEVELS.index(expected["max_priority"]):
        failures.append(f"priority {result.get('priority')} above maximum {expected['max_priority']}")
    if expected.get("conflict") is True and not result.get("conflicts"):
        failures.append("modality conflict not surfaced")
    if expected.get("conflict") is False and result.get("conflicts"):
        failures.append("spurious modality conflict")
    return failures


def _trim_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the parts of a fusion decision a reviewer needs, drop the bulk."""
    return {
        "status": result.get("status"),
        "priority": result.get("priority"),
        "score": result.get("score"),
        "agreement": result.get("agreement"),
        "message": result.get("message"),
        "evidence": [
            {k: e.get(k) for k in ("source", "available", "level_name", "confidence", "detail")}
            for e in result.get("evidence", [])
        ],
        "escalations": result.get("escalations", []),
        "conflicts": [c.get("description") for c in result.get("conflicts", [])],
        "human_review_reasons": result.get("human_review_reasons", []),
        "recommended_actions": result.get("recommended_actions", []),
        "warnings": result.get("warnings", []),
    }


class StressTester:
    """Run generated scenarios through the real fusion pipeline and score them."""

    def __init__(self, ml=None, dl=None, nlp=None, briefer: Briefer | None = None) -> None:
        self.decision = DecisionEngine(ml_engine=ml, dl_engine=dl, nlp_engine=nlp)
        self.briefer = briefer

    def run_zone(self, scenario: dict, zone: dict) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = zone["inputs"]
        started = time.perf_counter()
        crash = None
        try:
            result = self.decision.assess(sensors=inputs.get("sensors"), text=inputs.get("text"),
                                          image_path=inputs.get("image_path"),
                                          water_levels=inputs.get("water_levels"))
        except Exception as exc:  # a crash is a finding, not a reason to stop the run
            result = {"status": "crash"}
            crash = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000

        expected = zone["expected"]
        failures = [f"pipeline raised {crash}"] if crash else check_expectations(expected, result)
        true_index = PRIORITY_LEVELS.index(zone["true_severity"])
        ok = result.get("status") == "ok"
        predicted = result.get("priority_index") if ok else None
        row = {
            "scenario_id": scenario["scenario_id"], "scenario": scenario["name"],
            "archetype": scenario["archetype"], "zone_id": zone["zone_id"], "label": zone["label"],
            "true_severity": zone["true_severity"], "true_index": true_index,
            "status": result.get("status"), "priority": result.get("priority"),
            "priority_index": predicted, "score": result.get("score"),
            "agreement": result.get("agreement"),
            "sources_used": "; ".join(result.get("sources_used", [])),
            "sources_missing": "; ".join(result.get("sources_missing", [])),
            "n_conflicts": len(result.get("conflicts", [])),
            "n_escalations": len(result.get("escalations", [])),
            "human_review": bool(result.get("human_review_required")),
            "expected_status": expected.get("status"),
            "expected_min_priority": expected.get("min_priority"),
            "expected_max_priority": expected.get("max_priority"),
            "expected_human_review": expected.get("human_review"),
            "expected_conflict": expected.get("conflict"),
            "evidence_lost": "; ".join(zone["provenance"]["evidence_lost"]),
            "beyond_record_fields": "; ".join(zone["provenance"]["beyond_record_fields"]),
            "passed": not failures,
            "failures": "; ".join(failures),
            "exact_match": bool(ok and predicted == true_index),
            "within_one": bool(ok and abs(predicted - true_index) <= 1),
            # A truly URGENT/CRITICAL zone called ROUTINE/ELEVATED: the costliest error.
            "critical_miss": bool(ok and true_index >= 2 and predicted <= 1),
            "over_triage": bool(ok and true_index <= 1 and predicted >= true_index + 2),
            "latency_ms": round(latency_ms, 1),
        }
        log = inputs.get("incident_log")
        if self.briefer is not None and log:
            try:
                brief = self.briefer.brief(log)
                log_words = len(log.split())
                row.update({
                    "slm_priority": brief["priority"],
                    "slm_level": SLM_TO_LEVEL.get(brief["priority"]),
                    "slm_latency_ms": brief["latency_ms"],
                    "log_words": log_words,
                    "brief_words": brief["words"],
                    "read_savings_pct": round(100 * (1 - brief["words"] / max(1, log_words)), 1),
                })
            except Exception as exc:
                row["slm_error"] = f"{type(exc).__name__}: {exc}"
        return row, _trim_result(result)

    def run_scenario(self, scenario: dict) -> dict[str, Any]:
        rows, details = [], []
        for zone in scenario["zones"]:
            row, trimmed = self.run_zone(scenario, zone)
            rows.append(row)
            details.append({"zone_id": zone["zone_id"], "label": zone["label"],
                            "true_severity": zone["true_severity"],
                            "evidence_available": zone["provenance"]["evidence_available"],
                            "evidence_lost": zone["provenance"]["evidence_lost"],
                            "expected": zone["expected"], "passed": row["passed"],
                            "failures": row["failures"], "decision": trimmed})
        return {
            "scenario_id": scenario["scenario_id"], "name": scenario["name"],
            "archetype": scenario["archetype"], "blind_spots": scenario.get("blind_spots", []),
            "prompt": scenario.get("prompt", ""), "demand": scenario.get("demand", {}),
            "resources": scenario.get("resources", {}),
            "passed": all(r["passed"] for r in rows),
            "ranking_tau": ranking_tau(rows),
            "zones": rows, "details": details,
        }


def ranking_tau(rows: list[dict]) -> float | None:
    """Kendall tau between true severity and the system's triage order.

    Only meaningful when at least two scored zones differ in true severity; the
    fused score breaks ties within a priority level.
    """
    from scipy.stats import kendalltau

    scored = [r for r in rows if r["priority_index"] is not None]
    if len(scored) < 2 or len({r["true_index"] for r in scored}) < 2:
        return None
    predicted = [r["priority_index"] + (r["score"] or 0) / 10 for r in scored]
    tau = kendalltau([r["true_index"] for r in scored], predicted).statistic
    return None if tau is None or np.isnan(tau) else round(float(tau), 3)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarise(zone_df: pd.DataFrame, scenario_results: list[dict], slm_name: str | None,
              wildcard_df: pd.DataFrame | None = None) -> dict:
    ok = zone_df[zone_df["status"] == "ok"]
    high_risk = ok[ok["true_index"] >= 2]
    low_risk = ok[ok["true_index"] <= 1]
    # Only the wildcard has a zone with no evidence at all, so refusal handling
    # is measured over suite + wildcard; every other metric is suite-only.
    refusal_pool = pd.concat([zone_df, wildcard_df]) if wildcard_df is not None else zone_df
    expect_refusal = refusal_pool[refusal_pool["expected_status"] == "insufficient_evidence"]
    expect_conflict = zone_df[zone_df["expected_conflict"] == True]  # noqa: E712
    expect_review = zone_df[zone_df["expected_human_review"] == True]  # noqa: E712
    taus = [r["ranking_tau"] for r in scenario_results if r["ranking_tau"] is not None]
    latency = zone_df["latency_ms"]

    summary = {
        "scenarios": len(scenario_results),
        "scenarios_passed": sum(r["passed"] for r in scenario_results),
        "scenario_pass_rate": _rate(sum(r["passed"] for r in scenario_results), len(scenario_results)),
        "zones": int(len(zone_df)),
        "zones_passed": int(zone_df["passed"].sum()),
        "zone_pass_rate": _rate(int(zone_df["passed"].sum()), len(zone_df)),
        "crashes": int((zone_df["status"] == "crash").sum()),
        "priority_exact_accuracy": _rate(int(ok["exact_match"].sum()), len(ok)),
        "within_one_accuracy": _rate(int(ok["within_one"].sum()), len(ok)),
        "high_risk_zones_scored": int(len(high_risk)),
        "critical_misses": int(high_risk["critical_miss"].sum()),
        "critical_miss_rate": _rate(int(high_risk["critical_miss"].sum()), len(high_risk)),
        "over_triage_rate": _rate(int(low_risk["over_triage"].sum()), len(low_risk)),
        "insufficient_evidence_handled": _rate(
            int((expect_refusal["status"] == "insufficient_evidence").sum()), len(expect_refusal)),
        "insufficient_evidence_zones": int(len(expect_refusal)),
        "conflict_detection_rate": _rate(int((expect_conflict["n_conflicts"] > 0).sum()),
                                         len(expect_conflict)),
        "human_review_recall": _rate(int(expect_review["human_review"].sum()), len(expect_review)),
        "mean_ranking_tau": round(float(np.mean(taus)), 3) if taus else None,
        "ranking_scenarios": len(taus),
        "zones_beyond_record": int((zone_df["beyond_record_fields"] != "").sum()),
        "fusion_latency_ms": {"p50": round(float(latency.quantile(0.5)), 1),
                              "p95": round(float(latency.quantile(0.95)), 1),
                              "max": round(float(latency.max()), 1)},
    }
    if slm_name and "slm_priority" in zone_df:
        briefed = zone_df.dropna(subset=["slm_priority"])
        slm_high = briefed[briefed["true_index"] >= 2]
        summary["slm"] = {
            "model": slm_name,
            "briefings": int(len(briefed)),
            "priority_exact_accuracy": _rate(int((briefed["slm_level"] == briefed["true_index"]).sum()),
                                             len(briefed)),
            "high_risk_recall": _rate(int((slm_high["slm_level"] >= 2).sum()), len(slm_high)),
            "latency_ms": {"p50": round(float(briefed["slm_latency_ms"].quantile(0.5)), 1),
                           "p95": round(float(briefed["slm_latency_ms"].quantile(0.95)), 1)},
            "mean_read_savings_pct": round(float(briefed["read_savings_pct"].mean()), 1),
            "errors": int(zone_df.get("slm_error", pd.Series(dtype=object)).notna().sum()),
        }
    by_archetype = (zone_df.groupby("archetype")
                    .agg(zones=("passed", "size"), passed=("passed", "sum"),
                         critical_misses=("critical_miss", "sum"))
                    .reset_index())
    summary["by_archetype"] = by_archetype.to_dict(orient="records")
    return summary


# ===========================================================================
# A. Realism audit
# ===========================================================================

def naive_independent_sample(reference: pd.DataFrame, counts: dict[str, int], seed: int) -> pd.DataFrame:
    """Baseline generator: each feature drawn independently from its class marginal."""
    rng = np.random.default_rng(seed)
    frames = []
    for risk_class, n in counts.items():
        pool = reference[reference["zone_risk"] == risk_class]
        frame = pd.DataFrame({f: rng.choice(pool[f].to_numpy(), size=n) for f in GENAI.NUMERIC_FEATURES})
        frame["zone_risk"] = risk_class
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def c2st_auc(real: pd.DataFrame, synthetic: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """Classifier two-sample test: 5-fold ROC-AUC of a real-vs-synthetic classifier.

    Uses XGBoost, which Stage 01 already depends on. sklearn's tree ensembles
    are avoided on purpose: sklearn.ensemble imports a compiled sklearn.neighbors
    extension that application-control policies on locked-down Windows hosts
    can block, and the audit should not fail for that reason.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    features = GENAI.NUMERIC_FEATURES
    x = pd.concat([real[features], synthetic[features]], ignore_index=True).to_numpy()
    y = np.r_[np.zeros(len(real)), np.ones(len(synthetic))]
    scores = []
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y):
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                              random_state=seed, n_jobs=-1, eval_metric="logloss")
        model.fit(x[train], y[train])
        scores.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return {"auc_mean": round(float(np.mean(scores)), 4), "auc_std": round(float(np.std(scores)), 4),
            "classifier": "xgboost"}


def label_fidelity(ml_engine, reference: pd.DataFrame, synthetic: pd.DataFrame, seed: int) -> dict:
    """Does the independent Stage 01 model recover each sample's conditioning class?"""
    from sklearn.metrics import f1_score, recall_score

    rng = np.random.default_rng(seed)
    context = reference.iloc[rng.integers(0, len(reference), len(synthetic))][
        ["timestamp", "state", "district"]].reset_index(drop=True)
    records = pd.concat([context, synthetic[GENAI.NUMERIC_FEATURES].reset_index(drop=True)], axis=1)
    predictions = ml_engine.predict_batch(records.to_dict(orient="records"))["predictions"]
    predicted = [p["risk_category"] for p in predictions]
    truth = synthetic["zone_risk"].tolist()
    recalls = recall_score(truth, predicted, labels=GENAI.RISK_CLASSES, average=None, zero_division=0)
    return {"macro_f1": round(float(f1_score(truth, predicted, labels=GENAI.RISK_CLASSES,
                                              average="macro", zero_division=0)), 4),
            "recall_by_class": {c: round(float(r), 4) for c, r in zip(GENAI.RISK_CLASSES, recalls)}}


def text_fidelity(nlp_engine, generator, seed: int = SEED) -> dict:
    """Score generated messages with the independent Stage 03 model."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    state, district = "Assam", "Guwahati"
    clean, degraded = [], []
    for hazard in GENAI.HAZARDS:
        locations = generator.bank["hazards"][hazard]["locations"]
        for severity in GENAI.TEXT_LEVELS:
            for _ in range(TEXT_PROBES_PER_CELL):
                headcount = generator._headcount(severity, set(), rng)
                location = GENAI._weighted_choice(locations, rng)
                text = generator._dispatcher_text(hazard, severity, location, district, state, headcount, rng)
                probe = {"hazard": hazard, "severity": severity, "headcount": headcount}
                clean.append({**probe, "text": text})
                degraded.append({**probe, "text": GENAI.degrade_text(text, rng, truncate_probability=0.0)})

    def score(probes: list[dict]) -> dict:
        results = [nlp_engine.analyze(p["text"]) for p in probes]
        urgency = [r.get("urgency") or "NONE" for r in results]
        truth = [p["severity"] for p in probes]
        headcount_hits = sum(1 for r, p in zip(results, probes) if r.get("headcount") == p["headcount"])
        return {
            "n": len(probes),
            "hazard_accuracy": round(float(np.mean([r.get("hazard_type") == p["hazard"]
                                                    for r, p in zip(results, probes)])), 4),
            "urgency_accuracy": round(float(np.mean([u == t for u, t in zip(urgency, truth)])), 4),
            "urgency_macro_f1": round(float(f1_score(truth, urgency, labels=GENAI.TEXT_LEVELS,
                                                     average="macro", zero_division=0)), 4),
            "headcount_exact": round(headcount_hits / len(probes), 4),
        }

    implicit = []
    for severity in ("HIGH", "CRITICAL"):
        for template in GENAI.IMPLICIT_TEMPLATES[severity]:
            text = template.format(location="river bank", district=district, n=int(rng.integers(10, 80)))
            implicit.append(nlp_engine.analyze(text).get("urgency"))
    return {
        "clean": score(clean),
        "degraded_comms_noise": score(degraded),
        # Share of implied-urgency messages Stage 03 scores HIGH or CRITICAL.
        "implicit_urgency_recall": round(float(np.mean([u in {"HIGH", "CRITICAL"} for u in implicit])), 4),
        "implicit_urgency_labels": implicit,
    }


def realism_audit(reference: pd.DataFrame, synthetic: pd.DataFrame, engines: dict,
                  generator, seed: int = SEED) -> dict:
    from scipy.stats import ks_2samp

    features = GENAI.NUMERIC_FEATURES
    ks = {}
    for feature in features:
        per_class = {c: round(float(ks_2samp(reference.loc[reference.zone_risk == c, feature],
                                             synthetic.loc[synthetic.zone_risk == c, feature]).statistic), 4)
                     for c in GENAI.RISK_CLASSES}
        ks[feature] = {"overall": round(float(ks_2samp(reference[feature], synthetic[feature]).statistic), 4),
                       "per_class": per_class}
    real_corr = reference[features].corr()
    synth_corr = synthetic[features].corr()
    moments = {f: {"real_mean": round(float(reference[f].mean()), 3),
                   "synthetic_mean": round(float(synthetic[f].mean()), 3),
                   "real_std": round(float(reference[f].std()), 3),
                   "synthetic_std": round(float(synthetic[f].std()), 3)} for f in features}

    counts = synthetic["zone_risk"].value_counts().to_dict()
    naive = naive_independent_sample(reference, counts, seed)
    audit = {
        "n_real": int(len(reference)), "n_synthetic": int(len(synthetic)),
        "ks_statistic": ks,
        "mean_ks_overall": round(float(np.mean([v["overall"] for v in ks.values()])), 4),
        "moments": moments,
        "correlation": {
            "max_abs_delta": round(float((real_corr - synth_corr).abs().to_numpy().max()), 4),
            "rainfall_calls_real": round(float(real_corr.loc["rainfall_mm", "emergency_calls"]), 4),
            "rainfall_calls_synthetic": round(float(synth_corr.loc["rainfall_mm", "emergency_calls"]), 4),
            "rainfall_calls_naive_baseline": round(float(
                naive[features].corr().loc["rainfall_mm", "emergency_calls"]), 4),
        },
        "c2st": {"cvae": c2st_auc(reference, synthetic, seed), "naive_baseline": c2st_auc(reference, naive, seed)},
    }
    if engines.get("ml") is not None:
        audit["label_fidelity"] = {"cvae": label_fidelity(engines["ml"], reference, synthetic, seed),
                                   "naive_baseline": label_fidelity(engines["ml"], reference, naive, seed)}
    if engines.get("nlp") is not None:
        audit["text_fidelity"] = text_fidelity(engines["nlp"], generator, seed)
    return audit


# ===========================================================================
# Reporting
# ===========================================================================

def forecast_flat_probe(dl_engine, levels=tuple(float(v) for v in range(1, 10))) -> dict[str, Any]:
    """Feed the Stage 02 LSTM perfectly flat rivers and record what it projects.

    The stress suite showed calm zones escalated to URGENT because the LSTM
    projected steep rises from flat, low histories. A flat river is the
    simplest possible input, so any large projected change is the model's own
    bias. It is compared with real CWC gauge behaviour and with the fusion
    layer's escalation thresholds, imported rather than restated.
    """
    water_ref = json.loads(WATER_REFERENCE_JSON.read_text(encoding="utf-8"))
    real_q99 = water_ref.get("rise_6h_m", {}).get("q99")
    rows = []
    for level in levels:
        result = dl_engine.forecast_water_levels([level] * GENAI.LOOKBACK_HOURS, horizon=6)
        peak = float(max(result["forecast_water_levels"]))
        change = peak - level
        if change >= 3 * FORECAST_RISE_THRESHOLD_M:
            reading = "steep rise: escalates to URGENT"
        elif change >= FORECAST_RISE_THRESHOLD_M:
            reading = "rise: raises the score"
        elif change <= -FORECAST_RISE_THRESHOLD_M:
            reading = "falling"
        else:
            reading = "flat"
        rows.append({"flat_level_m": level, "peak_6h_m": round(peak, 3), "change_6h_m": round(change, 3),
                     "in_distribution": bool(result.get("in_distribution", True)),
                     "fusion_reading": reading})
    return {
        "real_gauge_rise_6h_q99_m": real_q99,
        "steep_rise_threshold_m": round(3 * FORECAST_RISE_THRESHOLD_M, 3),
        "rows": rows,
        "levels_projecting_implausible_rise": [r["flat_level_m"] for r in rows
                                               if real_q99 is not None and r["change_6h_m"] > real_q99],
        "levels_escalating_to_urgent": [r["flat_level_m"] for r in rows
                                        if r["fusion_reading"].startswith("steep")],
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def write_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Stage 05 -- Stress-Test Evaluation Report",
        "",
        f"Generated {report['generated_at']} (commit `{report.get('git_commit') or 'n/a'}`). "
        f"Stage status: " + ", ".join(f"{k}={v}" for k, v in report["stage_status"].items()) + ".",
        "",
        "Ground truth is the scenario designer's intent, not an observed outcome. Imagery comes "
        "from the Stage 02 CNN's likely training pool, so visual evidence is optimistic.",
        "",
        "## 1. Headline results (20-scenario suite)",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Scenarios passed | {s['scenarios_passed']}/{s['scenarios']} ({_pct(s['scenario_pass_rate'])}) |",
        f"| Zones passed | {s['zones_passed']}/{s['zones']} ({_pct(s['zone_pass_rate'])}) |",
        f"| Pipeline crashes | {s['crashes']} |",
        f"| Priority exact accuracy | {_pct(s['priority_exact_accuracy'])} |",
        f"| Within-one-level accuracy | {_pct(s['within_one_accuracy'])} |",
        f"| **Critical-miss rate** (true URGENT+/called <= ELEVATED) | **{_pct(s['critical_miss_rate'])}** "
        f"({s['critical_misses']}/{s['high_risk_zones_scored']}) |",
        f"| Over-triage rate (true <= ELEVATED, called 2+ levels higher) | {_pct(s['over_triage_rate'])} |",
        f"| No-evidence zones refused, not scored (suite + wildcard, n={s['insufficient_evidence_zones']}) | "
        f"{_pct(s['insufficient_evidence_handled'])} |",
        f"| Modality conflicts surfaced | {_pct(s['conflict_detection_rate'])} |",
        f"| Human-review recall | {_pct(s['human_review_recall'])} |",
        f"| Mean zone-ranking Kendall tau ({s['ranking_scenarios']} scenarios) | "
        f"{s['mean_ranking_tau'] if s['mean_ranking_tau'] is not None else 'n/a'} |",
        f"| Zones with values beyond the historical record | {s['zones_beyond_record']} |",
        f"| Fusion latency p50 / p95 | {s['fusion_latency_ms']['p50']} / {s['fusion_latency_ms']['p95']} ms |",
    ]
    if "slm" in s:
        slm = s["slm"]
        lines += [
            f"| Stage 04 briefing ({slm['model']}) priority accuracy | {_pct(slm['priority_exact_accuracy'])} |",
            f"| Stage 04 briefing high-risk recall | {_pct(slm['high_risk_recall'])} |",
            f"| Stage 04 read-time savings (mean) | {slm['mean_read_savings_pct']}% |",
        ]

    realism = report.get("realism")
    if realism:
        c2st = realism["c2st"]
        lines += [
            "", "## 2. Realism audit", "",
            "| Check | CVAE | Naive baseline |", "| --- | ---: | ---: |",
            f"| C2ST ROC-AUC (0.5 = indistinguishable) | {c2st['cvae']['auc_mean']} | "
            f"{c2st['naive_baseline']['auc_mean']} |",
            f"| Rainfall-calls correlation (real {realism['correlation']['rainfall_calls_real']}) | "
            f"{realism['correlation']['rainfall_calls_synthetic']} | "
            f"{realism['correlation']['rainfall_calls_naive_baseline']} |",
        ]
        if "label_fidelity" in realism:
            lf = realism["label_fidelity"]
            lines.append(f"| Stage 01 recovers conditioning class (macro F1) | {lf['cvae']['macro_f1']} | "
                         f"{lf['naive_baseline']['macro_f1']} |")
        lines += ["", f"Mean KS statistic across features: {realism['mean_ks_overall']}; "
                      f"max |delta r| between correlation matrices: {realism['correlation']['max_abs_delta']}."]
        if "text_fidelity" in realism:
            tf = realism["text_fidelity"]
            lines += [
                "", "| Stage 03 reading generated text | Clean | Degraded (comms noise) |",
                "| --- | ---: | ---: |",
                f"| Hazard accuracy | {tf['clean']['hazard_accuracy']} | {tf['degraded_comms_noise']['hazard_accuracy']} |",
                f"| Urgency macro F1 | {tf['clean']['urgency_macro_f1']} | {tf['degraded_comms_noise']['urgency_macro_f1']} |",
                f"| Headcount exact | {tf['clean']['headcount_exact']} | {tf['degraded_comms_noise']['headcount_exact']} |",
                "", f"Implicit-urgency messages scored HIGH/CRITICAL by Stage 03: "
                    f"{_pct(tf['implicit_urgency_recall'])}.",
            ]

    probe = report.get("forecast_probe")
    if probe:
        lines += [
            "", "## 2b. Stage 02 forecast probe (perfectly flat 72 h river)", "",
            f"Real CWC gauges rise more than {probe['real_gauge_rise_6h_q99_m']} m in 6 h in only 1% of "
            f"windows. Fusion escalates to URGENT on a projected rise of {probe['steep_rise_threshold_m']} m.",
            "", "| Flat level (m) | Forecast 6 h peak (m) | Projected change (m) | Fusion reads it as |",
            "| ---: | ---: | ---: | --- |",
        ]
        for row in probe["rows"]:
            lines.append(f"| {row['flat_level_m']:.1f} | {row['peak_6h_m']:.2f} | {row['change_6h_m']:+.2f} | "
                         f"{row['fusion_reading']} |")

    lines += ["", "## 3. Per-scenario results", "",
              "| ID | Scenario | Zones | Passed | Ranking tau | Blind spots |",
              "| --- | --- | ---: | :---: | ---: | --- |"]
    for sc in report["scenarios"]:
        passed = sum(z["passed"] for z in sc["zones"])
        zone_count = len(sc["zones"])
        result = "PASS" if sc["passed"] else f"{passed}/{zone_count}"
        lines.append(f"| {sc['scenario_id']} | {sc['name']} | {zone_count} | {result} | "
                     f"{sc['ranking_tau'] if sc['ranking_tau'] is not None else '-'} | "
                     f"{', '.join(sc['blind_spots'])} |")

    lines += ["", "## 4. Failure log", ""]
    if report["failures"]:
        lines += ["| Scenario | Zone | True | Called | Why it failed |", "| --- | --- | --- | --- | --- |"]
        for f in report["failures"]:
            lines.append(f"| {f['scenario_id']} | {f['zone_id']} {f['label']} | {f['true_severity']} | "
                         f"{f['priority'] or f['status']} | {f['failures']} |")
    else:
        lines.append("No zone failed its expectations.")

    wild = report.get("wildcard")
    if wild:
        lines += ["", f"## 5. Wildcard -- {wild['name']}", "", wild["prompt"], "",
                  "| Zone | True | Evidence left | Lost to outage | Decision | Review | Result |",
                  "| --- | --- | --- | --- | --- | :---: | :---: |"]
        for d in wild["details"]:
            kept = ", ".join(k for k, v in d["evidence_available"].items() if v) or "none"
            decision = d["decision"]
            called = decision.get("priority") or decision.get("status")
            lines.append(f"| {d['label']} | {d['true_severity']} | {kept} | "
                         f"{', '.join(d['evidence_lost']) or '-'} | {called} | "
                         f"{'yes' if decision.get('human_review_reasons') or decision.get('status') != 'ok' else 'no'} | "
                         f"{'PASS' if d['passed'] else 'FAIL: ' + d['failures']} |")
    return "\n".join(lines) + "\n"


def append_history(report: dict) -> None:
    s = report["summary"]
    row = {
        "run_at": report["generated_at"], "git_commit": report.get("git_commit"),
        "seed": report.get("seed"), "slm_model": s.get("slm", {}).get("model"),
        "scenario_pass_rate": s["scenario_pass_rate"], "zone_pass_rate": s["zone_pass_rate"],
        "critical_miss_rate": s["critical_miss_rate"],
        "priority_exact_accuracy": s["priority_exact_accuracy"], "crashes": s["crashes"],
        "c2st_auc": (report.get("realism") or {}).get("c2st", {}).get("cvae", {}).get("auc_mean"),
    }
    frame = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        frame = pd.concat([pd.read_csv(HISTORY_CSV), frame], ignore_index=True)
    frame.to_csv(HISTORY_CSV, index=False)


def run_stress_test(suite: list[dict], wildcard: dict | None, engines: dict) -> tuple[list, dict | None, pd.DataFrame]:
    tester = StressTester(engines.get("ml"), engines.get("dl"), engines.get("nlp"), engines.get("slm"))
    results = []
    for scenario in suite:
        result = tester.run_scenario(scenario)
        results.append(result)
        passed = sum(z["passed"] for z in result["zones"])
        print(f"  {result['scenario_id']} {result['name']:<36} {passed}/{len(result['zones'])} zones  "
              f"{'PASS' if result['passed'] else 'FAIL'}")
    wildcard_result = tester.run_scenario(wildcard) if wildcard else None
    zone_df = pd.DataFrame([row for r in results for row in r["zones"]])
    return results, wildcard_result, zone_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 realism audit + 20-scenario stress test")
    parser.add_argument("--slm", choices=["baseline", "qwen", "none"], default="baseline",
                        help="Stage 04 briefing model (qwen is slow: ~10 s per briefing)")
    parser.add_argument("--skip-realism", action="store_true")
    parser.add_argument("--suite", type=Path, default=SUITE_JSON)
    parser.add_argument("--wildcard", type=Path, default=WILDCARD_JSON)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.suite.exists():
        raise SystemExit("Scenario suite missing. Run `python Stage05_GenAI/03_genai_engineer.py` first.")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("[eval] loading Stage 01-04 adapters ...")
    engines, status = load_stage_engines(args.slm)
    for key, value in status.items():
        print(f"  {key:<4} {value}")

    suite = json.loads(args.suite.read_text(encoding="utf-8"))["scenarios"]
    wildcard = json.loads(args.wildcard.read_text(encoding="utf-8")) if args.wildcard.exists() else None

    print(f"[eval] stress test: {len(suite)} scenarios")
    results, wildcard_result, zone_df = run_stress_test(suite, wildcard, engines)
    slm_name = engines["slm"].name if engines.get("slm") else None
    wildcard_df = pd.DataFrame(wildcard_result["zones"]) if wildcard_result else None
    summary = summarise(zone_df, results, slm_name, wildcard_df)

    probe = None
    if engines.get("dl") is not None:
        print("[eval] Stage 02 forecast probe (flat rivers) ...")
        probe = forecast_flat_probe(engines["dl"])

    realism = None
    if not args.skip_realism:
        print("[eval] realism audit ...")
        reference = pd.read_csv(SENSOR_REFERENCE_CSV)
        synthetic = pd.read_csv(SYNTHETIC_SAMPLES_CSV)
        realism = realism_audit(reference, synthetic, engines, GENAI.ScenarioGenerator(), args.seed)
        REALISM_JSON.write_text(json.dumps(_jsonable(realism), indent=2), encoding="utf-8")

    failures = zone_df[~zone_df["passed"]][
        ["scenario_id", "zone_id", "label", "true_severity", "status", "priority", "failures"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "stage_status": status,
        "summary": summary,
        "realism": realism,
        "forecast_probe": probe,
        "scenarios": results,
        "wildcard": wildcard_result,
        "failures": failures.to_dict(orient="records"),
    }
    report = _jsonable(report)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(write_markdown(report), encoding="utf-8")
    zone_df.to_csv(ZONE_RESULTS_CSV, index=False)
    pd.DataFrame([{k: r[k] for k in ("scenario_id", "name", "archetype", "passed", "ranking_tau")}
                  | {"zones": len(r["zones"]), "zones_passed": sum(z["passed"] for z in r["zones"])}
                  for r in results]).to_csv(SCENARIO_RESULTS_CSV, index=False)
    append_history(report)

    s = report["summary"]
    print(f"[eval] scenarios passed {s['scenarios_passed']}/{s['scenarios']}  "
          f"zones {s['zones_passed']}/{s['zones']}  critical misses {s['critical_misses']}  "
          f"crashes {s['crashes']}")
    if realism:
        print(f"[eval] C2ST AUC cvae {realism['c2st']['cvae']['auc_mean']} vs naive "
              f"{realism['c2st']['naive_baseline']['auc_mean']}")
    if wildcard_result:
        print(f"[eval] wildcard: {sum(z['passed'] for z in wildcard_result['zones'])}/"
              f"{len(wildcard_result['zones'])} zones passed")
    print(f"[eval] report -> {REPORT_MD.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
