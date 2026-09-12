"""Stage 05 GenAI -- Data Engineer.

Assemble REFERENCE BASELINES from real historical disaster distributions.

The scenario generator (03_genai_engineer.py) is trained on and anchored to these
baselines, and the evaluation engineer (04_evaluation_engineer.py) measures how
far every synthetic scenario departs from them. A generator with no reference
point cannot tell "a realistic rare event" from "a physically absurd one".

Everything is READ from earlier stages' committed data. Nothing in Stages 01-04
is modified; all outputs are written under Stage05_GenAI/data/.

Inputs (read-only)
------------------
Stage01_ML/data/processed/Master_Dataset.csv              labelled zone sensor rows
Stage01_ML/data/raw/raw_historical_flood_log.csv          flood cause + duration
Stage03_NLP/data/processed/Dispatcher_Log_Master_60000_Processed.csv
Stage03_NLP/data/processed/Social_Feeds_India_Processed.csv
Stage02_DL/data/raw/03_RIVER_WATER_LEVEL_DATASET/*.csv    real hourly gauge telemetry
Stage02_DL/data/raw/01_SATELLITE_FLOOD_DATASET/           aerial images + water masks

Outputs
-------
data/processed/sensor_reference.csv          numeric sensor rows + label + cause
data/processed/reference_distributions.json  per-class quantiles, correlations, priors
data/processed/text_phrase_bank.json         dispatcher phrasing decomposed into slots
data/processed/water_level_reference.json    real hourly rise/fall rates
data/processed/imagery_manifest.csv          + data/raw/imagery/*.jpg
data/processed/data_manifest.json            lineage record for this run
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMAGERY_DIR = RAW_DIR / "imagery"

STAGE01_MASTER = REPO_ROOT / "Stage01_ML" / "data" / "processed" / "Master_Dataset.csv"
STAGE01_FLOOD_LOG = REPO_ROOT / "Stage01_ML" / "data" / "raw" / "raw_historical_flood_log.csv"
STAGE03_DISPATCHER = (
    REPO_ROOT / "Stage03_NLP" / "data" / "processed" / "Dispatcher_Log_Master_60000_Processed.csv"
)
STAGE03_SOCIAL = REPO_ROOT / "Stage03_NLP" / "data" / "processed" / "Social_Feeds_India_Processed.csv"
STAGE02_RIVER_DIR = REPO_ROOT / "Stage02_DL" / "data" / "raw" / "03_RIVER_WATER_LEVEL_DATASET"
STAGE02_SATELLITE_DIR = (
    REPO_ROOT / "Stage02_DL" / "data" / "raw" / "01_SATELLITE_FLOOD_DATASET" / "dataset"
)

SENSOR_REFERENCE_CSV = PROCESSED_DIR / "sensor_reference.csv"
DISTRIBUTIONS_JSON = PROCESSED_DIR / "reference_distributions.json"
PHRASE_BANK_JSON = PROCESSED_DIR / "text_phrase_bank.json"
WATER_REFERENCE_JSON = PROCESSED_DIR / "water_level_reference.json"
IMAGERY_MANIFEST_CSV = PROCESSED_DIR / "imagery_manifest.csv"
DATA_MANIFEST_JSON = PROCESSED_DIR / "data_manifest.json"

SEED = 42

# The nine numeric fields Stage 01 scores. The generator models exactly these.
NUMERIC_FEATURES = [
    "rainfall_mm",
    "river_level_m",
    "river_level_threshold_m",
    "emergency_calls",
    "road_closures",
    "bridge_closures",
    "flood_history_count",
    "population_affected",
    "water_level_change_m",
]
RISK_CLASSES = ["Low", "Moderate", "Severe"]
QUANTILES = {"q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q50": 0.50,
             "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99}

# Stage 03 dispatcher texts follow one template. Decomposing it into slots lets
# the generator recombine REAL phrasing instead of inventing a new register.
REQUEST_MAX_TEMPLATES = 12
PHRASE_TOP_N = 25

# A pixel is water when its mask value is non-zero. A scene is "flooded" for the
# imagery bank when at least this share of it is water, and "unflooded" when at
# most this share is. Scenes in between are ambiguous and left out on purpose.
FLOODED_MIN_WATER_FRACTION = 0.10
UNFLOODED_MAX_WATER_FRACTION = 0.01
IMAGE_MAX_SIDE_PX = 320

# Hourly gauge steps larger than this are logger resets, not hydrology.
MAX_PLAUSIBLE_CHANGE_1H_M = 3.0


def _quantile_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    summary = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
               "min": float(values.min()), "max": float(values.max())}
    for name, q in QUANTILES.items():
        summary[name] = float(values.quantile(q))
    return {key: round(value, 4) for key, value in summary.items()}


def _top(counter: Counter, n: int = PHRASE_TOP_N) -> dict[str, int]:
    return {key: int(count) for key, count in counter.most_common(n)}


# ---------------------------------------------------------------------------
# 1. Sensor reference
# ---------------------------------------------------------------------------

def build_sensor_reference() -> pd.DataFrame:
    """Labelled Stage 01 sensor rows joined with the flood log's cause/duration."""
    master = pd.read_csv(STAGE01_MASTER)
    missing = sorted(set(NUMERIC_FEATURES + ["timestamp", "state", "district", "zone_risk"])
                     .difference(master.columns))
    if missing:
        raise ValueError(f"Stage 01 master dataset is missing columns: {missing}")

    flood_log = pd.read_csv(STAGE01_FLOOD_LOG,
                            usecols=["timestamp", "state", "district", "cause", "duration_hours"])
    keys = ["timestamp", "state", "district"]
    flood_log = flood_log.drop_duplicates(subset=keys)
    reference = master.merge(flood_log, on=keys, how="left")

    parsed = pd.to_datetime(reference["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    reference["hour"] = parsed.dt.hour
    reference["month"] = parsed.dt.month
    reference["river_margin_m"] = reference["river_level_m"] - reference["river_level_threshold_m"]

    SENSOR_REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(SENSOR_REFERENCE_CSV, index=False)
    print(f"[data] sensor reference: {len(reference):,} rows -> {SENSOR_REFERENCE_CSV.name}")
    return reference


def summarise_distributions(reference: pd.DataFrame) -> dict:
    """Per-class quantiles, correlations and priors the generator is judged against."""
    per_class = {
        risk: {feature: _quantile_summary(group[feature]) for feature in NUMERIC_FEATURES}
        for risk, group in reference.groupby("zone_risk")
    }
    overall = {feature: _quantile_summary(reference[feature]) for feature in NUMERIC_FEATURES}
    correlation = reference[NUMERIC_FEATURES].corr().round(4)

    state_districts = {
        state: sorted(group["district"].unique().tolist())
        for state, group in reference.groupby("state")
    }
    distributions = {
        "source": str(STAGE01_MASTER.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_rows": int(len(reference)),
        "features": NUMERIC_FEATURES,
        "classes": RISK_CLASSES,
        "class_counts": {k: int(v) for k, v in reference["zone_risk"].value_counts().items()},
        "class_priors": {k: round(float(v), 4)
                         for k, v in reference["zone_risk"].value_counts(normalize=True).items()},
        "overall": overall,
        "per_class": per_class,
        "river_margin_by_class": {
            risk: _quantile_summary(group["river_margin_m"])
            for risk, group in reference.groupby("zone_risk")
        },
        "correlation": {row: correlation.loc[row].to_dict() for row in correlation.index},
        "state_districts": state_districts,
        "cause_priors": {k: round(float(v), 4)
                         for k, v in reference["cause"].value_counts(normalize=True).items()},
        "duration_hours": _quantile_summary(reference["duration_hours"]),
        "hour_distribution": {int(k): int(v) for k, v in
                              reference["hour"].value_counts().sort_index().items()},
        "month_distribution": {int(k): int(v) for k, v in
                               reference["month"].value_counts().sort_index().items()},
        "missing_values": int(reference[NUMERIC_FEATURES].isna().sum().sum()),
    }
    DISTRIBUTIONS_JSON.write_text(json.dumps(distributions, indent=2), encoding="utf-8")
    print(f"[data] reference distributions -> {DISTRIBUTIONS_JSON.name}")
    return distributions


# ---------------------------------------------------------------------------
# 2. Text phrase bank
# ---------------------------------------------------------------------------

_INJURY_RE = re.compile(r"(\d+) (\w+) affected, ([^.]+)\.")


def _decompose_dispatcher_text(row: pd.Series) -> dict | None:
    """Split one templated dispatcher message into reusable slots.

    Template (Stage 03 corpus):
      "<lead> (<status>) <prep> <location> in <district>, <state>. <n> <noun>
       affected, <injury>. <request sentence>"
    Returns None for any row that does not fit, so odd rows are skipped rather
    than contaminating the bank.
    """
    text = str(row["text"])
    location = str(row["location_entity"])
    lead, sep, rest = text.partition(" (")
    if not sep:
        return None
    status, sep, rest = rest.partition(") ")
    if not sep:
        return None
    loc_at = rest.find(f" {location} in ")
    if loc_at < 0:
        return None
    prep = rest[:loc_at].strip()
    injury_match = _INJURY_RE.search(rest)
    if injury_match is None:
        return None
    request = rest[injury_match.end():].strip()

    resources = [r.strip() for r in str(row["resource_entity"]).split(",") if r.strip()]
    template = None
    for joiner in (" and ", ", "):
        joined = joiner.join(resources)
        if joined and joined in request:
            template = request.replace(joined, "{resources}", 1)
            break
    return {
        "lead": lead.strip(),
        "status": status.strip(),
        "prep": prep,
        "noun": injury_match.group(2),
        "injury": injury_match.group(3).strip(),
        "request_template": template,
        "resources": resources,
    }


def build_text_phrase_bank(sample_rows: int = 20000) -> dict:
    columns = ["text", "severity", "hazard_type", "location_entity",
               "resource_entity", "headcount_entity"]
    dispatcher = pd.read_csv(STAGE03_DISPATCHER, usecols=columns)
    hazard_severity = pd.crosstab(dispatcher["hazard_type"], dispatcher["severity"])
    sample = dispatcher.sample(n=min(sample_rows, len(dispatcher)), random_state=SEED)

    leads: dict[str, Counter] = {}
    locations: dict[str, Counter] = {}
    resources: dict[str, Counter] = {}
    requests: dict[str, Counter] = {}
    statuses, preps, nouns = Counter(), Counter(), Counter()
    injuries: dict[str, Counter] = {}
    parsed = 0

    for _, row in sample.iterrows():
        slots = _decompose_dispatcher_text(row)
        if slots is None:
            continue
        parsed += 1
        hazard, severity = str(row["hazard_type"]), str(row["severity"])
        leads.setdefault(hazard, Counter())[slots["lead"]] += 1
        locations.setdefault(hazard, Counter())[str(row["location_entity"])] += 1
        for resource in slots["resources"]:
            resources.setdefault(hazard, Counter())[resource] += 1
        if slots["request_template"]:
            requests.setdefault(hazard, Counter())[slots["request_template"]] += 1
        statuses[slots["status"]] += 1
        preps[slots["prep"]] += 1
        nouns[slots["noun"]] += 1
        injuries.setdefault(severity, Counter())[slots["injury"]] += 1

    headcount = pd.to_numeric(dispatcher["headcount_entity"], errors="coerce")
    headcount_by_severity = {
        severity: _quantile_summary(headcount[dispatcher["severity"] == severity])
        for severity in sorted(dispatcher["severity"].dropna().unique())
    }

    bank = {
        "source": str(STAGE03_DISPATCHER.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rows_sampled": int(len(sample)),
        "rows_parsed": int(parsed),
        "template_parse_rate": round(parsed / max(1, len(sample)), 4),
        "hazards": {
            hazard: {
                "leads": _top(leads.get(hazard, Counter())),
                "locations": _top(locations.get(hazard, Counter())),
                "resources": _top(resources.get(hazard, Counter())),
                "requests": _top(requests.get(hazard, Counter()), REQUEST_MAX_TEMPLATES),
            }
            for hazard in sorted(leads)
        },
        "statuses": _top(statuses),
        "preps": _top(preps),
        "people_nouns": _top(nouns),
        "injury_by_severity": {sev: _top(counter) for sev, counter in injuries.items()},
        "headcount_by_severity": headcount_by_severity,
        "hazard_severity_counts": {
            hazard: {sev: int(count) for sev, count in row.items()}
            for hazard, row in hazard_severity.iterrows()
        },
        "social_templates": build_social_templates(),
    }
    PHRASE_BANK_JSON.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    print(f"[data] phrase bank: {parsed:,}/{len(sample):,} dispatcher rows decomposed "
          f"-> {PHRASE_BANK_JSON.name}")
    return bank


def build_social_templates(per_hazard: int = 30) -> dict[str, list[str]]:
    """Real social-feed posts with the place name replaced by a {district} slot."""
    social = pd.read_csv(STAGE03_SOCIAL, usecols=["text", "district", "location_mentioned",
                                                  "hazard_type", "urgency_level"])
    templates: dict[str, list[str]] = {}
    for hazard, group in social.groupby("hazard_type"):
        seen: list[str] = []
        for _, row in group.sample(frac=1.0, random_state=SEED).iterrows():
            text = str(row["text"])
            for place in {str(row["district"]), str(row["location_mentioned"])}:
                if place and place != "nan":
                    text = text.replace(place, "{district}")
            if "{district}" in text and text not in seen:
                seen.append(text)
            if len(seen) >= per_hazard:
                break
        if seen:
            templates[str(hazard)] = seen
    return templates


# ---------------------------------------------------------------------------
# 3. Real gauge dynamics
# ---------------------------------------------------------------------------

def build_water_level_reference() -> dict:
    """How fast do real rivers rise and fall, hour to hour?

    Stations report against different datums (one reads ~15 m), so absolute
    levels are not comparable with Stage 01's 1-9 m scale. Only CHANGES are
    kept; the generator uses them to shape plausible 72 h histories.
    """
    level_col = "River Water Level Telemetry Hourly (meter)"
    time_col = "Data Acquisition Time"
    rise_1h, rise_6h, fall_6h, range_72h = [], [], [], []
    stations = 0
    hours = 0
    dropped = 0
    for path in sorted(STAGE02_RIVER_DIR.glob("*.csv")):
        frame = pd.read_csv(path, usecols=["Station", time_col, level_col])
        frame[time_col] = pd.to_datetime(frame[time_col], format="%d-%m-%Y %H:%M", errors="coerce")
        frame[level_col] = pd.to_numeric(frame[level_col], errors="coerce")
        frame = frame.dropna()
        # One file holds several stations on different datums; differencing
        # across a station boundary produced "rises" of hundreds of metres.
        for _, station in frame.groupby("Station"):
            station = station.drop_duplicates(subset=time_col).sort_values(time_col)
            levels = station[level_col]
            # Telemetry carries sentinels (0.0, -1, 5e6). Keep readings within a
            # robust band around the station's own median.
            median = levels.median()
            mad = (levels - median).abs().median()
            keep = (levels - median).abs() <= max(10 * mad, 2.0)
            keep &= levels > 0
            dropped += int((~keep).sum())
            station = station[keep]
            if len(station) < 72:
                continue
            # Reindex to a strict hourly grid so a diff never spans a telemetry gap.
            series = station.set_index(time_col)[level_col].asfreq("h")
            d1 = series.diff(1)
            # A river does not move 3 m in one hour; such steps are logger resets.
            series = series.mask(d1.abs() > MAX_PLAUSIBLE_CHANGE_1H_M)
            stations += 1
            hours += int(series.notna().sum())
            d1 = series.diff(1).dropna()
            d6 = series.diff(6).dropna()
            rng72 = (series.rolling(72).max() - series.rolling(72).min()).dropna()
            rise_1h.append(d1.abs())
            rise_6h.append(d6[d6 > 0])
            fall_6h.append(-d6[d6 < 0])
            range_72h.append(rng72)

    def _q(parts: list[pd.Series]) -> dict[str, float]:
        values = pd.concat(parts) if parts else pd.Series(dtype=float)
        if values.empty:
            return {}
        out = {name: float(values.quantile(q)) for name, q in
               {"q50": 0.5, "q90": 0.9, "q99": 0.99, "q999": 0.999}.items()}
        out["max"] = float(values.max())
        return {k: round(v, 4) for k, v in out.items()}

    reference = {
        "source": str(STAGE02_RIVER_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "stations": stations,
        "hourly_readings": hours,
        "abs_change_1h_m": _q(rise_1h),
        "rise_6h_m": _q(rise_6h),
        "fall_6h_m": _q(fall_6h),
        "range_72h_m": _q(range_72h),
        "readings_dropped_as_sentinel_or_outlier": dropped,
        "max_plausible_change_1h_m": MAX_PLAUSIBLE_CHANGE_1H_M,
        "note": "Changes only, per station; station datums differ so absolute levels are not used.",
    }
    WATER_REFERENCE_JSON.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"[data] water-level reference: {stations} stations, {hours:,} hourly readings "
          f"({dropped:,} sentinel/outlier readings dropped) -> {WATER_REFERENCE_JSON.name}")
    return reference


# ---------------------------------------------------------------------------
# 4. Imagery bank
# ---------------------------------------------------------------------------

def build_imagery_bank(images_per_class: int = 12) -> pd.DataFrame:
    """Export a small labelled imagery bank for synthetic camera evidence.

    Labels come from each scene's WATER MASK, not from any model, so the Stage 02
    CNN is later scored against an independent reference. Caveat recorded in the
    manifest: these 500 scenes are very likely the Stage 02 CNN's own training
    pool, so visual evidence in the stress test is optimistic.
    """
    from datasets import load_from_disk
    from PIL import Image

    loaded = load_from_disk(str(STAGE02_SATELLITE_DIR))
    dataset = loaded["train"] if hasattr(loaded, "keys") and "train" in loaded else loaded

    fractions = []
    for index in range(len(dataset)):
        mask = np.asarray(dataset[index]["mask"].convert("L"))
        fractions.append(float((mask > 0).mean()))
    fractions = np.asarray(fractions)

    rng = np.random.default_rng(SEED)
    order = np.argsort(fractions)
    flooded_pool = [i for i in order[::-1] if fractions[i] >= FLOODED_MIN_WATER_FRACTION]
    unflooded_pool = [i for i in order if fractions[i] <= UNFLOODED_MAX_WATER_FRACTION]
    # Fall back to the most extreme scenes if a pool is thin, and record it.
    if len(flooded_pool) < images_per_class:
        flooded_pool = list(order[::-1][: images_per_class * 2])
    if len(unflooded_pool) < images_per_class:
        unflooded_pool = list(order[: images_per_class * 2])

    chosen = {
        "flooded": rng.choice(flooded_pool, size=min(images_per_class, len(flooded_pool)),
                              replace=False),
        "unflooded": rng.choice(unflooded_pool, size=min(images_per_class, len(unflooded_pool)),
                                replace=False),
    }

    IMAGERY_DIR.mkdir(parents=True, exist_ok=True)
    for stale in IMAGERY_DIR.glob("*.jpg"):
        stale.unlink()
    rows = []
    for label, indices in chosen.items():
        for index in sorted(int(i) for i in indices):
            record = dataset[index]
            image = record["image"].convert("RGB")
            image.thumbnail((IMAGE_MAX_SIDE_PX, IMAGE_MAX_SIDE_PX), Image.Resampling.LANCZOS)
            filename = f"{label}_{index:03d}.jpg"
            image.save(IMAGERY_DIR / filename, format="JPEG", quality=90)
            rows.append({
                "image_path": f"data/raw/imagery/{filename}",
                "label": label,
                "water_fraction": round(float(fractions[index]), 4),
                "source_index": index,
                "source_file_name": record.get("file_name"),
                "caveat": "likely in Stage 02 CNN training pool",
            })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(IMAGERY_MANIFEST_CSV, index=False)
    print(f"[data] imagery bank: {len(manifest)} scenes "
          f"({(manifest.label == 'flooded').sum()} flooded / "
          f"{(manifest.label == 'unflooded').sum()} unflooded) -> {IMAGERY_MANIFEST_CSV.name}")
    print(f"[data]   water-fraction distribution over all {len(fractions)} scenes: "
          f"median {np.median(fractions):.3f}, "
          f"{(fractions >= FLOODED_MIN_WATER_FRACTION).sum()} >= {FLOODED_MIN_WATER_FRACTION}, "
          f"{(fractions <= UNFLOODED_MAX_WATER_FRACTION).sum()} <= {UNFLOODED_MAX_WATER_FRACTION}")
    return manifest


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 GenAI data engineer")
    parser.add_argument("--skip-imagery", action="store_true",
                        help="Do not rebuild the imagery bank (slow: decodes 500 scenes)")
    parser.add_argument("--images-per-class", type=int, default=12)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reference = build_sensor_reference()
    distributions = summarise_distributions(reference)
    bank = build_text_phrase_bank()
    water = build_water_level_reference()
    imagery_count = None
    if not args.skip_imagery:
        imagery_count = int(len(build_imagery_bank(args.images_per_class)))
    elif IMAGERY_MANIFEST_CSV.exists():
        imagery_count = int(len(pd.read_csv(IMAGERY_MANIFEST_CSV)))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "inputs": [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in (
            STAGE01_MASTER, STAGE01_FLOOD_LOG, STAGE03_DISPATCHER, STAGE03_SOCIAL,
            STAGE02_RIVER_DIR, STAGE02_SATELLITE_DIR)],
        "sensor_rows": distributions["n_rows"],
        "class_counts": distributions["class_counts"],
        "dispatcher_template_parse_rate": bank["template_parse_rate"],
        "gauge_stations": water["stations"],
        "imagery_scenes": imagery_count,
    }
    DATA_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[data] done -> {DATA_MANIFEST_JSON.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
