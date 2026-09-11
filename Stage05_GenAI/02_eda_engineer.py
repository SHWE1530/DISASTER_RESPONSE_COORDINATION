"""Stage 05 GenAI -- EDA / Prompt Engineer.

Two jobs:

1. MEASURE THE BLIND SPOTS in the historical data -- conditions that are rare or
   entirely absent. A pipeline built on history is least tested exactly there,
   so that is where synthetic scenarios earn their keep. Every blind spot is
   computed from the data (or, for structural gaps, stated with the reason the
   data cannot represent it) rather than asserted.

2. CRAFT THE SCENARIO PROMPT LIBRARY that targets those blind spots. Each prompt
   carries a natural-language brief and a structured spec that
   03_genai_engineer.py realises. Each prompt names the blind spots it covers, so
   coverage is auditable: the script reports any rare/absent blind spot that no
   prompt targets.

Inputs:  data/processed/{sensor_reference.csv, reference_distributions.json,
                         text_phrase_bank.json, water_level_reference.json}
Outputs: data/outputs/eda/blind_spot_report.json
         data/outputs/eda/coverage_grid.csv
         data/outputs/eda/prompt_blind_spot_coverage.csv
         data/processed/scenario_prompts.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EDA_DIR = BASE_DIR / "data" / "outputs" / "eda"

SENSOR_REFERENCE_CSV = PROCESSED_DIR / "sensor_reference.csv"
DISTRIBUTIONS_JSON = PROCESSED_DIR / "reference_distributions.json"
PHRASE_BANK_JSON = PROCESSED_DIR / "text_phrase_bank.json"
WATER_REFERENCE_JSON = PROCESSED_DIR / "water_level_reference.json"
STAGE03_DISPATCHER = (
    REPO_ROOT / "Stage03_NLP" / "data" / "processed" / "Dispatcher_Log_Master_60000_Processed.csv"
)

BLIND_SPOT_JSON = EDA_DIR / "blind_spot_report.json"
COVERAGE_GRID_CSV = EDA_DIR / "coverage_grid.csv"
PROMPT_COVERAGE_CSV = EDA_DIR / "prompt_blind_spot_coverage.csv"
PROMPTS_JSON = PROCESSED_DIR / "scenario_prompts.json"

# A condition present in fewer than this share of rows is a "rare" blind spot.
RARE_SHARE = 0.02
# A coverage-grid cell holding fewer than this share of rows is "sparse".
SPARSE_CELL_SHARE = 0.005
MONSOON_MONTHS = {6, 7, 8, 9}
NIGHT_HOURS = set(range(0, 6))

# "no injuries reported" is template boilerplate, not negation of the hazard.
NEGATION_RE = re.compile(
    r"\b(?:not|never|false alarm|ignore|no (?:flood|flooding|water|one|rescue))\b", re.IGNORECASE
)


def _censored_spot(spot_id: str, name: str, values: pd.Series, detail: str) -> dict:
    """A feature whose upper tail was winsorised during Stage 01 cleaning.

    Values beyond the record are ABSENT by construction, and the rows pinned at
    the ceiling show the true extremes were clipped, not merely unobserved.
    """
    ceiling = float(values.max())
    pinned = int(np.isclose(values, ceiling, rtol=0, atol=1e-6).sum())
    return {"id": spot_id, "name": name, "kind": "statistical", "status": "absent",
            "count": 0, "share": 0.0, "ceiling": round(ceiling, 4),
            "rows_pinned_at_ceiling": pinned,
            "evidence": (f"No value exceeds the recorded ceiling of {ceiling:.4g}, and {pinned} rows "
                         f"({pinned / len(values):.2%}) sit exactly on it: the upper tail was capped "
                         f"during cleaning, so true extremes are censored. {detail}")}


def _status(count: int, share: float) -> str:
    if count == 0:
        return "absent"
    return "rare" if share < RARE_SHARE else "covered"


def _stat_spot(spot_id: str, name: str, mask: pd.Series, evidence: str) -> dict:
    count = int(mask.sum())
    share = float(mask.mean())
    return {"id": spot_id, "name": name, "kind": "statistical",
            "status": _status(count, share), "count": count,
            "share": round(share, 5), "evidence": evidence}


def _structural_spot(spot_id: str, name: str, evidence: str) -> dict:
    return {"id": spot_id, "name": name, "kind": "structural",
            "status": "absent", "count": 0, "share": 0.0, "evidence": evidence}


def find_blind_spots(ref: pd.DataFrame, dist: dict, bank: dict, water: dict,
                     dispatcher: pd.DataFrame) -> list[dict]:
    q = dist["overall"]
    margin = ref["river_margin_m"]
    severe = ref["zone_risk"] == "Severe"
    spots = []

    spots.append(_stat_spot(
        "BS01", "minority_low_class", ref["zone_risk"] == "Low",
        f"Only {dist['class_counts'].get('Low', 0)} of {dist['n_rows']} zone rows are Low risk; "
        "Stage 01's Low recall is its weakest (0.71). Calm conditions are the least rehearsed."))

    spots.append(_censored_spot(
        "BS02", "beyond_record_rainfall", ref["rainfall_mm"],
        "IMD classes >= 204.5 mm/day as 'extremely heavy'; if rainfall_mm is a daily total, the "
        "most extreme real events lie wholly outside the record."))

    spots.append(_stat_spot(
        "BS03", "urban_waterlogging",
        (ref["rainfall_mm"] >= q["rainfall_mm"]["q95"]) & (margin < -2.0),
        "Top-5% rainfall while the river sits > 2 m below its danger mark: streets flood from "
        "rain alone, a pattern a river-gauge-centred model rarely sees."))

    spots.append(_stat_spot(
        "BS04", "silent_rise",
        (margin > 2.0) & (ref["emergency_calls"] <= q["emergency_calls"]["q10"]),
        "River > 2 m above danger while calls are in the bottom decile -- a night-time rise "
        "while residents sleep."))

    spots.append(_stat_spot(
        "BS05", "receding_but_severe", severe & (ref["water_level_change_m"] < -0.2),
        "Severe zones with a clearly falling river: the forecast says 'calming', the present "
        "says 'emergency'."))

    spots.append(_stat_spot(
        "BS06", "dry_season_severe", severe & ~ref["month"].isin(MONSOON_MONTHS),
        "Severe events outside June-September. Stage 01 uses an is_monsoon feature, so an "
        "out-of-season flood tests whether that prior over-rides the sensors."))

    spots.append(_stat_spot(
        "BS07", "night_onset", ref["hour"].isin(NIGHT_HOURS),
        "Events recorded between 00:00 and 05:59 (the mission's 2 AM window)."))

    severe_per_ts = ref[severe].groupby("timestamp").size()
    all_ts = ref["timestamp"].nunique()
    concurrent = int((severe_per_ts >= 3).sum())
    spots.append({
        "id": "BS08", "name": "multi_zone_simultaneity", "kind": "statistical",
        "status": _status(concurrent, concurrent / max(1, all_ts)),
        "count": concurrent, "share": round(concurrent / max(1, all_ts), 5),
        "evidence": (f"Of {all_ts:,} distinct timestamps, {concurrent} have >= 3 Severe zones at "
                     f"once (max concurrent: {int(severe_per_ts.max()) if len(severe_per_ts) else 0}). "
                     "Every row is one zone in isolation; competing zones are not represented."),
    })

    spots.append(_structural_spot(
        "BS09", "missing_or_faulty_telemetry",
        f"The labelled sensor data has {dist['missing_values']} missing values: gauges never go "
        "offline, freeze, or report impossible readings in the record."))

    spots.append(_structural_spot(
        "BS10", "infrastructure_failure",
        "No field describes power, telecom or bridge integrity as a failing system. A grid "
        "outage that blinds cameras and gauges is unrepresentable."))

    spots.append(_structural_spot(
        "BS11", "cross_modal_conflict",
        "No joint dataset links sensors, imagery and text for the same event, so modality "
        "disagreement (a calm gauge beside people trapped) has never been exercised."))

    template_rate = bank.get("template_parse_rate", 1.0)
    untemplated = int(round((1 - template_rate) * bank.get("rows_sampled", 0)))
    spots.append({
        "id": "BS12", "name": "noisy_or_code_mixed_text", "kind": "statistical",
        "status": _status(untemplated, 1 - template_rate), "count": untemplated,
        "share": round(1 - template_rate, 5),
        "evidence": (f"{template_rate:.1%} of sampled dispatcher messages fit one fixed template. "
                     "Typos, all-caps panic, Hinglish and truncated SMS are absent."),
    })

    negation = dispatcher["text"].astype(str).str.contains(NEGATION_RE)
    spots.append(_stat_spot(
        "BS13", "negated_hazard_language", negation,
        "Messages that mention a hazard only to deny it ('no flooding here'). Stage 03 has no "
        "negation handling (documented limitation)."))

    spots.append(_censored_spot(
        "BS14", "beyond_record_water_change", ref["water_level_change_m"],
        f"Real CWC gauges (Stage 02 raw) show 6 h rises of "
        f"{water.get('rise_6h_m', {}).get('q99', 'n/a')} m at the 99th percentile; dam-release "
        "step changes sit beyond the labelled record."))

    headcount = pd.to_numeric(dispatcher["headcount_entity"], errors="coerce")
    spots.append(_stat_spot(
        "BS15", "mass_casualty_headcount", headcount > 150,
        f"Largest dispatcher headcount is {int(headcount.max())}; hospital-scale evacuations "
        "(hundreds of people) never appear."))

    multi = dispatcher["hazard_type"].astype(str).str.contains(r"[;,]")
    spots.append(_stat_spot(
        "BS16", "multi_hazard_message", multi,
        f"Every one of {len(dispatcher):,} dispatcher messages carries exactly one hazard label; "
        "compound flood + road + medical reports are absent."))

    spots.append(_structural_spot(
        "BS17", "forward_looking_labels",
        "zone_risk labels describe the present. Nothing marks a zone that is Moderate now but "
        "will be Severe in 6 h, so pre-positioning decisions are never supervised."))
    return spots


def build_coverage_grid(ref: pd.DataFrame, dist: dict) -> tuple[pd.DataFrame, list[dict]]:
    q = dist["overall"]["rainfall_mm"]
    rain_edges = [-np.inf, q["q25"], q["q50"], q["q75"], q["q95"], np.inf]
    rain_labels = ["<q25", "q25-q50", "q50-q75", "q75-q95", ">q95"]
    margin_edges = [-np.inf, -2.0, 0.0, 2.0, np.inf]
    margin_labels = ["< -2 m", "-2..0 m", "0..2 m", "> 2 m"]
    frame = ref.assign(
        rain_bin=pd.cut(ref["rainfall_mm"], rain_edges, labels=rain_labels),
        margin_bin=pd.cut(ref["river_margin_m"], margin_edges, labels=margin_labels),
    )
    grid = (frame.groupby(["rain_bin", "margin_bin", "zone_risk"], observed=False)
            .size().rename("count").reset_index())
    grid["share"] = (grid["count"] / len(ref)).round(5)
    sparse = grid[grid["share"] < SPARSE_CELL_SHARE]
    sparse_cells = [
        {"rain_bin": str(r.rain_bin), "margin_bin": str(r.margin_bin),
         "zone_risk": r.zone_risk, "count": int(r["count"])}
        for _, r in sparse.iterrows()
    ]
    return grid, sparse_cells


# ---------------------------------------------------------------------------
# Scenario prompt library
# ---------------------------------------------------------------------------
# Structured specs are the contract with 03_genai_engineer.py. Vocabulary:
#   true_severity  ROUTINE | ELEVATED | URGENT | CRITICAL  (designer's ground truth)
#   hazards        Flood | Rescue Emergency | Road Blockage | Medical Emergency
#   modifiers      see ZONE_MODIFIERS in 03_genai_engineer.py
#   evidence       sensors (bool), text (dispatcher|social|mixed|None),
#                  image (flooded|unflooded|None), water_history (shape|None)
#   expect         optional overrides of the default pass criteria

def _zone(label, severity, hazards, modifiers=(), sensors=True, text="dispatcher",
          image=None, water="flat", **extra):
    zone = {
        "label": label,
        "true_severity": severity,
        "hazards": list(hazards),
        "modifiers": list(modifiers),
        "evidence": {"sensors": sensors, "text": text, "image": image, "water_history": water},
    }
    zone.update(extra)
    return zone


SCENARIO_PROMPTS = [
    {
        "id": "S01", "name": "Baseline river overflow", "archetype": "baseline_river_overflow",
        "blind_spots": ["BS08"],
        "prompt": ("Generate a textbook monsoon river-overflow event across three Bihar districts "
                   "at staggered severities, with full sensor, text, imagery and gauge evidence, "
                   "as a sanity baseline the pipeline should handle cleanly."),
        "state": "Bihar", "start_time": "20-08-2026 14:00", "global_modifiers": [],
        "resources": {"ambulances": 4, "rescue_boats": 3, "shelter_beds": 200},
        "zones": [
            _zone("Riverside ward", "CRITICAL", ["Flood", "Rescue Emergency"], ["river_overflow"],
                  image="flooded", water="steady_rise"),
            _zone("Market ward", "URGENT", ["Flood"], ["river_overflow"], water="steady_rise"),
            _zone("Upland ward", "ELEVATED", ["Flood"]),
        ],
    },
    {
        "id": "S02", "name": "The 2 AM mission", "archetype": "night_flash_flood",
        "blind_spots": ["BS02", "BS08", "BS14"],
        "prompt": ("It is 2:00 AM. A river gauge spikes and water rises rapidly toward three "
                   "densely populated neighbourhoods of one Assam district. One ambulance fleet, "
                   "one 40-bed shelter. Generate the sensor surge, the 72 h gauge trace, and the "
                   "dispatcher and citizen messages as they would arrive."),
        "state": "Assam", "same_district": True, "start_time": "14-08-2026 02:00",
        "global_modifiers": ["night"],
        "resources": {"ambulances": 1, "rescue_boats": 1, "shelter_beds": 40},
        "zones": [
            _zone("Riverside settlement", "CRITICAL", ["Flood", "Rescue Emergency"], ["flash_flood"],
                  image="flooded", water="sharp_rise", location="riverside settlement"),
            _zone("Low-lying colony", "CRITICAL", ["Flood", "Rescue Emergency"],
                  ["flash_flood", "implicit_urgency"], text="social", water="sharp_rise",
                  location="low-lying area"),
            _zone("Old town", "URGENT", ["Flood", "Medical Emergency"], ["river_overflow"],
                  water="steady_rise", location="old town area"),
        ],
    },
    {
        "id": "S03", "name": "Cyclonic coastal landfall", "archetype": "cyclonic_rain",
        "blind_spots": ["BS02", "BS08"],
        "prompt": ("A landfalling cyclone drives extreme rain across four Odisha coastal districts; "
                   "rescue, medical and road-access demands arrive together."),
        "state": "Odisha", "start_time": "22-10-2026 18:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 4, "shelter_beds": 150},
        "zones": [
            _zone("Coastal block", "CRITICAL", ["Flood", "Rescue Emergency"], ["cyclonic_rain"],
                  image="flooded", water="steady_rise"),
            _zone("Fishing villages", "CRITICAL", ["Rescue Emergency", "Medical Emergency"],
                  ["cyclonic_rain"], text="mixed", water="steady_rise"),
            _zone("Highway corridor", "URGENT", ["Road Blockage", "Flood"], ["cyclonic_rain"],
                  water="steady_rise"),
            _zone("Inland town", "ELEVATED", ["Flood"], image="unflooded"),
        ],
    },
    {
        "id": "S04", "name": "Urban cloudburst waterlogging", "archetype": "urban_waterlogging",
        "blind_spots": ["BS03"],
        "prompt": ("An intense cloudburst floods city streets in two Maharashtra districts while "
                   "the river stays well below its danger level."),
        "state": "Maharashtra", "start_time": "05-07-2026 09:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 1, "shelter_beds": 120},
        "zones": [
            _zone("Central wards", "URGENT", ["Flood", "Road Blockage"], ["urban_waterlogging"],
                  image="flooded"),
            _zone("Suburban junctions", "ELEVATED", ["Road Blockage"], ["urban_waterlogging"]),
        ],
    },
    {
        "id": "S05", "name": "Negated flood rumour", "archetype": "negation_false_alarm",
        "blind_spots": ["BS13", "BS01"],
        "prompt": ("Calm conditions in two Kerala districts, but one receives a message that "
                   "explicitly denies a flood rumour -- full of flood vocabulary, yet negated."),
        "state": "Kerala", "start_time": "11-09-2026 16:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 80},
        "zones": [
            _zone("Backwater ward", "ROUTINE", ["Flood"], ["calm", "negation_text"],
                  image="unflooded", expect={"max_priority": "ELEVATED"}),
            _zone("Hill ward", "ROUTINE", ["Flood"], ["calm"], expect={"max_priority": "ELEVATED"}),
        ],
    },
    {
        "id": "S06", "name": "Silent rise while residents sleep", "archetype": "silent_rise",
        "blind_spots": ["BS04"],
        "prompt": ("The river crosses its danger level at 3 AM while residents sleep: sensors "
                   "scream, but almost nobody calls and no messages arrive."),
        "state": "Uttar Pradesh", "start_time": "03-08-2026 03:00", "global_modifiers": ["night"],
        "resources": {"ambulances": 2, "rescue_boats": 2, "shelter_beds": 100},
        "zones": [
            _zone("Floodplain village", "CRITICAL", ["Flood"], ["silent_rise"], text=None,
                  water="sharp_rise"),
            _zone("Riverside hamlet", "URGENT", ["Flood"], ["silent_rise"], text=None,
                  water="steady_rise"),
        ],
    },
    {
        "id": "S07", "name": "Telemetry dropout", "archetype": "sensor_dropout",
        "blind_spots": ["BS09"],
        "prompt": ("Gauge telemetry for two West Bengal districts drops out mid-event; responders "
                   "must rely on dispatcher and citizen text alone."),
        "state": "West Bengal", "start_time": "27-07-2026 11:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 2, "shelter_beds": 120},
        "zones": [
            _zone("Char lands", "CRITICAL", ["Flood", "Rescue Emergency"], ["river_overflow"],
                  sensors=False, water=None),
            _zone("Canal colony", "URGENT", ["Flood"], ["river_overflow", "implicit_urgency"],
                  sensors=False, text="social", water=None),
            _zone("Town centre", "ELEVATED", ["Flood"]),
        ],
    },
    {
        "id": "S08", "name": "Gauge logger malfunction", "archetype": "gauge_malfunction",
        "blind_spots": ["BS09"],
        "prompt": ("A gauge logger fails and reports an impossible 25 m spike, while an honest "
                   "neighbouring gauge shows a genuine steady rise."),
        "state": "Gujarat", "start_time": "16-08-2026 07:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 90},
        "zones": [
            _zone("Faulty-gauge reach", "ELEVATED", ["Flood"], ["gauge_malfunction"],
                  water="ood_spike", expect={"max_priority": "URGENT"}),
            _zone("Downstream reach", "URGENT", ["Flood"], ["river_overflow"], water="steady_rise"),
        ],
    },
    {
        "id": "S09", "name": "People versus sensors", "archetype": "text_sensor_conflict",
        "blind_spots": ["BS11"],
        "prompt": ("Sensors and people disagree. In one Karnataka district the gauges lag a real "
                   "flash flood people are already trapped in; in another, a prank message claims "
                   "mass casualties under calm skies."),
        "state": "Karnataka", "start_time": "09-08-2026 21:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 80},
        "zones": [
            _zone("Lagging-gauge valley", "CRITICAL", ["Rescue Emergency", "Flood"], [],
                  sensor_class="Low",
                  expect={"min_priority": "URGENT", "conflict": True, "human_review": True}),
            _zone("Prank-message town", "ROUTINE", ["Flood"], ["calm", "false_alarm_text"],
                  expect={"conflict": True, "human_review": True}),
        ],
    },
    {
        "id": "S10", "name": "Camera versus gauge", "archetype": "visual_sensor_conflict",
        "blind_spots": ["BS11", "BS06"],
        "prompt": ("Camera evidence contradicts sensors in two Tamil Nadu districts during the "
                   "north-east monsoon: one camera shows a flooded street under a 'moderate' gauge, "
                   "another gauge reads severe beside a dry street."),
        "state": "Tamil Nadu", "start_time": "02-12-2026 10:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 2, "shelter_beds": 120},
        "zones": [
            _zone("Flooded-camera ward", "URGENT", ["Flood"], [], sensor_class="Moderate",
                  image="flooded"),
            _zone("Dry-camera ward", "ELEVATED", ["Flood"], [], sensor_class="Severe",
                  text_severity="MEDIUM", image="unflooded",
                  expect={"conflict": True, "human_review": True}),
        ],
    },
    {
        "id": "S11", "name": "Receding but still flooded", "archetype": "receding_flood",
        "blind_spots": ["BS05"],
        "prompt": ("The river has peaked and is falling, but neighbourhoods remain under water: "
                   "the falling forecast must not talk the present-tense emergency down."),
        "state": "Bihar", "start_time": "30-08-2026 15:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 3, "shelter_beds": 150},
        "zones": [
            _zone("Inundated colony", "URGENT", ["Flood", "Rescue Emergency"], ["receding"],
                  image="flooded", water="falling", expect={"min_priority": "URGENT"}),
            _zone("Drying fringe", "ELEVATED", ["Flood"], ["receding"], water="falling"),
        ],
    },
    {
        "id": "S12", "name": "Bridge collapse isolation", "archetype": "bridge_collapse",
        "blind_spots": ["BS10"],
        "prompt": ("A bridge collapses under flood load, cutting a Karnataka hill district's only "
                   "access road; casualties need medical evacuation."),
        "state": "Karnataka", "start_time": "12-07-2026 13:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 60},
        "zones": [
            _zone("Cut-off valley", "CRITICAL", ["Road Blockage", "Rescue Emergency", "Medical Emergency"],
                  ["bridge_collapse", "river_overflow"], image="flooded", water="steady_rise"),
            _zone("Approach road", "URGENT", ["Road Blockage"], ["bridge_collapse"]),
        ],
    },
    {
        "id": "S13", "name": "Hospital ground floor flooding", "archetype": "mass_casualty",
        "blind_spots": ["BS15"],
        "prompt": ("Floodwater enters a district government hospital's ground floor before dawn; "
                   "hundreds of patients and staff need evacuation."),
        "state": "Maharashtra", "start_time": "25-07-2026 05:00", "global_modifiers": [],
        "resources": {"ambulances": 4, "rescue_boats": 2, "shelter_beds": 200},
        "zones": [
            _zone("Hospital campus", "CRITICAL", ["Flood", "Medical Emergency"],
                  ["river_overflow", "mass_casualty"], image="flooded", water="steady_rise",
                  location="government hospital"),
            _zone("Clinic district", "ELEVATED", ["Medical Emergency"]),
        ],
    },
    {
        "id": "S14", "name": "Night dam spillway release", "archetype": "dam_release",
        "blind_spots": ["BS14", "BS02"],
        "prompt": ("An upstream dam opens its spillway gates at night; downstream Kerala districts "
                   "see a step jump in river level within hours."),
        "state": "Kerala", "start_time": "16-08-2026 01:00", "global_modifiers": ["night"],
        "resources": {"ambulances": 3, "rescue_boats": 3, "shelter_beds": 150},
        "zones": [
            _zone("Spillway reach", "CRITICAL", ["Flood", "Rescue Emergency"], ["dam_release"],
                  image="flooded", water="step_jump"),
            _zone("Island ward", "CRITICAL", ["Flood"], ["dam_release"], text="mixed",
                  water="step_jump"),
            _zone("Lower reach", "URGENT", ["Flood"], ["river_overflow"], water="steady_rise"),
        ],
    },
    {
        "id": "S15", "name": "Five-state monsoon cascade", "archetype": "multi_state_ranking",
        "blind_spots": ["BS08"],
        "prompt": ("A monsoon trough floods five districts in five states at once, each at a "
                   "different severity: can the system rank them correctly?"),
        "state": None, "start_time": "18-08-2026 12:00", "global_modifiers": [],
        "resources": {"ambulances": 6, "rescue_boats": 6, "shelter_beds": 500},
        "zones": [
            _zone("Brahmaputra plain", "CRITICAL", ["Flood", "Rescue Emergency"], ["river_overflow"],
                  image="flooded", water="steady_rise", state="Assam"),
            _zone("Kosi basin", "URGENT", ["Flood"], ["river_overflow"], water="steady_rise",
                  state="Bihar"),
            _zone("Delta towns", "URGENT", ["Flood", "Road Blockage"], [], water="steady_rise",
                  state="West Bengal"),
            _zone("Mahanadi upland", "ELEVATED", ["Flood"], state="Odisha"),
            _zone("Doab farmland", "ROUTINE", ["Flood"], ["calm"], image="unflooded",
                  state="Uttar Pradesh"),
        ],
    },
    {
        "id": "S16", "name": "Out-of-season cloudburst", "archetype": "dry_season_anomaly",
        "blind_spots": ["BS06"],
        "prompt": ("An out-of-season February cloudburst triggers a flash flood where no monsoon "
                   "is expected."),
        "state": "Tamil Nadu", "start_time": "18-02-2026 14:00", "global_modifiers": ["dry_season"],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 80},
        "zones": [
            _zone("Cloudburst catchment", "URGENT", ["Flood"], ["flash_flood"], water="sharp_rise"),
            _zone("Neighbouring taluk", "ELEVATED", ["Flood"]),
        ],
    },
    {
        "id": "S17", "name": "Garbled panic messages", "archetype": "comms_noise",
        "blind_spots": ["BS12"],
        "prompt": ("Messages arrive garbled: typos, all caps, Hinglish, truncated SMS -- the "
                   "language of real panic, not of templates."),
        "state": "Assam", "start_time": "19-08-2026 20:00", "global_modifiers": ["comms_noise"],
        "resources": {"ambulances": 2, "rescue_boats": 2, "shelter_beds": 100},
        "zones": [
            _zone("Embankment breach", "CRITICAL", ["Flood", "Rescue Emergency"],
                  ["river_overflow", "implicit_urgency"], text="social", water="steady_rise"),
            _zone("Tea-garden lines", "URGENT", ["Flood"], ["river_overflow"], water="steady_rise"),
            _zone("Market town", "ELEVATED", ["Flood"]),
        ],
    },
    {
        "id": "S18", "name": "All quiet (do not cry wolf)", "archetype": "low_class_cluster",
        "blind_spots": ["BS01"],
        "prompt": ("Three genuinely calm Gujarat districts with routine reports -- the rare Low "
                   "class. A good system does not cry wolf."),
        "state": "Gujarat", "start_time": "10-03-2026 11:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 1, "shelter_beds": 80},
        "zones": [
            _zone("Coastal ward", "ROUTINE", ["Flood"], ["calm"], image="unflooded",
                  expect={"max_priority": "ELEVATED"}),
            _zone("Industrial belt", "ROUTINE", ["Road Blockage"], ["calm"],
                  expect={"max_priority": "ELEVATED"}),
            _zone("Old city", "ROUTINE", ["Medical Emergency"], ["calm"],
                  expect={"max_priority": "ELEVATED"}),
        ],
    },
    {
        "id": "S19", "name": "Four hazards, one district", "archetype": "multi_hazard_zone",
        "blind_spots": ["BS16"],
        "prompt": ("One Uttar Pradesh district faces flood, road blockage, medical emergency and "
                   "rescue at once; reports mix hazards in a single message."),
        "state": "Uttar Pradesh", "start_time": "28-07-2026 16:00", "global_modifiers": [],
        "resources": {"ambulances": 3, "rescue_boats": 2, "shelter_beds": 120},
        "zones": [
            _zone("Compound-hazard ward", "CRITICAL",
                  ["Flood", "Road Blockage", "Medical Emergency", "Rescue Emergency"],
                  ["river_overflow"], text="mixed", image="flooded", water="steady_rise"),
            _zone("Adjacent ward", "URGENT", ["Flood", "Medical Emergency"], [], water="steady_rise"),
        ],
    },
    {
        "id": "S20", "name": "Slow onset, pre-position now", "archetype": "slow_onset",
        "blind_spots": ["BS17"],
        "prompt": ("Conditions in a West Bengal district are only moderate now, but the river is "
                   "climbing fast: the right call is to pre-position before it becomes an emergency."),
        "state": "West Bengal", "start_time": "06-08-2026 10:00", "global_modifiers": [],
        "resources": {"ambulances": 2, "rescue_boats": 2, "shelter_beds": 100},
        "zones": [
            _zone("Rising-river ward", "URGENT", ["Flood"], [], sensor_class="Moderate",
                  text_severity="MEDIUM", water="sharp_rise", expect={"min_priority": "URGENT"}),
            _zone("Quiet ward", "ROUTINE", ["Flood"], ["calm"]),
        ],
    },
]

# The team's "Wildcard": a compound event nobody had considered. Defended in README.md.
WILDCARD_PROMPT = {
    "id": "W01", "name": "Flash flood + municipal blackout", "archetype": "wildcard_blackout",
    "blind_spots": ["BS10", "BS09", "BS11", "BS02", "BS08"],
    "prompt": ("At 2 AM a flash flood hits an Assam district at the same moment the municipal "
               "grid fails. Gauges, street cameras and cell towers go dark; only battery- and "
               "generator-backed equipment survives. Four neighbourhoods -- one goes completely "
               "silent. Silence must never be read as safety."),
    "state": "Assam", "same_district": True, "start_time": "14-08-2026 02:00",
    "global_modifiers": ["night", "power_outage"],
    "resources": {"ambulances": 1, "rescue_boats": 1, "shelter_beds": 40},
    "zones": [
        _zone("Riverside settlement (battery gauge)", "CRITICAL", ["Flood", "Rescue Emergency"],
              ["flash_flood", "battery_backup"], image="flooded", water="sharp_rise",
              location="riverside settlement"),
        _zone("Low-lying colony (grid down)", "CRITICAL", ["Flood", "Rescue Emergency"],
              ["flash_flood", "implicit_urgency"], text="social", image="flooded",
              water="sharp_rise", location="low-lying area"),
        _zone("District hospital (generator)", "URGENT", ["Medical Emergency", "Flood"],
              ["river_overflow", "generator_backup"], image="unflooded", water="steady_rise",
              location="government hospital"),
        _zone("Char island (no signal)", "CRITICAL", ["Flood"], ["flash_flood", "total_blackout"],
              image="flooded", water="sharp_rise", location="riverside settlement"),
    ],
}


def prompt_coverage(spots: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    known = {spot["id"] for spot in spots}
    prompts = SCENARIO_PROMPTS + [WILDCARD_PROMPT]
    for prompt in prompts:
        unknown = sorted(set(prompt["blind_spots"]) - known)
        if unknown:
            raise ValueError(f"Prompt {prompt['id']} targets unknown blind spots: {unknown}")
    matrix = pd.DataFrame(
        [{spot["id"]: int(spot["id"] in p["blind_spots"]) for spot in spots} | {"prompt_id": p["id"]}
         for p in prompts]
    ).set_index("prompt_id")
    needed = [spot["id"] for spot in spots if spot["status"] in {"absent", "rare"}]
    uncovered = [spot_id for spot_id in needed if matrix[spot_id].sum() == 0]
    return matrix, uncovered


def main() -> None:
    for path in (SENSOR_REFERENCE_CSV, DISTRIBUTIONS_JSON, PHRASE_BANK_JSON, WATER_REFERENCE_JSON):
        if not path.exists():
            raise SystemExit(f"Missing {path.name}. Run `python Stage05_GenAI/01_data_engineer.py` first.")
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(SENSOR_REFERENCE_CSV)
    dist = json.loads(DISTRIBUTIONS_JSON.read_text(encoding="utf-8"))
    bank = json.loads(PHRASE_BANK_JSON.read_text(encoding="utf-8"))
    water = json.loads(WATER_REFERENCE_JSON.read_text(encoding="utf-8"))
    dispatcher = pd.read_csv(STAGE03_DISPATCHER, usecols=["text", "hazard_type", "headcount_entity"])

    spots = find_blind_spots(ref, dist, bank, water, dispatcher)
    grid, sparse_cells = build_coverage_grid(ref, dist)
    grid.to_csv(COVERAGE_GRID_CSV, index=False)

    matrix, uncovered = prompt_coverage(spots)
    matrix.to_csv(PROMPT_COVERAGE_CSV)

    summary = {status: sum(1 for s in spots if s["status"] == status)
               for status in ("absent", "rare", "covered")}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rare_share_threshold": RARE_SHARE,
        "sparse_cell_threshold": SPARSE_CELL_SHARE,
        "summary": summary,
        "blind_spots": spots,
        "sparse_cells": sparse_cells,
        "uncovered_blind_spots": uncovered,
        "prompts_per_blind_spot": {col: int(matrix[col].sum()) for col in matrix.columns},
    }
    BLIND_SPOT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    PROMPTS_JSON.write_text(json.dumps({
        "generated_at": report["generated_at"],
        "blind_spot_report": str(BLIND_SPOT_JSON.relative_to(BASE_DIR)).replace("\\", "/"),
        "prompts": SCENARIO_PROMPTS,
        "wildcard": WILDCARD_PROMPT,
    }, indent=2), encoding="utf-8")

    print(f"[eda] blind spots: {summary['absent']} absent, {summary['rare']} rare, "
          f"{summary['covered']} covered")
    for spot in spots:
        share = f"{spot['share']:.2%}" if spot["kind"] == "statistical" else "structural"
        print(f"  {spot['id']} {spot['name']:<30} {spot['status']:<8} {share}")
    print(f"[eda] sparse coverage-grid cells (< {SPARSE_CELL_SHARE:.1%} of rows): {len(sparse_cells)}")
    print(f"[eda] prompt library: {len(SCENARIO_PROMPTS)} scenarios + wildcard -> {PROMPTS_JSON.name}")
    if uncovered:
        print(f"[eda] WARNING: rare/absent blind spots with no prompt: {uncovered}")
    else:
        print("[eda] every rare/absent blind spot is targeted by at least one prompt")


if __name__ == "__main__":
    main()
