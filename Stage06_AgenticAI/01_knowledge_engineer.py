"""Stage 06 Agentic AI -- Data / Knowledge Engineer.

Builds the knowledge base the agents reason over, and defines the tools they
are allowed to call.

Knowledge base (data/processed/)
  sop_index.joblib          TF-IDF index over the national disaster-management SOP
                            corpus (NDMA national plan, NIDM and state SOPs), the
                            same 1,697 passages Stage 03/04 ingested, de-noised here
  district_profiles.json    per-district flood history from the Stage 01 record:
                            rows, severe rate, danger threshold, typical impact
  resource_catalog.json     inventory types, which Stage 03 resource phrases each
                            one satisfies (from the Stage 04 domain dictionary),
                            the planning ratios, and a default inventory taken from
                            the Stage 05 scenario suite
  glossary.json             the Stage 04 domain dictionary (agencies, hazards, terms)
  tool_registry.json        every tool as an MCP tool definition (name, title,
                            description, inputSchema, annotations)
  knowledge_manifest.json   sources, filtering counts, retrieval check

KnowledgeBase
  search_sop(query, hazard, phase, top_k)   ranked, cited SOP passages
  district_profile(state, district)         historical context for one district
  resource_types / map_resource(phrase)     requested resource -> inventory type

Nothing in Stages 01-05 is modified; their data is read only.

Usage:
  python Stage06_AgenticAI/01_knowledge_engineer.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SOP_CSV = REPO_ROOT / "Stage04_SLM" / "data" / "processed" / "Safety_SOP_Dataset_PROCESSED.csv"
DICTIONARY_CSV = REPO_ROOT / "Stage04_SLM" / "data" / "processed" / "SLM_Domain_Dictionary.csv"
SENSOR_RECORD_CSV = REPO_ROOT / "Stage01_ML" / "data" / "processed" / "Master_Processed_Dataset.csv"
SCENARIO_SUITE_JSON = REPO_ROOT / "Stage05_GenAI" / "data" / "outputs" / "scenarios" / "scenario_suite.json"

SOP_INDEX = PROCESSED_DIR / "sop_index.joblib"
DISTRICT_JSON = PROCESSED_DIR / "district_profiles.json"
RESOURCE_JSON = PROCESSED_DIR / "resource_catalog.json"
GLOSSARY_JSON = PROCESSED_DIR / "glossary.json"
TOOL_REGISTRY_JSON = PROCESSED_DIR / "tool_registry.json"
MANIFEST_JSON = PROCESSED_DIR / "knowledge_manifest.json"

PRIORITY_LEVELS = ["ROUTINE", "ELEVATED", "URGENT", "CRITICAL"]
SOP_PHASES = ["Early_Warning", "General", "Preparedness", "Recovery", "Relief", "Response", "Trigger"]
RESOURCE_TYPES = ["rescue_boats", "ambulances", "shelter_beds"]

# SOP filtering: passages that cannot guide an action are removed, with counts.
MIN_SOP_CHARS = 60
GLOSSARY_SEPARATORS = 3  # "SEC : State Executive Committee SEOC : ..." is an acronym list

# Retrieval re-ranking. Cosine similarity carries the ranking; metadata only
# breaks near-ties toward passages about the right hazard, phase and action.
HAZARD_BOOST = 0.10
PHASE_BOOST = 0.05
ACTION_BOOST = 0.10

# Stage 03 hazard label -> what to prefer in the SOP corpus.
HAZARD_SOP_HINTS = {
    "Flood": {"hazard": "Flood", "action": "Search_and_Rescue"},
    "Rescue Emergency": {"hazard": "Flood", "action": "Search_and_Rescue"},
    "Medical Emergency": {"hazard": None, "action": "Medical_Care"},
    "Road Blockage": {"hazard": None, "action": "Resource_Management"},
}

# Planning ratios. Named policy constants, not learned values: one boat crew
# evacuates roughly this many people in a 12-hour operational period, one
# ambulance covers roughly this many affected people when injuries are reported.
PEOPLE_PER_BOAT = 50
PEOPLE_PER_AMBULANCE = 40

# Inventory type -> keywords that mark a Stage 03 resource phrase as needing it.
RESOURCE_KEYWORDS = {
    "rescue_boats": ["boat", "rescue", "life jacket", "rope"],
    "ambulances": ["ambulance", "medical", "paramedic", "first-aid"],
    "shelter_beds": ["relief", "shelter", "food"],
}

FALLBACK_INVENTORY = {"ambulances": 2, "rescue_boats": 2, "shelter_beds": 100}

# Where unmet demand is rerouted when the incident's own units run out
# (agencies named in the Stage 04 domain dictionary).
SECONDARY_RESPONDERS = {
    "rescue_boats": "SDRF / NDRF flood rescue team",
    "ambulances": "District EMS (108) / nearest hospital ambulance",
    "shelter_beds": "DDMA relief camp",
}

# Hierarchical chain of command an escalation can be routed to.
ESCALATION_LEVELS = ["field_supervisor", "district_eoc", "state_eoc", "incident_commander"]

# Retrieval sanity check: a query and the SOP label a relevant passage carries.
RETRIEVAL_PROBES = [
    ("set up relief camps with food and water for displaced people", "Relief"),
    ("issue early warning alerts to the population before landfall", "Early_Warning"),
    ("provide medical care and first aid to injured victims", "Medical_Care"),
    ("search and rescue of people trapped by flood water", "Search_and_Rescue"),
    ("mobilise equipment and stockpile resources for response", "Resource_Management"),
    ("assess damage to houses and infrastructure after the disaster", "Damage_Assessment"),
    ("activate the emergency operations centre", "Emergency_Operations"),
    ("communicate public information through media and messaging", "Communication"),
]

# ---------------------------------------------------------------------------
# Tool registry, in MCP tool-definition format. Project-specific facts live in
# _meta, which MCP reserves for exactly this.
# ---------------------------------------------------------------------------

_SENSOR_SCHEMA = {"type": "object", "description": "One Stage 01 sensor record (12 fields)."}
_LEVELS_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 72,
                  "description": "Hourly river levels in metres, oldest first (72 h)."}
_RESOURCE_ENUM = {"type": "string", "enum": RESOURCE_TYPES}


def _tool(name, title, description, properties, required, *, agent, stage, read_only=True,
          destructive=False, requires_approval=False):
    return {
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required,
                        "additionalProperties": False},
        "annotations": {"readOnlyHint": read_only, "destructiveHint": destructive,
                        "idempotentHint": read_only, "openWorldHint": False},
        "_meta": {"agent": agent, "backed_by": stage, "requires_human_approval": requires_approval},
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _tool("query_sensor_risk_score", "Sensor zone risk (Stage 01)",
          "Score a zone's sensor telemetry with the calibrated Stage 01 XGBoost model.",
          {"sensors": _SENSOR_SCHEMA}, ["sensors"], agent="dispatcher", stage="Stage01_ML"),
    _tool("predict_visual_flood", "Visual flood check (Stage 02 CNN)",
          "Classify a camera or drone image as flooded or unflooded.",
          {"image_path": {"type": "string", "minLength": 1}}, ["image_path"],
          agent="dispatcher", stage="Stage02_DL"),
    _tool("forecast_water_level", "River forecast (Stage 02 LSTM)",
          "Forecast the next 6 hours of river level, with an out-of-distribution check.",
          {"water_levels": _LEVELS_SCHEMA}, ["water_levels"], agent="dispatcher", stage="Stage02_DL"),
    _tool("parse_emergency_text", "Emergency text analysis (Stage 03)",
          "Urgency, hazard type, location, requested resources and headcount from a free-text report.",
          {"raw_text": {"type": "string", "minLength": 1, "maxLength": 5000}}, ["raw_text"],
          agent="dispatcher", stage="Stage03_NLP"),
    _tool("assess_zone", "Fused zone decision (fusion layer)",
          "Fuse any subset of sensors, text, image and gauge history into one priority with "
          "conflicts, escalations and a human-review flag.",
          {"sensors": _SENSOR_SCHEMA, "text": {"type": "string"}, "image_path": {"type": "string"},
           "water_levels": _LEVELS_SCHEMA}, [], agent="dispatcher", stage="fusion"),
    _tool("generate_tactical_briefing", "Tactical briefing (Stage 04)",
          "Condense a multi-entry incident log into situation, risk and actions.",
          {"incident_log": {"type": "string", "minLength": 1, "maxLength": 20000}}, ["incident_log"],
          agent="dispatcher", stage="Stage04_SLM"),
    _tool("get_district_profile", "District flood history",
          "Historical flood profile of a district from the Stage 01 sensor record.",
          {"state": {"type": "string", "minLength": 1}, "district": {"type": "string", "minLength": 1}},
          ["state", "district"], agent="dispatcher", stage="knowledge_base"),
    _tool("search_sop", "Search disaster-management SOPs",
          "Retrieve cited passages from the national disaster-management SOP corpus.",
          {"query": {"type": "string", "minLength": 3}, "hazard": {"type": "string"},
           "phase": {"type": "string", "enum": SOP_PHASES},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
          ["query"], agent="dispatcher", stage="knowledge_base"),
    _tool("recall_precedents", "Recall precedents",
          "Memory-augmented reasoning: past agent and commander decisions for the same district.",
          {"state": {"type": "string", "minLength": 1}, "district": {"type": "string", "minLength": 1},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
          ["state", "district"], agent="dispatcher", stage="agent_memory"),
    _tool("check_resource_inventory", "Check inventory",
          "Remaining and reserved units of each resource type in this incident.",
          {}, [], agent="allocator", stage="run_state"),
    _tool("reserve_resources", "Reserve resources",
          "Reserve units of one resource type for a zone. Fails rather than over-commit.",
          {"zone_id": {"type": "string", "minLength": 1}, "resource_type": _RESOURCE_ENUM,
           "quantity": {"type": "integer", "minimum": 1}},
          ["zone_id", "resource_type", "quantity"], agent="allocator", stage="run_state",
          read_only=False),
    _tool("release_resources", "Release resources",
          "Return a zone's reserved units of one resource type to the inventory.",
          {"zone_id": {"type": "string", "minLength": 1}, "resource_type": _RESOURCE_ENUM,
           "quantity": {"type": "integer", "minimum": 1}},
          ["zone_id", "resource_type", "quantity"], agent="coordinator", stage="run_state",
          read_only=False),
    _tool("escalate_to_human", "Escalate to human command",
          "Put a decision in front of a human operator: approval, verification, conflict or shortage.",
          {"zone_id": {"type": "string", "minLength": 1},
           "category": {"type": "string",
                        "enum": ["approval", "verification", "conflict", "shortage", "low_confidence"]},
           "reason": {"type": "string", "minLength": 3}, "action_id": {"type": "string"},
           "level": {"type": "string", "enum": ESCALATION_LEVELS}},
          ["zone_id", "category", "reason"], agent="auditor", stage="run_state", read_only=False),
    _tool("request_mutual_aid", "Reroute to a secondary responder",
          "Reroute demand the incident inventory cannot cover to a secondary responder "
          "(SDRF/NDRF, district EMS, DDMA relief camp).",
          {"zone_id": {"type": "string", "minLength": 1}, "state": {"type": "string"},
           "resource_type": _RESOURCE_ENUM, "shortfall": {"type": "integer", "minimum": 1}},
          ["zone_id", "resource_type", "shortfall"], agent="allocator", stage="run_state",
          read_only=False),
    _tool("issue_alert", "Issue standby alert",
          "Issue a low-consequence standby or monitoring alert for a zone.",
          {"zone_id": {"type": "string", "minLength": 1},
           "level": {"type": "string", "enum": PRIORITY_LEVELS},
           "message": {"type": "string", "minLength": 3}},
          ["zone_id", "level", "message"], agent="coordinator", stage="run_state", read_only=False),
    _tool("notify_shelter", "Notify shelter",
          "Tell the receiving shelter how many evacuees to expect from a zone.",
          {"zone_id": {"type": "string", "minLength": 1}, "evacuees": {"type": "integer", "minimum": 1},
           "beds_reserved": {"type": "integer", "minimum": 0}, "shelter": {"type": "string"}},
          ["zone_id", "evacuees", "beds_reserved"], agent="coordinator", stage="run_state", read_only=False),
    _tool("commit_rescue_dispatch", "Commit rescue dispatch",
          "Commit a planned dispatch. Actions the Safety Auditor held for sign-off need the "
          "approval token that only a human decision issues.",
          {"zone_id": {"type": "string", "minLength": 1}, "action_id": {"type": "string", "minLength": 1},
           "approval_token": {"type": "string"}},
          ["zone_id", "action_id"], agent="coordinator", stage="run_state", read_only=False,
          destructive=True, requires_approval=True),
    _tool("recall_dispatch", "Recall dispatch (emergency override)",
          "Recall a committed dispatch and release its units. Needs the one-time override token that only "
          "a human commander's override issues.",
          {"zone_id": {"type": "string", "minLength": 1}, "action_id": {"type": "string", "minLength": 1},
           "override_token": {"type": "string", "minLength": 16}, "reason": {"type": "string", "minLength": 3}},
          ["zone_id", "action_id", "override_token", "reason"], agent="commander", stage="run_state",
          read_only=False, destructive=True, requires_approval=True),
]

TOOL_NAMES = [tool["name"] for tool in TOOL_DEFINITIONS]


# ===========================================================================
# Builders
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_sop(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop passages that cannot guide an action, counting each reason."""
    counts = {"input": int(len(frame))}
    frame = frame.copy()
    frame["action_text"] = frame["action_text"].fillna("").astype(str).str.strip()

    short = frame["action_text"].str.len() < MIN_SOP_CHARS
    counts["too_short"] = int(short.sum())
    frame = frame[~short]

    listing = frame["section"].fillna("").str.upper().str.contains("LIST OF")
    counts["directory_listing"] = int(listing.sum())
    frame = frame[~listing]

    glossary = frame["action_text"].str.count(r"\s:\s") >= GLOSSARY_SEPARATORS
    counts["acronym_glossary"] = int(glossary.sum())
    frame = frame[~glossary]

    key = frame["action_text"].str.lower().str.replace(r"\s+", " ", regex=True)
    duplicate = key.duplicated()
    counts["duplicate"] = int(duplicate.sum())
    frame = frame[~duplicate]

    counts["kept"] = int(len(frame))
    return frame.reset_index(drop=True), counts


def build_sop_index(frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = (frame["section"].fillna("") + ". " + frame["action_text"] + ". "
              + frame["keywords"].fillna("").str.replace(";", " "))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                                 stop_words="english", max_features=30000)
    matrix = vectorizer.fit_transform(corpus)
    records = frame[["record_id", "source_document", "page_number", "section", "disaster_phase",
                     "hazard_type", "responsible_agency", "action_type", "action_text"]].to_dict("records")
    return {"vectorizer": vectorizer, "matrix": matrix, "records": records}


def build_district_profiles(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    frame = frame.copy()
    frame["above_threshold"] = frame["river_level_m"] > frame["river_level_threshold_m"]
    profiles: dict[str, dict[str, Any]] = {}
    for (state, district), group in frame.groupby(["state", "district"]):
        profiles[f"{state}|{district}"] = {
            "state": state,
            "district": district,
            "records": int(len(group)),
            "severe_rate": round(float((group["zone_risk"] == "Severe").mean()), 3),
            "low_rate": round(float((group["zone_risk"] == "Low").mean()), 3),
            "median_danger_threshold_m": round(float(group["river_level_threshold_m"].median()), 2),
            "p90_river_level_m": round(float(group["river_level_m"].quantile(0.9)), 2),
            "above_threshold_rate": round(float(group["above_threshold"].mean()), 3),
            "median_flood_history_count": float(group["flood_history_count"].median()),
            "median_population_affected": round(float(group["population_affected"].median()), 0),
            "p90_population_affected": round(float(group["population_affected"].quantile(0.9)), 0),
        }
    return profiles


def build_resource_catalog(dictionary: pd.DataFrame) -> dict[str, Any]:
    phrases = dictionary.loc[dictionary["category"] == "Resource", "term"].astype(str).tolist()
    mapping = {phrase: map_resource_phrase(phrase) for phrase in phrases}
    default_inventory, source = dict(FALLBACK_INVENTORY), "fallback constants"
    if SCENARIO_SUITE_JSON.exists():
        suite = json.loads(SCENARIO_SUITE_JSON.read_text(encoding="utf-8"))
        rows = [s.get("resources", {}) for s in suite.get("scenarios", [])]
        if rows:
            default_inventory = {k: int(np.median([r.get(k, 0) for r in rows])) for k in RESOURCE_TYPES}
            source = f"median over {len(rows)} Stage 05 suite scenarios"
    return {
        "resource_types": RESOURCE_TYPES,
        "phrase_to_inventory": mapping,
        "unmapped_phrases": sorted(p for p, t in mapping.items() if t is None),
        "planning_ratios": {"people_per_boat": PEOPLE_PER_BOAT, "people_per_ambulance": PEOPLE_PER_AMBULANCE,
                            "shelter_beds_per_evacuee": 1},
        "default_inventory": default_inventory,
        "default_inventory_source": source,
    }


def map_resource_phrase(phrase: str) -> str | None:
    lowered = phrase.lower()
    for resource_type, keywords in RESOURCE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return resource_type
    return None


# ===========================================================================
# Runtime knowledge base
# ===========================================================================

class KnowledgeBase:
    """Read-side of everything this script writes."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        processed_dir = Path(processed_dir)
        index = joblib.load(processed_dir / SOP_INDEX.name)
        self.vectorizer = index["vectorizer"]
        self.matrix = index["matrix"]
        self.records = index["records"]
        self.districts = json.loads((processed_dir / DISTRICT_JSON.name).read_text(encoding="utf-8"))
        self.resources = json.loads((processed_dir / RESOURCE_JSON.name).read_text(encoding="utf-8"))
        self.glossary = json.loads((processed_dir / GLOSSARY_JSON.name).read_text(encoding="utf-8"))
        self._district_lookup = {k.lower(): k for k in self.districts}

    # -- SOP retrieval -------------------------------------------------------

    def search_sop(self, query: str, hazard: str | None = None, phase: str | None = None,
                   top_k: int = 3) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        vector = self.vectorizer.transform([query])
        similarity = (self.matrix @ vector.T).toarray().ravel()
        hint = HAZARD_SOP_HINTS.get(hazard or "", {})
        sop_hazard = hint.get("hazard") or (hazard if hazard and hazard not in HAZARD_SOP_HINTS else None)
        action = hint.get("action")
        scores = similarity.copy()
        for i, record in enumerate(self.records):
            if similarity[i] <= 0:
                continue
            if sop_hazard and sop_hazard.lower() in str(record["hazard_type"]).lower():
                scores[i] += HAZARD_BOOST
            if phase and record["disaster_phase"] == phase:
                scores[i] += PHASE_BOOST
            if action and action in str(record["action_type"]):
                scores[i] += ACTION_BOOST
        order = np.argsort(-scores)[:top_k]
        results = []
        for i in order:
            if similarity[i] <= 0:
                break
            record = self.records[int(i)]
            text = record["action_text"]
            results.append({
                "record_id": record["record_id"],
                "source": f"{record['source_document']} p.{record['page_number']}",
                "section": record["section"],
                "phase": record["disaster_phase"],
                "hazard_type": record["hazard_type"],
                "agency": record["responsible_agency"],
                "action_type": record["action_type"],
                "excerpt": text if len(text) <= 400 else text[:397] + "...",
                "similarity": round(float(similarity[i]), 4),
                "score": round(float(scores[i]), 4),
            })
        return results

    # -- district context ----------------------------------------------------

    def district_profile(self, state: str, district: str) -> dict[str, Any] | None:
        key = self._district_lookup.get(f"{state}|{district}".lower())
        return self.districts.get(key) if key else None

    # -- resources -----------------------------------------------------------

    @property
    def resource_types(self) -> list[str]:
        return list(self.resources["resource_types"])

    @property
    def default_inventory(self) -> dict[str, int]:
        return dict(self.resources["default_inventory"])

    def map_resource(self, phrase: str) -> str | None:
        known = self.resources["phrase_to_inventory"]
        if phrase in known:
            return known[phrase]
        return map_resource_phrase(phrase)


def retrieval_check(kb: KnowledgeBase, top_k: int = 3) -> dict[str, Any]:
    """Label-match precision@k on probe queries, against the label's base rate."""
    rows = []
    for query, label in RETRIEVAL_PROBES:
        hits = kb.search_sop(query, top_k=top_k)
        relevant = sum(label in hit["action_type"] or label == hit["phase"] for hit in hits)
        base_rate = float(np.mean([label in r["action_type"] or label == r["disaster_phase"]
                                   for r in kb.records]))
        rows.append({"query": query, "label": label, "precision_at_k": round(relevant / top_k, 3),
                     "label_base_rate": round(base_rate, 3)})
    return {
        "k": top_k,
        "mean_precision_at_k": round(float(np.mean([r["precision_at_k"] for r in rows])), 3),
        "mean_label_base_rate": round(float(np.mean([r["label_base_rate"] for r in rows])), 3),
        "probes": rows,
        "note": "Relevance is judged by the SOP corpus's own action_type/phase labels, which are "
                "keyword-derived; this is a sanity check that retrieval beats chance, not a "
                "human-judged relevance score.",
    }


def build_all() -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sop_raw = pd.read_csv(SOP_CSV)
    sop, sop_counts = clean_sop(sop_raw)
    joblib.dump(build_sop_index(sop), SOP_INDEX)

    sensors = pd.read_csv(SENSOR_RECORD_CSV)
    profiles = build_district_profiles(sensors)
    DISTRICT_JSON.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    dictionary = pd.read_csv(DICTIONARY_CSV)
    catalog = build_resource_catalog(dictionary)
    RESOURCE_JSON.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    GLOSSARY_JSON.write_text(dictionary.to_json(orient="records", indent=2), encoding="utf-8")

    TOOL_REGISTRY_JSON.write_text(json.dumps({"tools": TOOL_DEFINITIONS}, indent=2), encoding="utf-8")

    kb = KnowledgeBase()
    retrieval = retrieval_check(kb)
    manifest = {
        "generated_at": _now(),
        "sources": {
            "sop_corpus": str(SOP_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sensor_record": str(SENSOR_RECORD_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
            "domain_dictionary": str(DICTIONARY_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
            "scenario_suite": str(SCENARIO_SUITE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "sop_filtering": sop_counts,
        "sop_vocabulary": int(len(kb.vectorizer.vocabulary_)),
        "sop_documents": sorted({r["source_document"] for r in kb.records}),
        "districts": len(profiles),
        "states": len({p["state"] for p in profiles.values()}),
        "resource_catalog": {"types": RESOURCE_TYPES, "mapped_phrases": sum(
            t is not None for t in catalog["phrase_to_inventory"].values()),
            "unmapped_phrases": len(catalog["unmapped_phrases"]),
            "default_inventory": catalog["default_inventory"]},
        "glossary_terms": int(len(dictionary)),
        "tools": TOOL_NAMES,
        "retrieval_check": retrieval,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print("=" * 60)
    print("Stage 06 Knowledge Engineer")
    print("=" * 60)
    manifest = build_all()
    f = manifest["sop_filtering"]
    print(f"  SOP passages: {f['input']} in, {f['kept']} kept "
          f"(short {f['too_short']}, listing {f['directory_listing']}, "
          f"glossary {f['acronym_glossary']}, duplicate {f['duplicate']})")
    print(f"  SOP vocabulary: {manifest['sop_vocabulary']} terms from {manifest['sop_documents']}")
    print(f"  District profiles: {manifest['districts']} across {manifest['states']} states")
    print(f"  Resource catalog: {manifest['resource_catalog']}")
    print(f"  Tools registered: {len(manifest['tools'])}")
    r = manifest["retrieval_check"]
    print(f"  Retrieval precision@{r['k']}: {r['mean_precision_at_k']} "
          f"(label base rate {r['mean_label_base_rate']})")
    for probe in r["probes"]:
        print(f"    {probe['label']:<22} P@k {probe['precision_at_k']:.2f}  base {probe['label_base_rate']:.3f}")


if __name__ == "__main__":
    main()
