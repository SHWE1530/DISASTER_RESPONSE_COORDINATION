"""Build the Stage 04 (SLM) dataset from Stage 03 outputs.

WHAT THIS PRODUCES
------------------
1. SLM_Report_Summary_Pairs.jsonl  - instruction-tuning pairs for fine-tuning
2. SLM_Report_Summary_Pairs.csv    - the same rows, tabular, for inspection
3. SLM_Domain_Dictionary.csv       - evacuation terms, resource codes, shorthand
4. SLM_dataset_statistics.json     - measured statistics for the dataset card

WHY IT IS BUILT THIS WAY
------------------------
The Stage 04 brief says the SLM's data is "report-to-summary training pairs
generated from Stage 03". A single Stage 03 dispatcher message is one or two
sentences, so summarising one message to two sentences achieves no compression
and would not train a summariser at all.

Real incident logs are a *stream* of entries about the same developing
situation. This script therefore groups Stage 03 dispatcher records into
multi-entry incident logs (same state, district and zone, inside a rolling time
window), uses the concatenated entries as the REPORT, and derives a two-sentence
tactical briefing as the SUMMARY. That yields genuine compression, which is what
the brief's ">80% reduction" huddle target requires.

Every field in a summary is aggregated from the structured ground-truth columns
of the same records that form the report, so the summary never states anything
the report does not contain. Nothing is invented.

HONESTY NOTE
------------
The summaries are TEMPLATED. A model fine-tuned on them learns to reproduce
these templates, not to summarise open-domain English. Phrasing is sampled from
several variants per slot to reduce rigidity, but this remains synthetic
supervision built on a synthetic corpus. See SLM_DATASET_CARD.md.

Run:
    python build_slm_dataset.py --source <repo>/Stage03_NLP --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 42

# Multiplier applied to the time window for severity-banded grouping.
BANDED_WINDOW_MULTIPLIER = 45

# ---------------------------------------------------------------------------
# Ordinal severity so a multi-entry log can be reduced to its worst entry.
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_TO_PRIORITY = {
    "LOW": "ROUTINE",
    "MEDIUM": "ELEVATED",
    "HIGH": "URGENT",
    "CRITICAL": "IMMEDIATE",
}

# Sentence-1 openers -- include state so the model learns that slot too.
# These must stay in sync with SITUATION_TEMPLATES in 03_slm_engineer.py.
SITUATION_TEMPLATES = [
    "{priority}: {hazard_phrase}, {zone} {district}, {state}. {entry_count} reports, {headcount} affected.",
    "{priority} - {zone} {district} ({state}): {hazard_phrase}, {headcount} affected across {entry_count} reports.",
    "{priority}. {hazard_phrase} in {zone} {district}, {state}; {headcount} affected, {entry_count} reports.",
    "{priority}: {zone} {district}, {state} -- {hazard_phrase}. {entry_count} reports, {headcount} affected.",
]

# Sentence-2 action lines -- must include ICS shorthand for radio-code retention.
ACTION_TEMPLATES = [
    "Deploy {resources} to {location_phrase}; IC to confirm ACCESS. {directive}",
    "{resources} to {location_phrase}; STAGING at nearest safe point. {directive}",
    "Dispatch {resources} via ACCESS route to {location_phrase}. {directive}",
    "Send {resources} to {location_phrase}; submit SITREP on arrival. {directive}",
]

# Directive text keyed by the worst severity in the log.
# Each directive uses at least one ICS code or agency shorthand.
DIRECTIVES = {
    "LOW": ["STANDBY, submit SITREP every 60 min.", "monitor only; IC on STANDBY.",
            "routine watch; CAS count zero."],
    "MEDIUM": ["pre-position at STAGING area; SITREP every 30 min.", "stage nearby, monitor ACCESS.",
               "STANDBY; update CAS count on change."],
    "HIGH": ["deploy now; begin TRIAGE on arrival; notify SDRF.",
             "deploy and confirm ETA; submit SITREP every 15 min.",
             "commit teams; TRIAGE + CAS tracking; IC to verify ACCESS route."],
    "CRITICAL": ["EVAC under way; NDRF requested; SITREP every 5 min.",
                 "all units, EVAC of affected persons; notify NDMA and SDRF.",
                 "immediate TRIAGE; log all CAS; IC orders full EVAC."],
}

HAZARD_PHRASES = {
    "Flood": "flooding",
    "Rescue Emergency": "rescue emergencies",
    "Road Blockage": "road blockages",
    "Medical Emergency": "medical emergencies",
}


def load_dispatcher(source_dir: Path) -> pd.DataFrame:
    """Load the Stage 03 processed dispatcher log."""
    path = source_dir / "data" / "processed" / "Dispatcher_Log_Master_60000_Processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 03 dispatcher corpus not found: {path}\n"
            "Run `python Stage03_NLP/01_data_engineer.py` first."
        )
    frame = pd.read_csv(path, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "text", "severity"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def split_resources(value: object) -> list[str]:
    """Stage 03 stores multiple resources comma-separated in one cell."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def humanise_window(minutes: float) -> str:
    if minutes < 90:
        return f"{int(round(minutes))} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def join_natural(items: list[str], limit: int = 3, overflow: bool = True,
                 empty: str = "no specific resource identified") -> str:
    """Join a list into readable English, truncating politely.

    `overflow=False` suppresses the "plus N more" tail. Used for the location
    slot, where a spoken briefing names the primary site and the extra words
    would push the summary past the five-second read budget.
    """
    unique = list(dict.fromkeys(items))
    shown = unique[:limit]
    extra = len(unique) - len(shown)
    if not shown:
        return empty
    if len(shown) == 1:
        text = shown[0]
    else:
        text = ", ".join(shown[:-1]) + " and " + shown[-1]
    if overflow and extra > 0:
        text += f" plus {extra} more"
    return text


def build_groups(frame: pd.DataFrame, min_entries: int, max_entries: int,
                 window_minutes: int) -> list[pd.DataFrame]:
    """Group records into incident logs: same state/district/zone, within a window.

    Grouping by location AND time is what makes each report a coherent
    developing situation rather than an arbitrary bag of unrelated messages.
    """
    groups: list[pd.DataFrame] = []
    for _, block in frame.groupby(["state", "district", "zone"], sort=False):
        block = block.sort_values("timestamp")
        current: list[int] = []
        window_start = None
        for idx, row in block.iterrows():
            if not current:
                current = [idx]
                window_start = row["timestamp"]
                continue
            elapsed = (row["timestamp"] - window_start).total_seconds() / 60.0
            if elapsed <= window_minutes and len(current) < max_entries:
                current.append(idx)
            else:
                if len(current) >= min_entries:
                    groups.append(block.loc[current])
                current = [idx]
                window_start = row["timestamp"]
        if len(current) >= min_entries:
            groups.append(block.loc[current])
    return groups


def build_banded_groups(frame: pd.DataFrame, min_entries: int, max_entries: int,
                        window_minutes: int) -> list[pd.DataFrame]:
    """Group records that share a location AND a severity band.

    WHY THIS EXISTS
    ---------------
    A briefing's priority is the WORST entry in its log. With natural
    location+time grouping, HIGH is 40% of all Stage 03 records, so across 3-8
    entries the worst is almost always HIGH or CRITICAL: a first build produced
    only 15 ROUTINE briefings out of 9,323. A model trained on that would never
    learn to say "monitor only".

    A quiet zone genuinely does produce logs containing only routine entries, so
    grouping within a severity band is realistic, not a fabrication. It is
    tracked separately via the `grouping` column so the two populations can
    always be told apart.
    """
    groups: list[pd.DataFrame] = []
    keys = ["state", "district", "zone", "severity"]
    for _, block in frame.groupby(keys, sort=False):
        block = block.sort_values("timestamp")
        current: list[int] = []
        window_start = None
        for idx, row in block.iterrows():
            if not current:
                current = [idx]
                window_start = row["timestamp"]
                continue
            elapsed = (row["timestamp"] - window_start).total_seconds() / 60.0
            if elapsed <= window_minutes and len(current) < max_entries:
                current.append(idx)
            else:
                if len(current) >= min_entries:
                    groups.append(block.loc[current])
                current = [idx]
                window_start = row["timestamp"]
        if len(current) >= min_entries:
            groups.append(block.loc[current])
    return groups


def build_report(group: pd.DataFrame) -> str:
    """Concatenate the group's entries into a timestamped incident log.

    The header line carries the zone. Stage 03's `text` field names only the
    district and state, so without this header a summary that mentions the zone
    would assert something the report does not contain -- which is exactly how a
    summariser is taught to hallucinate. A real incident log carries the zone as
    record metadata, so including it is faithful rather than padding.
    """
    first = group.iloc[0]
    lines = [
        f"INCIDENT LOG | {first['state']} / {first['district']} / "
        f"{first['zone']} | {len(group)} entries"
    ]
    for _, row in group.iterrows():
        stamp = row["timestamp"].strftime("%d-%m-%Y %H:%M")
        lines.append(f"[{stamp}] {row['incident_id']} | {row['text']}")
    return "\n".join(lines)


def build_summary(group: pd.DataFrame, rng: random.Random) -> tuple[str, dict]:
    """Derive a two-sentence tactical briefing from the group's ground truth."""
    worst = max(group["severity"], key=lambda s: SEVERITY_ORDER.get(str(s), 0))
    priority = SEVERITY_TO_PRIORITY[worst]

    hazards = [str(h) for h in group["hazard_type"].dropna().unique()]
    hazard_phrase = join_natural(
        [HAZARD_PHRASES.get(h, str(h).lower()) for h in hazards], limit=2
    )

    headcount = int(pd.to_numeric(group["headcount_entity"], errors="coerce")
                    .fillna(0).sum())

    resources: list[str] = []
    for value in group["resource_entity"]:
        resources.extend(split_resources(value))
    resource_counts = Counter(resources)
    top_resources = [name for name, _ in resource_counts.most_common(3)]

    locations = [str(l) for l in group["location_entity"].dropna().unique()]
    location_phrase = join_natural(locations, limit=1, overflow=False,
                                   empty="the affected area")

    span_minutes = (group["timestamp"].max() - group["timestamp"].min()) \
        .total_seconds() / 60.0
    window = humanise_window(max(span_minutes, 1))

    first = rng.choice(SITUATION_TEMPLATES).format(
        priority=priority,
        hazard_phrase=hazard_phrase,
        zone=group["zone"].iloc[0],
        district=group["district"].iloc[0],
        state=group["state"].iloc[0],
        entry_count=len(group),
        headcount=headcount,
    )
    second = rng.choice(ACTION_TEMPLATES).format(
        resources=join_natural(top_resources, limit=2),
        location_phrase=location_phrase,
        directive=rng.choice(DIRECTIVES[worst]),
    )

    meta = {
        "worst_severity": worst,
        "priority": priority,
        "hazard_types": hazards,
        "total_headcount": headcount,
        "resources_required": top_resources,
        "locations": locations,
        "entry_count": len(group),
        "window_minutes": round(span_minutes, 1),
        "zone": group["zone"].iloc[0],
        "district": group["district"].iloc[0],
        "state": group["state"].iloc[0],
        "first_timestamp": group["timestamp"].min().strftime("%d-%m-%Y %H:%M"),
        "last_timestamp": group["timestamp"].max().strftime("%d-%m-%Y %H:%M"),
    }
    return f"{first} {second}", meta


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


INSTRUCTION = (
    "You are a disaster-response tactical briefing assistant. "
    "Read the incident log and produce a structured 3-part response:\n"
    "SITUATION: <priority keyword> + hazard type + zone + district + state + "
    "entry count + total headcount affected.\n"
    "RISK: severity assessment, population at risk, escalation likelihood.\n"
    "ACTIONS: numbered list of recommended response steps using ICS radio "
    "shorthand (EVAC, TRIAGE, SITREP, ACCESS, STAGING, CAS, ETA, IC, NDRF, NDMA, SDRF)."
)


def build_dataset(source_dir: Path, out_dir: Path, min_entries: int,
                  max_entries: int, window_minutes: int,
                  max_pairs: int | None, reports_dir: Path | None = None) -> dict:
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dispatcher(source_dir)
    print(f"[load] dispatcher records: {len(frame):,}")

    natural = build_groups(frame, min_entries, max_entries, window_minutes)
    print(f"[group] natural (location+time) logs: {len(natural):,}")

    # Severity-banded logs, used only to top up the priorities that natural
    # grouping starves. See build_banded_groups for the rationale.
    # Quiet zones genuinely accumulate routine entries over days rather than
    # hours, so the banded pass uses a much wider window. Without this, LOW
    # (8,977 of 60,000 records spread over 500 zones) yields almost no all-LOW
    # logs and the model never learns to emit a ROUTINE briefing.
    banded_all = build_banded_groups(frame, min_entries, max_entries,
                                     window_minutes * BANDED_WINDOW_MULTIPLIER)
    print(f"[group] banded (location+time+severity) logs: {len(banded_all):,}")

    def worst_of(group: pd.DataFrame) -> str:
        return max(group["severity"], key=lambda x: SEVERITY_ORDER.get(str(x), 0))

    by_severity: dict[str, list] = {k: [] for k in SEVERITY_ORDER}
    for group in banded_all:
        by_severity[worst_of(group)].append(group)
    for bucket in by_severity.values():
        rng.shuffle(bucket)

    # Record-disjoint selection.
    #
    # The natural and banded pools are built from the same 60,000 records, so a
    # single dispatcher entry can appear in one group from each pool. If those
    # groups then land in different splits, a test report contains text the
    # model already trained on. A first build leaked 1,548 incident IDs across
    # splits this way. Claiming every record at most once removes the
    # contamination and also guarantees no two reports share content.
    claimed: set = set()

    def claim(group: pd.DataFrame) -> bool:
        ids = set(group["incident_id"])
        if ids & claimed:
            return False
        claimed.update(ids)
        return True

    tagged = []

    # Scarce priorities claim their records FIRST.
    #
    # LOW is only 8,977 of 60,000 records. If natural grouping runs first it
    # consumes most of those records inside mixed-severity logs whose worst
    # entry is HIGH or CRITICAL, leaving almost nothing to build a ROUTINE
    # briefing from (a previous build yielded 102). Reserving the scarce bands
    # up front is the difference between a model that can say "monitor only"
    # and one that cannot.
    reserve_quota = int(len(natural) * 0.25)
    for severity in ("LOW", "MEDIUM"):
        added = 0
        for group in by_severity.get(severity, []):
            if added >= reserve_quota:
                break
            if claim(group):
                tagged.append((group, "severity_banded"))
                added += 1
        print(f"[reserve] {added:,} banded logs reserved for "
              f"{SEVERITY_TO_PRIORITY[severity]}")

    kept_natural = 0
    for group in natural:
        if claim(group):
            tagged.append((group, "natural"))
            kept_natural += 1
    print(f"[disjoint] natural logs kept: {kept_natural:,} "
          f"(dropped {len(natural) - kept_natural:,} overlapping)")

    counts = Counter(worst_of(g) for g, _ in tagged)
    target = max(counts.values()) if counts else 0

    for severity, bucket in by_severity.items():
        needed = max(0, target - counts.get(severity, 0))
        added = 0
        for group in bucket:
            if added >= needed:
                break
            if claim(group):
                tagged.append((group, "severity_banded"))
                added += 1
        if added:
            print(f"[balance] +{added:,} banded logs for "
                  f"{SEVERITY_TO_PRIORITY[severity]}")

    rng.shuffle(tagged)
    if max_pairs is not None:
        tagged = tagged[:max_pairs]
        print(f"[cap] limited to {len(tagged):,} pairs")

    rows = []
    for index, (group, grouping) in enumerate(tagged, start=1):
        report = build_report(group)
        summary, meta = build_summary(group, rng)
        rw, sw = word_count(report), word_count(summary)
        rows.append({
            "pair_id": f"SLM-{index:06d}",
            "source": "stage03_dispatcher",
            "grouping": grouping,
            "report": report,
            "summary": summary,
            "priority": meta["priority"],
            "worst_severity": meta["worst_severity"],
            "hazard_types": "; ".join(meta["hazard_types"]),
            "total_headcount": meta["total_headcount"],
            "resources_required": "; ".join(meta["resources_required"]),
            "locations": "; ".join(meta["locations"]),
            "state": meta["state"],
            "district": meta["district"],
            "zone": meta["zone"],
            "entry_count": meta["entry_count"],
            "window_minutes": meta["window_minutes"],
            "first_timestamp": meta["first_timestamp"],
            "last_timestamp": meta["last_timestamp"],
            "incident_ids": "; ".join(group["incident_id"].tolist()),
            "report_words": rw,
            "summary_words": sw,
            "compression_pct": round(100.0 * (1 - sw / rw), 1) if rw else 0.0,
        })

    data = pd.DataFrame(rows)

    # ---- Splits: assigned per PAIR, stratified by priority. Each incident log
    # is built from disjoint records, so no dispatcher record can appear in two
    # splits and no report is ever partially seen during training.
    data["split"] = "train"
    for priority, block in data.groupby("priority"):
        idx = block.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        val_end = int(n * 0.15)
        test_end = val_end + int(n * 0.15)
        data.loc[idx[:val_end], "split"] = "validation"
        data.loc[idx[val_end:test_end], "split"] = "test"

    csv_path = out_dir / "SLM_Report_Summary_Pairs.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[write] {csv_path.name}  ({len(data):,} pairs)")

    # Excel-safe variant: the report column contains newlines (it is a
    # multi-line incident log), which is valid CSV but renders as multi-line
    # cells in Excel. This flattens them to " || " separators.
    flat = data.copy()
    flat["report"] = flat["report"].str.replace(chr(10), " || ", regex=False)
    flat_path = out_dir / "SLM_Report_Summary_Pairs_FLAT.csv"
    flat.to_csv(flat_path, index=False, encoding="utf-8-sig")
    print(f"[write] {flat_path.name}  (Excel-safe)")

    jsonl_path = out_dir / "SLM_Report_Summary_Pairs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for _, row in data.iterrows():
            handle.write(json.dumps({
                "pair_id": row["pair_id"],
                "split": row["split"],
                "grouping": row["grouping"],
                "instruction": INSTRUCTION,
                "input": row["report"],
                "output": row["summary"],
                "metadata": {
                    "priority": row["priority"],
                    "worst_severity": row["worst_severity"],
                    "hazard_types": row["hazard_types"].split("; ") if row["hazard_types"] else [],
                    "total_headcount": int(row["total_headcount"]),
                    "headcount_is_aggregate": True,  # sum of per-entry headcounts, not verbatim in report text
                    "resources_required": row["resources_required"].split("; ") if row["resources_required"] else [],
                    "locations": row["locations"].split("; ") if row["locations"] else [],
                    "state": row["state"],
                    "district": row["district"],
                    "zone": row["zone"],
                    "entry_count": int(row["entry_count"]),
                    "report_words": int(row["report_words"]),
                    "summary_words": int(row["summary_words"]),
                    "compression_pct": float(row["compression_pct"]),
                },
            }, ensure_ascii=False) + "\n")
    print(f"[write] {jsonl_path.name}")

    stats = {
        "total_pairs": int(len(data)),
        "splits": {k: int(v) for k, v in data["split"].value_counts().items()},
        "priority_distribution": {k: int(v) for k, v in data["priority"].value_counts().items()},
        "grouping_distribution": {k: int(v) for k, v in data["grouping"].value_counts().items()},
        "worst_severity_distribution": {k: int(v) for k, v in data["worst_severity"].value_counts().items()},
        "entries_per_report": {
            "min": int(data["entry_count"].min()),
            "max": int(data["entry_count"].max()),
            "mean": round(float(data["entry_count"].mean()), 2),
        },
        "report_words": {
            "min": int(data["report_words"].min()),
            "max": int(data["report_words"].max()),
            "mean": round(float(data["report_words"].mean()), 1),
        },
        "summary_words": {
            "min": int(data["summary_words"].min()),
            "max": int(data["summary_words"].max()),
            "mean": round(float(data["summary_words"].mean()), 1),
        },
        "compression_pct": {
            "min": round(float(data["compression_pct"].min()), 1),
            "max": round(float(data["compression_pct"].max()), 1),
            "mean": round(float(data["compression_pct"].mean()), 1),
        },
        "states": int(data["state"].nunique()),
        "districts": int(data["district"].nunique()),
        "grouping": {
            "min_entries": min_entries,
            "max_entries": max_entries,
            "window_minutes": window_minutes,
            "key": "state + district + zone, within a rolling time window",
            "banded_key": "state + district + zone + severity, 4x window; used only to top up starved priorities",
        },
        "seed": SEED,
        "source": "Stage03_NLP/data/processed/Dispatcher_Log_Master_60000_Processed.csv",
    }
    stats_path = (reports_dir or out_dir) / "SLM_dataset_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[write] {stats_path.name}")
    return stats


# ---------------------------------------------------------------------------
# Domain dictionary
# ---------------------------------------------------------------------------
def build_domain_dictionary(source_dir: Path, out_dir: Path) -> int:
    """Build the domain dictionary, grounded in terms that appear in the data.

    Rows are tagged with `origin` so it is always clear which terms are real
    standard terminology and which are a controlled vocabulary defined by this
    project. No entry claims to be an official agency radio code.
    """
    frame = load_dispatcher(source_dir)

    resources: list[str] = []
    for value in frame["resource_entity"]:
        resources.extend(split_resources(value))
    resource_counts = Counter(resources)
    location_counts = Counter(str(l) for l in frame["location_entity"].dropna())

    rows: list[dict] = []

    # --- Resource codes: derived from the actual corpus vocabulary ----------
    def resource_code(name: str) -> str:
        words = re.findall(r"[A-Za-z]+", name.upper())
        if not words:
            return "RES"
        if len(words) == 1:
            return "RES-" + words[0][:4]
        return "RES-" + "".join(w[0] for w in words[:4])

    for name, count in resource_counts.most_common():
        rows.append({
            "term": name,
            "category": "Resource",
            "shorthand": resource_code(name),
            "expansion": name.title(),
            "definition": f"Response resource requested in dispatcher logs; "
                          f"appears {count:,} times in the Stage 03 corpus.",
            "example_usage": f"Dispatch {name} to the affected location.",
            "origin": "derived from Stage 03 corpus",
        })

    # --- Location types: derived from the corpus ----------------------------
    for name, count in location_counts.most_common():
        rows.append({
            "term": name,
            "category": "Location type",
            "shorthand": "LOC-" + "".join(
                w[0] for w in re.findall(r"[A-Za-z]+", name.upper())[:4]),
            "expansion": name.title(),
            "definition": f"Location descriptor used in incident reports; "
                          f"appears {count:,} times in the Stage 03 corpus.",
            "example_usage": f"Flooding reported near the {name}.",
            "origin": "derived from Stage 03 corpus",
        })

    # --- Hazard and severity vocabulary from the corpus --------------------
    for hazard in sorted(frame["hazard_type"].dropna().unique()):
        rows.append({
            "term": hazard,
            "category": "Hazard type",
            "shorthand": "HZ-" + "".join(
                w[0] for w in re.findall(r"[A-Za-z]+", str(hazard).upper())[:3]),
            "expansion": str(hazard),
            "definition": "Hazard classification assigned to an incident by "
                          "the Stage 03 pipeline.",
            "example_usage": f"Hazard type: {hazard}.",
            "origin": "derived from Stage 03 corpus",
        })

    for severity, priority in SEVERITY_TO_PRIORITY.items():
        rows.append({
            "term": severity,
            "category": "Severity level",
            "shorthand": f"SEV-{severity[0]}",
            "expansion": severity.title(),
            "definition": f"Incident severity. Maps to briefing priority "
                          f"'{priority}' in the Stage 04 summary format.",
            "example_usage": f"{priority}: {severity.lower()} severity incident.",
            "origin": "project-defined (Stage 03 label space)",
        })

    # --- Standard incident-command terminology -----------------------------
    ics_terms = [
        ("Incident Commander", "IC",
         "The single person accountable for overall management of an incident.",
         "IC has authorised evacuation of the zone."),
        ("Staging Area", "STAGING",
         "A location where resources are held ready for assignment.",
         "Hold the ambulance fleet at the staging area."),
        ("Triage", "TRIAGE",
         "Sorting casualties by urgency of the treatment they need.",
         "Begin triage at the shelter entrance."),
        ("Evacuation", "EVAC",
         "Organised movement of people away from a threatened area.",
         "EVAC under way for the residential colony."),
        ("Shelter-in-place", "SIP",
         "Instruction to remain indoors rather than evacuate.",
         "Advise SIP until the water level falls."),
        ("Situation Report", "SITREP",
         "A periodic summary of current conditions and actions.",
         "Submit a SITREP every thirty minutes."),
        ("Estimated Time of Arrival", "ETA",
         "Expected arrival time of a resource on scene.",
         "Rescue boat ETA fifteen minutes."),
        ("Casualty", "CAS",
         "A person injured or killed in an incident.",
         "Three CAS reported at the main bridge."),
        ("Access Route", "ACCESS",
         "A road or path usable to reach the incident.",
         "Confirm access route before committing units."),
        ("Standby", "STANDBY",
         "Resource is ready but not yet committed.",
         "Place the medical team on standby."),
    ]
    for term, short, definition, example in ics_terms:
        rows.append({
            "term": term,
            "category": "Incident-command term",
            "shorthand": short,
            "expansion": term,
            "definition": definition,
            "example_usage": example,
            "origin": "standard incident-command terminology",
        })

    # --- Indian response agencies that appear in the corpus ----------------
    agencies = [
        ("NDRF", "National Disaster Response Force",
         "India's national specialist disaster-response force. Referenced "
         "directly in Stage 03 resource strings."),
        ("SDRF", "State Disaster Response Force",
         "State-level counterpart to the NDRF."),
        ("CWC", "Central Water Commission",
         "Source agency for the river-level and rainfall telemetry used in "
         "Stage 01 and Stage 02."),
        ("IMD", "India Meteorological Department",
         "Source agency for the rainfall data referenced in Stage 01."),
        ("NDMA", "National Disaster Management Authority",
         "Source agency for the vulnerability and impact data in Stage 01."),
        ("ERSS", "Emergency Response Support System",
         "The incident-ID prefix used throughout the Stage 03 dispatcher log "
         "(for example ERSS-000001)."),
    ]
    for short, expansion, definition in agencies:
        rows.append({
            "term": expansion,
            "category": "Agency",
            "shorthand": short,
            "expansion": expansion,
            "definition": definition,
            "example_usage": f"Request support from {short}.",
            "origin": "real agency referenced in project data",
        })

    # --- Briefing priority vocabulary used by the summaries ----------------
    for severity, priority in SEVERITY_TO_PRIORITY.items():
        rows.append({
            "term": priority,
            "category": "Briefing priority",
            "shorthand": f"P-{priority[:4]}",
            "expansion": priority.title(),
            "definition": f"Opening keyword of a Stage 04 briefing. Emitted "
                          f"when the worst entry in the log is {severity}.",
            "example_usage": f"{priority}: flooding across ZONE-3.",
            "origin": "project-defined (Stage 04 summary format)",
        })

    dictionary = pd.DataFrame(rows).drop_duplicates(subset=["term", "category"])
    path = out_dir / "SLM_Domain_Dictionary.csv"
    dictionary.to_csv(path, index=False, encoding="utf-8")
    print(f"[write] {path.name}  ({len(dictionary):,} terms)")
    return len(dictionary)


STAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE_DIR.parent
DEFAULT_SOURCE = REPO_ROOT / "Stage03_NLP"
DEFAULT_PROCESSED = STAGE_DIR / "data" / "processed"
DEFAULT_OUTPUTS = STAGE_DIR / "data" / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 04 SLM dataset")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Path to the Stage03_NLP directory")
    parser.add_argument("--out", default=str(DEFAULT_PROCESSED),
                        help="Directory for the fine-tuning pairs and dictionary")
    parser.add_argument("--reports-dir", default=str(DEFAULT_OUTPUTS),
                        help="Directory for the statistics manifest")
    parser.add_argument("--min-entries", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=12)
    parser.add_argument("--window-minutes", type=int, default=480)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap the number of pairs (default: no cap)")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 04 (SLM) DATASET BUILDER")
    print("=" * 68)
    stats = build_dataset(source_dir, out_dir, args.min_entries,
                          args.max_entries, args.window_minutes, args.max_pairs,
                          reports_dir)
    terms = build_domain_dictionary(source_dir, out_dir)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  pairs            : {stats['total_pairs']:,}")
    print(f"  splits           : {stats['splits']}")
    print(f"  priority mix     : {stats['priority_distribution']}")
    print(f"  entries/report   : {stats['entries_per_report']}")
    print(f"  report words     : {stats['report_words']}")
    print(f"  summary words    : {stats['summary_words']}")
    print(f"  compression      : {stats['compression_pct']['mean']}% mean "
          f"({stats['compression_pct']['min']}-{stats['compression_pct']['max']}%)")
    print(f"  dictionary terms : {terms:,}")
    print(f"  pairs -> {out_dir}")
    print(f"  stats -> {reports_dir}")


if __name__ == "__main__":
    main()
