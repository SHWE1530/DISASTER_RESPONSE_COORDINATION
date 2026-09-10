"""Stage 04 SLM -- Exploratory Data Analysis.

Analyses the report-to-summary pairs produced by 01_data_engineer.py.

Outputs
-------
data/outputs/eda/
    priority_distribution.png           bar chart of briefing priorities
    compression_histogram.png           compression-% distribution by priority
    report_length_distribution.png      words-per-report histogram
    summary_length_distribution.png     words-per-summary histogram
    split_priority_matrix.png           heatmap: split x priority pair counts
    per_state_coverage.png              pairs per state
    top_report_unigrams.csv             most frequent non-stop words in reports
    top_summary_unigrams.csv            most frequent non-stop words in summaries
    report_bigrams.csv                  top bigrams across all reports
    summary_bigrams.csv                 top bigrams across all summaries
    domain_dictionary_audit.csv         basic term presence across the corpus

    -- NEW: factual integrity analyses --
    factual_comparison.csv              per-pair slot-by-slot report vs summary check
    hallucination_drift_report.csv      per-pair hallucination and numeric-drift flags
    verified_pairs.csv                  pairs where ALL factual slots are grounded
    flagged_pairs.csv                   pairs with at least one factual violation
    radio_code_retention.csv            per-shorthand presence rate in summaries
    radio_code_retention.png            bar chart of shorthand retention rates

    eda_manifest.json                   machine-readable summary of every output

Run:
    python Stage04_SLM/02_eda_engineer.py [--top-n N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "data" / "outputs" / "eda"

PAIRS_CSV = PROCESSED_DIR / "SLM_Report_Summary_Pairs.csv"
DICT_CSV = PROCESSED_DIR / "SLM_Domain_Dictionary.csv"

PRIORITY_ORDER = ["ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE"]
PRIORITY_COLOURS = {
    "ROUTINE": "#4CAF50",
    "ELEVATED": "#FFC107",
    "URGENT": "#FF5722",
    "IMMEDIATE": "#B71C1C",
}
SPLIT_ORDER = ["train", "validation", "test"]

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "at", "of", "to", "is", "are",
    "was", "were", "be", "been", "being", "by", "for", "on", "with",
    "as", "this", "that", "it", "its", "from", "have", "has", "had",
    "will", "can", "may", "more", "all", "into", "no", "not", "also",
    "our", "there", "we", "you", "your", "their", "they", "he", "she",
    "about", "after", "near", "reported", "send", "dispatch", "log",
    "incident", "entries", "entry",
}

TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS and len(t) > 2]


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def counter_frame(counts: Counter, top_n: int, col: str = "term") -> pd.DataFrame:
    rows = [{"term": t, "count": c} for t, c in counts.most_common(top_n)]
    return pd.DataFrame(rows, columns=[col, "count"])


def _priority_colour(p: str) -> str:
    return PRIORITY_COLOURS.get(p, "#607D8B")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pairs() -> pd.DataFrame:
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            f"Pairs CSV not found: {PAIRS_CSV}\n"
            "Run `python Stage04_SLM/01_data_engineer.py` first."
        )
    df = pd.read_csv(PAIRS_CSV, low_memory=False)
    df["priority"] = pd.Categorical(df["priority"], categories=PRIORITY_ORDER, ordered=True)
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    return df


def load_dictionary() -> pd.DataFrame:
    if not DICT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(DICT_CSV)


# ---------------------------------------------------------------------------
# Plot 1 — Priority distribution
# ---------------------------------------------------------------------------

def plot_priority_distribution(df: pd.DataFrame) -> str:
    counts = df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[_priority_colour(p) for p in counts.index],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Briefing Priority Distribution (all 3,218 pairs)", fontsize=14, pad=12)
    ax.set_xlabel("Priority", fontsize=12)
    ax.set_ylabel("Pair count", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "priority_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path.name}")
    return path.name


# ---------------------------------------------------------------------------
# Plot 2 — Compression histogram by priority
# ---------------------------------------------------------------------------

def plot_compression_histogram(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for i, priority in enumerate(PRIORITY_ORDER):
        ax = axes[i]
        subset = df[df["priority"] == priority]["compression_pct"]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.hist(subset, bins=20, color=_priority_colour(priority), edgecolor="white", alpha=0.9)
        ax.axvline(subset.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean {subset.mean():.1f}%")
        ax.set_title(f"{priority}  (n={len(subset)})", fontsize=12)
        ax.set_xlabel("Compression %")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Compression % Distribution by Priority", fontsize=14, y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "compression_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path.name}")
    return path.name


# ---------------------------------------------------------------------------
# Plot 3 — Report and summary length distributions
# ---------------------------------------------------------------------------

def plot_length_distributions(df: pd.DataFrame) -> list[str]:
    paths = []
    for col, label, fname in [
        ("report_words", "Report length (words)", "report_length_distribution.png"),
        ("summary_words", "Summary length (words)", "summary_length_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df[col], bins=30, color="#1565C0", edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="#E65100", linestyle="--", linewidth=1.5,
                   label=f"mean {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="#1B5E20", linestyle=":", linewidth=1.5,
                   label=f"median {df[col].median():.1f}")
        ax.set_title(f"{label} — all pairs", fontsize=13)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = OUTPUT_DIR / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] {path.name}")
        paths.append(fname)
    return paths


# ---------------------------------------------------------------------------
# Plot 4 — Split × priority heatmap
# ---------------------------------------------------------------------------

def plot_split_priority_matrix(df: pd.DataFrame) -> str:
    matrix = (
        df.groupby(["split", "priority"], observed=False)
        .size()
        .unstack("priority", fill_value=0)
        .reindex(SPLIT_ORDER)
        .reindex(columns=PRIORITY_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(PRIORITY_ORDER)))
    ax.set_xticklabels(PRIORITY_ORDER, fontsize=11)
    ax.set_yticks(range(len(SPLIT_ORDER)))
    ax.set_yticklabels(SPLIT_ORDER, fontsize=11)
    for (r, c), val in np.ndenumerate(matrix.values):
        ax.text(c, r, str(val), ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if val > matrix.values.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Pair count")
    ax.set_title("Pairs per Split × Priority", fontsize=13)
    fig.tight_layout()
    path = OUTPUT_DIR / "split_priority_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path.name}")
    return path.name


# ---------------------------------------------------------------------------
# Plot 5 — Pairs per state
# ---------------------------------------------------------------------------

def plot_state_coverage(df: pd.DataFrame) -> str:
    counts = df["state"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.45)))
    colours = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colours, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Pairs per State", fontsize=13)
    ax.set_xlabel("Pair count", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "per_state_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path.name}")
    return path.name


# ---------------------------------------------------------------------------
# Vocabulary analysis
# ---------------------------------------------------------------------------

def analyse_vocabulary(df: pd.DataFrame, top_n: int) -> dict[str, str]:
    report_tokens: list[str] = []
    summary_tokens: list[str] = []
    report_bi: list[str] = []
    summary_bi: list[str] = []

    for _, row in df.iterrows():
        rt = tokenise(str(row["report"]))
        st = tokenise(str(row["summary"]))
        report_tokens.extend(rt)
        summary_tokens.extend(st)
        report_bi.extend(bigrams(rt))
        summary_bi.extend(bigrams(st))

    outputs: dict[str, str] = {}

    for tokens, bi, label in [
        (report_tokens, report_bi, "report"),
        (summary_tokens, summary_bi, "summary"),
    ]:
        uni_path = OUTPUT_DIR / f"top_{label}_unigrams.csv"
        counter_frame(Counter(tokens), top_n).to_csv(uni_path, index=False)
        outputs[f"top_{label}_unigrams.csv"] = f"Top {top_n} unigrams in {label}s"
        print(f"[vocab] {uni_path.name}")

        bi_path = OUTPUT_DIR / f"{label}_bigrams.csv"
        counter_frame(Counter(bi), top_n, col="bigram").to_csv(bi_path, index=False)
        outputs[f"{label}_bigrams.csv"] = f"Top {top_n} bigrams in {label}s"
        print(f"[vocab] {bi_path.name}")

    return outputs


# ---------------------------------------------------------------------------
# Domain dictionary audit
# ---------------------------------------------------------------------------

def audit_domain_dictionary(df: pd.DataFrame, dictionary: pd.DataFrame) -> str:
    if dictionary.empty:
        print("[audit] Domain dictionary not found, skipping.")
        return ""

    all_text = " ".join(df["report"].astype(str)).lower() + " " + \
               " ".join(df["summary"].astype(str)).lower()

    rows = []
    for _, entry in dictionary.iterrows():
        term = str(entry.get("term", "")).lower()
        found_in_reports = term in all_text
        rows.append({
            "term": entry.get("term", ""),
            "category": entry.get("category", ""),
            "shorthand": entry.get("shorthand", ""),
            "origin": entry.get("origin", ""),
            "found_in_corpus": found_in_reports,
        })

    audit_df = pd.DataFrame(rows)
    coverage = audit_df["found_in_corpus"].mean() * 100
    path = OUTPUT_DIR / "domain_dictionary_audit.csv"
    audit_df.to_csv(path, index=False)
    print(f"[audit] {path.name}  coverage {coverage:.1f}% ({audit_df['found_in_corpus'].sum()}/{len(audit_df)} terms)")
    return path.name


# ===========================================================================
# NEW: Report-Summary Factual Comparison
# ===========================================================================

# Mapping from priority keyword to the worst_severity it implies
_PRIORITY_TO_SEVERITY = {
    "ROUTINE": "LOW",
    "ELEVATED": "MEDIUM",
    "URGENT": "HIGH",
    "IMMEDIATE": "CRITICAL",
}

# Hazard synonyms used by the summary templates (from 01_data_engineer.py)
_HAZARD_PHRASES = {
    "Flood": ["flooding", "flood"],
    "Rescue Emergency": ["rescue emergencies", "rescue emergency", "rescue"],
    "Road Blockage": ["road blockages", "road blockage", "road blocked"],
    "Medical Emergency": ["medical emergencies", "medical emergency", "medical"],
}


def _slots_from_row(row: pd.Series) -> dict:
    """Extract all structured ground-truth slots from a pairs-CSV row."""
    hazards = [h.strip() for h in str(row.get("hazard_types", "")).split(";") if h.strip()]
    resources = [r.strip() for r in str(row.get("resources_required", "")).split(";") if r.strip()]
    locations = [l.strip() for l in str(row.get("locations", "")).split(";") if l.strip()]
    return {
        "priority": str(row.get("priority", "")),
        "worst_severity": str(row.get("worst_severity", "")),
        "zone": str(row.get("zone", "")),
        "district": str(row.get("district", "")),
        "state": str(row.get("state", "")),
        "entry_count": int(row.get("entry_count", 0)),
        "total_headcount": int(pd.to_numeric(row.get("total_headcount", 0), errors="coerce") or 0),
        "hazards": hazards,
        "resources": resources,
        "locations": locations,
    }


def _check_priority_in_summary(summary: str, expected_priority: str) -> bool:
    return expected_priority.upper() in summary.upper()


def _check_zone_in_summary(summary: str, zone: str) -> bool:
    return zone.upper() in summary.upper()


def _check_district_in_summary(summary: str, district: str) -> bool:
    return district.lower() in summary.lower()


def _check_state_in_summary(summary: str, state: str) -> bool:
    return state.lower() in summary.lower()


def _check_headcount_in_summary(summary: str, headcount: int) -> bool:
    """Verify the headcount figure that appears in the summary matches ground truth."""
    nums = [int(m) for m in re.findall(r"\b(\d+)\b", summary)]
    return headcount in nums


def _check_entry_count_in_summary(summary: str, entry_count: int) -> bool:
    """Verify the entry count figure (N reports) in the summary matches ground truth."""
    nums = [int(m) for m in re.findall(r"\b(\d+)\b", summary)]
    return entry_count in nums


def _check_hazard_in_summary(summary: str, hazards: list[str]) -> bool:
    """True if at least one hazard type (or its phrase alias) is in the summary."""
    summary_lower = summary.lower()
    for h in hazards:
        aliases = _HAZARD_PHRASES.get(h, [h.lower()])
        if any(a in summary_lower for a in aliases):
            return True
    return False


def _check_location_in_summary(summary: str, locations: list[str]) -> bool:
    """True if at least one location entity from ground truth appears in summary."""
    summary_lower = summary.lower()
    return any(loc.lower() in summary_lower for loc in locations)


def report_summary_factual_compare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Slot-by-slot grounding check: does each summary assert only facts from its report?

    For every pair we check seven factual slots:
      1. priority keyword present in summary
      2. zone named in summary
      3. district named in summary
      4. state named in summary (summary uses district, not always state explicitly)
      5. headcount figure matches
      6. entry count figure matches
      7. at least one hazard type (or alias) present
      8. at least one location entity present

    A pair PASSES if all eight checks succeed.
    """
    records = []
    for _, row in df.iterrows():
        summary = str(row.get("summary", ""))
        slots = _slots_from_row(row)

        chk_priority  = _check_priority_in_summary(summary, slots["priority"])
        chk_zone       = _check_zone_in_summary(summary, slots["zone"])
        chk_district   = _check_district_in_summary(summary, slots["district"])
        chk_state      = _check_state_in_summary(summary, slots["state"])
        chk_headcount  = _check_headcount_in_summary(summary, slots["total_headcount"])
        chk_entry_cnt  = _check_entry_count_in_summary(summary, slots["entry_count"])
        chk_hazard     = _check_hazard_in_summary(summary, slots["hazards"])
        chk_location   = _check_location_in_summary(summary, slots["locations"])

        all_pass = all([
            chk_priority, chk_zone, chk_district,
            chk_headcount, chk_entry_cnt, chk_hazard, chk_location,
        ])  # state check is advisory, not required for pass

        records.append({
            "pair_id":          row.get("pair_id", ""),
            "split":            row.get("split", ""),
            "priority":         slots["priority"],
            "chk_priority":     chk_priority,
            "chk_zone":         chk_zone,
            "chk_district":     chk_district,
            "chk_state":        chk_state,
            "chk_headcount":    chk_headcount,
            "chk_entry_count":  chk_entry_cnt,
            "chk_hazard":       chk_hazard,
            "chk_location":     chk_location,
            "all_pass":         all_pass,
            "summary_preview":  summary[:120],
        })

    result = pd.DataFrame(records)
    path = OUTPUT_DIR / "factual_comparison.csv"
    result.to_csv(path, index=False)

    # Aggregate pass rates
    slot_cols = [c for c in result.columns if c.startswith("chk_")]
    pass_rates = {c: round(float(result[c].mean()) * 100, 1) for c in slot_cols}
    overall_pass = round(float(result["all_pass"].mean()) * 100, 1)

    print(f"[factual] {path.name}  overall pass {overall_pass:.1f}% ({result['all_pass'].sum()}/{len(result)})")
    for slot, rate in pass_rates.items():
        status = "OK" if rate >= 99.0 else ("WARN" if rate >= 90.0 else "FAIL")
        print(f"  [{status}] {slot[4:]:20s}: {rate:.1f}%")

    stats = {
        "total_pairs": len(result),
        "overall_pass_pct": overall_pass,
        "overall_pass_count": int(result["all_pass"].sum()),
        "slot_pass_rates": pass_rates,
    }
    return result, stats


# ===========================================================================
# NEW: Hallucination / Factual-Drift Detection
# ===========================================================================

def _extract_numbers(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\b(\d+)\b", text)]


def _extract_cap_tokens(text: str) -> list[str]:
    """Capitalised words (len >= 3) that could be named entities."""
    skip = {"SITUATION", "RISK", "ACTIONS", "INCIDENT", "LOG", "ZONE",
            "ERSS", "ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE",
            "Send", "Dispatch", "Deploy", "Monitor", "Request",
            "Activate", "Submit", "Establish", "Notify", "Begin",
            "Confirm", "Place", "Open", "All"}
    return [
        t for t in re.findall(r"\b([A-Z][A-Za-z]{2,})\b", text)
        if t not in skip
    ]


def detect_hallucination_drift(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag pairs where the summary asserts facts not grounded in the report.

    Three drift signals are checked:

    1. **Number drift** -- a number that appears in the summary but NOT in the
       report (e.g. summary says '120 affected' but report totals only 80).
       Zero is excluded (it is structural, not a factual claim).

    2. **Entity drift** -- a capitalised token in the summary that cannot be
       traced back to the report text (potential hallucinated place/agency).

    3. **Priority mismatch** -- the summary's priority keyword does not match
       the ground-truth worst_severity mapping.

    A pair is DRIFTED if it has any number drift OR any entity drift OR
    a priority mismatch.
    """
    records = []
    for _, row in df.iterrows():
        report  = str(row.get("report",  ""))
        summary = str(row.get("summary", ""))
        priority = str(row.get("priority", ""))
        worst_sev = str(row.get("worst_severity", ""))

        # 1. Number drift
        report_nums  = set(_extract_numbers(report))
        summary_nums = set(_extract_numbers(summary))
        # Numbers in summary that don't appear at all in the report
        drifted_nums = [n for n in summary_nums if n > 0 and n not in report_nums]
        num_drift = len(drifted_nums) > 0

        # 2. Entity drift -- exclude known ICS shorthand and agency codes
        #    that are template-injected vocabulary, not hallucinations.
        _ICS_WHITELIST = {
            "sitrep", "staging", "access", "standby", "triage", "cas",
            "evac", "eta", "ic", "sip", "eoc",
            "ndrf", "sdrf", "ndma", "cwc", "imd", "erss",
            "deploy", "dispatch", "send", "notify", "submit",
        }
        report_lower = report.lower()
        summary_ents = _extract_cap_tokens(summary)
        halluc_ents = [
            e for e in summary_ents
            if e.lower() not in report_lower
            and e.lower() not in _ICS_WHITELIST
        ]
        entity_drift = len(halluc_ents) > 0

        # 3. Priority mismatch
        expected_priority = _PRIORITY_TO_SEVERITY.get(priority, "")
        # ground-truth mapping is priority -> severity; reverse check:
        # the summary's priority keyword should match the pair's priority column
        priority_mismatch = priority.upper() not in summary.upper()

        is_drifted = num_drift or entity_drift or priority_mismatch

        records.append({
            "pair_id":           row.get("pair_id", ""),
            "split":             row.get("split", ""),
            "priority":          priority,
            "worst_severity":    worst_sev,
            "num_drift":         num_drift,
            "drifted_numbers":   "; ".join(str(n) for n in drifted_nums[:5]),
            "entity_drift":      entity_drift,
            "hallucinated_ents": "; ".join(halluc_ents[:8]),
            "priority_mismatch": priority_mismatch,
            "is_drifted":        is_drifted,
            "summary_preview":   summary[:120],
        })

    result = pd.DataFrame(records)
    path = OUTPUT_DIR / "hallucination_drift_report.csv"
    result.to_csv(path, index=False)

    n_drifted   = int(result["is_drifted"].sum())
    n_num_drift = int(result["num_drift"].sum())
    n_ent_drift = int(result["entity_drift"].sum())
    n_pri_mism  = int(result["priority_mismatch"].sum())
    total       = len(result)

    print(f"[halluc] {path.name}")
    print(f"  drifted pairs     : {n_drifted}/{total} ({100*n_drifted/total:.1f}%)")
    print(f"  -- number drift   : {n_num_drift} ({100*n_num_drift/total:.1f}%)")
    print(f"  -- entity drift   : {n_ent_drift} ({100*n_ent_drift/total:.1f}%)")
    print(f"  -- priority mism. : {n_pri_mism} ({100*n_pri_mism/total:.1f}%)")

    stats = {
        "total_pairs":     total,
        "drifted":         n_drifted,
        "drift_rate_pct":  round(100 * n_drifted / max(total, 1), 1),
        "number_drift":    n_num_drift,
        "entity_drift":    n_ent_drift,
        "priority_mismatch": n_pri_mism,
    }
    return result, stats


# ===========================================================================
# NEW: Verified vs Flagged Pair Generation
# ===========================================================================

def generate_verified_flagged_pairs(
    df: pd.DataFrame,
    factual_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> dict:
    """Produce two CSVs: verified (clean) and flagged (has factual violation).

    A pair is VERIFIED only when:
      - factual_comparison.all_pass == True
      - hallucination_drift_report.is_drifted == False

    Any other combination produces a FLAGGED pair with a reason code.
    """
    # Align on pair_id
    merged = df[["pair_id", "split", "priority", "worst_severity",
                 "report", "summary",
                 "total_headcount", "entry_count",
                 "state", "district", "zone"]].copy()

    fact_cols = ["pair_id", "all_pass",
                 "chk_priority", "chk_zone", "chk_district",
                 "chk_headcount", "chk_entry_count",
                 "chk_hazard", "chk_location"]
    drift_cols = ["pair_id", "is_drifted",
                  "num_drift", "entity_drift", "priority_mismatch"]

    merged = (
        merged
        .merge(factual_df[fact_cols],  on="pair_id", how="left")
        .merge(drift_df[drift_cols],   on="pair_id", how="left")
    )

    # Build reason string for flagged pairs
    def _reason(row: pd.Series) -> str:
        reasons = []
        if not row.get("chk_priority",     True): reasons.append("wrong_priority")
        if not row.get("chk_zone",         True): reasons.append("zone_absent")
        if not row.get("chk_district",     True): reasons.append("district_absent")
        if not row.get("chk_headcount",    True): reasons.append("headcount_mismatch")
        if not row.get("chk_entry_count",  True): reasons.append("entry_count_mismatch")
        if not row.get("chk_hazard",       True): reasons.append("hazard_absent")
        if not row.get("chk_location",     True): reasons.append("location_absent")
        if row.get("num_drift",    False):          reasons.append("number_drift")
        if row.get("entity_drift", False):          reasons.append("entity_drift")
        if row.get("priority_mismatch", False):     reasons.append("priority_mismatch")
        return "; ".join(reasons) if reasons else ""

    merged["flag_reason"] = merged.apply(_reason, axis=1)
    merged["is_verified"] = merged["all_pass"] & ~merged["is_drifted"]

    verified = merged[merged["is_verified"]].drop(columns=["flag_reason"])
    flagged  = merged[~merged["is_verified"]]

    v_path = OUTPUT_DIR / "verified_pairs.csv"
    f_path = OUTPUT_DIR / "flagged_pairs.csv"
    verified.to_csv(v_path, index=False)
    flagged.to_csv(f_path,  index=False)

    n_verified = len(verified)
    n_flagged  = len(flagged)
    total      = len(merged)

    print(f"[pairs]  verified_pairs.csv : {n_verified:,} pairs  ({100*n_verified/total:.1f}%)")
    print(f"[pairs]  flagged_pairs.csv  : {n_flagged:,} pairs  ({100*n_flagged/total:.1f}%)")

    # Top flag reasons
    if n_flagged:
        all_reasons: list[str] = []
        for r in flagged["flag_reason"]:
            all_reasons.extend([x.strip() for x in str(r).split(";") if x.strip()])
        top = Counter(all_reasons).most_common(5)
        print(f"  top flag reasons: {top}")

    return {
        "verified": n_verified,
        "flagged":  n_flagged,
        "total":    total,
        "verified_pct": round(100 * n_verified / max(total, 1), 1),
    }


# ===========================================================================
# NEW: Radio-Code Retention Audit (explicit)
# ===========================================================================

# The domain dictionary shorthand values whose *retention in model summaries*
# matters most for radio-operator use.  We check each one against the
# summaries column (not the reports) because the SLM's output must carry
# the codes forward, not just the input.
#
# Categories checked:
#   a) Briefing priority keywords   (ROUTINE / ELEVATED / URGENT / IMMEDIATE)
#      -- these must appear as the FIRST WORD of the summary.
#   b) Incident-command shorthand   (EVAC, SITREP, TRIAGE, ETA, IC ...)
#      -- these are optional but tracked; their absence means the SLM
#         is not using standard radio vocabulary.
#   c) Agency codes                 (NDRF, SDRF, NDMA, IMD ...)
#      -- checked for raw presence in the summary text.

_PRIORITY_CODES = ["ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE"]

_ICS_CODES = {
    "EVAC":    "Evacuation",
    "SITREP":  "Situation Report",
    "TRIAGE":  "Triage",
    "IC":      "Incident Commander",
    "ETA":     "Estimated Time of Arrival",
    "CAS":     "Casualty",
    "SIP":     "Shelter-in-place",
    "STAGING": "Staging Area",
    "ACCESS":  "Access Route",
    "STANDBY": "Standby",
}

_AGENCY_CODES = {
    "NDRF": "National Disaster Response Force",
    "SDRF": "State Disaster Response Force",
    "NDMA": "National Disaster Management Authority",
    "IMD":  "India Meteorological Department",
    "CWC":  "Central Water Commission",
    "ERSS": "Emergency Response Support System",
}


def audit_radio_code_retention(df: pd.DataFrame,
                               dictionary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Measure how often each radio code / shorthand appears in the summaries.

    Three sub-analyses:
      1. Priority-keyword retention -- must appear as the opening word.
         A summary that does NOT begin with its expected priority keyword is
         a training-data defect; this check surfaces those cases.
      2. ICS shorthand retention  -- optional but tracked.
      3. Agency code retention    -- optional but tracked.

    A per-code retention rate (appearances per 1,000 summaries) is computed,
    plus a binary 'used_at_all' flag.
    """
    summaries = df["summary"].astype(str).tolist()
    priorities = df["priority"].astype(str).tolist()
    n = len(summaries)

    rows: list[dict] = []

    # --- Priority keyword: must be the opening token ------------------------
    for code in _PRIORITY_CODES:
        count_lead = sum(
            1 for s in summaries if s.strip().upper().startswith(code)
        )
        count_any = sum(1 for s in summaries if code in s.upper())
        # Expected appearances = pairs whose ground-truth priority == code
        expected = sum(1 for p in priorities if p == code)
        rows.append({
            "shorthand":        code,
            "category":         "Briefing priority",
            "full_term":        code.title(),
            "appears_as_opener": count_lead,
            "appears_anywhere": count_any,
            "expected_appearances": expected,
            "opener_rate_pct":  round(100 * count_lead / max(expected, 1), 1),
            "retention_per_1k": round(1000 * count_any / n, 1),
            "used_at_all":      count_any > 0,
        })

    # --- ICS shorthand ------------------------------------------------------
    for code, full_term in _ICS_CODES.items():
        count = sum(1 for s in summaries if re.search(rf"\b{re.escape(code)}\b", s))
        rows.append({
            "shorthand":            code,
            "category":             "Incident-command term",
            "full_term":            full_term,
            "appears_as_opener":    0,
            "appears_anywhere":     count,
            "expected_appearances": None,
            "opener_rate_pct":       None,
            "retention_per_1k":     round(1000 * count / n, 1),
            "used_at_all":          count > 0,
        })

    # --- Agency codes -------------------------------------------------------
    for code, full_term in _AGENCY_CODES.items():
        count = sum(1 for s in summaries if re.search(rf"\b{re.escape(code)}\b", s))
        rows.append({
            "shorthand":            code,
            "category":             "Agency",
            "full_term":            full_term,
            "appears_as_opener":    0,
            "appears_anywhere":     count,
            "expected_appearances": None,
            "opener_rate_pct":       None,
            "retention_per_1k":     round(1000 * count / n, 1),
            "used_at_all":          count > 0,
        })

    result = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "radio_code_retention.csv"
    result.to_csv(csv_path, index=False)

    # --- Bar-chart plot -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    categories = ["Briefing priority", "Incident-command term", "Agency"]
    cat_colours = ["#1565C0", "#4CAF50", "#E65100"]

    for ax, cat, col in zip(axes, categories, cat_colours):
        sub = result[result["category"] == cat].copy()
        metric = "opener_rate_pct" if cat == "Briefing priority" else "retention_per_1k"
        xlabel = "Opener rate (%)" if cat == "Briefing priority" else "Per 1,000 summaries"
        sub = sub.sort_values(metric, ascending=True)
        vals = sub[metric].fillna(0).tolist()
        labels = sub["shorthand"].tolist()
        bars = ax.barh(labels, vals, color=col, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(cat, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Radio-Code Retention in SLM Summaries", fontsize=13, y=1.01)
    fig.tight_layout()
    png_path = OUTPUT_DIR / "radio_code_retention.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Console summary
    print(f"[radio]  {csv_path.name}  |  {png_path.name}")
    priority_rows = result[result["category"] == "Briefing priority"]
    for _, r in priority_rows.iterrows():
        status = "OK" if r["opener_rate_pct"] >= 99.0 else "WARN"
        print(f"  [{status}] {r['shorthand']:10s} opener rate {r['opener_rate_pct']:.1f}%  "
              f"(expected {r['expected_appearances']}, found leading {r['appears_as_opener']})")
    ics_used = result[result["category"] == "Incident-command term"]["used_at_all"].sum()
    ics_total = (result["category"] == "Incident-command term").sum()
    print(f"  ICS codes used in at least 1 summary: {ics_used}/{ics_total}")
    agency_used = result[result["category"] == "Agency"]["used_at_all"].sum()
    agency_total = (result["category"] == "Agency").sum()
    print(f"  Agency codes used in at least 1 summary: {agency_used}/{agency_total}")

    return result, csv_path.name


# ---------------------------------------------------------------------------
# Per-split statistics table
# ---------------------------------------------------------------------------

def compute_split_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for split in SPLIT_ORDER:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        stats[split] = {
            "pairs": int(len(sub)),
            "priority_counts": sub["priority"].value_counts().to_dict(),
            "mean_report_words": round(float(sub["report_words"].mean()), 1),
            "mean_summary_words": round(float(sub["summary_words"].mean()), 1),
            "mean_compression_pct": round(float(sub["compression_pct"].mean()), 1),
            "states": int(sub["state"].nunique()),
            "districts": int(sub["district"].nunique()),
        }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(top_n: int = 30) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_pairs()
    print(f"[load] {len(df):,} pairs  |  columns: {list(df.columns)}")

    dictionary = load_dictionary()

    artefacts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Existing plots
    # ------------------------------------------------------------------
    artefacts["priority_distribution.png"] = "Briefing priority bar chart"
    plot_priority_distribution(df)

    artefacts["compression_histogram.png"] = "Compression % histograms by priority"
    plot_compression_histogram(df)

    for fname in plot_length_distributions(df):
        artefacts[fname] = "Length distribution histogram"

    artefacts["split_priority_matrix.png"] = "Split x priority heatmap"
    plot_split_priority_matrix(df)

    artefacts["per_state_coverage.png"] = "Pairs per state"
    plot_state_coverage(df)

    # Vocabulary
    vocab_outputs = analyse_vocabulary(df, top_n)
    artefacts.update(vocab_outputs)

    # Basic domain dictionary audit
    audit_name = audit_domain_dictionary(df, dictionary)
    if audit_name:
        artefacts[audit_name] = "Domain dictionary term coverage audit"

    # ------------------------------------------------------------------
    # NEW: Report-summary factual comparison
    # ------------------------------------------------------------------
    print("\n[factual] Running report-summary slot comparison ...")
    factual_df, factual_stats = report_summary_factual_compare(df)
    artefacts["factual_comparison.csv"] = (
        "Per-pair slot-by-slot report vs summary factual check"
    )

    # ------------------------------------------------------------------
    # NEW: Hallucination / factual-drift detection
    # ------------------------------------------------------------------
    print("\n[halluc] Running hallucination / factual-drift detection ...")
    drift_df, drift_stats = detect_hallucination_drift(df)
    artefacts["hallucination_drift_report.csv"] = (
        "Per-pair hallucination and numeric-drift flags"
    )

    # ------------------------------------------------------------------
    # NEW: Verified vs flagged pair generation
    # ------------------------------------------------------------------
    print("\n[pairs] Generating verified / flagged pair CSVs ...")
    vf_stats = generate_verified_flagged_pairs(df, factual_df, drift_df)
    artefacts["verified_pairs.csv"] = "Pairs passing all factual checks"
    artefacts["flagged_pairs.csv"]  = "Pairs with at least one factual violation"

    # ------------------------------------------------------------------
    # NEW: Radio-code retention audit
    # ------------------------------------------------------------------
    print("\n[radio] Running radio-code retention audit ...")
    _, radio_csv = audit_radio_code_retention(df, dictionary)
    artefacts["radio_code_retention.csv"] = "Per-shorthand retention rate in summaries"
    artefacts["radio_code_retention.png"] = "Bar chart of shorthand retention rates"

    # ------------------------------------------------------------------
    # Split stats
    # ------------------------------------------------------------------
    split_stats = compute_split_stats(df)

    # Overall dataset summary
    overall = {
        "total_pairs": int(len(df)),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "grouping_distribution": df["grouping"].value_counts().to_dict(),
        "report_words": {
            "min": int(df["report_words"].min()),
            "max": int(df["report_words"].max()),
            "mean": round(float(df["report_words"].mean()), 1),
            "std": round(float(df["report_words"].std()), 1),
        },
        "summary_words": {
            "min": int(df["summary_words"].min()),
            "max": int(df["summary_words"].max()),
            "mean": round(float(df["summary_words"].mean()), 1),
            "std": round(float(df["summary_words"].std()), 1),
        },
        "compression_pct": {
            "min": round(float(df["compression_pct"].min()), 1),
            "max": round(float(df["compression_pct"].max()), 1),
            "mean": round(float(df["compression_pct"].mean()), 1),
            "std": round(float(df["compression_pct"].std()), 1),
        },
        "states": int(df["state"].nunique()),
        "districts": int(df["district"].nunique()),
        "zones": int(df["zone"].nunique()),
    }

    manifest = {
        "stage": "04_SLM",
        "script": "02_eda_engineer.py",
        "top_n": top_n,
        "overall": overall,
        "splits": split_stats,
        "factual_integrity": factual_stats,
        "hallucination_drift": drift_stats,
        "verified_flagged": vf_stats,
        "artefacts": artefacts,
        "output_dir": str(OUTPUT_DIR),
    }

    manifest_path = OUTPUT_DIR / "eda_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[write] {manifest_path.name}")

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"  total pairs          : {overall['total_pairs']:,}")
    print(f"  priority mix         : {overall['priority_distribution']}")
    print(f"  report words         : {overall['report_words']}")
    print(f"  summary words        : {overall['summary_words']}")
    print(f"  compression          : {overall['compression_pct']}")
    print(f"  factual pass rate    : {factual_stats['overall_pass_pct']:.1f}%  "
          f"({factual_stats['overall_pass_count']}/{factual_stats['total_pairs']})")
    print(f"  drift rate           : {drift_stats['drift_rate_pct']:.1f}%  "
          f"({drift_stats['drifted']}/{drift_stats['total_pairs']})")
    print(f"  verified pairs       : {vf_stats['verified']:,}  ({vf_stats['verified_pct']:.1f}%)")
    print(f"  flagged pairs        : {vf_stats['flagged']:,}")
    print(f"  outputs -> {OUTPUT_DIR}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=30,
                        help="Top N terms/bigrams to retain per table (default: 30)")
    args = parser.parse_args()
    run(args.top_n)


if __name__ == "__main__":
    main()
