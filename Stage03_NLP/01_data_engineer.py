import os
import re
import random
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DISPATCHER_REQUIRED_COLS = [
    "incident_id", "timestamp", "state", "district", "zone",
    "call_type", "severity", "hazard_type",
]

LOCATION_PHRASES = [
    ("near the local market", "local market"),
    ("close to the river bank", "river bank"),
    ("on the national highway", "national highway"),
    ("in a residential colony", "residential colony"),
    ("near the bus stand", "bus stand"),
    ("at the railway crossing", "railway crossing"),
    ("in the low-lying area", "low-lying area"),
    ("near the village school", "village school"),
    ("at the highway junction", "highway junction"),
    ("in the riverside settlement", "riverside settlement"),
    ("near the industrial area", "industrial area"),
    ("in the hilly terrain", "hilly terrain"),
    ("near the main bridge", "main bridge"),
    ("in the old town area", "old town area"),
    ("near the government hospital", "government hospital"),
]

DISPATCH_RESOURCES = {
    "Flooding": ["rescue boat", "NDRF flood rescue team", "water pump",
                 "relief supplies", "life jackets"],
    "Road Blockage": ["tow truck", "traffic police team",
                       "earthmover (JCB)", "road clearance crew"],
    "Rescue": ["NDRF rescue team", "rescue rope and ladder", "ambulance",
               "fire and rescue unit"],
    "Waterlogging": ["water pump", "sandbags", "road clearance team",
                      "drainage team"],
    "Medical Emergency": ["ambulance", "paramedic team",
                            "medical first-aid kit",
                            "emergency medical team"],
}

CALL_VERB = {
    "Flooding": "Flooding reported",
    "Road Blockage": "Road blockage reported",
    "Rescue": "Rescue operation required",
    "Waterlogging": "Waterlogging reported",
    "Medical Emergency": "Medical emergency reported",
}

REPORT_PREFIXES = [
    "Incident reported",
    "Emergency report received",
    "Situation update",
]
DISPATCH_PHRASES = [
    "Dispatch requested for",
    "Please dispatch",
    "Response team requested: ",
]

HEADCOUNT_RANGE = {
    "LOW": (1, 3),
    "MEDIUM": (3, 8),
    "HIGH": (8, 20),
    "CRITICAL": (15, 80),
}

CALL_TYPE_HAZARDS = {
    "Flooding": "Flood",
    "Road Blockage": "Road Blockage",
    "Rescue": "Rescue Emergency",
    "Waterlogging": "Flood",
    "Medical Emergency": "Medical Emergency",
}

def people_phrase(severity: str, n: int) -> str:
    if severity == "LOW":
        return f"{n} person(s) affected, no major injuries reported"
    if severity == "MEDIUM":
        return f"{n} people affected, minor injuries reported"
    if severity == "HIGH":
        return (f"{n} people affected including elderly and children, "
                 "some injuries reported")
    return (f"{n} people trapped/affected, multiple injuries reported, "
             "urgent evacuation needed")

def generate_dispatcher_processed(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    rows_text, rows_loc_entity, rows_res_entity = [], [], []
    rows_headcount, rows_people_ind, rows_loc_ind = [], [], []

    for _, row in df.iterrows():
        call_type = row["call_type"]
        severity = row["severity"]
        district = row["district"]
        state = row["state"]

        if call_type not in DISPATCH_RESOURCES:
            raise ValueError(
                f"Unrecognised call_type '{call_type}'. Expected one of "
                f"{list(DISPATCH_RESOURCES.keys())}"
            )
        if severity not in HEADCOUNT_RANGE:
            raise ValueError(
                f"Unrecognised severity '{severity}'. Expected one of "
                f"{list(HEADCOUNT_RANGE.keys())}"
            )

        loc_phrase, loc_entity = random.choice(LOCATION_PHRASES)

        n_res = random.choice([1, 1, 2])
        pool = DISPATCH_RESOURCES[call_type]
        resources = random.sample(pool, k=min(n_res, len(pool)))
        resource_str = " and ".join(resources)

        lo, hi = HEADCOUNT_RANGE[severity]
        n = random.randint(lo, hi)
        p_phrase = people_phrase(severity, n)

        report_prefix = random.choice(REPORT_PREFIXES)
        dispatch_phrase = random.choice(DISPATCH_PHRASES)
        text = (f"{CALL_VERB[call_type]} ({report_prefix}) {loc_phrase} in "
            f"{district}, {state}. {p_phrase}. {dispatch_phrase} "
            f"{resource_str}.")

        rows_text.append(text)
        rows_loc_entity.append(loc_entity)
        rows_res_entity.append(", ".join(resources))
        rows_headcount.append(n)
        rows_people_ind.append(p_phrase)
        rows_loc_ind.append(loc_phrase)

    out = df.copy()
    if out["hazard_type"].nunique(dropna=False) <= 1:
        out["hazard_type"] = out["call_type"].map(CALL_TYPE_HAZARDS).fillna(
            out["hazard_type"]
        )
    out["text"] = rows_text
    out["location_entity"] = rows_loc_entity
    out["resource_entity"] = rows_res_entity
    out["headcount_entity"] = rows_headcount
    out["people_indicators"] = rows_people_ind
    out["location_indicators"] = rows_loc_ind
    return out

def make_dispatcher_raw(processed_df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = ["incident_id", "timestamp", "state", "district", "zone",
                "call_type", "text", "severity", "hazard_type"]
    return processed_df[raw_cols].copy()

def build_dispatcher_dataset(source_csv: str, output_dir: str):
    if not source_csv or not os.path.exists(source_csv):
        print(f"[dispatcher] source CSV not found ({source_csv}); skipping stage 1.")
        return None, None

    df = pd.read_csv(source_csv)

    missing_cols = [c for c in DISPATCHER_REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[dispatcher] source CSV is missing required columns: {missing_cols}. "
            f"Expected at least: {DISPATCHER_REQUIRED_COLS}"
        )

    processed_df = generate_dispatcher_processed(df)
    raw_df = make_dispatcher_raw(processed_df)

    os.makedirs(output_dir, exist_ok=True)
    processed_path = os.path.join(output_dir, "Dispatcher_Log_Master_NLP_FILLED.csv")
    raw_path = os.path.join(output_dir, "Dispatcher_Log_Master_RAW.csv")

    processed_df.to_csv(processed_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    print(f"[dispatcher] processed dataset -> {processed_path}  shape={processed_df.shape}")
    print(f"[dispatcher] raw dataset       -> {raw_path}  shape={raw_df.shape}")
    return raw_df, processed_df

INDIAN_LOCATIONS = {
    "Tamil Nadu": ["Chennai", "Coimbatore", "Erode", "Madurai",
                   "Tiruchirappalli", "Salem", "Tirunelveli", "Thanjavur"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode",
               "Alappuzha", "Kollam", "Thrissur"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Puri", "Balasore", "Sambalpur"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Kolhapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
}

HAZARDS = ["Flood", "Cyclone", "Heavy Rainfall", "Landslide",
           "Earthquake", "Lightning", "Forest Fire", "Tsunami"]

PLATFORMS = ["X", "Facebook", "Instagram", "WhatsApp", "Telegram"]

AUTHOR_TYPES = ["Citizen", "Resident", "Volunteer", "Local Reporter", "Community Group"]

POST_TYPES = ["Distress Call", "Help Request", "Situation Report",
              "Safety Update", "Missing Person", "Resource Request"]

TEXT_TEMPLATES = {
    "Flood": [
        "{location} is severely flooded after continuous rainfall. "
        "People are trapped and need immediate rescue.",
        "Water has entered houses in {location}. "
        "Please send rescue teams and boats.",
        "Flood situation is getting worse in {location}. "
        "Families need drinking water and food.",
        "Residents near {location} are requesting emergency assistance. "
        "Roads are under water.",
    ],
    "Cyclone": [
        "Strong cyclone conditions reported in {location}. "
        "People are requesting emergency shelter.",
        "Heavy winds and rain affecting {location}. "
        "Please avoid travelling and stay indoors.",
        "Cyclone impact reported around {location}. "
        "Emergency teams are needed.",
        "Several families in {location} need food, water and shelter.",
    ],
    "Heavy Rainfall": [
        "Continuous heavy rainfall reported in {location}. "
        "Roads are becoming difficult to use.",
        "Heavy rain is causing waterlogging in {location}. "
        "Residents need emergency assistance.",
        "Rainfall situation is worsening in {location}. "
        "Please send rescue support if required.",
        "Several areas around {location} are affected by heavy rain.",
    ],
    "Landslide": [
        "Landslide reported near {location}. "
        "Road access is blocked and people need help.",
        "A landslide has affected roads near {location}. "
        "Emergency rescue support is required.",
        "Residents near {location} are reporting a major landslide. "
        "Please send emergency teams.",
        "Debris is blocking the road in {location}. "
        "People are requesting immediate assistance.",
    ],
    "Earthquake": [
        "Strong tremors reported in {location}. "
        "Residents are requesting emergency assistance.",
        "Buildings have been affected by an earthquake in {location}. "
        "Rescue teams are needed.",
        "People in {location} are reporting earthquake damage. "
        "Medical assistance may be required.",
        "Emergency support requested from {location} after earthquake tremors.",
    ],
    "Lightning": [
        "Lightning activity reported around {location}. "
        "Residents are advised to stay indoors.",
        "Severe lightning reported in {location}. "
        "Emergency assistance may be required.",
        "Lightning incident reported near {location}. "
        "Please avoid open areas.",
    ],
    "Forest Fire": [
        "Forest fire reported near {location}. "
        "Fire and rescue teams are urgently needed.",
        "Smoke and fire spreading near {location}. "
        "Residents may need evacuation support.",
        "Wildfire situation reported around {location}. "
        "Emergency response is required.",
        "Fire is spreading near forest areas around {location}. "
        "Please send fire and rescue teams.",
    ],
    "Tsunami": [
        "Tsunami warning situation reported near {location}. "
        "Residents should move to safer areas.",
        "Coastal residents near {location} are requesting emergency support.",
        "Tsunami-related emergency reported near {location}. "
        "Evacuation assistance may be required.",
    ],
}

URGENCY_TRIGGER_WORDS = ["trapped", "urgently", "immediate", "emergency"]

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_hazard(text: str) -> str:
    text = text.lower()
    if "flood" in text:
        return "Flood"
    if "cyclone" in text:
        return "Cyclone"
    if "rain" in text:
        return "Heavy Rainfall"
    if "landslide" in text:
        return "Landslide"
    if "earthquake" in text or "tremor" in text:
        return "Earthquake"
    if "lightning" in text:
        return "Lightning"
    if "fire" in text or "wildfire" in text:
        return "Forest Fire"
    if "tsunami" in text:
        return "Tsunami"
    return "Unknown"

def generate_social_feed_dataset(n_records: int, output_dir: str, seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    start_time = datetime(2026, 1, 1, 8, 0, 0)
    records = []

    for i in range(n_records):
        state = random.choice(list(INDIAN_LOCATIONS.keys()))
        location = random.choice(INDIAN_LOCATIONS[state])
        hazard = random.choice(HAZARDS)
        platform = random.choice(PLATFORMS)
        author_type = random.choice(AUTHOR_TYPES)
        post_type = random.choice(POST_TYPES)

        template = random.choice(TEXT_TEMPLATES[hazard])
        text = template.format(location=location)

        timestamp = start_time + timedelta(minutes=random.randint(0, 60 * 24 * 180))
        source_event_id = f"IND_EVENT_{random.randint(1000, 9999)}"

        records.append({
            "post_id": f"IND_SOCIAL_{i + 1:05d}",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform,
            "text": text,
            "author_type": author_type,
            "post_type": post_type,
            "state": state,
            "district": location,
            "location_mentioned": location,
            "language": "English",
            "country": "India",
            "source_type": "Simulated Social Feed",
            "source_portal": "India Disaster Information Sources",
            "source_url": "",
            "source_event_id": source_event_id,
        })

    raw_df = pd.DataFrame(records)

    processed_df = raw_df.copy()
    processed_df["clean_text"] = processed_df["text"].apply(clean_text)
    processed_df["hazard_type"] = processed_df["text"].apply(detect_hazard)

    urgency_values, resource_values, people_values = [], [], []
    for _, row in processed_df.iterrows():
        text = row["text"].lower()

        if any(word in text for word in URGENCY_TRIGGER_WORDS):
            urgency = random.choice(["Critical", "High"])
        else:
            urgency = random.choice(["Medium", "Low", "High"])
        urgency_values.append(urgency)

        if "rescue" in text:
            resource = "Rescue Team"
        elif "boat" in text:
            resource = "Boat"
        elif "food" in text:
            resource = "Food"
        elif "water" in text:
            resource = "Drinking Water"
        elif "medical" in text:
            resource = "Medical Aid"
        elif "shelter" in text:
            resource = "Shelter"
        elif "fire" in text:
            resource = "Fire and Rescue Team"
        elif "ambulance" in text:
            resource = "Ambulance"
        else:
            resource = "Emergency Support"
        resource_values.append(resource)

        if urgency == "Critical":
            people = random.randint(10, 100)
        elif urgency == "High":
            people = random.randint(5, 50)
        elif urgency == "Medium":
            people = random.randint(1, 20)
        else:
            people = random.randint(0, 10)
        people_values.append(people)

    processed_df["urgency_level"] = urgency_values
    processed_df["resource_needed"] = resource_values
    processed_df["people_affected"] = people_values

    processed_df = processed_df[[
        "post_id", "timestamp", "platform", "text", "clean_text",
        "author_type", "post_type", "country", "state", "district",
        "location_mentioned", "hazard_type", "urgency_level",
        "resource_needed", "people_affected", "language",
        "source_type", "source_portal", "source_url", "source_event_id",
    ]]

    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "Social_Feeds_India_Raw.csv")
    processed_path = os.path.join(output_dir, "Social_Feeds_India_Processed.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")
    processed_df.to_csv(processed_path, index=False, encoding="utf-8")

    print(f"[social-feed] raw dataset       -> {raw_path}  shape={raw_df.shape}")
    print(f"[social-feed] processed dataset -> {processed_path}  shape={processed_df.shape}")

    try:
        from google.colab import files as colab_files
        colab_files.download(raw_path)
        colab_files.download(processed_path)
        print("[social-feed] Colab detected -> download started.")
    except ImportError:
        pass

    return raw_df, processed_df

INVALID_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

NER_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b")


def annotate_bio(text: str, entities: dict[str, object]) -> dict[str, list[str]]:
    """Create token-level BIO tags from known entity values in one text."""
    matches = []
    for entity_type, value in entities.items():
        if value is None or pd.isna(value):
            continue
        values = str(value).replace(",", "|").split("|")
        for entity in values:
            entity = entity.strip()
            if not entity:
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", re.IGNORECASE)
            matches.extend((match.start(), match.end(), entity_type) for match in pattern.finditer(text))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    tokens, tags = [], []
    for token_match in NER_TOKEN_PATTERN.finditer(text):
        token_tag = "O"
        overlapping = [
            match for match in matches
            if match[0] < token_match.end() and match[1] > token_match.start()
        ]
        if overlapping:
            start, _, entity_type = max(overlapping, key=lambda item: item[1] - item[0])
            token_tag = ("B-" if start == token_match.start() else "I-") + entity_type
        tokens.append(token_match.group(0))
        tags.append(token_tag)
    return {"tokens": tokens, "ner_tags": tags}


def write_ner_annotations(frame: pd.DataFrame, dataset: str, text_column: str,
                          entity_columns: dict[str, str], output_dir: str) -> str:
    """Write one JSON-lines record per text with tokens and BIO NER labels."""
    records = []
    for _, row in frame.iterrows():
        entities = {
            entity_type: row[column]
            for entity_type, column in entity_columns.items()
            if column in frame.columns
        }
        annotation = annotate_bio(str(row.get(text_column, "")), entities)
        records.append({
            "record_id": row.get("incident_id", row.get("post_id", row.get("record_id", ""))),
            "dataset": dataset,
            "text": str(row.get(text_column, "")),
            **annotation,
        })
    path = os.path.join(output_dir, f"{dataset}_ner_bio.jsonl")
    pd.DataFrame(records).to_json(path, orient="records", lines=True, force_ascii=False)
    return path

def run_data_quality_audit(df: pd.DataFrame, name: str = "dataset") -> dict:
    print("=" * 60)
    print(f"DATA QUALITY AUDIT: {name}")
    print("=" * 60)
    print(f"\nTotal Rows    : {len(df)}")
    print(f"Total Columns : {len(df.columns)}")

    total_missing = 0
    duplicate_count = 0
    empty_action = 0
    short_action = 0
    total_invalid = 0
    duplicate_ids = None

    print("\n" + "=" * 60)
    print("1. MISSING VALUE CHECK")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_table = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percentage": (missing.values / max(len(df), 1) * 100).round(2),
    })
    print(missing_table.to_string(index=False))
    total_missing = int(df.isnull().sum().sum())
    print(f"\nTOTAL MISSING VALUES: {total_missing}")
    print("STATUS: PASS" if total_missing == 0 else "STATUS: REVIEW")

    print("\n" + "=" * 60)
    print("2. DUPLICATE ROW CHECK")
    print("=" * 60)
    duplicate_count = int(df.duplicated().sum())
    print("Exact duplicate rows:", duplicate_count)
    print("STATUS: PASS" if duplicate_count == 0 else "STATUS: REVIEW")

    print("\n" + "=" * 60)
    print("3. DUPLICATE TEXT CHECK")
    print("=" * 60)
    text_col_candidates = [c for c in ["text_clean", "clean_text", "text"] if c in df.columns]
    if text_col_candidates:
        col = text_col_candidates[0]
        text_dupes = int(df.duplicated(subset=[col], keep=False).sum())
        print(f"Using column '{col}'. Rows with repeated text:", text_dupes)
    else:
        print("No recognised text column found (looked for text_clean/clean_text/text).")

    print("\n" + "=" * 60)
    print("4. EMPTY / SHORT TEXT CHECK")
    print("=" * 60)
    action_col_candidates = [c for c in ["action_text", "text", "clean_text"] if c in df.columns]
    if action_col_candidates:
        col = action_col_candidates[0]
        action_text = df[col].fillna("").astype(str).str.strip()
        empty_action = int((action_text == "").sum())
        short_action = int((action_text.str.len() < 10).sum())
        print(f"Using column '{col}'.")
        print("Empty rows            :", empty_action)
        print("Very short (<10 chars) :", short_action)
        print("STATUS: PASS" if empty_action == 0 else "STATUS: REVIEW")
    else:
        print("No text-like column found to check.")

    print("\n" + "=" * 60)
    print("5. INVALID CHARACTER CHECK")
    print("=" * 60)
    invalid_results = {}
    for col in df.columns:
        if df[col].dtype == "object":
            count = df[col].astype(str).apply(lambda x: bool(INVALID_CHAR_PATTERN.search(x))).sum()
            invalid_results[col] = int(count)
    invalid_table = pd.DataFrame(list(invalid_results.items()),
                                  columns=["column", "rows_with_invalid_characters"])
    print(invalid_table.to_string(index=False) if not invalid_table.empty else "No object columns.")
    total_invalid = sum(invalid_results.values())
    print("\nTotal invalid-character occurrences:", total_invalid)

    print("\n" + "=" * 60)
    print("6. EXTRA WHITESPACE CHECK")
    print("=" * 60)
    whitespace_results = {}
    for col in df.columns:
        if df[col].dtype == "object":
            series = df[col].fillna("").astype(str)
            whitespace_results[col] = int(series.str.contains(r"\s{2,}", regex=True).sum())
    whitespace_table = pd.DataFrame(list(whitespace_results.items()),
                                     columns=["column", "rows_with_extra_whitespace"])
    print(whitespace_table.to_string(index=False) if not whitespace_table.empty else "No object columns.")

    for label_col, title in [
        ("disaster_phase", "7. DISASTER PHASE LABEL DISTRIBUTION"),
        ("hazard_type", "8. HAZARD LABEL DISTRIBUTION"),
        ("action_type", "9. ACTION TYPE DISTRIBUTION"),
        ("responsible_agency", "10. RESPONSIBLE AGENCY DISTRIBUTION"),
        ("urgency_level", "10b. URGENCY LEVEL DISTRIBUTION"),
        ("resource_needed", "10c. RESOURCE NEEDED DISTRIBUTION"),
    ]:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        if label_col in df.columns:
            print(df[label_col].fillna("MISSING").value_counts().head(30).to_string())
            print(f"\nUnique label combinations: {df[label_col].nunique()}")
        else:
            print(f"{label_col} column not found.")

    print("\n" + "=" * 60)
    print("11. OUTLIER CHECK (IQR)")
    print("=" * 60)
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        print("Numeric columns found:", numeric_columns)
        for col in numeric_columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            print(f"{col}: {outliers} potential IQR outliers")
    else:
        print("No numeric columns requiring outlier treatment.")

    print("\n" + "=" * 60)
    print("12. ID UNIQUENESS CHECK")
    print("=" * 60)
    id_col_candidates = [c for c in df.columns if c.endswith("_id")]
    if id_col_candidates:
        col = id_col_candidates[0]
        duplicate_ids = int(df[col].duplicated().sum())
        print(f"Using column '{col}'. Duplicate IDs:", duplicate_ids)
        print("STATUS: PASS" if duplicate_ids == 0 else "STATUS: REVIEW")
    else:
        print("No *_id column found.")

    print("\n" + "=" * 60)
    print("13. SOURCE DISTRIBUTION")
    print("=" * 60)
    source_col_candidates = [c for c in ["source_document", "source_portal", "source_type"] if c in df.columns]
    if source_col_candidates:
        col = source_col_candidates[0]
        print(df[col].value_counts().to_string())
    else:
        print("No source column found.")

    print("\n" + "=" * 60)
    print("FINAL DATA QUALITY SUMMARY")
    print("=" * 60)
    print(f"""
Total rows                  : {len(df)}
Total columns                : {len(df.columns)}
Total missing values        : {total_missing}
Exact duplicate rows        : {duplicate_count}
Empty short-text rows       : {empty_action}
Invalid character issues    : {total_invalid}
Duplicate IDs                : {duplicate_ids if duplicate_ids is not None else "N/A"}
""")

    overall_pass = (total_missing == 0 and duplicate_count == 0 and empty_action == 0)
    print("OVERALL STATUS:", "PASS" if overall_pass else "REVIEW")
    print("=" * 60)

    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "total_missing": total_missing,
        "duplicate_rows": duplicate_count,
        "empty_short_text_rows": empty_action,
        "invalid_char_issues": total_invalid,
        "duplicate_ids": duplicate_ids,
        "overall_status": "PASS" if overall_pass else "REVIEW",
    }

def main():
    parser = argparse.ArgumentParser(description="Unified data engineering pipeline")
    parser.add_argument("--dispatcher-source", default=None,
                         help="Path to Dispatcher_Log_Master_*.csv (source with empty NLP columns)")
    parser.add_argument("--output-dir", default="./output", help="Directory to write all output CSVs")
    parser.add_argument("--social-records", type=int, default=5000,
                         help="Number of simulated social-feed records to generate")
    parser.add_argument("--skip-dispatcher", action="store_true",
                         help="Skip stage 1 (dispatcher log) entirely")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audit_reports = []

    if not args.skip_dispatcher:
        raw_disp, processed_disp = build_dispatcher_dataset(args.dispatcher_source, args.output_dir)
        if processed_disp is not None:
            audit_reports.append(run_data_quality_audit(processed_disp, "Dispatcher Log (NLP-filled)"))
            write_ner_annotations(
                processed_disp,
                "dispatcher",
                "text",
                {"LOCATION": "location_entity", "RESOURCE": "resource_entity",
                 "HEADCOUNT": "headcount_entity"},
                args.output_dir,
            )

    raw_social, processed_social = generate_social_feed_dataset(args.social_records, args.output_dir)
    audit_reports.append(run_data_quality_audit(processed_social, "Social Feeds (India, processed)"))
    write_ner_annotations(
        processed_social,
        "social_feeds",
        "text",
        {"LOCATION": "location_mentioned", "HAZARD": "hazard_type",
         "RESOURCE": "resource_needed", "HEADCOUNT": "people_affected"},
        args.output_dir,
    )

    sop_path = os.path.join(os.path.dirname(__file__), "data", "processed",
                            "Safety_SOP_NLP_Dataset_PROCESSED.csv")
    if os.path.exists(sop_path):
        sop_df = pd.read_csv(sop_path, low_memory=False)
        write_ner_annotations(
            sop_df,
            "safety_sop",
            "text_clean",
            {"HAZARD": "hazard_type", "AGENCY": "responsible_agency",
             "ACTION": "action_type"},
            args.output_dir,
        )

    print("\n" + "#" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("#" * 60)
    for r in audit_reports:
        print(f"- {r['name']}: {r['rows']} rows, {r['columns']} cols, "
              f"status={r['overall_status']}")

if __name__ == "__main__":
    main()
