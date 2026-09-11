"""Stage 05 GenAI -- GenAI Engineer.

The generative scenario pipeline.

1. SensorCVAE -- a conditional variational autoencoder trained on the 10,000
   real labelled sensor rows. It learns the JOINT distribution of the nine
   Stage 01 sensor fields given a risk class, so a sampled zone keeps real
   correlations (rainfall vs emergency calls, r = 0.82) instead of drawing each
   field independently, which is how naive "random scenario" generators produce
   heavy rain with no calls at all.

2. ScenarioGenerator -- realises a scenario PROMPT (02_eda_engineer.py) as a
   multi-zone compound disaster. Per zone it produces:
     sensors        a Stage 01 payload (CVAE sample + stress modifiers)
     water_levels   a 72 h gauge history shaped by real CWC rise/fall rates
     text entries   dispatcher / citizen messages recombined from real Stage 03
                    phrasing, plus implicit-urgency, negation and noise variants
     incident_log   the same entries in Stage 04's INCIDENT LOG format
     image_path     a mask-labelled scene from the imagery bank
     expected       ground-truth pass criteria for the stress test

Stress modifiers deliberately push some fields BEYOND the historical record --
that is the point of a stress test. Every push is logged per zone in
provenance.sensor_adjustments / beyond_record_fields, so extrapolation is never
passed off as typical.

Ground truth is the SCENARIO DESIGNER'S INTENT, not an observed outcome. The
stress test measures whether the pipeline responds the way the designer (and
the fusion policy's own documented rules) say it should.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
SCENARIO_DIR = OUTPUT_DIR / "scenarios"

SENSOR_REFERENCE_CSV = PROCESSED_DIR / "sensor_reference.csv"
DISTRIBUTIONS_JSON = PROCESSED_DIR / "reference_distributions.json"
PHRASE_BANK_JSON = PROCESSED_DIR / "text_phrase_bank.json"
WATER_REFERENCE_JSON = PROCESSED_DIR / "water_level_reference.json"
IMAGERY_MANIFEST_CSV = PROCESSED_DIR / "imagery_manifest.csv"
PROMPTS_JSON = PROCESSED_DIR / "scenario_prompts.json"

CVAE_PATH = MODEL_DIR / "sensor_cvae.pt"
TRAINING_MANIFEST_JSON = OUTPUT_DIR / "genai_training_manifest.json"
SUITE_JSON = SCENARIO_DIR / "scenario_suite.json"
WILDCARD_JSON = SCENARIO_DIR / "wildcard_scenario.json"
ZONES_CSV = SCENARIO_DIR / "scenario_zones.csv"
SYNTHETIC_SAMPLES_CSV = OUTPUT_DIR / "synthetic_sensor_samples.csv"

SEED = 42

NUMERIC_FEATURES = [
    "rainfall_mm", "river_level_m", "river_level_threshold_m", "emergency_calls",
    "road_closures", "bridge_closures", "flood_history_count", "population_affected",
    "water_level_change_m",
]
INTEGER_FEATURES = ["emergency_calls", "road_closures", "bridge_closures",
                    "flood_history_count", "population_affected"]
RISK_CLASSES = ["Low", "Moderate", "Severe"]

# Physical hard limits. Stress modifiers may exceed the historical record but
# never these -- a 1,000 mm hourly rainfall is not a stress test, it is noise.
PHYSICAL_LIMITS = {
    "rainfall_mm": (0.0, 400.0),
    "river_level_m": (0.5, 12.0),
    "river_level_threshold_m": (3.0, 8.0),
    "emergency_calls": (0.0, 500.0),
    "road_closures": (0.0, 12.0),
    "bridge_closures": (0.0, 6.0),
    "flood_history_count": (0.0, 20.0),
    "population_affected": (0.0, 20000.0),
    "water_level_change_m": (-3.0, 3.0),
}

# Must match fusion/decision_engine.PRIORITY_LEVELS.
PRIORITY_LEVELS = ["ROUTINE", "ELEVATED", "URGENT", "CRITICAL"]
TEXT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEVERITY_TO_CLASS = {"ROUTINE": "Low", "ELEVATED": "Moderate", "URGENT": "Severe", "CRITICAL": "Severe"}
SEVERITY_TO_TEXT = {"ROUTINE": "LOW", "ELEVATED": "MEDIUM", "URGENT": "HIGH", "CRITICAL": "CRITICAL"}
ENTRIES_BY_TEXT_LEVEL = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

HAZARDS = ("Flood", "Rescue Emergency", "Road Blockage", "Medical Emergency")
ZONE_MODIFIERS = {
    "flash_flood", "river_overflow", "cyclonic_rain", "urban_waterlogging", "silent_rise",
    "receding", "bridge_collapse", "dam_release", "mass_casualty", "calm",
    "gauge_malfunction", "negation_text", "false_alarm_text", "implicit_urgency",
    "battery_backup", "generator_backup", "total_blackout",
}
GLOBAL_MODIFIERS = {"night", "power_outage", "comms_noise", "dry_season"}
TEXT_STYLES = {"dispatcher", "social", "mixed"}
IMAGE_LABELS = {"flooded", "unflooded"}
WATER_SHAPES = {"flat", "stuck", "steady_rise", "sharp_rise", "falling", "step_jump", "ood_spike"}
EXPECT_KEYS = {"status", "min_priority", "max_priority", "human_review", "conflict"}
EVIDENCE_KEYS = {"sensors", "text", "image", "water_history"}
MAX_ZONES = 8
MAX_SPEC_BYTES = 20_000

LOOKBACK_HOURS = 72
# Stage 02's LSTM flags windows beyond 4 sigma of its training series
# (3.12 +/- 1.56 m => 9.36 m). Ordinary histories stay below this so only the
# deliberate ood_spike shape trips the out-of-distribution guard.
LSTM_SAFE_MAX_M = 9.2
OOD_SPIKE_LEVEL_M = 25.0

# CVAE architecture / training.
LATENT_DIM = 6
HIDDEN = 64
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
KL_WARMUP_FRACTION = 0.3
CANDIDATES_PER_ZONE = 24

# Citizen messages where urgency is IMPLIED, never stated -- the gap Stage 03's
# team huddle calls out. No "urgent", "critical" or "emergency" tokens.
IMPLICIT_TEMPLATES = {
    "CRITICAL": [
        "Water is at chest level inside our house near the {location} in {district}. {n} of us are "
        "on the roof with an old man who cannot walk. Phone battery almost gone.",
        "Current is very strong near the {location}, {district}. Our boat turned over and {n} "
        "people are holding on to a tree. Can anyone reach us",
        "We can hear people shouting from the {location} in {district} but cannot get to them. "
        "Water still rising. {n} families live there.",
    ],
    "HIGH": [
        "Water entering houses near the {location} in {district}. {n} people moved to the upper "
        "floor, small children with us. The road outside is gone.",
        "Stuck at the {location}, {district} -- {n} of us, water all around the vehicle and the "
        "engine is dead.",
    ],
    "MEDIUM": [
        "Knee-deep water on the road near the {location} in {district}. Shops are closing and {n} "
        "people are waiting at the bus shelter.",
    ],
    "LOW": [
        "Light rain near the {location} in {district}. A few puddles on the road but traffic is "
        "moving normally.",
    ],
}
NEGATION_TEMPLATES = [
    "No flooding at the {location} in {district}. Only light rain and the roads are clear. "
    "Please ignore the forwarded message about a flood here.",
    "False alarm near the {location}, {district} -- the siren was a pump test. No flood and "
    "no one is affected.",
]
PRANK_TEMPLATES = [
    "EMERGENCY!!! Massive flood at the {location} in {district}, hundreds trapped, send all "
    "rescue teams NOW",
    "Whole {location} in {district} washed away, {n} dead, army needed immediately!!!",
]
OUTAGE_SOCIAL_TEMPLATES = [
    "Total darkness near the {location} in {district}, power has been out for hours. Water "
    "coming into the ground floor, {n} of us upstairs, mobile network going in and out.",
    "No electricity and no light at the {location}, {district}. We can hear water but cannot "
    "see how high it is. {n} people here, torch batteries finishing.",
]
OUTAGE_CLAUSE = " Power supply is down; street lights and mobile towers are offline."
BRIDGE_LEAD = "Bridge collapse reported"
HINGLISH = {"water": "paani", "help": "madad", "people": "log", "quickly": "jaldi",
            "house": "ghar", "please": "plz", "come": "aao", "fast": "jaldi"}


# ===========================================================================
# 1. Conditional VAE over sensor readings
# ===========================================================================

class SensorCVAE(nn.Module):
    """p(sensor readings | risk class), with a small latent space."""

    def __init__(self, n_features: int, n_classes: int,
                 latent_dim: int = LATENT_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.n_features, self.n_classes, self.latent_dim = n_features, n_classes, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features + n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, c), mu, logvar


def _one_hot(indices: np.ndarray, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.as_tensor(indices, dtype=torch.long), n_classes).float()


def _cvae_loss(recon, x, mu, logvar, beta):
    recon_loss = ((recon - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon_loss + beta * kl, recon_loss, kl


def train_cvae(reference: pd.DataFrame, epochs: int = 150, seed: int = SEED,
               output_path: Path = CVAE_PATH, verbose: bool = True) -> dict[str, Any]:
    """Train the CVAE and save a weights-only-loadable bundle."""
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)
    x = reference[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    y = reference["zone_risk"].map({c: i for i, c in enumerate(RISK_CLASSES)}).to_numpy()
    if np.isnan(x).any() or pd.isna(y).any():
        raise ValueError("Sensor reference contains missing values or unknown risk classes")

    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    xs = (x - mean) / std
    train_idx, val_idx = train_test_split(np.arange(len(xs)), test_size=0.1,
                                          stratify=y, random_state=seed)
    x_train, y_train = torch.tensor(xs[train_idx]), y[train_idx]
    x_val, c_val = torch.tensor(xs[val_idx]), _one_hot(y[val_idx], len(RISK_CLASSES))
    c_train = _one_hot(y_train, len(RISK_CLASSES))

    # Low is 1.8% of rows. Inverse-sqrt-frequency sampling shows the decoder the
    # rare class more often without memorising its ~160 training rows.
    counts = np.bincount(y_train, minlength=len(RISK_CLASSES))
    weights = 1.0 / np.sqrt(counts[y_train])
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(y_train),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, c_train), batch_size=BATCH_SIZE, sampler=sampler)

    model = SensorCVAE(len(NUMERIC_FEATURES), len(RISK_CLASSES))
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    warmup = max(1, int(epochs * KL_WARMUP_FRACTION))
    history, best_val, best_state = [], float("inf"), None
    started = time.time()

    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / warmup)
        model.train()
        train_total = 0.0
        for xb, cb in loader:
            recon, mu, logvar = model(xb, cb)
            loss, _, _ = _cvae_loss(recon, xb, mu, logvar, beta)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_total += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(x_val, c_val)
            val_loss, val_recon, val_kl = _cvae_loss(recon, x_val, mu, logvar, 1.0)
        history.append({"epoch": epoch, "beta": round(beta, 3),
                        "train_loss": round(train_total / len(y_train), 4),
                        "val_loss": round(float(val_loss), 4),
                        "val_recon": round(float(val_recon), 4),
                        "val_kl": round(float(val_kl), 4)})
        # Only judge "best" once the KL term is at full weight; earlier epochs
        # have artificially low loss and a latent space that cannot be sampled.
        if beta >= 1.0 and float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(f"[cvae] epoch {epoch:>4}  beta {beta:.2f}  train {history[-1]['train_loss']:.3f}"
                  f"  val {history[-1]['val_loss']:.3f} (recon {history[-1]['val_recon']:.3f},"
                  f" kl {history[-1]['val_kl']:.3f})")

    # A Gaussian-likelihood VAE generates x = decoder(z) + noise. Decoding the
    # mean alone puts all nine fields on a 6-D surface: correlations inflate
    # (rainfall-calls r reached 0.996 against a real 0.824) and samples become
    # trivially separable from real rows. The per-feature residual of the
    # validation reconstructions is that noise scale.
    model.load_state_dict(best_state or model.state_dict())
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_val, c_val)
        residual_std = (x_val - model.decode(mu, c_val)).std(dim=0)

    bundle = {
        "state_dict": best_state or model.state_dict(),
        "residual_std": [float(v) for v in residual_std],
        "config": {"n_features": len(NUMERIC_FEATURES), "n_classes": len(RISK_CLASSES),
                   "latent_dim": LATENT_DIM, "hidden": HIDDEN},
        "feature_names": list(NUMERIC_FEATURES),
        "classes": list(RISK_CLASSES),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "hist_min": {f: float(reference[f].min()) for f in NUMERIC_FEATURES},
        "hist_max": {f: float(reference[f].max()) for f in NUMERIC_FEATURES},
        "trained_rows": int(len(train_idx)),
        "epochs": int(epochs),
        "seed": int(seed),
        "best_val_loss": round(best_val, 4) if best_state else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return {"bundle_path": str(output_path), "history": history,
            "training_seconds": round(time.time() - started, 1),
            "best_val_loss": bundle["best_val_loss"]}


class SensorSampler:
    """Draw class-conditional sensor readings from a trained CVAE bundle."""

    def __init__(self, bundle_path: Path = CVAE_PATH) -> None:
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"CVAE bundle not found at {bundle_path}. Run `python Stage05_GenAI/03_genai_engineer.py`.")
        # weights_only=True: the bundle holds tensors and plain containers only.
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        config = bundle["config"]
        self.model = SensorCVAE(config["n_features"], config["n_classes"],
                                config["latent_dim"], config["hidden"])
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()
        self.features = list(bundle["feature_names"])
        self.classes = list(bundle["classes"])
        self.mean = np.asarray(bundle["mean"], dtype=np.float64)
        self.std = np.asarray(bundle["std"], dtype=np.float64)
        self.hist_min = dict(bundle["hist_min"])
        self.hist_max = dict(bundle["hist_max"])
        self.latent_dim = config["latent_dim"]
        # Bundles written before observation noise was added decode the mean only.
        self.residual_std = np.asarray(bundle.get("residual_std") or [0.0] * len(self.features))
        self.bundle_path = bundle_path

    def sample_raw(self, risk_class: str, n: int, temperature: float = 1.0,
                   seed: int = SEED, observation_noise: bool = True) -> np.ndarray:
        if risk_class not in self.classes:
            raise ValueError(f"Unknown risk class {risk_class!r}; expected one of {self.classes}")
        generator = torch.Generator().manual_seed(int(seed))
        z = torch.randn(n, self.latent_dim, generator=generator) * float(temperature)
        c = _one_hot(np.full(n, self.classes.index(risk_class)), len(self.classes))
        with torch.no_grad():
            decoded = self.model.decode(z, c).numpy().astype(np.float64)
        if observation_noise:
            noise = torch.randn(n, len(self.features), generator=generator).numpy()
            decoded = decoded + noise * self.residual_std
        return decoded * self.std + self.mean

    def postprocess(self, raw: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
        """Clip decoded values into the historical range and round count fields.

        Plain samples are clipped to the RECORD, so an unmodified sample never
        extrapolates. Only named stress modifiers may push past it, and they log
        it. The number of values that needed clipping is returned as a quality
        signal for the evaluation engineer.
        """
        frame = pd.DataFrame(raw, columns=self.features)
        clipped = 0
        physical_violations = 0
        for feature in self.features:
            lo, hi = self.hist_min[feature], self.hist_max[feature]
            values = frame[feature]
            clipped += int(((values < lo) | (values > hi)).sum())
            p_lo, p_hi = PHYSICAL_LIMITS[feature]
            physical_violations += int(((values < p_lo) | (values > p_hi)).sum())
            frame[feature] = values.clip(lo, hi)
        for feature in INTEGER_FEATURES:
            frame[feature] = np.rint(frame[feature]).astype(int)
        return frame, {"values_clipped_to_record": clipped,
                       "raw_physical_violations": physical_violations,
                       "values_total": int(raw.size)}

    def sample(self, risk_class: str, n: int, temperature: float = 1.0,
               seed: int = SEED) -> tuple[pd.DataFrame, dict[str, int]]:
        return self.postprocess(self.sample_raw(risk_class, n, temperature, seed))


# ===========================================================================
# 2. Text synthesis helpers
# ===========================================================================

def _weighted_choice(options: dict[str, int], rng: np.random.Generator) -> str:
    keys = list(options)
    weights = np.asarray([options[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=weights / weights.sum()))]


def degrade_text(text: str, rng: np.random.Generator, truncate_probability: float = 0.35) -> str:
    """Make a message look like a real panicked / failing-network message.

    Hinglish substitutions, adjacent-letter swaps, all-caps shouting and a
    truncated tail (a dropped SMS). Deterministic for a given generator state.
    """
    words = []
    for word in text.split():
        core = word.lower().strip(".,!?-")
        if core in HINGLISH and rng.random() < 0.6:
            word = word.lower().replace(core, HINGLISH[core])
        elif len(word) > 3 and word.isalpha() and rng.random() < 0.10:
            j = int(rng.integers(1, len(word) - 1))
            word = word[:j] + word[j + 1] + word[j] + word[j + 2:]
        words.append(word)
    degraded = " ".join(words)
    if rng.random() < 0.3:
        degraded = degraded.upper()
    if rng.random() < truncate_probability:
        cut = int(len(degraded) * rng.uniform(0.55, 0.8))
        degraded = degraded[:cut].rstrip()
    return degraded


# ===========================================================================
# 3. Scenario generator
# ===========================================================================

def validate_spec(spec: Any) -> None:
    """Reject malformed scenario specs with a message a caller can act on."""
    if not isinstance(spec, dict):
        raise ValueError("Scenario spec must be a JSON object")
    if len(json.dumps(spec, default=str)) > MAX_SPEC_BYTES:
        raise ValueError(f"Scenario spec exceeds {MAX_SPEC_BYTES} bytes")
    zones = spec.get("zones")
    if not isinstance(zones, list) or not 1 <= len(zones) <= MAX_ZONES:
        raise ValueError(f"'zones' must be a list of 1-{MAX_ZONES} zone objects")
    try:
        datetime.strptime(str(spec.get("start_time")), "%d-%m-%Y %H:%M")
    except ValueError as exc:
        raise ValueError("'start_time' must be formatted DD-MM-YYYY HH:MM") from exc
    unknown_global = set(spec.get("global_modifiers") or []) - GLOBAL_MODIFIERS
    if unknown_global:
        raise ValueError(f"Unknown global modifiers: {sorted(unknown_global)}")
    for index, zone in enumerate(zones, start=1):
        where = f"zone {index}"
        if not isinstance(zone, dict):
            raise ValueError(f"{where} must be an object")
        if zone.get("true_severity") not in PRIORITY_LEVELS:
            raise ValueError(f"{where}: true_severity must be one of {PRIORITY_LEVELS}")
        hazards = zone.get("hazards")
        if not isinstance(hazards, list) or not hazards or set(hazards) - set(HAZARDS):
            raise ValueError(f"{where}: hazards must be a non-empty subset of {list(HAZARDS)}")
        unknown = set(zone.get("modifiers") or []) - ZONE_MODIFIERS
        if unknown:
            raise ValueError(f"{where}: unknown modifiers {sorted(unknown)}")
        evidence = zone.get("evidence") or {}
        if not isinstance(evidence, dict) or set(evidence) - EVIDENCE_KEYS:
            raise ValueError(f"{where}: evidence keys must be a subset of {sorted(EVIDENCE_KEYS)}")
        if evidence.get("text") not in TEXT_STYLES | {None}:
            raise ValueError(f"{where}: evidence.text must be one of {sorted(TEXT_STYLES)} or null")
        if evidence.get("image") not in IMAGE_LABELS | {None}:
            raise ValueError(f"{where}: evidence.image must be one of {sorted(IMAGE_LABELS)} or null")
        if evidence.get("water_history") not in WATER_SHAPES | {None}:
            raise ValueError(f"{where}: evidence.water_history must be one of {sorted(WATER_SHAPES)} or null")
        if zone.get("sensor_class") not in set(RISK_CLASSES) | {None}:
            raise ValueError(f"{where}: sensor_class must be one of {RISK_CLASSES}")
        if zone.get("text_severity") not in set(TEXT_LEVELS) | {None}:
            raise ValueError(f"{where}: text_severity must be one of {TEXT_LEVELS}")
        expect = zone.get("expect") or {}
        if not isinstance(expect, dict) or set(expect) - EXPECT_KEYS:
            raise ValueError(f"{where}: expect keys must be a subset of {sorted(EXPECT_KEYS)}")
        for key in ("min_priority", "max_priority"):
            if expect.get(key) not in set(PRIORITY_LEVELS) | {None}:
                raise ValueError(f"{where}: expect.{key} must be one of {PRIORITY_LEVELS}")


def build_expectations(true_severity: str, available: dict[str, bool],
                       overrides: dict | None = None) -> dict[str, Any]:
    """Default pass criteria for one zone, derived from its ground truth.

    Tolerance is one level below the truth (URGENT is an acceptable call for a
    CRITICAL zone because fusion still forces human review there). A zone left
    with NO evidence must be refused as insufficient_evidence -- never scored.
    """
    if not any(available.values()):
        expected = {"status": "insufficient_evidence", "min_priority": None,
                    "max_priority": None, "human_review": True, "conflict": None}
    else:
        index = PRIORITY_LEVELS.index(true_severity)
        expected = {
            "status": "ok",
            "min_priority": PRIORITY_LEVELS[index - 1] if index >= 2 else None,
            "max_priority": None,
            "human_review": True if index >= 2 else None,
            "conflict": None,
        }
    expected.update(overrides or {})
    return expected


class ScenarioGenerator:
    """Prompt spec -> multi-zone compound disaster scenario."""

    def __init__(self, bundle_path: Path = CVAE_PATH, processed_dir: Path = PROCESSED_DIR) -> None:
        self.sampler = SensorSampler(bundle_path)
        processed_dir = Path(processed_dir)
        self.distributions = json.loads((processed_dir / DISTRIBUTIONS_JSON.name).read_text(encoding="utf-8"))
        self.bank = json.loads((processed_dir / PHRASE_BANK_JSON.name).read_text(encoding="utf-8"))
        self.water_ref = json.loads((processed_dir / WATER_REFERENCE_JSON.name).read_text(encoding="utf-8"))
        manifest = processed_dir / IMAGERY_MANIFEST_CSV.name
        self.imagery = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame(
            columns=["image_path", "label"])
        self.state_districts: dict[str, list[str]] = self.distributions["state_districts"]
        q = self.distributions["overall"]
        self.q = {feature: q[feature] for feature in NUMERIC_FEATURES}

    # ---- public API --------------------------------------------------------

    def generate(self, spec: dict, seed: int = SEED) -> dict[str, Any]:
        validate_spec(spec)
        spec = copy.deepcopy(spec)
        if spec.get("state") and spec["state"] not in self.state_districts:
            raise ValueError(f"Unknown state {spec['state']!r}; known: {sorted(self.state_districts)}")
        start = datetime.strptime(spec["start_time"], "%d-%m-%Y %H:%M")
        global_mods = set(spec.get("global_modifiers") or [])
        rng = np.random.default_rng([seed, 0])
        default_state = spec.get("state") or str(rng.choice(sorted(self.state_districts)))
        shared_district = (str(rng.choice(self.state_districts[default_state]))
                           if spec.get("same_district") else None)

        used: dict[str, set[str]] = {}
        zones = []
        for index, zone_spec in enumerate(spec["zones"]):
            zrng = np.random.default_rng([seed, index + 1])
            state = zone_spec.get("state") or default_state
            if state not in self.state_districts:
                raise ValueError(f"Unknown state {state!r} in zone {index + 1}")
            if shared_district and state == default_state:
                district = shared_district
            else:
                pool = [d for d in self.state_districts[state] if d not in used.get(state, set())]
                district = str(zrng.choice(pool or self.state_districts[state]))
            used.setdefault(state, set()).add(district)
            zones.append(self._build_zone(zone_spec, index, state, district, start,
                                          global_mods, zrng, seed))

        total_headcount = int(sum(z["headcount_reported"] for z in zones))
        beds = int((spec.get("resources") or {}).get("shelter_beds", 0))
        return {
            "scenario_id": spec.get("id", "CUSTOM"),
            "name": spec.get("name", "Custom scenario"),
            "archetype": spec.get("archetype", "custom"),
            "prompt": spec.get("prompt", ""),
            "blind_spots": spec.get("blind_spots", []),
            "start_time": spec["start_time"],
            "global_modifiers": sorted(global_mods),
            "resources": spec.get("resources", {}),
            "demand": {"total_headcount_reported": total_headcount, "shelter_beds": beds,
                       "shelter_gap": max(0, total_headcount - beds)},
            "generator": {"model": "SensorCVAE", "bundle": "data/models/sensor_cvae.pt",
                          "seed": int(seed)},
            "zones": zones,
        }

    def generate_suite(self, prompts: list[dict], base_seed: int = SEED) -> list[dict]:
        return [self.generate(spec, seed=base_seed + index) for index, spec in enumerate(prompts)]

    # ---- zone assembly -----------------------------------------------------

    def _build_zone(self, zone_spec, index, state, district, start, global_mods, zrng, seed):
        true_severity = zone_spec["true_severity"]
        mods = set(zone_spec.get("modifiers") or [])
        hazards = list(zone_spec["hazards"])
        evidence = {"sensors": True, "text": "dispatcher", "image": None, "water_history": None}
        evidence.update(zone_spec.get("evidence") or {})
        zone_id = f"ZONE-{index + 1}"
        lost: list[str] = []

        # Evidence lost to infrastructure failure. Cameras need mains power;
        # gauges survive on batteries; a generator keeps everything running.
        if "total_blackout" in mods:
            for key in ("sensors", "text", "image", "water_history"):
                if evidence[key]:
                    lost.append(key)
                    evidence[key] = None if key != "sensors" else False
        elif "power_outage" in global_mods and "generator_backup" not in mods:
            if evidence["image"]:
                lost.append("image")
                evidence["image"] = None
            if "battery_backup" not in mods:
                for key in ("sensors", "water_history"):
                    if evidence[key]:
                        lost.append(key)
                        evidence[key] = None if key != "sensors" else False

        location = zone_spec.get("location") or _weighted_choice(
            self.bank["hazards"][hazards[0]]["locations"], zrng)
        sensor_time = start + timedelta(minutes=int(zrng.integers(0, 20)))

        # Sensors are always sampled so the gauge history has a consistent end
        # level; they are only EXPOSED when the zone's telemetry survives.
        reading, adjustments, beyond, sensor_class = self._sensor_reading(
            zone_spec, true_severity, mods, global_mods, zrng)
        sensors = None
        if evidence["sensors"]:
            sensors = {"timestamp": sensor_time.strftime("%d-%m-%Y %H:%M"),
                       "state": state, "district": district}
            sensors.update({f: (int(reading[f]) if f in INTEGER_FEATURES else round(float(reading[f]), 3))
                            for f in NUMERIC_FEATURES})

        water_levels = None
        if evidence["water_history"]:
            water_levels = self._water_history(float(reading["river_level_m"]),
                                               evidence["water_history"], zrng)

        entries = []
        if evidence["text"]:
            entries = self._text_entries(zone_spec, true_severity, mods, global_mods, hazards,
                                         evidence["text"], location, state, district, start,
                                         index, seed, zrng)
        text = None
        if entries:
            top = max(range(len(entries)),
                      key=lambda k: (TEXT_LEVELS.index(entries[k]["severity"]), k))
            text = entries[top]["text"]
        incident_log = self._incident_log(entries, state, district, zone_id) if entries else None

        image_path = None
        if evidence["image"]:
            pool = self.imagery[self.imagery["label"] == evidence["image"]]
            if pool.empty:
                lost.append("image (imagery bank empty)")
            else:
                image_path = str(pool.iloc[int(zrng.integers(0, len(pool)))]["image_path"])

        available = {"sensors": sensors is not None, "text": text is not None,
                     "image": image_path is not None, "water_history": water_levels is not None}
        return {
            "zone_id": zone_id,
            "label": zone_spec.get("label", zone_id),
            "state": state,
            "district": district,
            "location": location,
            "true_severity": true_severity,
            "hazards": hazards,
            "modifiers": sorted(mods),
            "inputs": {"sensors": sensors, "text": text, "image_path": image_path,
                       "water_levels": water_levels, "incident_log": incident_log},
            "text_entries": entries,
            "headcount_reported": int(sum(e["headcount"] for e in entries)),
            "provenance": {
                "sensor_class": sensor_class,
                "sensor_adjustments": adjustments,
                "beyond_record_fields": beyond,
                "evidence_lost": lost,
                "evidence_available": available,
                "image_label": evidence["image"],
                "water_history_shape": evidence["water_history"],
            },
            "expected": build_expectations(true_severity, available, zone_spec.get("expect")),
        }

    def _sensor_reading(self, zone_spec, true_severity, mods, global_mods, zrng):
        sensor_class = zone_spec.get("sensor_class") or SEVERITY_TO_CLASS[true_severity]
        candidates, _ = self.sampler.sample(sensor_class, CANDIDATES_PER_ZONE,
                                            seed=int(zrng.integers(0, 2**31 - 1)))
        margin = (candidates["river_level_m"] - candidates["river_level_threshold_m"]).to_numpy()
        # Within the Severe class, CRITICAL zones are drawn from the upper tail
        # of river margin and URGENT from the middle, so the two stay distinct.
        if sensor_class == "Severe" and "sensor_class" not in zone_spec and true_severity == "CRITICAL":
            top = np.argsort(margin)[-max(1, CANDIDATES_PER_ZONE // 4):]
            pick = int(zrng.choice(top))
        elif sensor_class == "Severe" and "sensor_class" not in zone_spec:
            pick = int(np.argsort(margin)[len(margin) // 2])
        else:
            pick = int(zrng.integers(0, len(candidates)))
        row = candidates.iloc[pick].astype(float).to_dict()
        adjustments: list[str] = []
        q = self.q

        def put(field: str, value: float, why: str) -> None:
            old = row[field]
            row[field] = float(value)
            adjustments.append(f"{field}: {old:.2f} -> {float(value):.2f} ({why})")

        def u(lo: float, hi: float) -> float:
            return float(zrng.uniform(lo, hi))

        thr = row["river_level_threshold_m"]
        night = "night" in global_mods
        # Fixed order so a spec's result never depends on set iteration order.
        if "calm" in mods:
            put("rainfall_mm", row["rainfall_mm"] * 0.3, "calm")
            put("river_level_m", max(q["river_level_m"]["min"], thr - u(3.0, 4.5)), "calm")
            put("emergency_calls", u(q["emergency_calls"]["min"], q["emergency_calls"]["q25"]), "calm")
            put("water_level_change_m", u(-0.1, 0.05), "calm")
            put("road_closures", 0, "calm")
            put("bridge_closures", 0, "calm")
        if "flash_flood" in mods:
            put("rainfall_mm", row["rainfall_mm"] * u(1.6, 2.2), "flash flood")
            put("water_level_change_m", u(q["water_level_change_m"]["q99"],
                                          1.4 * q["water_level_change_m"]["max"]), "flash flood")
            put("river_level_m", min(thr + u(1.5, 3.0), LSTM_SAFE_MAX_M), "flash flood")
            if not night:
                put("emergency_calls", row["emergency_calls"] * 1.3, "flash flood call surge")
        if "river_overflow" in mods:
            put("river_level_m", min(thr + u(0.8, 2.5), LSTM_SAFE_MAX_M), "river overflow")
            put("water_level_change_m", u(q["water_level_change_m"]["q75"],
                                          q["water_level_change_m"]["q95"]), "river overflow")
        if "cyclonic_rain" in mods:
            put("rainfall_mm", row["rainfall_mm"] * u(1.4, 1.9), "cyclonic rain")
            put("road_closures", row["road_closures"] + 2, "cyclonic rain")
            put("population_affected", row["population_affected"] * 1.3, "cyclonic rain")
        if "urban_waterlogging" in mods:
            put("rainfall_mm", max(row["rainfall_mm"], u(q["rainfall_mm"]["q95"],
                                                         1.3 * q["rainfall_mm"]["max"])), "cloudburst")
            put("river_level_m", max(q["river_level_m"]["min"], thr - u(1.5, 3.0)), "river below danger")
            put("road_closures", int(zrng.integers(3, 6)), "waterlogged streets")
            put("emergency_calls", row["emergency_calls"] * 1.4, "waterlogging calls")
        if "silent_rise" in mods:
            put("river_level_m", min(thr + u(2.0, 3.5), LSTM_SAFE_MAX_M), "silent rise")
            put("emergency_calls", u(q["emergency_calls"]["min"], q["emergency_calls"]["q05"]),
                "residents asleep")
            put("water_level_change_m", u(q["water_level_change_m"]["q90"],
                                          q["water_level_change_m"]["max"]), "silent rise")
        if "receding" in mods:
            put("water_level_change_m", -u(0.3, 0.8), "receding river")
        if "bridge_collapse" in mods:
            put("bridge_closures", 3, "bridge collapse")
            put("road_closures", 5, "access cut")
        if "dam_release" in mods:
            put("water_level_change_m", u(1.6, 2.4), "dam spillway release")
            put("river_level_m", min(thr + u(2.0, 3.0), LSTM_SAFE_MAX_M), "dam spillway release")
        if "mass_casualty" in mods:
            put("population_affected", row["population_affected"] * 2.2, "mass casualty")
        if night and not {"flash_flood", "silent_rise"} & mods:
            put("emergency_calls", row["emergency_calls"] * 0.7, "night: fewer callers awake")
        if "power_outage" in global_mods and "generator_backup" not in mods:
            put("emergency_calls", row["emergency_calls"] * 0.5, "phones dying in outage")

        for feature, (lo, hi) in PHYSICAL_LIMITS.items():
            row[feature] = float(np.clip(row[feature], lo, hi))
        for feature in INTEGER_FEATURES:
            row[feature] = float(np.rint(row[feature]))
        beyond = [f for f in NUMERIC_FEATURES
                  if row[f] > self.sampler.hist_max[f] + 1e-9 or row[f] < self.sampler.hist_min[f] - 1e-9]
        return row, adjustments, beyond, sensor_class

    def _water_history(self, end_level: float, shape: str, zrng: np.random.Generator) -> list[float]:
        """72 hourly levels ending at the zone's current river level.

        Rise magnitudes are drawn from real CWC gauge statistics (72 h range,
        90th-99th percentile) so ordinary shapes stay physically plausible.
        """
        n = LOOKBACK_HOURS
        range72 = self.water_ref.get("range_72h_m", {})
        change_q90 = self.water_ref.get("abs_change_1h_m", {}).get("q90", 0.024)
        noise_sd = float(np.clip(change_q90 / 2, 0.005, 0.03))
        end = float(min(end_level, LSTM_SAFE_MAX_M))
        floor = 0.8

        if shape in {"flat", "stuck"}:
            base = np.full(n, end)
        elif shape == "steady_rise":
            rise = zrng.uniform(range72.get("q90", 0.9), range72.get("q99", 2.7))
            base = np.linspace(max(floor, end - rise), end, n)
        elif shape == "sharp_rise":
            start = max(floor, end - zrng.uniform(1.5, 2.8))
            ramp = 10
            base = np.full(n, start)
            base[-ramp:] = start + (end - start) * (np.arange(1, ramp + 1) / ramp) ** 1.5
        elif shape == "falling":
            peak = min(LSTM_SAFE_MAX_M, end + zrng.uniform(1.0, 2.0))
            base = np.concatenate([np.full(12, peak), np.linspace(peak, end, n - 12)])
        elif shape == "step_jump":
            start = max(floor, end - zrng.uniform(1.8, 3.0))
            base = np.full(n, start)
            base[-3:] = start + (end - start) * np.array([0.5, 0.85, 1.0])
        elif shape == "ood_spike":
            rise = zrng.uniform(range72.get("q50", 0.1), range72.get("q90", 0.9))
            base = np.linspace(max(floor, end - rise), end, n)
        else:
            raise ValueError(f"Unknown water-history shape {shape!r}")

        levels = base if shape == "stuck" else base + zrng.normal(0.0, noise_sd, n)
        levels = np.clip(levels, 0.5, LSTM_SAFE_MAX_M)
        levels[-1] = end
        if shape == "ood_spike":
            # A failed logger: four impossible readings at the end of the window.
            levels[-4:] = OOD_SPIKE_LEVEL_M
        return [round(float(v), 3) for v in levels]

    def _text_entries(self, zone_spec, true_severity, mods, global_mods, hazards, style,
                      location, state, district, start, zone_index, seed, zrng):
        target = zone_spec.get("text_severity") or SEVERITY_TO_TEXT[true_severity]
        target_index = TEXT_LEVELS.index(target)
        count = ENTRIES_BY_TEXT_LEVEL[target] + int(zrng.integers(0, 2))
        outage = "power_outage" in global_mods and "generator_backup" not in mods
        if outage:
            count = max(1, count // 2)  # phones die, fewer messages get through

        entries = []
        clock = start
        for k in range(count):
            clock += timedelta(minutes=int(zrng.integers(4, 23)))
            # Messages escalate toward the zone's target severity.
            severity = TEXT_LEVELS[max(0, target_index - (count - 1 - k) // 2)]
            hazard = hazards[k % len(hazards)]
            entry_style = style if style != "mixed" else ("dispatcher" if k % 2 == 0 else "social")
            headcount = self._headcount(severity, mods, zrng)

            if "false_alarm_text" in mods:
                severity, entry_style = "CRITICAL", "prank"
                text = str(zrng.choice(PRANK_TEMPLATES)).format(location=location, district=district,
                                                                n=headcount)
            elif "negation_text" in mods and k == count - 1:
                severity, entry_style = "LOW", "negation"
                headcount = 0
                text = str(zrng.choice(NEGATION_TEMPLATES)).format(location=location, district=district)
            elif entry_style == "social":
                text = self._social_text(hazard, severity, mods, outage, location, district,
                                         headcount, zrng)
            else:
                lead = None
                if "bridge_collapse" in mods and hazard == "Road Blockage":
                    lead = BRIDGE_LEAD
                elif len(hazards) >= 3 and k == 0:
                    # One message naming two hazards ("Flooding reported with road
                    # blockage") -- the compound report BS16 says never occurs.
                    second = _weighted_choice(self.bank["hazards"][hazards[1]]["leads"], zrng)
                    second = re.sub(r"\s+(?:reported|required)$", "", second.lower())
                    lead = (f"{_weighted_choice(self.bank['hazards'][hazard]['leads'], zrng)} "
                            f"with {second}")
                text = self._dispatcher_text(hazard, severity, location, district, state,
                                             headcount, zrng, lead)
                if outage:
                    text += OUTAGE_CLAUSE

            if "comms_noise" in global_mods:
                text = degrade_text(text, zrng)
            elif outage and entry_style == "social":
                text = degrade_text(text, zrng, truncate_probability=0.5)

            entries.append({
                "timestamp": clock.strftime("%d-%m-%Y %H:%M"),
                "incident_id": f"ERSS-{900000 + (seed % 900) * 100 + zone_index * 10 + k:06d}",
                "severity": severity,
                "hazard": hazard,
                "style": entry_style,
                "headcount": int(headcount),
                "text": text,
            })
        return entries

    def _headcount(self, severity: str, mods: set[str], zrng: np.random.Generator) -> int:
        if "mass_casualty" in mods and severity in {"HIGH", "CRITICAL"}:
            return int(zrng.integers(200, 451))
        stats = self.bank["headcount_by_severity"].get(severity, {"q25": 1, "q75": 5})
        lo, hi = int(max(1, stats["q25"])), int(max(2, stats["q75"]))
        return int(zrng.integers(lo, hi + 1))

    def _dispatcher_text(self, hazard, severity, location, district, state, headcount, zrng, lead=None):
        slots = self.bank["hazards"][hazard]
        lead = lead or _weighted_choice(slots["leads"], zrng)
        status = _weighted_choice(self.bank["statuses"], zrng)
        prep = _weighted_choice(self.bank["preps"], zrng)
        noun = _weighted_choice(self.bank["people_nouns"], zrng)
        injury = _weighted_choice(self.bank["injury_by_severity"][severity], zrng)
        resources = list(slots["resources"])
        weights = np.asarray([slots["resources"][r] for r in resources], dtype=float)
        picked = zrng.choice(len(resources), size=int(zrng.integers(1, 3)), replace=False,
                             p=weights / weights.sum())
        request = _weighted_choice(slots["requests"], zrng).replace(
            "{resources}", " and ".join(resources[int(i)] for i in picked))
        return (f"{lead} ({status}) {prep} {location} in {district}, {state}. "
                f"{headcount} {noun} affected, {injury}. {request}")

    def _social_text(self, hazard, severity, mods, outage, location, district, headcount, zrng):
        if outage:
            template = str(zrng.choice(OUTAGE_SOCIAL_TEMPLATES))
        elif "implicit_urgency" in mods or hazard != "Flood":
            template = str(zrng.choice(IMPLICIT_TEMPLATES[severity]))
        else:
            corpus = self.bank.get("social_templates", {}).get("Flood", [])
            if corpus and severity in {"MEDIUM", "HIGH"} and zrng.random() < 0.5:
                return str(zrng.choice(corpus)).replace("{district}", district)
            template = str(zrng.choice(IMPLICIT_TEMPLATES[severity]))
        return template.format(location=location, district=district, n=headcount)

    @staticmethod
    def _incident_log(entries, state, district, zone_id) -> str:
        header = f"INCIDENT LOG | {state} / {district} / {zone_id} | {len(entries)} entries"
        lines = [f"[{e['timestamp']}] {e['incident_id']} | {e['text']}" for e in entries]
        return "\n".join([header, *lines])


def flatten_zones(scenarios: list[dict]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for zone in scenario["zones"]:
            sensors = zone["inputs"]["sensors"] or {}
            rows.append({
                "scenario_id": scenario["scenario_id"], "scenario": scenario["name"],
                "zone_id": zone["zone_id"], "label": zone["label"], "state": zone["state"],
                "district": zone["district"], "true_severity": zone["true_severity"],
                "hazards": "; ".join(zone["hazards"]), "modifiers": "; ".join(zone["modifiers"]),
                "has_sensors": zone["provenance"]["evidence_available"]["sensors"],
                "has_text": zone["provenance"]["evidence_available"]["text"],
                "has_image": zone["provenance"]["evidence_available"]["image"],
                "has_water_history": zone["provenance"]["evidence_available"]["water_history"],
                "text_entries": len(zone["text_entries"]),
                "headcount_reported": zone["headcount_reported"],
                "beyond_record_fields": "; ".join(zone["provenance"]["beyond_record_fields"]),
                "evidence_lost": "; ".join(zone["provenance"]["evidence_lost"]),
                **{f: sensors.get(f) for f in NUMERIC_FEATURES},
                "expected_status": zone["expected"]["status"],
                "expected_min_priority": zone["expected"]["min_priority"],
                "expected_max_priority": zone["expected"]["max_priority"],
            })
    return pd.DataFrame(rows)


def load_prompts(path: Path = PROMPTS_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python Stage05_GenAI/02_eda_engineer.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 GenAI engineer: CVAE + scenario generator")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--skip-train", action="store_true", help="Reuse the saved CVAE bundle")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    reference = pd.read_csv(SENSOR_REFERENCE_CSV)

    if args.skip_train and CVAE_PATH.exists():
        print(f"[genai] reusing {CVAE_PATH.relative_to(BASE_DIR)}")
    else:
        result = train_cvae(reference, epochs=args.epochs, seed=args.seed)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TRAINING_MANIFEST_JSON.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "SensorCVAE",
            "architecture": {"latent_dim": LATENT_DIM, "hidden": HIDDEN, "features": NUMERIC_FEATURES,
                             "classes": RISK_CLASSES},
            "training": {"epochs": args.epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                         "kl_warmup_fraction": KL_WARMUP_FRACTION, "seed": args.seed,
                         "sampler": "inverse-sqrt class frequency",
                         "seconds": result["training_seconds"],
                         "best_val_loss": result["best_val_loss"]},
            "history": result["history"],
        }, indent=2), encoding="utf-8")
        print(f"[genai] CVAE trained in {result['training_seconds']} s, best val loss "
              f"{result['best_val_loss']} -> {CVAE_PATH.relative_to(BASE_DIR)}")

    generator = ScenarioGenerator()

    # Realism batch: same class counts as the real record, for the evaluation engineer.
    counts = reference["zone_risk"].value_counts()
    frames = []
    for offset, risk_class in enumerate(RISK_CLASSES):
        frame, stats = generator.sampler.sample(risk_class, int(counts.get(risk_class, 0)),
                                                seed=args.seed + 1000 + offset)
        frame["zone_risk"] = risk_class
        frames.append(frame)
        print(f"[genai] realism batch {risk_class:<8} n={len(frame):>5}  clipped-to-record "
              f"{stats['values_clipped_to_record'] / max(1, stats['values_total']):.2%}  "
              f"raw physical violations {stats['raw_physical_violations']}")
    pd.concat(frames, ignore_index=True).to_csv(SYNTHETIC_SAMPLES_CSV, index=False)

    library = load_prompts()
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    suite = generator.generate_suite(library["prompts"], base_seed=args.seed)
    wildcard = generator.generate(library["wildcard"], seed=args.seed + 100)
    SUITE_JSON.write_text(json.dumps({"seed": args.seed, "scenarios": suite}, indent=2), encoding="utf-8")
    WILDCARD_JSON.write_text(json.dumps(wildcard, indent=2), encoding="utf-8")
    flatten_zones(suite + [wildcard]).to_csv(ZONES_CSV, index=False)

    zones = sum(len(s["zones"]) for s in suite)
    extrapolated = sum(1 for s in suite for z in s["zones"] if z["provenance"]["beyond_record_fields"])
    print(f"[genai] stress suite: {len(suite)} scenarios, {zones} zones "
          f"({extrapolated} with fields beyond the historical record) -> {SUITE_JSON.relative_to(BASE_DIR)}")
    print(f"[genai] wildcard '{wildcard['name']}': {len(wildcard['zones'])} zones -> "
          f"{WILDCARD_JSON.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
