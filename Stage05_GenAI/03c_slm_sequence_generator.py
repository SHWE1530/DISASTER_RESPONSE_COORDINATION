"""Stage 05 GenAI -- sequence generation with a domain SLM trained from scratch.

This is the LLM/sequence family of Day 5, realised with a small language model
that is trained here, on this project's own disaster corpus, rather than
downloaded. It exists because the Qwen2.5-3B checkpoint in Stage04_SLM is
corrupt (33 of 36 layers damaged), and because the corpus turns out to be well
suited to a small model: 60,000 dispatcher messages over a 308-word vocabulary,
none longer than 34 words.

WHAT MAKES THIS THE SEQUENCE FAMILY, not the statistical one. Each training
example is ONE token sequence carrying the conditions and the report together:

    <bos> SEV_CRITICAL HAZ_Flood ST_Assam DI_Nagaon
          RAIN_b37 RIVER_b52 THR_b28 CALLS_b44 ROAD_b03 ...
          <sep> flooding reported near the river bank in nagaon , assam . ...
    <eos>

The model is a decoder-only transformer trained on next-token prediction over
that sequence. At generation time it is conditioned on the scenario spec and
then samples the REST -- the nine sensor values and the dispatcher narrative --
in a single autoregressive pass. The numbers and the words are drawn from one
joint distribution, which is the property the CVAE-plus-phrase-bank path cannot
have: there, the sensors and the text are sampled independently and only
*look* related because both were conditioned on the same severity label.

WHAT IT DOES AND DOES NOT DO. Five of the eight sequence techniques are real
here: autoregressive decoding, prompt engineering (the spec is the conditioning
prefix), a fine-tuned domain model (it is trained on nothing else), constrained
decoding (logits are masked to the grammar, so a malformed scenario cannot be
emitted at all), and temperature sampling. Few-shot in-context learning,
retrieval-augmented grounding and chain-of-thought need a large pretrained
model and are NOT claimed -- they live in 03b_llm_scenario_generator.py, which
implements all eight against Qwen and runs as soon as a sound checkpoint exists.

Run:
    python Stage05_GenAI/03c_slm_sequence_generator.py --train      # ~5 min GPU
    python Stage05_GenAI/03c_slm_sequence_generator.py              # generate suite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
SCENARIO_DIR = OUTPUT_DIR / "scenarios"

DISPATCHER_CORPUS = (REPO_ROOT / "Stage03_NLP" / "data" / "processed"
                     / "Dispatcher_Log_Master_60000_Processed.csv")
SENSOR_REFERENCE_CSV = PROCESSED_DIR / "sensor_reference.csv"
SLM_PATH = MODEL_DIR / "scenario_slm.pt"

SUITE_JSON = SCENARIO_DIR / "slm_scenario_suite.json"
WILDCARD_JSON = SCENARIO_DIR / "slm_wildcard_scenario.json"
ZONES_CSV = SCENARIO_DIR / "slm_scenario_zones.csv"
MANIFEST_JSON = OUTPUT_DIR / "slm_generation_manifest.json"

SEED = 42

# Model. Small on purpose: a 308-word corpus does not need capacity, it needs
# enough context to carry the spec through to the narrative.
N_LAYER, N_HEAD, D_MODEL, BLOCK = 4, 4, 256, 64
DROPOUT = 0.1
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
N_BINS = 64          # quantile bins per sensor field
MAX_TEXT_WORDS = 36

TECHNIQUES = [
    "autoregressive_slm",
    "prompt_engineering",
    "fine_tuned_domain_model",
    "constrained_decoding",
    "sampling_temperature",
]
TECHNIQUES_NOT_CLAIMED = ["few_shot", "retrieval_augmented", "chain_of_thought"]

# Dispatcher severity -> Stage 01 risk class, so a message can be paired with a
# sensor row from the same district at a compatible condition.
SEVERITY_TO_RISK = {"LOW": "Low", "MEDIUM": "Moderate", "HIGH": "Severe", "CRITICAL": "Severe"}

# CRITICAL and HIGH both map to the Severe class, so the class alone cannot
# teach the model to tell them apart. Within Severe, rank candidate rows by
# river margin (level minus danger threshold) and draw each severity from its
# own band -- the same rule 03_genai_engineer.py uses when it picks a CVAE
# sample for a zone. Without it the model sees CRITICAL paired with rivers
# metres below their threshold, and learns to generate exactly that.
MARGIN_BAND = {"CRITICAL": (0.75, 1.00), "HIGH": (0.35, 0.80),
               "MEDIUM": (0.0, 1.0), "LOW": (0.0, 1.0)}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENAI = _load_module("stage05_genai_engineer", BASE_DIR / "03_genai_engineer.py")
FIELDS = GENAI.NUMERIC_FEATURES


# ===========================================================================
# 1. Vocabulary and serialization
# ===========================================================================

class ScenarioVocab:
    """Word-level vocabulary over specs, binned sensor values and report text.

    Sensor values become BIN tokens rather than digits. Two reasons: it keeps
    the sequence short enough that the narrative stays inside the context
    window, and it makes constrained decoding exact -- at a sensor position the
    only legal tokens are that field's bins, so the model cannot emit a
    malformed scenario even in principle.
    """

    PAD, BOS, SEP, EOS, UNK = "<pad>", "<bos>", "<sep>", "<eos>", "<unk>"

    def __init__(self, sensors: pd.DataFrame, texts: pd.Series,
                 states: list[str], districts: list[str], hazards: list[str],
                 severities: list[str]) -> None:
        self.specials = [self.PAD, self.BOS, self.SEP, self.EOS, self.UNK]
        self.severity_tokens = [f"SEV_{s}" for s in severities]
        self.hazard_tokens = [f"HAZ_{h.replace(' ', '-')}" for h in hazards]
        self.state_tokens = [f"ST_{s.replace(' ', '-')}" for s in states]
        self.district_tokens = [f"DI_{d.replace(' ', '-')}" for d in districts]

        # Quantile bin edges per field, from the real record.
        self.edges: dict[str, np.ndarray] = {}
        self.field_tokens: dict[str, list[str]] = {}
        for field in FIELDS:
            qs = np.linspace(0, 1, N_BINS + 1)
            edges = np.unique(np.quantile(sensors[field].to_numpy(), qs))
            if len(edges) < 2:
                edges = np.array([float(sensors[field].min()), float(sensors[field].max()) + 1e-6])
            self.edges[field] = edges
            self.field_tokens[field] = [f"{field}_b{i:02d}" for i in range(len(edges) - 1)]

        words = sorted({w for t in texts for w in str(t).lower().split()})
        self.word_tokens = words

        ordered = (self.specials + self.severity_tokens + self.hazard_tokens
                   + self.state_tokens + self.district_tokens
                   + [t for f in FIELDS for t in self.field_tokens[f]]
                   + self.word_tokens)
        self.itos = ordered
        self.stoi = {tok: i for i, tok in enumerate(ordered)}
        self.word_ids = torch.tensor([self.stoi[w] for w in self.word_tokens], dtype=torch.long)

    # ---- helpers -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.itos)

    def id(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.UNK])

    def bin_of(self, field: str, value: float) -> int:
        edges = self.edges[field]
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        return int(np.clip(idx, 0, len(edges) - 2))

    def value_of(self, field: str, bin_index: int, rng: np.random.Generator) -> float:
        """Decode a bin back to a value by sampling uniformly inside it."""
        edges = self.edges[field]
        bin_index = int(np.clip(bin_index, 0, len(edges) - 2))
        low, high = float(edges[bin_index]), float(edges[bin_index + 1])
        value = float(rng.uniform(low, high)) if high > low else low
        low_limit, high_limit = GENAI.PHYSICAL_LIMITS[field]
        value = float(np.clip(value, low_limit, high_limit))
        return int(round(value)) if field in GENAI.INTEGER_FEATURES else round(value, 3)

    def field_token_ids(self, field: str) -> torch.Tensor:
        return torch.tensor([self.stoi[t] for t in self.field_tokens[field]], dtype=torch.long)

    # ---- serialization -----------------------------------------------------

    def prefix_ids(self, severity: str, hazard: str, state: str, district: str) -> list[int]:
        return [self.id(self.BOS),
                self.id(f"SEV_{severity}"),
                self.id(f"HAZ_{hazard.replace(' ', '-')}"),
                self.id(f"ST_{state.replace(' ', '-')}"),
                self.id(f"DI_{district.replace(' ', '-')}")]

    def encode_example(self, severity, hazard, state, district,
                       row: pd.Series, text: str) -> list[int]:
        ids = self.prefix_ids(severity, hazard, state, district)
        for field in FIELDS:
            ids.append(self.id(self.field_tokens[field][self.bin_of(field, float(row[field]))]))
        ids.append(self.id(self.SEP))
        ids += [self.id(w) for w in str(text).lower().split()[:MAX_TEXT_WORDS]]
        ids.append(self.id(self.EOS))
        return ids

    def decode_words(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def state_dict(self) -> dict:
        return {"itos": self.itos,
                "edges": {f: self.edges[f].tolist() for f in FIELDS},
                "field_tokens": self.field_tokens,
                "word_tokens": self.word_tokens}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ScenarioVocab":
        vocab = cls.__new__(cls)
        vocab.specials = [cls.PAD, cls.BOS, cls.SEP, cls.EOS, cls.UNK]
        vocab.itos = state["itos"]
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        vocab.edges = {f: np.asarray(e) for f, e in state["edges"].items()}
        vocab.field_tokens = state["field_tokens"]
        vocab.word_tokens = state["word_tokens"]
        vocab.word_ids = torch.tensor([vocab.stoi[w] for w in vocab.word_tokens], dtype=torch.long)
        return vocab


# ===========================================================================
# 2. The model -- a small decoder-only transformer
# ===========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block: int) -> None:
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, block: int) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(DROPOUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ScenarioSLM(nn.Module):
    """Decoder-only transformer over (spec, sensors, narrative) sequences."""

    def __init__(self, vocab_size: int, block: int = BLOCK) -> None:
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(block, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEAD, block) for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=0)
        return logits, loss


# ===========================================================================
# 3. Training corpus -- pairing real messages with real sensor rows
# ===========================================================================

def _pick_by_margin(pool: pd.DataFrame, severity: str, rng: np.random.Generator) -> int:
    """Index of a row from the band of river margin that fits this severity."""
    margin = (pool["river_level_m"] - pool["river_level_threshold_m"]).to_numpy()
    order = np.argsort(margin)
    low_q, high_q = MARGIN_BAND[severity]
    lo = int(low_q * (len(order) - 1))
    hi = max(lo + 1, int(high_q * (len(order) - 1)) + 1)
    return int(order[int(rng.integers(lo, hi))])


def build_corpus(seed: int = SEED, limit: int | None = None) -> tuple[ScenarioVocab, np.ndarray, dict]:
    """Pair each dispatcher message with a real sensor row from its district.

    The two datasets are not natively joined -- Stage 01's sensor record and
    Stage 03's message corpus were collected separately -- so a message is
    matched to a sensor row from the SAME district at a compatible risk class.
    That is the same correspondence the rest of Stage 05 already assumes when it
    scores a generated zone, made explicit and used as training signal.
    """
    rng = np.random.default_rng(seed)
    messages = pd.read_csv(DISPATCHER_CORPUS,
                           usecols=["text", "severity", "hazard_type", "state", "district"]).dropna()
    if limit:
        messages = messages.head(limit)
    sensors = pd.read_csv(SENSOR_REFERENCE_CSV)

    vocab = ScenarioVocab(
        sensors, messages["text"],
        states=sorted(set(messages["state"]) | set(sensors["state"])),
        districts=sorted(set(messages["district"]) | set(sensors["district"])),
        hazards=sorted(messages["hazard_type"].unique()),
        severities=sorted(messages["severity"].unique()))

    pools: dict[tuple[str, str], pd.DataFrame] = {}
    for (district, risk), group in sensors.groupby(["district", "zone_risk"]):
        pools[(district, risk)] = group.reset_index(drop=True)
    by_risk = {risk: group.reset_index(drop=True)
               for risk, group in sensors.groupby("zone_risk")}

    sequences, skipped = [], 0
    for _, message in messages.iterrows():
        severity = message["severity"]
        risk = SEVERITY_TO_RISK[severity]
        pool = pools.get((message["district"], risk))
        if pool is None or pool.empty:
            pool = by_risk.get(risk)
        if pool is None or pool.empty:
            skipped += 1
            continue
        row = pool.iloc[_pick_by_margin(pool, severity, rng)]
        ids = vocab.encode_example(message["severity"], message["hazard_type"],
                                   message["state"], message["district"], row, message["text"])
        if len(ids) <= BLOCK:
            sequences.append(ids)
        else:
            skipped += 1

    padded = np.zeros((len(sequences), BLOCK), dtype=np.int64)
    for i, ids in enumerate(sequences):
        padded[i, :len(ids)] = ids
    stats = {"messages": int(len(messages)), "paired": len(sequences), "skipped": skipped,
             "vocab_size": len(vocab), "block": BLOCK,
             "mean_length": round(float(np.mean([len(s) for s in sequences])), 1)}
    return vocab, padded, stats


def train(seed: int = SEED, epochs: int = EPOCHS, limit: int | None = None,
          verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, data, stats = build_corpus(seed, limit)
    if verbose:
        print(f"[slm] paired {stats['paired']:,} of {stats['messages']:,} messages | "
              f"vocab {stats['vocab_size']} | mean length {stats['mean_length']}")

    split = int(0.95 * len(data))
    perm = np.random.default_rng(seed).permutation(len(data))
    train_ids = torch.tensor(data[perm[:split]])
    val_ids = torch.tensor(data[perm[split:]])

    model = ScenarioSLM(len(vocab)).to(device)
    params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[slm] {params / 1e6:.2f}M parameters on {device.type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    t0 = time.time()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_ids))
        total, batches = 0.0, 0
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = train_ids[order[start:start + BATCH_SIZE]].to(device)
            _, loss = model(batch[:, :-1], batch[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
            batches += 1

        model.eval()
        with torch.inference_mode():
            val_losses = []
            for start in range(0, len(val_ids) - BATCH_SIZE + 1, BATCH_SIZE):
                batch = val_ids[start:start + BATCH_SIZE].to(device)
                _, loss = model(batch[:, :-1], batch[:, 1:])
                val_losses.append(float(loss))
        train_loss = total / max(batches, 1)
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4)})
        if verbose:
            print(f"[slm] epoch {epoch + 1}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab.state_dict(),
                "config": {"n_layer": N_LAYER, "n_head": N_HEAD, "d_model": D_MODEL,
                           "block": BLOCK, "n_bins": N_BINS, "params": params},
                "corpus": stats, "history": history, "seed": seed}, SLM_PATH)
    manifest = {"trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "seconds": round(time.time() - t0, 1), "parameters": params,
                "corpus": stats, "history": history,
                "final_val_loss": history[-1]["val_loss"] if history else None,
                "final_val_perplexity": round(math.exp(history[-1]["val_loss"]), 2) if history else None}
    if verbose:
        print(f"[slm] trained in {manifest['seconds']}s | val perplexity "
              f"{manifest['final_val_perplexity']} -> {SLM_PATH.name}")
    return manifest


# ===========================================================================
# 4. Constrained autoregressive sampling
# ===========================================================================

class SLMSampler:
    """Loads the trained SLM and samples a zone from a conditioning prefix.

    Decoding is CONSTRAINED by position: the nine sensor slots accept only that
    field's bin tokens, and the narrative accepts only corpus words and <eos>.
    A structurally invalid scenario therefore cannot be produced, which is why
    this path needs no repair-retry loop -- unlike prompting a general LLM for
    JSON, where the schema is a request rather than a guarantee.
    """

    def __init__(self, path: Path = SLM_PATH) -> None:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = ScenarioVocab.from_state_dict(bundle["vocab"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScenarioSLM(len(self.vocab), bundle["config"]["block"]).to(self.device)
        self.model.load_state_dict(bundle["model"])
        self.model.eval()
        self.config = bundle["config"]
        self.history = bundle.get("history", [])
        self.corpus = bundle.get("corpus", {})
        self.model_id = (f"ScenarioSLM {self.config['params'] / 1e6:.1f}M "
                         f"({self.device.type})")
        self._field_ids = {f: self.vocab.field_token_ids(f).to(self.device) for f in FIELDS}
        self._text_ids = torch.cat([self.vocab.word_ids.to(self.device),
                                    torch.tensor([self.vocab.id(ScenarioVocab.EOS)],
                                                 device=self.device)])

    def _next(self, ids: list[int], allowed: torch.Tensor, temperature: float,
              generator: torch.Generator) -> int:
        window = ids[-self.config["block"]:]
        x = torch.tensor([window], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits, _ = self.model(x)
        logits = logits[0, -1]
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]                    # constrained decoding
        probs = F.softmax(masked / max(temperature, 1e-4), dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))

    def sample(self, severity: str, hazard: str, state: str, district: str,
               temperature: float = 0.9, seed: int = SEED) -> tuple[dict, str]:
        generator = torch.Generator(device=self.device).manual_seed(int(seed) % (2**31 - 1))
        rng = np.random.default_rng(seed)
        ids = self.vocab.prefix_ids(severity, hazard, state, district)

        sensors: dict[str, float] = {}
        for field in FIELDS:
            token = self._next(ids, self._field_ids[field], temperature, generator)
            ids.append(token)
            bin_index = self.vocab.field_tokens[field].index(self.vocab.itos[token])
            sensors[field] = self.vocab.value_of(field, bin_index, rng)

        ids.append(self.vocab.id(ScenarioVocab.SEP))
        eos = self.vocab.id(ScenarioVocab.EOS)
        words: list[int] = []
        for _ in range(MAX_TEXT_WORDS):
            token = self._next(ids, self._text_ids, temperature, generator)
            if token == eos:
                break
            ids.append(token)
            words.append(token)
        return sensors, self.vocab.decode_words(words)


# ===========================================================================
# 5. Scenario generator
# ===========================================================================

def _prettify(text: str, district: str, state: str) -> str:
    """Restore the capitalisation the lowercased vocabulary dropped."""
    text = " ".join(text.split()).replace(" ,", ",").replace(" .", ".")
    if not text:
        return text
    out = text[0].upper() + text[1:]
    for name in {district, state}:
        out = out.replace(name.lower(), name)
    return out


class SLMScenarioGenerator:
    """Prompt spec -> multi-zone scenario, sensors and narrative from the SLM.

    Mirrors LLMScenarioGenerator: the scenario scaffolding (districts, evidence
    loss under blackout, gauge histories, imagery, expectations) is reused from
    ScenarioGenerator, and only the generative core is replaced.
    """

    def __init__(self, sampler: SLMSampler | None = None,
                 processed_dir: Path = PROCESSED_DIR,
                 temperature: float = 0.9, verbose: bool = False) -> None:
        self.base = GENAI.ScenarioGenerator(processed_dir=processed_dir)
        self.sampler = sampler
        self.temperature = float(temperature)
        self.verbose = bool(verbose)
        self.stats: dict[str, Any] = {"zones": 0, "slm_zones": 0, "fallback_zones": 0,
                                      "latency_ms": []}

    def generate(self, spec: dict, seed: int = SEED) -> dict[str, Any]:
        scenario = self.base.generate(spec, seed=seed)
        for index, zone in enumerate(scenario["zones"]):
            self._rewrite_zone(zone, seed=seed + index)
        total = int(sum(z["headcount_reported"] for z in scenario["zones"]))
        scenario["demand"]["total_headcount_reported"] = total
        scenario["demand"]["shelter_gap"] = max(
            0, total - int(scenario["demand"].get("shelter_beds", 0)))
        scenario["generator"] = {
            "family": "sequence_slm",
            "model": self.sampler.model_id if self.sampler else "unavailable (CVAE fallback)",
            "techniques": TECHNIQUES,
            "techniques_not_claimed": TECHNIQUES_NOT_CLAIMED,
            "temperature": self.temperature,
            "seed": int(seed),
            "trained_on": "Stage 03 dispatcher corpus x Stage 01 sensor record",
            "fallback_path": "SensorCVAE + phrase bank (03_genai_engineer.py)",
        }
        return scenario

    def generate_suite(self, prompts: list[dict], base_seed: int = SEED) -> list[dict]:
        out = []
        for index, spec in enumerate(prompts):
            out.append(self.generate(spec, seed=base_seed + index * 10))
            if self.verbose:
                print(f"  [{spec.get('id', '?')}] {len(out[-1]['zones'])} zones", flush=True)
        return out

    def _rewrite_zone(self, zone: dict, seed: int) -> None:
        self.stats["zones"] += 1
        available = zone["provenance"]["evidence_available"]
        if not (available["sensors"] or available["text"]):
            zone["provenance"]["slm"] = {"status": "skipped", "reason": "no surviving evidence"}
            return
        if self.sampler is None:
            self.stats["fallback_zones"] += 1
            zone["provenance"]["slm"] = {"status": "fallback", "reason": "no SLM trained"}
            return

        severity = GENAI.SEVERITY_TO_TEXT[zone["true_severity"]]
        hazard = zone["hazards"][0]
        started = time.time()
        sensors, text = self.sampler.sample(severity, hazard, zone["state"], zone["district"],
                                            temperature=self.temperature, seed=seed)
        latency = (time.time() - started) * 1000
        self.stats["latency_ms"].append(round(latency, 1))
        self.stats["slm_zones"] += 1

        if zone["inputs"]["sensors"] is not None:
            zone["inputs"]["sensors"].update(sensors)
        message = _prettify(text, zone["district"], zone["state"])
        if zone["inputs"]["text"] is not None and zone["text_entries"] and message:
            top = max(range(len(zone["text_entries"])),
                      key=lambda k: (GENAI.TEXT_LEVELS.index(zone["text_entries"][k]["severity"]), k))
            zone["text_entries"][top]["text"] = message
            zone["text_entries"][top]["source"] = "slm"
            zone["inputs"]["text"] = message
            zone["inputs"]["incident_log"] = GENAI.ScenarioGenerator._incident_log(
                zone["text_entries"], zone["state"], zone["district"], zone["zone_id"])
            zone["headcount_reported"] = int(sum(e["headcount"] for e in zone["text_entries"]))

        history = zone["inputs"].get("water_levels")
        if history and "river_level_m" in sensors:
            offset = float(sensors["river_level_m"]) - float(history[-1])
            zone["inputs"]["water_levels"] = [round(float(v) + offset, 3) for v in history]

        zone["provenance"]["slm"] = {
            "status": "ok", "model": self.sampler.model_id, "techniques": TECHNIQUES,
            "constrained": True, "temperature": self.temperature,
            "latency_ms": round(latency, 1),
        }

    def summary(self) -> dict[str, Any]:
        latencies = self.stats["latency_ms"]
        attempted = self.stats["slm_zones"] + self.stats["fallback_zones"]
        return {
            "zones": self.stats["zones"],
            "slm_generated_zones": self.stats["slm_zones"],
            "fallback_zones": self.stats["fallback_zones"],
            "fallback_rate": round(self.stats["fallback_zones"] / max(attempted, 1), 4),
            "latency_ms": {
                "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else None,
                "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            },
        }


# ===========================================================================
# 6. CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 domain-SLM sequence generation")
    parser.add_argument("--train", action="store_true", help="train the SLM before generating")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit-corpus", type=int, default=0, help="train on the first N messages")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N prompts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    training = None
    if args.train or not SLM_PATH.exists():
        training = train(args.seed, args.epochs, args.limit_corpus or None, not args.quiet)

    sampler = SLMSampler() if SLM_PATH.exists() else None
    generator = SLMScenarioGenerator(sampler, temperature=args.temperature,
                                     verbose=not args.quiet)
    library = GENAI.load_prompts()
    prompts = library["prompts"][:args.limit] if args.limit else library["prompts"]

    print(f"[slm] generating {len(prompts)} scenarios + wildcard ...")
    suite = generator.generate_suite(prompts, base_seed=args.seed)
    wildcard = generator.generate(library["wildcard"], seed=args.seed + 100)

    SUITE_JSON.write_text(json.dumps({"seed": args.seed, "generator": "sequence_slm",
                                      "scenarios": suite}, indent=2), encoding="utf-8")
    WILDCARD_JSON.write_text(json.dumps(wildcard, indent=2), encoding="utf-8")
    GENAI.flatten_zones(suite + [wildcard]).to_csv(ZONES_CSV, index=False)

    summary = generator.summary()
    MANIFEST_JSON.write_text(json.dumps({
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "family": "sequence_slm",
        "model": sampler.model_id if sampler else "unavailable",
        "techniques": TECHNIQUES, "techniques_not_claimed": TECHNIQUES_NOT_CLAIMED,
        "temperature": args.temperature, "seed": args.seed,
        "training": training, "summary": summary,
    }, indent=2), encoding="utf-8")

    print(f"[slm] zones {summary['zones']} | SLM {summary['slm_generated_zones']} | "
          f"fallback {summary['fallback_zones']} ({summary['fallback_rate']:.1%})")
    if summary["latency_ms"]["p50"] is not None:
        print(f"[slm] latency p50 {summary['latency_ms']['p50']} ms per zone")
    print(f"[slm] -> {SUITE_JSON.name}, {WILDCARD_JSON.name}, {MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
