"""Project-wide configuration: paths, IDs, and model slugs.

The values in this module are the single source of truth for everything else
in the pipeline. Model slugs read from the environment so the user can swap
them via .env without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"
DATA_DIR = ROOT / "data"
SOLO_DIR = DATA_DIR / "solo"
DEBATE_DIR = DATA_DIR / "debate"
RATINGS_DIR = DATA_DIR / "ratings"
RESULTS_DIR = ROOT / "results"
LOG_PATH = DATA_DIR / "log.jsonl"

PERSONAS = ["participant_1", "participant_2", "participant_3"]
PERSONA_LEANING = {
    "participant_1": "utilitarian-leaning (high IB)",
    "participant_2": "mixed (median OUS)",
    "participant_3": "deontological-leaning (low IH)",
}

SCENARIOS = ["switch", "footbridge", "loved_one"]

MODELS = {
    "claude": os.getenv("MODEL_CLAUDE", "anthropic/claude-opus-4.7"),
    "gemini": os.getenv("MODEL_GEMINI", "google/gemini-pro-latest"),
    "gpt": os.getenv("MODEL_GPT", "openai/gpt-chat-latest"),
}

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

MIN_TRANSCRIPT_WORDS = 2000

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
    "X-Title": os.getenv("OPENROUTER_X_TITLE", "Persona-Conditioned Trolley Debate"),
}


def ensure_dirs() -> None:
    for d in (TRANSCRIPTS_DIR, DATA_DIR, SOLO_DIR, DEBATE_DIR, RATINGS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def transcript_path(persona: str) -> Path:
    return TRANSCRIPTS_DIR / f"{persona}.txt"


def solo_path(persona: str, model_key: str, scenario: str) -> Path:
    return SOLO_DIR / f"solo_{persona}_{model_key}_{scenario}.json"


def debate_path(model_key: str, scenario: str) -> Path:
    return DEBATE_DIR / f"debate_{model_key}_{scenario}.json"


def ratings_path(persona: str) -> Path:
    return RATINGS_DIR / f"ratings_{persona}.json"
