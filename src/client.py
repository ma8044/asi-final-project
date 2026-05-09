"""Thin OpenRouter client wrapping the OpenAI Python SDK.

OpenRouter is OpenAI-API-compatible \u2014 we just point ``base_url`` at it and
use the same ``client.chat.completions.create`` call for every model family.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import config


@dataclass
class CallResult:
    text: str
    model_version: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    raw: Dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    def __init__(self) -> None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.sample to .env and "
                "fill in your key."
            )
        self._client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers={k: v for k, v in config.OPENROUTER_HEADERS.items() if v},
        )

    def chat(
        self,
        *,
        model_slug: str,
        system: str,
        user: str,
        temperature: float = config.TEMPERATURE,
        request_json: bool = True,
    ) -> CallResult:
        kwargs: Dict[str, Any] = {
            "model": model_slug,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if request_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - retried by caller
            if request_json:
                # Some routes reject json_object; retry without it.
                kwargs.pop("response_format", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise exc

        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return CallResult(
            text=text,
            model_version=getattr(resp, "model", model_slug),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )


def append_log(record: Dict[str, Any]) -> None:
    """Append a single JSONL line to the authoritative call log (§4.4)."""
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_transcript(persona: str) -> str:
    """Load and lightly validate a participant transcript."""
    path = config.transcript_path(persona)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing transcript for {persona}: expected {path}. "
            "Place the cleaned interview transcript there."
        )
    text = path.read_text(encoding="utf-8").strip()
    word_count = len(text.split())
    if word_count < config.MIN_TRANSCRIPT_WORDS:
        raise ValueError(
            f"Transcript for {persona} is only {word_count} words "
            f"(need >= {config.MIN_TRANSCRIPT_WORDS}). Per the spec a 2,000-word "
            "minimum is enforced."
        )
    return text
