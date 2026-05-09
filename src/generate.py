"""Solo verdict generation: 3 personas \u00d7 3 models \u00d7 3 scenarios = 27 calls.

Each call is persisted as one JSON file in ``data/solo/`` and one line in
``data/log.jsonl``. Rerunning the script skips any (persona, model, scenario)
that already has a file on disk, so partial failures are recoverable.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from . import config
from .client import OpenRouterClient, append_log, load_transcript, utc_now_iso
from .parse import parse_response
from .prompts import build_user_message, system_message_from_transcript


def _record(
    *,
    persona: str,
    model_key: str,
    model_slug: str,
    scenario: str,
    system: str,
    user: str,
    text: str,
    parsed: Dict[str, object],
    model_version: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    generation: str,
) -> Dict[str, object]:
    return {
        "model_key": model_key,
        "model_slug": model_slug,
        "model_version": model_version,
        "persona": persona,
        "scenario": scenario,
        "generation": generation,
        "temperature": config.TEMPERATURE,
        "timestamp": utc_now_iso(),
        "prompt": {"system": system, "user": user},
        "response": {
            "raw": text,
            "reasoning": parsed.get("reasoning"),
            "action": parsed.get("action"),
            "confidence": parsed.get("confidence"),
            "parse_ok": parsed.get("parse_ok"),
        },
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def run_solo_one(
    client: OpenRouterClient,
    *,
    persona: str,
    model_key: str,
    scenario: str,
    overwrite: bool = False,
) -> Dict[str, object]:
    out_path = config.solo_path(persona, model_key, scenario)
    if out_path.exists() and not overwrite:
        return json.loads(out_path.read_text(encoding="utf-8"))

    transcript = load_transcript(persona)
    system = system_message_from_transcript(transcript, persona)
    user = build_user_message(scenario)
    model_slug = config.MODELS[model_key]

    result = client.chat(model_slug=model_slug, system=system, user=user)
    parsed = parse_response(result.text)

    record = _record(
        persona=persona,
        model_key=model_key,
        model_slug=model_slug,
        scenario=scenario,
        system=system,
        user=user,
        text=result.text,
        parsed=parsed,
        model_version=result.model_version,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        generation="solo",
    )

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(record)
    return record


def run_solo_all(
    client: OpenRouterClient,
    *,
    overwrite: bool = False,
    progress=print,
) -> Dict[str, Dict[str, object]]:
    """Run all 27 solo generations. Returns a dict keyed by file stem."""
    results: Dict[str, Dict[str, object]] = {}
    total = len(config.PERSONAS) * len(config.MODELS) * len(config.SCENARIOS)
    i = 0
    for persona in config.PERSONAS:
        for model_key in config.MODELS:
            for scenario in config.SCENARIOS:
                i += 1
                key = f"{persona}_{model_key}_{scenario}"
                out_path = config.solo_path(persona, model_key, scenario)
                if out_path.exists() and not overwrite:
                    progress(f"[{i}/{total}] skip (exists) {key}")
                    results[key] = json.loads(out_path.read_text(encoding="utf-8"))
                    continue
                progress(f"[{i}/{total}] generating {key}")
                results[key] = run_solo_one(
                    client,
                    persona=persona,
                    model_key=model_key,
                    scenario=scenario,
                    overwrite=overwrite,
                )
    return results
