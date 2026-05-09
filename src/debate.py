"""Within-model 3-round debate (\u00a74.4): 3 models \u00d7 3 scenarios = 9 debates.

Each debate file contains all three rounds and all three personas, so the
file is a self-contained record of one (model, scenario) arena.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import config
from .client import OpenRouterClient, append_log, load_transcript, utc_now_iso
from .parse import parse_response
from .prompts import (
    debate_final_user,
    debate_opening_user,
    debate_response_user,
    system_message_from_transcript,
)


def _round_record(
    *,
    persona: str,
    model_key: str,
    model_slug: str,
    scenario: str,
    round_idx: int,
    round_name: str,
    system: str,
    user: str,
    text: str,
    parsed: Dict[str, object],
    model_version: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "model_key": model_key,
        "model_slug": model_slug,
        "model_version": model_version,
        "persona": persona,
        "scenario": scenario,
        "generation": f"debate-r{round_idx}",
        "round_name": round_name,
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
    if "stance_changed" in parsed:
        rec["response"]["stance_changed"] = parsed["stance_changed"]
    if "stance_change_reason" in parsed:
        rec["response"]["stance_change_reason"] = parsed["stance_change_reason"]
    return rec


def run_debate_one(
    client: OpenRouterClient,
    *,
    model_key: str,
    scenario: str,
    overwrite: bool = False,
    progress=print,
) -> Dict[str, object]:
    out_path = config.debate_path(model_key, scenario)
    if out_path.exists() and not overwrite:
        return json.loads(out_path.read_text(encoding="utf-8"))

    model_slug = config.MODELS[model_key]
    systems: Dict[str, str] = {
        p: system_message_from_transcript(load_transcript(p), p)
        for p in config.PERSONAS
    }

    rounds: List[Dict[str, object]] = []

    # ---- Round 1: opening ----------------------------------------------------
    progress(f"  round 1 (opening) ...")
    opening_records: Dict[str, Dict[str, object]] = {}
    for persona in config.PERSONAS:
        user = debate_opening_user(scenario)
        result = client.chat(model_slug=model_slug, system=systems[persona], user=user)
        parsed = parse_response(result.text)
        rec = _round_record(
            persona=persona,
            model_key=model_key,
            model_slug=model_slug,
            scenario=scenario,
            round_idx=1,
            round_name="opening",
            system=systems[persona],
            user=user,
            text=result.text,
            parsed=parsed,
            model_version=result.model_version,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        append_log(rec)
        opening_records[persona] = rec
    rounds.append({"round": 1, "name": "opening", "by_persona": opening_records})

    # ---- Round 2: response ---------------------------------------------------
    progress(f"  round 2 (response) ...")
    response_records: Dict[str, Dict[str, object]] = {}
    opening_texts = {p: opening_records[p]["response"]["raw"] for p in config.PERSONAS}
    for persona in config.PERSONAS:
        others = {p: opening_texts[p] for p in config.PERSONAS if p != persona}
        user = debate_response_user(
            scenario,
            self_persona=persona,
            opening_self=opening_texts[persona],
            opening_others=others,
        )
        result = client.chat(model_slug=model_slug, system=systems[persona], user=user)
        parsed = parse_response(result.text)
        rec = _round_record(
            persona=persona,
            model_key=model_key,
            model_slug=model_slug,
            scenario=scenario,
            round_idx=2,
            round_name="response",
            system=systems[persona],
            user=user,
            text=result.text,
            parsed=parsed,
            model_version=result.model_version,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        append_log(rec)
        response_records[persona] = rec
    rounds.append({"round": 2, "name": "response", "by_persona": response_records})

    # ---- Round 3: final ------------------------------------------------------
    progress(f"  round 3 (final) ...")
    final_records: Dict[str, Dict[str, object]] = {}
    response_texts = {p: response_records[p]["response"]["raw"] for p in config.PERSONAS}
    for persona in config.PERSONAS:
        others = {p: response_texts[p] for p in config.PERSONAS if p != persona}
        user = debate_final_user(
            scenario,
            self_persona=persona,
            round1_self=opening_texts[persona],
            round2_self=response_texts[persona],
            round2_others=others,
        )
        result = client.chat(model_slug=model_slug, system=systems[persona], user=user)
        parsed = parse_response(result.text)
        rec = _round_record(
            persona=persona,
            model_key=model_key,
            model_slug=model_slug,
            scenario=scenario,
            round_idx=3,
            round_name="final",
            system=systems[persona],
            user=user,
            text=result.text,
            parsed=parsed,
            model_version=result.model_version,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        append_log(rec)
        final_records[persona] = rec
    rounds.append({"round": 3, "name": "final", "by_persona": final_records})

    payload = {
        "model_key": model_key,
        "model_slug": model_slug,
        "scenario": scenario,
        "temperature": config.TEMPERATURE,
        "timestamp": utc_now_iso(),
        "rounds": rounds,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_debate_all(
    client: OpenRouterClient,
    *,
    overwrite: bool = False,
    progress=print,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    pairs = [(m, s) for m in config.MODELS for s in config.SCENARIOS]
    for i, (model_key, scenario) in enumerate(pairs, 1):
        out_path = config.debate_path(model_key, scenario)
        if out_path.exists() and not overwrite:
            progress(f"[{i}/{len(pairs)}] skip (exists) debate {model_key}/{scenario}")
            results[f"{model_key}_{scenario}"] = json.loads(out_path.read_text(encoding="utf-8"))
            continue
        progress(f"[{i}/{len(pairs)}] running debate {model_key}/{scenario}")
        results[f"{model_key}_{scenario}"] = run_debate_one(
            client, model_key=model_key, scenario=scenario, overwrite=overwrite, progress=progress
        )
    return results
