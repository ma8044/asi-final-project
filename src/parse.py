"""Tolerant extraction of {action, confidence, reasoning} from model output.

Per §4.3 of the spec, refusal-prone outputs are coded REF and never excluded.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


REFUSAL_HINTS = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "i'm not able",
    "i am not able",
    "as an ai",
    "i don't feel comfortable",
    "i do not feel comfortable",
)


def _extract_json_blob(text: str) -> Optional[str]:
    """Find the first balanced top-level JSON object in *text*."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _normalise_action(value: Any) -> str:
    if not isinstance(value, str):
        return "REF"
    v = value.strip().lower()
    if v in {"intervene", "act", "pull", "push", "yes", "true"}:
        return "intervene"
    if v in {"abstain", "do nothing", "refrain", "no", "false"}:
        return "abstain"
    if "interven" in v or "pull" in v or "push" in v:
        return "intervene"
    if "abstain" in v or "do not" in v or "don't" in v or "nothing" in v:
        return "abstain"
    return "REF"


def _normalise_confidence(value: Any) -> Optional[int]:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(10, n))


def parse_response(raw: str) -> Dict[str, Any]:
    """Return {action, confidence, reasoning, stance_changed?, stance_change_reason?, parse_ok}.

    On any failure to recover an action, sets action="REF" and parse_ok=False.
    """
    parsed: Dict[str, Any] = {
        "action": "REF",
        "confidence": None,
        "reasoning": raw or "",
        "parse_ok": False,
    }

    if not raw or not raw.strip():
        return parsed

    blob = _extract_json_blob(raw)
    if blob is not None:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            action = _normalise_action(obj.get("action"))
            confidence = _normalise_confidence(obj.get("confidence"))
            reasoning = obj.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = raw
            parsed.update(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                parse_ok=action != "REF",
            )
            if "stance_changed" in obj:
                parsed["stance_changed"] = bool(obj.get("stance_changed"))
            if "stance_change_reason" in obj:
                reason = obj.get("stance_change_reason")
                parsed["stance_change_reason"] = reason if isinstance(reason, str) else ""
            return parsed

    # Fallback: hard refusal detection.
    lower = raw.lower()
    if any(h in lower for h in REFUSAL_HINTS):
        parsed["action"] = "REF"
        return parsed

    # Last-resort heuristic on raw text.
    parsed["action"] = _normalise_action(raw)
    parsed["parse_ok"] = parsed["action"] != "REF"
    return parsed
