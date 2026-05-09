"""Participant fidelity rating UI (\u00a74.5).

For each participant we collect:

1. Their own verdict + 1\u201310 confidence on each of the three trolley scenarios
   \u2014 this is the self-report ground-truth used for verdict-match.
2. A 1\u20135 agreement rating for each of the three conditioned-model arguments
   produced for *their* persona on each scenario (3 models \u00d7 3 scenarios = 9
   ratings per participant), plus an optional comment.

Per the protocol participants only see arguments generated for their own
persona \u2014 they are not asked to evaluate other participants' personas.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from . import config
from .prompts import SCENARIO_DISPLAY, SCENARIO_TEXTS


def _load_solo_for_persona(persona: str) -> Dict[str, Dict[str, Any]]:
    """Return {scenario: {model_key: solo_record}} for one persona."""
    out: Dict[str, Dict[str, Any]] = {sc: {} for sc in config.SCENARIOS}
    for scenario in config.SCENARIOS:
        for model_key in config.MODELS:
            p = config.solo_path(persona, model_key, scenario)
            if p.exists():
                out[scenario][model_key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _load_existing_ratings(persona: str) -> Dict[str, Any]:
    p = config.ratings_path(persona)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"persona": persona, "self_report": {}, "argument_ratings": {}}


def _save(persona: str, payload: Dict[str, Any]) -> None:
    config.RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    config.ratings_path(persona).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def rating_form(persona: str) -> widgets.Widget:
    """Build the ipywidgets form for one participant.

    The form rehydrates from any saved JSON so participants can pause and
    resume.
    """
    if persona not in config.PERSONAS:
        raise ValueError(f"Unknown persona {persona!r}")

    solo = _load_solo_for_persona(persona)
    existing = _load_existing_ratings(persona)
    self_prev = existing.get("self_report", {})
    args_prev = existing.get("argument_ratings", {})

    title = widgets.HTML(
        f"<h2 style='margin-bottom:6px'>Fidelity rating \u2014 {persona}</h2>"
        f"<p style='color:#666;margin-top:0'>Leaning: "
        f"{config.PERSONA_LEANING.get(persona, '\u2014')}. "
        f"Your responses save to <code>{config.ratings_path(persona)}</code>.</p>"
    )

    self_report_widgets: Dict[str, Dict[str, widgets.Widget]] = {}
    arg_widgets: Dict[str, Dict[str, Dict[str, widgets.Widget]]] = {}

    sections = []

    # ---- Self-report section -------------------------------------------------
    sr_header = widgets.HTML(
        "<h3>1. Your own verdicts</h3>"
        "<p style='color:#444'>Read each scenario below and record your own "
        "verdict and confidence. This is the ground-truth used for the "
        "verdict-match metric. Do <em>not</em> read the model arguments in "
        "section 2 before completing this section.</p>"
    )
    sections.append(sr_header)

    for scenario in config.SCENARIOS:
        prev = self_prev.get(scenario, {})
        scenario_box = widgets.HTML(
            f"<details><summary><b>{SCENARIO_DISPLAY[scenario]}</b> "
            f"\u2014 click to read scenario</summary>"
            f"<p style='white-space:pre-wrap;background:#f6f6f6;padding:8px;"
            f"border-radius:6px'>{SCENARIO_TEXTS[scenario]}</p></details>"
        )
        action = widgets.RadioButtons(
            options=[("Intervene (act)", "intervene"), ("Abstain (do nothing)", "abstain")],
            value=prev.get("action") if prev.get("action") in ("intervene", "abstain") else None,
            description="Verdict:",
            style={"description_width": "100px"},
        )
        confidence = widgets.IntSlider(
            value=prev.get("confidence") or 5,
            min=1,
            max=10,
            step=1,
            description="Confidence (1\u201310):",
            style={"description_width": "150px"},
            continuous_update=False,
        )
        notes = widgets.Textarea(
            value=prev.get("notes", ""),
            description="Notes (optional):",
            placeholder="Any reasoning you want to record alongside your verdict",
            layout=widgets.Layout(width="100%", height="60px"),
            style={"description_width": "150px"},
        )
        self_report_widgets[scenario] = {"action": action, "confidence": confidence, "notes": notes}
        sections.append(
            widgets.VBox(
                [scenario_box, action, confidence, notes],
                layout=widgets.Layout(border="1px solid #ddd", padding="10px", margin="6px 0"),
            )
        )

    # ---- Argument-rating section ---------------------------------------------
    sections.append(
        widgets.HTML(
            "<h3>2. Rate the model arguments produced for your persona</h3>"
            "<p style='color:#444'>For each model argument below, rate how "
            "much you agree with the <em>reasoning</em> (not just the "
            "conclusion) on a 1\u20135 scale: 1 = strongly disagree, "
            "3 = neutral, 5 = strongly agree.</p>"
        )
    )

    for scenario in config.SCENARIOS:
        sections.append(widgets.HTML(f"<h4>{SCENARIO_DISPLAY[scenario]}</h4>"))
        scenario_args = args_prev.get(scenario, {})
        arg_widgets[scenario] = {}
        for model_key in config.MODELS:
            rec = solo.get(scenario, {}).get(model_key)
            if rec is None:
                sections.append(
                    widgets.HTML(
                        f"<p style='color:#a00'>(no solo generation found for "
                        f"{model_key}/{scenario} \u2014 run the solo phase first)</p>"
                    )
                )
                continue
            response = rec.get("response", {})
            argument_text = response.get("reasoning") or response.get("raw") or ""
            verdict = response.get("action")
            conf = response.get("confidence")
            prev_for_model = scenario_args.get(model_key, {})

            arg_html = widgets.HTML(
                f"<div style='background:#fafafa;border-left:3px solid #88a;"
                f"padding:8px 12px;margin:6px 0'>"
                f"<div style='color:#666;font-size:90%'><b>{model_key}</b> \u2014 "
                f"verdict: <code>{verdict}</code>, confidence: {conf}</div>"
                f"<div style='white-space:pre-wrap;margin-top:6px'>{argument_text}</div>"
                f"</div>"
            )
            score = widgets.RadioButtons(
                options=[
                    ("1 \u2014 strongly disagree", 1),
                    ("2 \u2014 disagree", 2),
                    ("3 \u2014 neutral", 3),
                    ("4 \u2014 agree", 4),
                    ("5 \u2014 strongly agree", 5),
                ],
                value=prev_for_model.get("score"),
                description="Agreement:",
                style={"description_width": "100px"},
            )
            comment = widgets.Textarea(
                value=prev_for_model.get("comment", ""),
                description="Comment:",
                placeholder="Optional: what felt right or wrong about this reasoning?",
                layout=widgets.Layout(width="100%", height="50px"),
                style={"description_width": "100px"},
            )
            arg_widgets[scenario][model_key] = {"score": score, "comment": comment}
            sections.append(
                widgets.VBox(
                    [arg_html, score, comment],
                    layout=widgets.Layout(border="1px solid #eee", padding="6px", margin="4px 0"),
                )
            )

    status = widgets.HTML("")
    save_btn = widgets.Button(description="Save ratings", button_style="primary")

    def _on_save(_):
        payload: Dict[str, Any] = {
            "persona": persona,
            "self_report": {},
            "argument_ratings": {},
        }
        missing = []
        for scenario, ws in self_report_widgets.items():
            action = ws["action"].value
            if action is None:
                missing.append(f"self-report verdict for {scenario}")
            payload["self_report"][scenario] = {
                "action": action,
                "confidence": int(ws["confidence"].value),
                "notes": ws["notes"].value,
            }
        for scenario, by_model in arg_widgets.items():
            payload["argument_ratings"][scenario] = {}
            for model_key, ws in by_model.items():
                score = ws["score"].value
                if score is None:
                    missing.append(f"agreement rating for {model_key}/{scenario}")
                payload["argument_ratings"][scenario][model_key] = {
                    "score": score,
                    "comment": ws["comment"].value,
                }
        _save(persona, payload)
        if missing:
            status.value = (
                "<p style='color:#a60'>Saved partial ratings. Still missing: "
                + "; ".join(missing)
                + "</p>"
            )
        else:
            status.value = (
                f"<p style='color:#070'>\u2713 Saved to "
                f"<code>{config.ratings_path(persona)}</code></p>"
            )

    save_btn.on_click(_on_save)
    sections.extend([save_btn, status])

    return widgets.VBox([title] + sections)


def display_form(persona: str) -> None:
    display(rating_form(persona))


# ---------------------------------------------------------------------------
# Verdict-match (requires self-report from rating files)
# ---------------------------------------------------------------------------

def verdict_match_table() -> Optional[pd.DataFrame]:
    """Compare each model verdict against the participant's own verdict.

    Returns None if no rating files exist yet.
    """
    rows = []
    any_self_report = False
    for persona in config.PERSONAS:
        rating = _load_existing_ratings(persona)
        self_report = rating.get("self_report", {})
        if not self_report:
            continue
        any_self_report = True
        for scenario in config.SCENARIOS:
            sr = self_report.get(scenario, {})
            sr_action = sr.get("action")
            for model_key in config.MODELS:
                p = config.solo_path(persona, model_key, scenario)
                if not p.exists():
                    continue
                rec = json.loads(p.read_text(encoding="utf-8"))
                model_action = rec["response"]["action"]
                rows.append(
                    {
                        "persona": persona,
                        "scenario": scenario,
                        "model": model_key,
                        "self_report": sr_action,
                        "model_verdict": model_action,
                        "match": (
                            sr_action is not None
                            and model_action is not None
                            and sr_action == model_action
                        ),
                    }
                )
    if not any_self_report:
        return None
    return pd.DataFrame(rows)


def fidelity_summary() -> Optional[pd.DataFrame]:
    """Aggregate the 1\u20135 agreement scores per (persona, model, scenario)."""
    rows = []
    found = False
    for persona in config.PERSONAS:
        rating = _load_existing_ratings(persona)
        for scenario, by_model in rating.get("argument_ratings", {}).items():
            for model_key, entry in by_model.items():
                score = entry.get("score")
                if score is None:
                    continue
                found = True
                rows.append(
                    {
                        "persona": persona,
                        "scenario": scenario,
                        "model": model_key,
                        "agreement_1_to_5": score,
                    }
                )
    if not found:
        return None
    return pd.DataFrame(rows)
