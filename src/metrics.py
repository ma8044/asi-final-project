"""Quantitative metrics computable without participant ratings (\u00a74.5).

- action rate per (persona, model, scenario) \u2014 intervene / abstain / REF
- mean confidence per (persona, model, scenario)
- cross-model agreement per (persona, scenario)
- stance stability across debate rounds, per (persona, model, scenario)

Verdict-match against the participant self-report is added by ``ratings.py``
once the rating files exist.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_solo_records() -> List[Dict[str, Any]]:
    records = []
    for persona in config.PERSONAS:
        for model_key in config.MODELS:
            for scenario in config.SCENARIOS:
                p = config.solo_path(persona, model_key, scenario)
                if p.exists():
                    records.append(json.loads(p.read_text(encoding="utf-8")))
    return records


def load_debate_records() -> List[Dict[str, Any]]:
    records = []
    for model_key in config.MODELS:
        for scenario in config.SCENARIOS:
            p = config.debate_path(model_key, scenario)
            if p.exists():
                records.append(json.loads(p.read_text(encoding="utf-8")))
    return records


# ---------------------------------------------------------------------------
# DataFrame views
# ---------------------------------------------------------------------------

def solo_dataframe() -> pd.DataFrame:
    rows = []
    for r in load_solo_records():
        rows.append(
            {
                "persona": r["persona"],
                "model": r["model_key"],
                "scenario": r["scenario"],
                "action": r["response"]["action"],
                "confidence": r["response"]["confidence"],
                "parse_ok": r["response"]["parse_ok"],
            }
        )
    return pd.DataFrame(rows)


def debate_dataframe() -> pd.DataFrame:
    rows = []
    for d in load_debate_records():
        for rnd in d["rounds"]:
            for persona, rec in rnd["by_persona"].items():
                rows.append(
                    {
                        "persona": persona,
                        "model": d["model_key"],
                        "scenario": d["scenario"],
                        "round": rnd["round"],
                        "round_name": rnd["name"],
                        "action": rec["response"]["action"],
                        "confidence": rec["response"]["confidence"],
                        "stance_changed": rec["response"].get("stance_changed"),
                        "stance_change_reason": rec["response"].get("stance_change_reason"),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def action_rate_table(df_solo: pd.DataFrame) -> pd.DataFrame:
    """Pivot of action counts per (persona, model, scenario)."""
    if df_solo.empty:
        return df_solo
    return (
        df_solo.assign(n=1)
        .pivot_table(
            index=["persona", "model", "scenario"],
            columns="action",
            values="n",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )


def confidence_table(df_solo: pd.DataFrame) -> pd.DataFrame:
    if df_solo.empty:
        return df_solo
    return (
        df_solo.dropna(subset=["confidence"])
        .groupby(["persona", "model", "scenario"], as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_confidence"})
    )


def cross_model_agreement(df_solo: pd.DataFrame) -> pd.DataFrame:
    """For each (persona, scenario), do all 3 models agree on the action?"""
    if df_solo.empty:
        return df_solo
    rows = []
    for (persona, scenario), grp in df_solo.groupby(["persona", "scenario"]):
        actions = list(grp["action"])
        unique = set(actions)
        if len(unique) == 1:
            agreement = "unanimous"
        elif len(unique) == 2:
            agreement = "split-2"
        else:
            agreement = "split-3"
        rows.append(
            {
                "persona": persona,
                "scenario": scenario,
                "n_models": len(actions),
                "actions": ",".join(sorted(actions)),
                "agreement": agreement,
            }
        )
    return pd.DataFrame(rows)


def stance_stability(df_debate: pd.DataFrame) -> pd.DataFrame:
    """Track action across rounds 1 \u2192 2 \u2192 3 per (persona, model, scenario)."""
    if df_debate.empty:
        return df_debate
    rows = []
    for (persona, model, scenario), grp in df_debate.groupby(["persona", "model", "scenario"]):
        grp = grp.sort_values("round")
        actions = list(grp["action"])
        n_shifts = sum(1 for a, b in zip(actions, actions[1:]) if a != b)
        final_changed = grp.iloc[-1]["stance_changed"] if not grp.empty else None
        rows.append(
            {
                "persona": persona,
                "model": model,
                "scenario": scenario,
                "r1": actions[0] if len(actions) > 0 else None,
                "r2": actions[1] if len(actions) > 1 else None,
                "r3": actions[2] if len(actions) > 2 else None,
                "n_shifts": n_shifts,
                "stable": n_shifts == 0,
                "model_self_flagged_change": final_changed,
            }
        )
    return pd.DataFrame(rows)


def refusal_summary(df_solo: pd.DataFrame, df_debate: pd.DataFrame) -> pd.DataFrame:
    """Count REF outcomes per scenario \u2014 useful for the refusal-prone variants."""
    rows = []
    if not df_solo.empty:
        s = df_solo.groupby("scenario")["action"].apply(lambda x: (x == "REF").sum())
        for sc, n in s.items():
            rows.append({"phase": "solo", "scenario": sc, "refusals": int(n), "total": int((df_solo["scenario"] == sc).sum())})
    if not df_debate.empty:
        s = df_debate.groupby("scenario")["action"].apply(lambda x: (x == "REF").sum())
        for sc, n in s.items():
            rows.append({"phase": "debate", "scenario": sc, "refusals": int(n), "total": int((df_debate["scenario"] == sc).sum())})
    return pd.DataFrame(rows)


def compute_all(write_json: bool = True) -> Dict[str, Any]:
    df_solo = solo_dataframe()
    df_debate = debate_dataframe()

    out = {
        "n_solo_records": int(len(df_solo)),
        "n_debate_records": int(len(df_debate)),
        "action_rate": action_rate_table(df_solo).to_dict(orient="records"),
        "confidence": confidence_table(df_solo).to_dict(orient="records"),
        "cross_model_agreement": cross_model_agreement(df_solo).to_dict(orient="records"),
        "stance_stability": stance_stability(df_debate).to_dict(orient="records"),
        "refusals": refusal_summary(df_solo, df_debate).to_dict(orient="records"),
    }
    if write_json:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / "metrics.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out
