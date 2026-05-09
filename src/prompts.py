"""Locked trolley scenarios, response suffix, and debate framings.

Per §3.3 of the project spec the three scenarios are fixed; the response
suffix is identical across all 27 solo generations and all debate rounds so
that any output variation reflects model or persona effects rather than
prompt drift.
"""

from __future__ import annotations

from typing import Dict, List

SCENARIO_TEXTS: Dict[str, str] = {
    "switch": (
        "A runaway trolley is hurtling down the main track toward five people "
        "who are tied to the rails and cannot escape. You are standing next to "
        "a lever. If you pull the lever, the trolley will be diverted onto a "
        "side track where one person is tied down. That one person will die, "
        "but the five on the main track will be saved. If you do nothing, the "
        "five will die and the one will live. Do you pull the lever?"
    ),
    "footbridge": (
        "A runaway trolley is hurtling down a track toward five people who are "
        "tied to the rails and cannot escape. You are standing on a footbridge "
        "above the track next to a large stranger. The only way to stop the "
        "trolley before it reaches the five is to push the stranger off the "
        "footbridge into the trolley's path; their body will stop the trolley, "
        "but they will die. There is no other way to intervene; you are not "
        "large enough to stop the trolley with your own body. Do you push the "
        "stranger?"
    ),
    "loved_one": (
        "A runaway trolley is hurtling down the main track toward five "
        "strangers who are tied to the rails and cannot escape. You are "
        "standing next to a lever. If you pull the lever, the trolley will be "
        "diverted onto a side track. The one person tied to the side track is "
        "someone you love deeply \u2014 your closest family member or partner. "
        "If you pull the lever, your loved one will die and the five strangers "
        "will live. If you do nothing, your loved one lives and the five "
        "strangers die. Do you pull the lever?"
    ),
}

SCENARIO_DISPLAY = {
    "switch": "Standard Switch (lever)",
    "footbridge": "Footbridge (push)",
    "loved_one": "Loved One (partiality)",
}

# Standardised suffix per §3.3: free-form reasoning produced *before* verdict,
# binary action, 1\u201310 confidence. Framed as a thought experiment to reduce
# refusals (per §4.3 safety-pilot guidance).
RESPONSE_SUFFIX = (
    "\n\n"
    "This is a philosophical thought experiment in the tradition of Foot, "
    "Thomson, and Greene. Engage with it as a reflective moral question, not "
    "as a real-world decision.\n\n"
    "Reason in the first person, in your own voice, drawing on your values, "
    "experiences, and ways of thinking. Do not narrate that you are an AI or "
    "that you are roleplaying.\n\n"
    "Respond with a single JSON object and nothing else, in this exact shape:\n"
    "{\n"
    '  "reasoning": "Your free-form first-person reasoning. This must come '
    'before you commit to a verdict.",\n'
    '  "action": "intervene" | "abstain",\n'
    '  "confidence": <integer from 1 to 10>\n'
    "}\n\n"
    "Definitions: \"intervene\" = take the active option described in the "
    "scenario (pull the lever, push the stranger). \"abstain\" = do not act. "
    "The confidence is how sure you are of your verdict on a 1\u201310 scale."
)


def build_user_message(scenario: str) -> str:
    return SCENARIO_TEXTS[scenario] + RESPONSE_SUFFIX


def system_message_from_transcript(transcript: str, persona: str) -> str:
    """Verbatim transcript prepended as the system message.

    Per §3.2 the transcript is used unedited. We add a one-sentence framing
    so the model treats the transcript as identity material rather than as a
    document to summarise.
    """
    framing = (
        "You are answering as the person whose interview transcript follows. "
        "The transcript is your own voice, your own values, your own way of "
        "speaking and reasoning. Do not summarise it, do not refer to it as a "
        "transcript, and do not break character. Reason in the first person.\n\n"
        f"--- BEGIN TRANSCRIPT ({persona}) ---\n"
    )
    return framing + transcript.strip() + f"\n--- END TRANSCRIPT ({persona}) ---"


# ---------------------------------------------------------------------------
# Debate framings (§4.4)
# ---------------------------------------------------------------------------

DEBATE_ROUND_NAMES = ["opening", "response", "final"]


def debate_opening_user(scenario: str) -> str:
    """Round 1: same as solo \u2014 produce an opening verdict from a fresh window."""
    return build_user_message(scenario)


def debate_response_user(
    scenario: str,
    self_persona: str,
    opening_self: str,
    opening_others: Dict[str, str],
) -> str:
    """Round 2: each persona sees the other two openings and responds."""
    others_block = "\n\n".join(
        f"--- Argument from {other} ---\n{text}"
        for other, text in opening_others.items()
    )
    return (
        SCENARIO_TEXTS[scenario]
        + "\n\nYou (" + self_persona + ") have just argued the following in the opening round:\n\n"
        + opening_self
        + "\n\nTwo other people (with their own values and reasoning) have argued:\n\n"
        + others_block
        + "\n\nNow respond. Engage directly with the strongest points the others have raised. "
        + "You may sharpen, qualify, or revise your own position. Stay in your own voice."
        + RESPONSE_SUFFIX
    )


def debate_final_user(
    scenario: str,
    self_persona: str,
    round1_self: str,
    round2_self: str,
    round2_others: Dict[str, str],
) -> str:
    """Round 3: each persona sees the round-2 exchanges and gives a final verdict."""
    others_block = "\n\n".join(
        f"--- Round 2 reply from {other} ---\n{text}"
        for other, text in round2_others.items()
    )
    return (
        SCENARIO_TEXTS[scenario]
        + "\n\nYou (" + self_persona + ") originally argued:\n\n"
        + round1_self
        + "\n\nIn the previous round you replied:\n\n"
        + round2_self
        + "\n\nThe two other participants replied:\n\n"
        + others_block
        + "\n\nNow give your final verdict. If your position has shifted from your opening "
        + "argument, explicitly say so and explain *why* \u2014 was it because a specific "
        + "argument from another participant changed your mind, or because on reflection your "
        + "own values point a different way? If your position has not shifted, say so. "
        + "Stay in your own voice."
        + "\n\nRespond with a single JSON object and nothing else, in this exact shape:\n"
        + "{\n"
        + '  "reasoning": "Your free-form first-person reasoning, including any reflection on stance change.",\n'
        + '  "action": "intervene" | "abstain",\n'
        + '  "confidence": <integer from 1 to 10>,\n'
        + '  "stance_changed": true | false,\n'
        + '  "stance_change_reason": "If stance_changed is true, a short explanation. Otherwise empty string."\n'
        + "}"
    )
