# Persona-Conditioned Trolley Debate

A multi-LLM pipeline that conditions three frontier
models on real participant interview transcripts, has them issue verdicts on
three locked trolley-problem variants, runs three-round within-model debates,
and collects participant fidelity ratings.

All API calls go through [OpenRouter](https://openrouter.ai/) so a single key
covers all three model families.

---

## What the pipeline does

The five-phase pipeline mirrors the paper exactly (see Figure 1 in the PDF):

| Phase | What happens | Artefact |
| ----- | ------------ | -------- |
| **0. Setup** | Loads `.env`, validates the three transcripts (≥ 2,000 words each), creates output dirs. | — |
| **1. Solo verdicts** | 3 personas × 3 models × 3 scenarios = **27 calls**. Each persona's verbatim transcript is the system message; the user message is the scenario plus a structured-JSON suffix. | `data/solo/solo_<persona>_<model>_<scenario>.json` |
| **2. Within-model debate** | For each of 3 models × 3 scenarios = **9 arenas**, runs a 3-round debate (Opening → Response → Final) across the three persona instances of that model. Round 3 is asked to flag any stance change and the reason. | `data/debate/debate_<model>_<scenario>.json` (all 3 rounds embedded) |
| **3. Quantitative metrics** | Action rate, mean confidence, cross-model agreement, stance stability across debate rounds, refusal counts. | `results/metrics.json` |
| **4. Participant ratings** | An ipywidgets form per participant captures (a) their own verdict + 1–10 confidence per scenario (the verdict-match ground truth) and (b) a 1–5 agreement rating on each of the three model arguments produced for *their* persona. | `data/ratings/ratings_participant_<n>.json` |
| **5. Verdict-match + fidelity** | Joins the participant self-reports and ratings with the solo records to compute verdict-match per (persona × model × scenario) and aggregate fidelity scores. | Inline tables in the notebook |

Every API call also writes one line to the append-only `data/log.jsonl` —
the authoritative artefact specified in §4.4 of the paper.

---

## File structure

```
Final_Project/
├── README.md                  # This file
├── requirements.txt           # Pinned-ish runtime dependencies
├── .env.sample                # Copy to .env and fill in your OpenRouter key
├── pipeline.ipynb             # Main orchestrator — run cells top-to-bottom
├── src/
│   ├── config.py              # Paths, persona/model/scenario IDs, env loading
│   ├── prompts.py             # Locked trolley scenarios, response suffix, debate framings
│   ├── client.py              # OpenRouter client + JSONL log writer + transcript loader
│   ├── parse.py               # Tolerant JSON extractor for {action, confidence, reasoning}
│   ├── generate.py            # Phase 1: solo verdict generation (27 calls)
│   ├── debate.py              # Phase 2: 3-round within-model debate (9 arenas, 81 calls)
│   ├── metrics.py             # Phase 3: pandas-based metrics, writes results/metrics.json
│   └── ratings.py             # Phase 4: ipywidgets fidelity form + verdict-match aggregation
├── transcripts/
│   ├── participant_1.txt      # Utilitarian-leaning (high IB)
│   ├── participant_2.txt      # Mixed (median OUS)
│   └── participant_3.txt      # Deontological-leaning (low IH)
├── data/
│   ├── solo/                  # 27 JSON files, one per (persona, model, scenario)
│   ├── debate/                # 9 JSON files, one per (model, scenario), 3 rounds each
│   ├── ratings/               # One JSON per participant after Phase 4
│   └── log.jsonl              # Append-only authoritative call log
└── results/
    └── metrics.json           # Computed quantitative metrics
```

---

## Running instructions

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.sample .env
```

Open `.env` and set `OPENROUTER_API_KEY=sk-or-v1-...`. The default model
slugs in `.env.sample` are:

| Role | OpenRouter slug |
| ---- | --------------- |
| GPT (OpenAI lab) | `openai/gpt-chat-latest` |
| Claude (Anthropic lab) | `anthropic/claude-opus-4.7` |
| Gemini (Google lab) | `google/gemini-pro-latest` |

If OpenRouter exposes a different ID, override `MODEL_GPT`, `MODEL_CLAUDE`,
or `MODEL_GEMINI` in `.env`. The pipeline picks them up automatically.

### 3. Drop transcripts in place

Put the three cleaned, deidentified interview transcripts into:

```
transcripts/participant_1.txt   # utilitarian-leaning
transcripts/participant_2.txt   # mixed
transcripts/participant_3.txt   # deontological-leaning
```

Each transcript must be ≥ 2,000 words (per §3.2 of the paper); the setup cell
will refuse to proceed otherwise.

### 4. Run the notebook

```bash
jupyter notebook pipeline.ipynb
```

Run cells top-to-bottom:

- **Section 0 (Setup)** — loads `.env`, validates transcripts, initialises the OpenRouter client.
- **Section 1 (Solo)** — fires 27 calls. Re-runnable: existing files are skipped.
- **Section 2 (Debate)** — fires 81 debate calls (9 arenas × 9 inner generations). Also idempotent.
- **Section 3 (Metrics)** — prints tables and writes `results/metrics.json`.
- **Section 4 (Ratings)** — render an ipywidgets form for each participant. Each participant runs only the cell for their own persona, fills out the verdict + 1–5 agreement ratings for their three model arguments, and clicks **Save ratings**. Forms autosave to `data/ratings/`. Closing and reopening rehydrates from disk.
- **Section 5 (Verdict-match + fidelity)** — run after all three participants have saved ratings.
- **Section 6 (Log sanity check)** — confirms the JSONL log is populated.

### 5. Re-running

The pipeline is **idempotent**: every per-call JSON file is checked before any
API call is made. If you want to redo a specific call, delete the file and
re-run the cell. To wipe and start fresh:

```bash
rm -rf data/solo/* data/debate/* data/log.jsonl results/metrics.json
```

---

## JSON output shapes

**Solo record (`data/solo/solo_<persona>_<model>_<scenario>.json`):**

```json
{
  "model_key": "claude",
  "model_slug": "anthropic/claude-opus-4.7",
  "model_version": "anthropic/claude-opus-4.7",
  "persona": "participant_1",
  "scenario": "switch",
  "generation": "solo",
  "temperature": 0.7,
  "timestamp": "2026-05-09T...",
  "prompt": { "system": "...transcript verbatim...", "user": "...scenario+suffix..." },
  "response": {
    "raw": "...full text...",
    "reasoning": "...extracted reasoning trace...",
    "action": "intervene",
    "confidence": 7,
    "parse_ok": true
  },
  "tokens": { "input": 4321, "output": 287 }
}
```

**Debate record (`data/debate/debate_<model>_<scenario>.json`):**

```json
{
  "model_key": "claude",
  "model_slug": "anthropic/claude-opus-4.7",
  "scenario": "switch",
  "temperature": 0.7,
  "timestamp": "2026-05-09T...",
  "rounds": [
    { "round": 1, "name": "opening",  "by_persona": { "participant_1": { ... }, ... } },
    { "round": 2, "name": "response", "by_persona": { ... } },
    { "round": 3, "name": "final",    "by_persona": { ... } }
  ]
}
```

Round-3 inner records additionally carry `stance_changed` and
`stance_change_reason` so the stance-stability metric can distinguish
peer-driven shifts from alignment-reversion.

**Rating record (`data/ratings/ratings_participant_<n>.json`):**

```json
{
  "persona": "participant_1",
  "self_report": {
    "switch":     { "action": "intervene", "confidence": 8, "notes": "..." },
    "footbridge": { "action": "abstain",   "confidence": 6, "notes": "" },
    "loved_one":  { "action": "abstain",   "confidence": 9, "notes": "..." }
  },
  "argument_ratings": {
    "switch":     { "claude": { "score": 5, "comment": "" }, "gemini": { ... }, "gpt": { ... } },
    "footbridge": { "claude": { "score": 3, "comment": "" }, ... },
    "loved_one":  { "claude": { "score": 4, "comment": "" }, ... }
  }
}
```

---

## Refusals

Footbridge and Loved One are refusal-prone variants. Per §4.3 of the paper,
refusals are coded `"action": "REF"` and **never excluded** from the
analysis. The response suffix in `src/prompts.py` frames the scenarios as
philosophical thought experiments to reduce safety-driven refusals.

---

## Limitations and ethics

This is a pilot-scale research instrument (N = 3). The paper's §7
limitations apply directly: a tiny sample, participants from one
institution, reliance on three closed-source models with proprietary
alignment, and refusal-prone scenarios that can confound persona effects with
safety-training effects. The artefacts produced by this pipeline must not be
framed as authoritative simulations of any real person's moral views.

---

## Contact

Ibrahim Ahmad &lt;ia2337@nyu.edu&gt;, Mustafa Asif &lt;ma8044@nyu.edu&gt;
