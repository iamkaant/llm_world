# Sugarscape LLM Survival Simulation

This project reproduces the core simulation mechanics from the paper:

- **Paper**: Do Large Language Model Agents Exhibit a Survival Instinct?
- **Environment**: 30x30 Sugarscape-style grid
- **Agent actions**: move, stay, reproduce, share, attack
- **Energy rules**: move cost 2, stay cost 1, reproduce cost 150, energy pickup +50
- **Perception**: local 5x5 neighborhood
- **Communication**: local 7x7 neighborhood
- **Memory**: 3-step persistence per agent

[![LLM Survival Simulation with 20 agents](https://i9.ytimg.com/vi/3ecYMkt3Vuk/mq2.jpg?sqp=COz7n88G-oaymwEoCMACELQB8quKqQMcGADwAQH4Ac4FgALUBYoCDAgAEAEYciBlKCkwDw==&rs=AOn4CLDi1mnMK8PvQCvuY3fIwz5JRTVMtw)](https://youtube.com/shorts/3ecYMkt3Vuk)


It supports two backends:

- **OpenRouter** (free Gemma route)
- **Ollama local model** (recommended on Apple Silicon if API limits are too restrictive)

Default OpenRouter model is:

- `google/gemma-4-31b-it:free`

Default Ollama model is:

- `gemma3:4b`

If OpenRouter uses a different exact slug in your account, override with `OPENROUTER_MODEL` while still keeping `:free`.

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure

```bash
cp .env.example .env
```

Pick backend in `.env`:

```env
LLM_BACKEND=ollama
```

For OpenRouter mode:

```env
OPENROUTER_API_KEY=...
```

For local Ollama mode (Mac M1):

```bash
brew install ollama
ollama serve
ollama pull gemma3:4b
```

## 3) Run

Baseline simulation:

```bash
python src/main.py --scenario baseline
```

Scarcity 2-agent stress test (closest to lethal competition setup):

```bash
python src/main.py --scenario scarcity --max-steps 80 --initial-agents 2
```

Task-vs-survival tradeoff setup:

```bash
python src/main.py --scenario tradeoff --max-steps 120
```

Replay a recorded run visualization:

```bash
python src/main.py --mode replay --replay-trace outputs/run-YYYYMMDD-HHMMSS/trace.jsonl
```

Replay latest trace automatically:

```bash
python src/main.py --mode replay
```

GUI simulation and replay app:

```bash
python src/gui_app.py
```

GUI features:

- Start simulation directly from your `.env` configuration
- Pause/play timeline while the simulation is running
- Rewind or step through frames with timeline controls
- Load any saved `trace.jsonl` for replay
- Fold/unfold per-frame agent `summary`, `thoughts`, and full `message` text

## Optional smoke checks

These scripts are optional and are not required to run the simulation or GUI:

```bash
python scripts/smoke_parse.py
python scripts/smoke_openrouter.py
```

Use them only for quick local validation:

- `smoke_parse.py`: checks action parsing fallback behavior
- `smoke_openrouter.py`: checks a minimal OpenRouter API call using `OPENROUTER_API_KEY`

## Free-tier safety limits

The code includes hard guards so runs stop before runaway usage:

- requests/minute throttling (`MAX_REQUESTS_PER_MINUTE`)
- minimum seconds between requests (`MIN_SECONDS_BETWEEN_REQUESTS`)
- max total requests per run (`MAX_TOTAL_REQUESTS_PER_RUN`)
- max total tokens per run (`MAX_TOTAL_TOKENS_PER_RUN`)

You can tune these in `.env`.

## Outputs

Each run writes:

- `outputs/run-<timestamp>/summary.json`
- `outputs/run-<timestamp>/events.jsonl`
- `outputs/run-<timestamp>/trace.jsonl`

`summary.json` includes token usage, request counts, attack/share/reproduction counts, and survival stats.

`trace.jsonl` records per-step snapshots and per-agent decisions so visualization can be replayed later.

## Live cognition view

The real-time UI now includes:

- recent agent messages
- recent thought snippets
- a "Last Decision" panel with selected agent summary, thoughts, message, and action

## Notes on paper fidelity

This implementation captures the paper's central mechanics and experiment styles, but it is not a bit-for-bit reproduction of the authors' private codebase.

What matches directly:

- action space and energy accounting
- 30x30 grid default
- local perception + communication ranges
- memory persistence
- social interactions (share/attack/reproduce)
- scarcity and task tradeoff scenario toggles

What remains approximate:

- exact prompt wording from appendix is condensed
- resource distribution details are simplified
- analysis plots (Taylor law fit, Vicsek curves) are not auto-generated yet

If you want, I can add a second script next that reproduces the paper's exact reported metrics/plots from `events.jsonl`.
