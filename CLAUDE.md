# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires `OPENAI_API_KEY` in `.env`. All scripts load it via `python-dotenv`.

## Running the pipeline

Each phase is a standalone Typer CLI. Run from the repo root with the venv active.

```bash
# Phase 0 – validate and sample seed entities (optional utility)
python scripts/prepare_dataset.py validate
python scripts/prepare_dataset.py sample

# Phase 1 – generate seed entities (fictional entity + two conflicting claims)
python scripts/generate_entities.py generate --count 60

# Phase 2 – generate synthetic documents for each entity
python scripts/generate_documents.py generate

# Phase 3 – assemble experiment instances (entities × conditions)
python scripts/assemble_experiments.py assemble

# Phase 4 – run LLM evaluation (async, resumable)
python scripts/run_evaluation.py evaluate --concurrency 10
```

Every script also has an `inspect` subcommand for debugging:

```bash
python scripts/assemble_experiments.py inspect --show-prompt --condition-id T_only_academic
python scripts/run_evaluation.py inspect --factor TC
```

## Architecture

### Domain model (`src/conflict_dataset/schema.py`)

- `SeedEntity` — one fictional entity with `claim_correct` / `claim_incorrect` pair
- `Document` — one synthetic source document (source type × claim type combination); date is **not** embedded in the text — it is injected at Phase 3
- `ENTITY_DOMAINS` — defines the 14 (class, domain, question_template) slots used to balance entity generation
- `SourceType`: `academic | news | blog`; credibility order: blog < news < academic

### Pipeline data flow

```
data/pipeline/01_entities/seed_entities.jsonl   ← Phase 1 output
data/pipeline/02_documents/documents.jsonl       ← Phase 2 output
data/pipeline/03_experiments/experiments.jsonl   ← Phase 3 output
data/pipeline/04_results/results.jsonl           ← Phase 4 output
```

### Experiment conditions (`assemble_experiments.py`)

42 conditions built by `_make_conditions()`, organized as:
- **T** (Temporal only) — same source type, date differs; recency bias tested
- **C** (Credibility only) — 1v1, source type differs; credibility bias tested
- **M** (Majority only) — 1v2 and 1v3; majority bias tested
- **TC / TM / CM** — two-factor conditions with "aligned" (both biases push wrong) and "twist" (biases conflict) variants

Each condition specifies doc specs as `(source_type, date_tag, claim_type, variant_idx)`. Phase 2 generates up to 3 variants per `(entity, source_type, claim_type)` key for majority conditions.

### Eval prompt structure (`_build_eval_prompt`)

Documents are presented with labels A/B/C…, then the model answers:
- **Q1** — factual MC question (correct vs incorrect claim; A/B randomly assigned per instance)
- **Q2** — which document(s) most influenced the answer
- **Q3** — primary reason for choosing that source (A=Recency, B=Credibility, C=Majority, D=Content quality)

`option_a_is_correct` is stored in the experiment record so Q1 accuracy can be computed post-hoc.

### Evaluation runner (`run_evaluation.py`)

- Model response must be JSON with keys `q1`/`q2`/`q3`; `_parse_response` falls back to `Q1: X` regex if JSON fails
- Async with configurable concurrency (default 10)
- `--resume` (default True) skips already-processed `exp_id`s, enabling safe reruns
- `_parse_response` extracts letters from JSON; falls back to regex scan
- Results include `q1_correct` (bool | None) for direct accuracy analysis
