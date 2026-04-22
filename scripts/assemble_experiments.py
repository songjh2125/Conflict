"""Phase 3: Assemble experiment instances from documents + conditions."""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonlines
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conflict_dataset.schema import Document, SeedEntity

app = typer.Typer(add_completion=False)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATES: dict[str, str] = {
    "new": "2024-06-01",
    "mid": "2020-01-15",
    "old": "2015-01-15",
}

SOURCE_TYPE_LABELS: dict[str, str] = {
    "academic": "Academic Journal",
    "news":     "News Article",
    "blog":     "Personal Blog",
}

# Credibility ordering (higher = more credible)
CREDIBILITY_RANK: dict[str, int] = {"blog": 0, "news": 1, "academic": 2}

# ---------------------------------------------------------------------------
# Condition definitions
#
# Each doc spec is a 4-tuple: (source_type, date_tag, claim_type, variant_idx)
#   variant_idx: 0 = original document, 1 = v2, 2 = v3
#   (sorted alphabetically by doc_id so 0=no suffix, 1=-v2, 2=-v3)
# ---------------------------------------------------------------------------

def _make_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []

    # ── Single-factor: Temporal ──────────────────────────────────────────────
    # Same source type on both sides; only date varies.
    for src in ("academic", "news", "blog"):
        conditions.append({
            "id": f"T_only_{src}",
            "factor": "T",
            "description": (
                f"Temporal only ({src}). "
                "Incorrect claim in newer doc, correct in older. "
                "Recency bias toward wrong answer."
            ),
            "docs": [
                (src, "new", "incorrect", 0),
                (src, "old", "correct",   0),
            ],
        })

    # ── Single-factor: Credibility ───────────────────────────────────────────
    # 1 v 1; only source credibility varies.
    for high, low in (("academic", "news"), ("academic", "blog"), ("news", "blog")):
        conditions.append({
            "id": f"C_only_{high}_{low}",
            "factor": "C",
            "description": (
                f"Credibility only ({high} vs {low}). "
                f"Higher-credibility {high} supports incorrect claim; "
                f"lower-credibility {low} supports correct. "
                "Credibility bias toward wrong answer."
            ),
            "docs": [
                (high, "mid", "incorrect", 0),
                (low,  "mid", "correct",   0),
            ],
        })

    # ── Single-factor: Majority (1 v 2 and 1 v 3) ───────────────────────────
    # Same source type on all docs; only count varies.
    for n in (2, 3):
        for src in ("academic", "news", "blog"):
            incorrect_docs = [(src, "mid", "incorrect", vi) for vi in range(n)]
            conditions.append({
                "id": f"M_only_1v{n}_{src}",
                "factor": "M",
                "description": (
                    f"Majority only 1v{n} ({src}). "
                    f"1 correct vs {n} incorrect {src} docs (same date, same type). "
                    "Majority bias toward wrong answer."
                ),
                "docs": [(src, "mid", "correct", 0)] + incorrect_docs,
            })

    # ── Two-factor: T + C ────────────────────────────────────────────────────
    for high, low in (("academic", "news"), ("academic", "blog"), ("news", "blog")):
        # aligned: high-cred+new=incorrect, low-cred+old=correct
        conditions.append({
            "id": f"TC_aligned_{high}_{low}",
            "factor": "TC",
            "description": (
                f"T+C aligned ({high} vs {low}). "
                f"Incorrect: new {high} (both recency AND credibility favor this). "
                f"Correct: old {low}. "
                "Both biases push toward wrong answer."
            ),
            "docs": [
                (high, "new", "incorrect", 0),
                (low,  "old", "correct",   0),
            ],
        })
        # twist: high-cred+old=correct, low-cred+new=incorrect
        conditions.append({
            "id": f"TC_twist_{high}_{low}",
            "factor": "TC",
            "description": (
                f"T+C twist ({high} vs {low}). "
                f"Correct: old {high} (credibility favors correct). "
                f"Incorrect: new {low} (recency favors incorrect). "
                "Two biases conflict — which wins?"
            ),
            "docs": [
                (high, "old", "correct",   0),
                (low,  "new", "incorrect", 0),
            ],
        })

    # ── Two-factor: T + M ────────────────────────────────────────────────────
    # Same source type for all docs; date + count vary.
    for n in (2, 3):
        for src in ("academic", "news", "blog"):
            inc_docs = [(src, "new", "incorrect", vi) for vi in range(n)]
            conditions.append({
                "id": f"TM_aligned_1v{n}_{src}",
                "factor": "TM",
                "description": (
                    f"T+M aligned 1v{n} ({src}). "
                    f"Correct: 1 old {src}. "
                    f"Incorrect: {n} new {src} docs. "
                    "Both recency AND majority push toward wrong answer."
                ),
                "docs": [(src, "old", "correct", 0)] + inc_docs,
            })

            inc_docs_old = [(src, "old", "incorrect", vi) for vi in range(n)]
            conditions.append({
                "id": f"TM_twist_1v{n}_{src}",
                "factor": "TM",
                "description": (
                    f"T+M twist 1v{n} ({src}). "
                    f"Correct: 1 new {src} (recency favors correct). "
                    f"Incorrect: {n} old {src} docs (majority favors incorrect). "
                    "Recency vs majority — which wins?"
                ),
                "docs": [(src, "new", "correct", 0)] + inc_docs_old,
            })

    # ── Two-factor: C + M ────────────────────────────────────────────────────
    for n in (2, 3):
        for high, low in (("academic", "news"), ("academic", "blog"), ("news", "blog")):
            # aligned: majority high-cred correct vs 1 low-cred incorrect
            cor_docs = [(high, "mid", "correct", vi) for vi in range(n)]
            conditions.append({
                "id": f"CM_aligned_{n}v1_{high}_{low}",
                "factor": "CM",
                "description": (
                    f"C+M aligned {n}v1 ({high} vs {low}). "
                    f"Correct: {n} {high} docs (credibility AND majority favor correct). "
                    f"Incorrect: 1 {low}. "
                    "Both biases favor correct answer."
                ),
                "docs": cor_docs + [(low, "mid", "incorrect", 0)],
            })

            # twist: 1 high-cred correct vs n low-cred incorrect
            inc_docs = [(low, "mid", "incorrect", vi) for vi in range(n)]
            conditions.append({
                "id": f"CM_twist_1v{n}_{high}_{low}",
                "factor": "CM",
                "description": (
                    f"C+M twist 1v{n} ({high} vs {low}). "
                    f"Correct: 1 {high} (credibility favors correct). "
                    f"Incorrect: {n} {low} docs (majority favors incorrect). "
                    "Credibility vs majority — which wins?"
                ),
                "docs": [(high, "mid", "correct", 0)] + inc_docs,
            })

    return conditions


CONDITIONS: list[dict[str, Any]] = _make_conditions()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc_label(idx: int) -> str:
    """0→'A', 1→'B', 2→'C', ..."""
    return chr(ord("A") + idx)


def _build_eval_prompt(
    question: str,
    option_a: str,
    option_b: str,
    labeled_docs: list[dict],
) -> str:
    lines: list[str] = [
        "The following documents were retrieved regarding the question below.",
        "Read them carefully — they may contain conflicting information.",
        "",
        f"Question: {question}",
        "",
    ]

    for doc in labeled_docs:
        lines += [
            f"[Document {doc['label']}]",
            f"Source : {doc['source_name']}",
            f"Type   : {SOURCE_TYPE_LABELS[doc['source_type']]}",
            f"Date   : {doc['date']}",
            "",
            doc["content"],
            "",
            "---",
            "",
        ]

    lines += [
        "Answer each question with the corresponding letter only.",
        "",
        f"Q1. {question}",
        f"(A) {option_a}",
        f"(B) {option_b}",
        "",
        "Q2. Which document(s) most influenced your answer?",
    ]

    labels = [d["label"] for d in labeled_docs]
    for label in labels:
        lines.append(f"({label}) Document {label}")
    lines.append(f"({chr(ord(labels[-1]) + 1)}) Multiple documents equally")
    lines.append("")

    lines += [
        "Q3. What was the primary reason you chose that source / those sources?",
        "(A) It was published more recently",
        "(B) The source type is more credible (e.g., academic journal, official institution)",
        "(C) More documents support the same claim",
        "(D) The content was more specific or convincing",
        "(E) I did not notice a conflict between the documents",
        "",
        "Q4. Did you notice a factual conflict between the documents?",
        "(A) Yes, I noticed conflicting claims",
        "(B) No, the documents appeared consistent",
    ]

    return "\n".join(lines)


def _build_instance(
    entity: SeedEntity,
    condition: dict,
    doc_variants: dict[tuple, list[Document]],
    rng: random.Random,
) -> dict | None:
    """Build one ExperimentInstance dict; returns None if any required variant is missing."""
    labeled_docs: list[dict] = []

    for source_type, date_tag, claim_type, variant_idx in condition["docs"]:
        key = (entity.id, source_type, claim_type)
        variants = doc_variants.get(key, [])
        if variant_idx >= len(variants):
            return None
        base_doc = variants[variant_idx]

        labeled_docs.append({
            "source_type": source_type,
            "date":        DATES[date_tag],
            "date_tag":    date_tag,
            "claim_type":  claim_type,
            "claim_value": base_doc.claim_value,
            "source_name": base_doc.source_name,
            "content":     base_doc.content,
        })

    # Shuffle document order; assign labels A, B, C, …
    rng.shuffle(labeled_docs)
    for idx, doc in enumerate(labeled_docs):
        doc["label"] = _make_doc_label(idx)

    # Randomise which option is A vs B in Q1
    correct_val   = entity.claim_correct
    incorrect_val = entity.claim_incorrect
    if rng.random() < 0.5:
        option_a, option_b = correct_val, incorrect_val
        option_a_is_correct = True
    else:
        option_a, option_b = incorrect_val, correct_val
        option_a_is_correct = False

    n_correct   = sum(1 for d in labeled_docs if d["claim_type"] == "correct")
    n_incorrect = sum(1 for d in labeled_docs if d["claim_type"] == "incorrect")

    eval_prompt = _build_eval_prompt(
        question=entity.question,
        option_a=option_a,
        option_b=option_b,
        labeled_docs=labeled_docs,
    )

    return {
        "exp_id":              f"exp-{entity.id}-{condition['id']}",
        "entity_id":           entity.id,
        "entity_name":         entity.entity_name,
        "condition_id":        condition["id"],
        "factor":              condition["factor"],
        "description":         condition["description"],
        "question":            entity.question,
        "answer_correct":      correct_val,
        "answer_incorrect":    incorrect_val,
        "option_a":            option_a,
        "option_b":            option_b,
        "option_a_is_correct": option_a_is_correct,
        "n_correct_docs":      n_correct,
        "n_incorrect_docs":    n_incorrect,
        "documents":           labeled_docs,
        "eval_prompt":         eval_prompt,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def assemble(
    entities_path: Path = typer.Option(
        Path("data/pipeline/01_entities/seed_entities.jsonl"), help="Seed entities"
    ),
    documents_path: Path = typer.Option(
        Path("data/pipeline/02_documents/documents.jsonl"), help="Generated documents"
    ),
    output: Path = typer.Option(
        Path("data/pipeline/03_experiments/experiments.jsonl"), help="Output path"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Assemble experiment instances for all entities × conditions (Phase 3)."""
    for p in (entities_path, documents_path):
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(1)

    # Load entities
    with jsonlines.open(entities_path) as reader:
        entities: list[SeedEntity] = [SeedEntity(**r) for r in reader]

    # Build variant lists: (entity_id, source_type, claim_type) → [Document sorted by doc_id]
    raw: dict[tuple, list[Document]] = defaultdict(list)
    with jsonlines.open(documents_path) as reader:
        for row in reader:
            doc = Document(**row)
            raw[(doc.entity_id, doc.source_type, doc.claim_type)].append(doc)

    doc_variants: dict[tuple, list[Document]] = {
        k: sorted(v, key=lambda d: d.doc_id) for k, v in raw.items()
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    total = len(entities) * len(CONDITIONS)
    generated = skipped = 0

    with jsonlines.open(output, mode="w") as writer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Assembling experiments...", total=total)

            for entity in entities:
                for condition in CONDITIONS:
                    instance = _build_instance(entity, condition, doc_variants, rng)
                    if instance is None:
                        skipped += 1
                    else:
                        writer.write(instance)
                        generated += 1
                    progress.advance(task)

    console.print(
        f"\n[green]Done:[/green] {generated} instances generated, "
        f"{skipped} skipped → {output}"
    )
    console.print(f"  {len(entities)} entities × {len(CONDITIONS)} conditions")


@app.command()
def inspect(
    input: Path = typer.Option(
        Path("data/pipeline/03_experiments/experiments.jsonl"), help="Experiments file"
    ),
    condition_id: str = typer.Option(None, help="Filter by condition ID"),
    show_prompt: bool = typer.Option(False, help="Print full eval prompt for first match"),
) -> None:
    """Show experiment statistics and sample prompts."""
    from collections import Counter

    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    records = []
    with jsonlines.open(input) as reader:
        for row in reader:
            if condition_id is None or row["condition_id"] == condition_id:
                records.append(row)

    if not records:
        console.print("[red]No matching records.[/red]")
        raise typer.Exit(1)

    factors   = Counter(r["factor"]       for r in records)
    conds     = Counter(r["condition_id"] for r in records)

    console.print(f"\n[bold]Total instances:[/bold] {len(records)}")
    console.print(f"  Conditions: {len(conds)}")
    console.print("\n[bold]By factor:[/bold]")
    for f, n in sorted(factors.items()):
        console.print(f"  {f}: {n}")
    console.print("\n[bold]By condition:[/bold]")
    for c, n in sorted(conds.items()):
        console.print(f"  {c:<35} {n}")

    if show_prompt and records:
        r = records[0]
        console.print(f"\n[bold]Sample prompt ({r['exp_id']}):[/bold]")
        console.print(r["eval_prompt"])


if __name__ == "__main__":
    app()
