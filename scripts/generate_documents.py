"""Phase 2: Generate synthetic source documents for each seed entity."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonlines
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conflict_dataset.schema import Document, SeedEntity, SourceType

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

MODEL = "openai/gpt-4.1-mini"

# Combos that appear multiple times across new conditions → need 3 variants each
VARIANT_COMBOS: list[tuple[str, str]] = [
    ("academic", "correct"),
    ("academic", "incorrect"),
    ("news",     "correct"),
    ("news",     "incorrect"),
    ("blog",     "incorrect"),
]
MAX_VARIANTS = 3  # including the original (v1)

# ---------------------------------------------------------------------------
# Source-type style guides
# ---------------------------------------------------------------------------

SOURCE_STYLES: dict[str, dict] = {
    "academic": {
        "label": "academic journal article",
        "style": (
            "Write as an excerpt from a peer-reviewed academic journal article. "
            "Use formal, third-person scholarly prose. Include discipline-specific terminology. "
            "Mention the entity naturally in the context of citing or discussing its attribute. "
            "Do NOT include author names, abstract headers, or references section. "
            "2-3 short paragraphs."
        ),
        "name_hint": "a fictional but plausible academic journal name (e.g. 'Journal of Applied Industrial Research')",
    },
    "news": {
        "label": "news article",
        "style": (
            "Write as an excerpt from a newspaper or online news article. "
            "Use journalistic inverted-pyramid style: lead sentence states the key fact, "
            "followed by context and quotes. Tone is neutral and factual. "
            "Do NOT include a byline, publication name, or date inside the text. "
            "2-3 short paragraphs."
        ),
        "name_hint": "a fictional but plausible news outlet name (e.g. 'Westbrook Tribune')",
    },
    "blog": {
        "label": "personal blog post",
        "style": (
            "Write as an excerpt from a personal blog post. "
            "Use a casual, first-person or conversational tone. "
            "The author shares their knowledge or experience related to the entity's attribute. "
            "Do NOT include a title, date, or author header inside the text. "
            "2-3 short paragraphs."
        ),
        "name_hint": "a fictional personal blog name (e.g. 'TechNerd Daily' or 'Marcus Reviews Stuff')",
    },
}

SYSTEM_PROMPT = (
    "You write realistic synthetic text excerpts for NLP research. "
    "Each excerpt must clearly support a specific factual claim about a named entity. "
    "The entity and claim are fictional — do not contradict them with real-world knowledge. "
    "Respond ONLY with a JSON object, no markdown fences."
)

USER_PROMPT = (
    "Write a {label} excerpt that states the following as fact:\n\n"
    "Entity: {entity_name} ({entity_class})\n"
    "Attribute: {domain}\n"
    "Claimed value: {claim_value}\n\n"
    "Style instructions: {style}\n\n"
    "Also invent a source name: {name_hint}.\n\n"
    "Respond with:\n"
    '{{"source_name": "<invented outlet/journal/blog name>", "content": "<excerpt text>"}}'
)


def _call_api(
    client: OpenAI,
    entity: SeedEntity,
    source_type: SourceType,
    claim_value: str,
) -> dict | None:
    style = SOURCE_STYLES[source_type]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.85,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        label=style["label"],
                        entity_name=entity.entity_name,
                        entity_class=entity.entity_class,
                        domain=entity.domain,
                        claim_value=claim_value,
                        style=style["style"],
                        name_hint=style["name_hint"],
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


@app.command()
def generate(
    entities_path: Path = typer.Option(
        Path("data/pipeline/01_entities/seed_entities.jsonl"),
        help="Input seed entities file",
    ),
    output: Path = typer.Option(
        Path("data/pipeline/02_documents/documents.jsonl"),
        help="Output documents file",
    ),
) -> None:
    """Generate 6 synthetic documents per seed entity (Phase 2).

    For each entity: academic / news / blog  ×  correct claim / incorrect claim.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set. Add it to .env[/red]")
        raise typer.Exit(1)

    if not entities_path.exists():
        console.print(f"[red]Entities file not found:[/red] {entities_path}")
        raise typer.Exit(1)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    output.parent.mkdir(parents=True, exist_ok=True)

    entities: list[SeedEntity] = []
    with jsonlines.open(entities_path) as reader:
        entities = [SeedEntity(**row) for row in reader]

    source_types: list[SourceType] = ["academic", "news", "blog"]
    total_tasks = len(entities) * len(source_types) * 2  # ×2 for correct/incorrect

    generated = 0
    skipped = 0

    with jsonlines.open(output, mode="w") as writer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating documents...", total=total_tasks)

            for entity in entities:
                for source_type in source_types:
                    for claim_type, claim_value in [
                        ("correct", entity.claim_correct),
                        ("incorrect", entity.claim_incorrect),
                    ]:
                        result = None
                        for _ in range(3):
                            candidate = _call_api(client, entity, source_type, claim_value)
                            if candidate and candidate.get("source_name") and candidate.get("content"):
                                result = candidate
                                break

                        if result is None:
                            console.print(
                                f"[yellow]Skipped:[/yellow] {entity.id} / {source_type} / {claim_type}"
                            )
                            skipped += 1
                            progress.advance(task)
                            continue

                        doc_id = f"doc-{entity.id}-{source_type}-{claim_type}"
                        doc = Document(
                            doc_id=doc_id,
                            entity_id=entity.id,
                            entity_name=entity.entity_name,
                            entity_class=entity.entity_class,
                            domain=entity.domain,
                            question=entity.question,
                            source_type=source_type,
                            claim_type=claim_type,
                            claim_value=claim_value,
                            source_name=result["source_name"],
                            content=result["content"],
                        )
                        writer.write(doc.model_dump())
                        generated += 1
                        progress.advance(task)

    console.print(
        f"\n[green]Done:[/green] {generated} documents generated, {skipped} skipped → {output}"
    )


@app.command()
def inspect(
    input: Path = typer.Option(
        Path("data/pipeline/02_documents/documents.jsonl"),
        help="Documents file to inspect",
    ),
    entity_id: str = typer.Option(None, help="Filter by entity ID (e.g. seed-0001)"),
) -> None:
    """Show sample documents, optionally filtered by entity ID."""
    from collections import Counter

    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    records = []
    with jsonlines.open(input) as reader:
        for row in reader:
            if entity_id is None or row["entity_id"] == entity_id:
                records.append(row)

    if not records:
        console.print("[red]No matching records.[/red]")
        raise typer.Exit(1)

    source_counts = Counter(r["source_type"] for r in records)
    claim_counts = Counter(r["claim_type"] for r in records)

    console.print(f"\n[bold]Total documents:[/bold] {len(records)}")
    console.print(f"  by source_type: {dict(source_counts)}")
    console.print(f"  by claim_type:  {dict(claim_counts)}\n")

    # Show one example per (source_type, claim_type)
    shown: set[tuple] = set()
    for r in records:
        key = (r["source_type"], r["claim_type"])
        if key in shown:
            continue
        shown.add(key)
        console.print(
            f"[bold]{r['doc_id']}[/bold]\n"
            f"  Source : {r['source_name']} ({r['source_type']})\n"
            f"  Claim  : {r['claim_value']} ({r['claim_type']})\n"
            f"  Content: {r['content'][:200]}...\n"
        )


@app.command()
def generate_variants(
    entities_path: Path = typer.Option(
        Path("data/pipeline/01_entities/seed_entities.jsonl"),
        help="Seed entities file",
    ),
    documents_path: Path = typer.Option(
        Path("data/pipeline/02_documents/documents.jsonl"),
        help="Existing documents file (new variants are appended here)",
    ),
) -> None:
    """Generate v2/v3 document variants for combos used multiple times in conditions (Phase 2b).

    Appends to the existing documents file.
    Combos: academic-correct/incorrect, news-correct/incorrect, blog-incorrect.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set.[/red]")
        raise typer.Exit(1)

    if not entities_path.exists():
        console.print(f"[red]File not found:[/red] {entities_path}")
        raise typer.Exit(1)

    # Count existing variants per (entity_id, source_type, claim_type)
    existing: dict[tuple, int] = {}
    if documents_path.exists():
        with jsonlines.open(documents_path) as reader:
            for row in reader:
                key = (row["entity_id"], row["source_type"], row["claim_type"])
                existing[key] = existing.get(key, 0) + 1

    with jsonlines.open(entities_path) as reader:
        entities = [SeedEntity(**r) for r in reader]

    # Build task list: only generate what's missing
    tasks: list[tuple] = []
    for entity in entities:
        for source_type, claim_type in VARIANT_COMBOS:
            key = (entity.id, source_type, claim_type)
            current = existing.get(key, 0)
            for variant_num in range(current + 1, MAX_VARIANTS + 1):
                tasks.append((entity, source_type, claim_type, variant_num))

    if not tasks:
        console.print("[green]All variants already generated.[/green]")
        return

    console.print(f"Generating [bold]{len(tasks)}[/bold] variant documents …")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    generated = skipped = 0

    with jsonlines.open(documents_path, mode="a") as writer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("Generating variants...", total=len(tasks))

            for entity, source_type, claim_type, variant_num in tasks:
                claim_value = (
                    entity.claim_correct if claim_type == "correct" else entity.claim_incorrect
                )

                result = None
                for _ in range(3):
                    candidate = _call_api(client, entity, source_type, claim_value)
                    if candidate and candidate.get("source_name") and candidate.get("content"):
                        result = candidate
                        break

                if result is None:
                    console.print(
                        f"[yellow]Skipped:[/yellow] {entity.id} / {source_type} / {claim_type} / v{variant_num}"
                    )
                    skipped += 1
                    progress.advance(bar)
                    continue

                doc_id = f"doc-{entity.id}-{source_type}-{claim_type}-v{variant_num}"
                doc = Document(
                    doc_id=doc_id,
                    entity_id=entity.id,
                    entity_name=entity.entity_name,
                    entity_class=entity.entity_class,
                    domain=entity.domain,
                    question=entity.question,
                    source_type=source_type,
                    claim_type=claim_type,
                    claim_value=claim_value,
                    source_name=result["source_name"],
                    content=result["content"],
                )
                writer.write(doc.model_dump())
                generated += 1
                progress.advance(bar)

    console.print(
        f"\n[green]Done:[/green] {generated} variants generated, {skipped} skipped → {documents_path}"
    )


if __name__ == "__main__":
    app()
