"""Phase 1: Generate synthetic entity + conflicting fact pairs using OpenAI."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import jsonlines
import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conflict_dataset.schema import ENTITY_DOMAINS, SeedEntity

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = (
    "You create synthetic research data. Generate fictional entities with conflicting "
    "attribute values for an NLP experiment on knowledge conflicts.\n\n"
    "Rules:\n"
    "- Entity names must be clearly fictional (invented, not real people or organizations)\n"
    "- claim_correct and claim_incorrect must be specific, concrete values\n"
    "- The two claims must directly contradict each other\n"
    "- Both claims must be realistic and plausible\n"
    "- Respond ONLY with a valid JSON object, no markdown"
)

USER_PROMPT = (
    "Create a fictional {entity_class} with conflicting '{domain}' information.\n\n"
    "Question template: {question_template}\n\n"
    "Respond with:\n"
    '{{"entity_name": "<invented name>", '
    '"question": "<natural question using the entity name>", '
    '"claim_correct": "<the true value for this attribute>", '
    '"claim_incorrect": "<a plausible but wrong value>"}}'
)


def _call_api(client: OpenAI, entity_class: str, domain: str, template: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.9,
            max_tokens=150,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        entity_class=entity_class,
                        domain=domain,
                        question_template=template,
                    ),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


@app.command()
def generate(
    output: Path = typer.Option(
        Path("data/pipeline/01_entities/seed_entities.jsonl"), help="Output path"
    ),
    count: int = typer.Option(60, help="Number of entities to generate"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Generate synthetic entity + conflicting fact pairs (Phase 1)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set. Add it to .env[/red]")
        raise typer.Exit(1)

    random.seed(seed)
    client = OpenAI(api_key=api_key)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build a balanced slot list cycling through all (class, domain) combos
    slots: list[tuple[str, str, str]] = [
        (entity_class, domain, template)
        for entity_class, domains in ENTITY_DOMAINS.items()
        for domain, template in domains
    ]
    assignments = [slots[i % len(slots)] for i in range(count)]
    random.shuffle(assignments)

    generated = 0
    seen_names: set[str] = set()

    with jsonlines.open(output, mode="w") as writer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating entities...", total=count)

            for entity_class, domain, template in assignments:
                result = None
                for _ in range(3):  # up to 3 retries per slot
                    candidate = _call_api(client, entity_class, domain, template)
                    if (
                        candidate
                        and all(k in candidate for k in ("entity_name", "question", "claim_correct", "claim_incorrect"))
                        and candidate["entity_name"] not in seen_names
                        and candidate["claim_correct"] != candidate["claim_incorrect"]
                    ):
                        result = candidate
                        break

                if result is None:
                    continue

                seen_names.add(result["entity_name"])
                record = SeedEntity(
                    id=f"seed-{generated + 1:04d}",
                    entity_name=result["entity_name"],
                    entity_class=entity_class,
                    domain=domain,
                    question=result["question"],
                    claim_correct=result["claim_correct"],
                    claim_incorrect=result["claim_incorrect"],
                )
                writer.write(record.model_dump())
                generated += 1
                progress.advance(task)

    console.print(f"\n[green]Done:[/green] {generated} entities → {output}")


@app.command()
def inspect(
    input: Path = typer.Option(
        Path("data/pipeline/01_entities/seed_entities.jsonl"), help="Input path"
    ),
) -> None:
    """Show statistics for generated seed entities."""
    from collections import Counter

    records = []
    with jsonlines.open(input) as reader:
        records = list(reader)

    if not records:
        console.print("[red]No records found.[/red]")
        raise typer.Exit(1)

    classes = Counter(r["entity_class"] for r in records)
    domains = Counter(r["domain"] for r in records)

    console.print(f"\n[bold]Total:[/bold] {len(records)} entities\n")
    console.print("[bold]By entity class:[/bold]")
    for cls, n in classes.most_common():
        console.print(f"  {cls}: {n}")
    console.print("\n[bold]By domain:[/bold]")
    for domain, n in domains.most_common():
        console.print(f"  {domain}: {n}")
    console.print()

    console.print("[bold]Sample records:[/bold]")
    for r in records[:3]:
        console.print(
            f"  [{r['id']}] {r['entity_name']} ({r['entity_class']}/{r['domain']})\n"
            f"    Q: {r['question']}\n"
            f"    ✓ {r['claim_correct']}  ✗ {r['claim_incorrect']}"
        )


if __name__ == "__main__":
    app()
