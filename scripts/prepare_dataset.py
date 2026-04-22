"""Validate and inspect seed entities against the pipeline schema."""
from __future__ import annotations

import sys
from pathlib import Path

import jsonlines
import typer
from pydantic import ValidationError
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from conflict_dataset.schema import SeedEntity

app = typer.Typer(add_completion=False)
console = Console()

_DEFAULT = Path("data/pipeline/01_entities/seed_entities.jsonl")


@app.command()
def validate(
    input: Path = typer.Option(_DEFAULT, help="Seed entities file to validate"),
) -> None:
    """Validate seed_entities.jsonl against the SeedEntity schema."""
    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    ok, errors = 0, 0
    with jsonlines.open(input) as reader:
        for i, row in enumerate(reader, start=1):
            try:
                SeedEntity(**row)
                ok += 1
            except ValidationError as e:
                console.print(f"[red]Row {i} invalid:[/red] {e.error_count()} error(s)")
                for err in e.errors():
                    console.print(f"  {err['loc']} → {err['msg']}")
                errors += 1

    if errors == 0:
        console.print(f"[green]All {ok} records valid.[/green]")
    else:
        console.print(f"\n[yellow]{ok} valid / {errors} invalid[/yellow]")
        raise typer.Exit(1)


@app.command()
def sample(
    input: Path = typer.Option(_DEFAULT, help="Seed entities file to inspect"),
    n: int = typer.Option(5, help="Number of records to show"),
) -> None:
    """Print N sample records from seed_entities.jsonl."""
    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    with jsonlines.open(input) as reader:
        for i, row in enumerate(reader):
            if i >= n:
                break
            console.print(
                f"[bold]{row['id']}[/bold] {row['entity_name']} "
                f"({row['entity_class']} / {row['domain']})\n"
                f"  Q: {row['question']}\n"
                f"  [green]✓[/green] {row['claim_correct']}  "
                f"[red]✗[/red] {row['claim_incorrect']}\n"
            )


if __name__ == "__main__":
    app()
