"""Phase 4: Run experiment instances through GPT-4.1-mini and collect Q1–Q3 responses."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import jsonlines
import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

MODEL = "openai/gpt-4.1-mini"

SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. "
    "You will be given several documents and a multiple-choice question. "
    "Answer each sub-question (Q1, Q2, Q3) using ONLY the corresponding option letter. "
    "Respond with a JSON object containing exactly three keys: \"q1\", \"q2\", \"q3\". "
    "Each value must be a single uppercase letter. No explanation needed."
)


def _extract_letter(raw: str) -> str | None:
    """Extract first uppercase letter A–Z from a string."""
    m = re.search(r"\b([A-Z])\b", raw)
    return m.group(1) if m else None


def _parse_response(raw: str) -> dict[str, str | None]:
    """Parse model response into q1–q3 letter answers."""
    try:
        data = json.loads(raw)
        return {
            "q1": _extract_letter(str(data.get("q1", ""))),
            "q2": _extract_letter(str(data.get("q2", ""))),
            "q3": _extract_letter(str(data.get("q3", ""))),
        }
    except (json.JSONDecodeError, AttributeError):
        # Fallback: scan for Q1:X patterns
        answers: dict[str, str | None] = {}
        for q in ("q1", "q2", "q3"):
            m = re.search(rf"{q}\s*[:\-]\s*([A-Z])", raw, re.IGNORECASE)
            answers[q] = m.group(1).upper() if m else None
        return answers


async def _call_api_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    instance: dict,
) -> dict:
    """Call the API for a single experiment instance; returns result dict."""
    exp_id = instance["exp_id"]

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    temperature=0.0,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": instance["eval_prompt"]},
                    ],
                )
                raw = response.choices[0].message.content or ""
                answers = _parse_response(raw)

                # Determine if Q1 was answered correctly
                q1 = answers["q1"]
                option_a_is_correct: bool = instance["option_a_is_correct"]
                if q1 == "A":
                    q1_correct = option_a_is_correct
                elif q1 == "B":
                    q1_correct = not option_a_is_correct
                else:
                    q1_correct = None  # unparseable

                return {
                    "exp_id":          exp_id,
                    "entity_id":       instance["entity_id"],
                    "entity_name":     instance["entity_name"],
                    "condition_id":    instance["condition_id"],
                    "factor":          instance["factor"],
                    "question":        instance["question"],
                    "answer_correct":  instance["answer_correct"],
                    "answer_incorrect": instance["answer_incorrect"],
                    "option_a":        instance["option_a"],
                    "option_b":        instance["option_b"],
                    "option_a_is_correct": option_a_is_correct,
                    "n_correct_docs":  instance["n_correct_docs"],
                    "n_incorrect_docs": instance["n_incorrect_docs"],
                    "q1_answer":       answers["q1"],
                    "q2_answer":       answers["q2"],
                    "q3_answer":       answers["q3"],
                    "q1_correct":      q1_correct,
                    "raw_response":    raw,
                    "status":          "ok",
                }
            except Exception as exc:
                if attempt == 2:
                    return {
                        "exp_id":    exp_id,
                        "entity_id": instance["entity_id"],
                        "condition_id": instance["condition_id"],
                        "factor":    instance["factor"],
                        "status":    "error",
                        "error":     str(exc),
                    }
                await asyncio.sleep(2 ** attempt)  # 1s, 2s backoff


async def _run_all(
    instances: list[dict],
    client: AsyncOpenAI,
    concurrency: int,
    progress_task,
    progress,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def _task(inst: dict) -> None:
        result = await _call_api_async(client, semaphore, inst)
        results.append(result)
        progress.advance(progress_task)

    await asyncio.gather(*[_task(inst) for inst in instances])
    return results


@app.command()
def evaluate(
    experiments_path: Path = typer.Option(
        Path("data/pipeline/03_experiments/experiments.jsonl"),
        help="Input experiments file",
    ),
    output: Path = typer.Option(
        Path("data/pipeline/04_results/results.jsonl"),
        help="Output results file",
    ),
    concurrency: int = typer.Option(
        10, help="Number of concurrent API calls"
    ),
    limit: int = typer.Option(
        0, help="Process only first N instances (0 = all)"
    ),
    resume: bool = typer.Option(
        True, help="Skip already-processed exp_ids found in output file"
    ),
) -> None:
    """Run all experiment instances through the model and collect Q1–Q3 answers (Phase 4)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set. Add it to .env[/red]")
        raise typer.Exit(1)

    if not experiments_path.exists():
        console.print(f"[red]File not found:[/red] {experiments_path}")
        raise typer.Exit(1)

    # Load experiments
    instances: list[dict] = []
    with jsonlines.open(experiments_path) as reader:
        instances = list(reader)

    if limit > 0:
        instances = instances[:limit]

    # Resume: skip already-done exp_ids
    done_ids: set[str] = set()
    if resume and output.exists():
        with jsonlines.open(output) as reader:
            for row in reader:
                done_ids.add(row["exp_id"])
        if done_ids:
            console.print(f"[cyan]Resuming:[/cyan] skipping {len(done_ids)} already-processed instances")
        instances = [i for i in instances if i["exp_id"] not in done_ids]

    if not instances:
        console.print("[green]Nothing to process.[/green]")
        raise typer.Exit(0)

    console.print(
        f"Processing [bold]{len(instances)}[/bold] instances "
        f"with concurrency=[bold]{concurrency}[/bold] …"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Run async evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(instances))
        results = asyncio.run(
            _run_all(instances, client, concurrency, task, progress)
        )

    # Append to output (preserve prior results when resuming)
    mode = "a" if (resume and done_ids) else "w"
    ok = error = 0
    with jsonlines.open(output, mode=mode) as writer:
        for r in results:
            writer.write(r)
            if r.get("status") == "ok":
                ok += 1
            else:
                error += 1

    console.print(
        f"\n[green]Done:[/green] {ok} ok, {error} errors → {output}"
    )
    if error:
        console.print("[yellow]Re-run with --resume to retry failed instances.[/yellow]")


@app.command()
def inspect(
    input: Path = typer.Option(
        Path("data/pipeline/04_results/results.jsonl"),
        help="Results file to inspect",
    ),
    condition_id: str = typer.Option(None, help="Filter by condition ID"),
    factor: str = typer.Option(None, help="Filter by factor (T, C, M, TC, TM, CM)"),
) -> None:
    """Show summary statistics for evaluation results."""
    from collections import Counter, defaultdict

    if not input.exists():
        console.print(f"[red]File not found:[/red] {input}")
        raise typer.Exit(1)

    records = []
    with jsonlines.open(input) as reader:
        for row in reader:
            if row.get("status") != "ok":
                continue
            if condition_id and row.get("condition_id") != condition_id:
                continue
            if factor and row.get("factor") != factor:
                continue
            records.append(row)

    if not records:
        console.print("[red]No matching records (status=ok).[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Total results (ok):[/bold] {len(records)}")

    # Overall accuracy
    q1_answers = [r for r in records if r.get("q1_correct") is not None]
    if q1_answers:
        acc = sum(1 for r in q1_answers if r["q1_correct"]) / len(q1_answers)
        console.print(f"[bold]Overall Q1 accuracy:[/bold] {acc:.1%} ({sum(1 for r in q1_answers if r['q1_correct'])}/{len(q1_answers)})")

    # Accuracy per condition
    console.print("\n[bold]Accuracy by condition:[/bold]")
    by_cond: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("q1_correct") is not None:
            by_cond[r["condition_id"]].append(r["q1_correct"])
    for cond, vals in sorted(by_cond.items()):
        acc = sum(vals) / len(vals)
        console.print(f"  {cond:<20} {acc:.1%}  (n={len(vals)})")

    # Q3 reasoning distribution
    console.print("\n[bold]Q3 reasoning distribution:[/bold]")
    q3_map = {
        "A": "Recency",
        "B": "Credibility",
        "C": "Majority",
        "D": "Content quality",
    }
    q3_counts = Counter(r.get("q3_answer") for r in records if r.get("q3_answer"))
    for letter, label in q3_map.items():
        n = q3_counts.get(letter, 0)
        pct = n / len(records) if records else 0
        console.print(f"  ({letter}) {label:<22} {n:>4}  {pct:.1%}")


if __name__ == "__main__":
    app()
