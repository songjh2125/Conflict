# Conflict Dataset Builder

A benchmark dataset for measuring how LLMs prioritize conflicting source documents.

Three bias factors are analyzed:
- **Temporal**: preference for more recent documents
- **Credibility**: preference for higher-credibility sources (academic > news > blog)
- **Majority**: preference for the larger number of documents

The key contribution is analyzing which factor dominates when two or more **conflict**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

| Phase | Script | Description |
|---|---|---|
| 1 | `generate_entities.py` | Generate fictional entities |
| 2 | `generate_documents.py` | Generate documents per entity |
| 3 | `assemble_experiments.py` | Assemble experiment instances |
| 4 | `run_evaluation.py` | Run LLM evaluation |
