"""Pydantic models and domain constants for the conflict dataset pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EntityClass = Literal["person", "organization", "product"]

# (domain, question_template) pairs per entity class.
# question_template uses {name} as the entity placeholder.
ENTITY_DOMAINS: dict[str, list[tuple[str, str]]] = {
    "person": [
        ("occupation", "What is {name}'s occupation?"),
        ("employer", "Which organization does {name} work for?"),
        ("nationality", "What is {name}'s nationality?"),
        ("research_area", "What academic field does {name} specialize in?"),
        ("academic_affiliation", "Which university is {name} affiliated with?"),
    ],
    "organization": [
        ("headquarters", "Where is {name} headquartered?"),
        ("founding_year", "When was {name} founded?"),
        ("industry", "What industry does {name} operate in?"),
        ("ceo", "Who currently leads {name}?"),
        ("parent_company", "Which company owns {name}?"),
    ],
    "product": [
        ("manufacturer", "Who manufactures {name}?"),
        ("release_year", "When was {name} first released?"),
        ("category", "What type of product is {name}?"),
        ("country_of_origin", "Where is {name} produced?"),
    ],
}


SourceType = Literal["academic", "news", "blog"]
ClaimType = Literal["correct", "incorrect"]


class SeedEntity(BaseModel):
    """Phase 1 output: a synthetic entity with two conflicting claims."""

    id: str
    entity_name: str
    entity_class: EntityClass
    domain: str
    question: str
    claim_correct: str
    claim_incorrect: str
    generation_method: str = "llm_synthetic"


class Document(BaseModel):
    """Phase 2 output: a synthetic document for one (entity, source_type, claim) combination."""

    doc_id: str
    entity_id: str
    entity_name: str
    entity_class: EntityClass
    domain: str
    question: str
    source_type: SourceType
    claim_type: ClaimType
    claim_value: str        # the actual value this document asserts
    source_name: str        # fictional publication / outlet name
    content: str            # full document text (no date embedded — added in Phase 3)
