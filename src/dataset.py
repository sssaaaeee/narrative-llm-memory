# elements→story→distractor→qa の整形

# src/dataset.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple
import random


# ---------
# Data model
# ---------
@dataclass(frozen=True)
class Event:
    location: str
    temporal: str  # "YYYY-MM-DD"
    entity: str
    first_name: str
    content: str
    content_single_detail: str  # e.g., "did something."


def sample_events(
    elements: Dict[str, Sequence[str]],
    *,
    num_events: int,
    rng: random.Random,
) -> List[Event]:
    """Sample events from common elements."""
    locations = elements["locations"]
    temporals = elements["temporals"]
    entities = elements["entities"]
    contents = elements["contents"]

    events: List[Event] = []
    for _ in range(num_events):
        location = rng.choice(locations)
        temporal = rng.choice(temporals)
        entity = rng.choice(entities)
        content = rng.choice(contents)
        events.append(
            Event(
                location=location,
                temporal=temporal,
                entity=entity,
                content=content
            )
        )
    return events


def events_to_dicts(events: Sequence[Event]) -> List[dict]:
    return [asdict(e) for e in events]


# -------------------------
# Constraints / verification
# -------------------------
def verify_verbatim_inclusion(events: Sequence[Event], text: str) -> Tuple[bool, List[str]]:
    """
    Verify that each event's key strings appear in the generated text.
    Minimal & robust given your prompt constraints:
      - location appears (verbatim)
      - temporal "YYYY-MM-DD" appears (verbatim)
      - entity appears (verbatim)
      - "FirstName {content}." appears (verbatim)
    """
    missing: List[str] = []
    for i, e in enumerate(events, start=1):
        if e.location not in text:
            missing.append(f"[event {i}] missing location: {e.location}")
        if e.temporal not in text:
            missing.append(f"[event {i}] missing temporal: {e.temporal}")
        if e.entity not in text:
            missing.append(f"[event {i}] missing entity: {e.entity}")
        detail = f"{e.first_name} {e.content_single_detail}"
        if detail not in text:
            missing.append(f"[event {i}] missing detail: {detail}")

    return (len(missing) == 0), missing


# -----------------
# Prompt generation
# -----------------
def build_event_instructions(events: Sequence[Event]) -> str:
    lines: List[str] = []
    for i, e in enumerate(events, start=1):
        lines.append(
            "\n".join(
                [
                    f"- In paragraph {i}, the following must appear verbatim and only in that paragraph:",
                    f"  - Full location '{e.location}'",
                    f"  - Full date '{e.temporal}'",
                    f"  - Full name '{e.entity}'",
                    f"  - Full detail that '{e.first_name} {e.content_single_detail}'",
                ]
            )
        )
    return "\n".join(lines)


def build_high_narrativity_prompts(
    events: Sequence[Event],
    *,
    k_paragraphs: int,
    style: str = "Shakespearean",
    style_description: str = (
        "emulating the poetic language, dramatic flair, and rhythmic cadence "
        "characteristic of Shakespeare's plays, including the use of archaic expressions "
        "and heightened emotion"
    ),
) -> Tuple[str, str]:
    system = (
        "You are a creative fiction writer specializing in detailed, atmospheric novel excerpts. "
        "Your task is to generate vivid, immersive scenes based on specific prompts."
    )
    instructions = build_event_instructions(events)

    user = f"""
    Write a complete chapter in a {style} style ({style_description}), consisting of exactly {k_paragraphs} numbered paragraphs (1)–({k_paragraphs}).
    Each paragraph centers on one distinct event. The story must be strictly chronological: (1) is the earliest, ({k_paragraphs}) the latest.

    === HARD CONTENT CONSTRAINTS (PER PARAGRAPH i=1..{k_paragraphs}) ===
    {instructions}
    - You MUST include ALL four verbatim elements (location, date, entity, detail) for each event in the corresponding paragraph, EXACTLY AS GIVEN (no paraphrase, no pronouns, no coreference).
    - The four verbatim strings for paragraph (i) must appear EXACTLY ONCE in that paragraph and in no other paragraph.
    - For the temporal/date, write the date exactly as provided (e.g., "2017-12-19"). Do not spell it out.
    - Do not introduce any other locations, dates, or times; do not paraphrase the four verbatim strings.
    - No background lore beyond what the event requires.

    === STYLE CONSTRAINTS ===
    - Immersive, sensory descriptions and emotions are allowed.
    - Number each paragraph (1), (2), …, ({k_paragraphs}). Output the chapter ONLY (no headers, no meta-talk).
    """.strip()

    return system, user


def build_low_narrativity_prompts(
    events: Sequence[Event],
    *,
    k_events: int,
) -> Tuple[str, str]:
    system = (
        "You are a logical thinker who lacks sensitivity and expresses yourself logically. "
        "Your task is to state the facts calmly based on specific instructions."
    )
    instructions = build_event_instructions(events)

    user = f"""
    Produce a low-narrativity version of the same {k_events} events. One bullet per event, strictly one sentence per bullet.

    === CONTENT CONSTRAINTS (PER BULLET i=1..{k_events}) ===
    {instructions}
    - You MUST include ALL four verbatim elements (location, date, entity, detail) for each event in the corresponding bullet, EXACTLY AS GIVEN (no paraphrase, no pronouns, no coreference).
    - Do NOT omit or alter any element; every bullet must contain all four.
    - Do NOT add any other locations/dates/times/entities or extra facts.

    === STYLE CONSTRAINTS ===
    - Low narrativity: no causal or temporal connectives (e.g., "therefore", "then", "so", "after that"; also Japanese equivalents like だから/そして/しかし/それで/その後/まず/次に/最後に are prohibited).
    - One sentence per bullet; one bullet per event; no cross-references; no background info.
    - Format: lines starting with "- " for each of the {k_events} bullets.

    Output ONLY the bullet list (no meta text).
    """.strip()

    return system, user
