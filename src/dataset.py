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
    content: str


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
    """
    missing: List[str] = []
    for i, e in enumerate(events, start=1):
        if e.location not in text:
            missing.append(f"[event {i}] missing location: {e.location}")
        if e.temporal not in text:
            missing.append(f"[event {i}] missing temporal: {e.temporal}")
        if e.entity not in text:
            missing.append(f"[event {i}] missing entity: {e.entity}")
        if e.content not in text:
            missing.append(f"[event {i}] missing content: {e.content}")

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
                    f"  - Full content '{e.content}'",
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
    include_style_description: bool = True,
    include_event_instructions: bool = True,
) -> Tuple[str, str]:
    """
    Build prompts for high-narrativity generation.

    - If include_style_description is False, the verbose style_description is omitted from the user prompt.
    - If include_event_instructions is False, event-specific verbatim strings are not embedded in the prompt;
      only generic/high-level constraints remain (so you can supply the event details separately).
    """
    system = (
        "You are a creative fiction writer specializing in detailed, atmospheric novel excerpts. "
        "Your task is to generate vivid, immersive scenes based on specific prompts."
    )

    # Optionally include the detailed per-event instructions (which contain verbatim strings).
    if include_event_instructions:
        instructions = build_event_instructions(events)
        instructions_block = (
            f"=== HARD CONTENT CONSTRAINTS (PER PARAGRAPH i=1..{k_paragraphs}) ===\n"
            f"{instructions}\n"
            "- You MUST include ALL four verbatim elements (location, date, entity, detail) for each event in the corresponding paragraph, EXACTLY AS GIVEN (no paraphrase, no pronouns, no coreference).\n"
            "- The four verbatim strings for paragraph (i) must appear EXACTLY ONCE in that paragraph and in no other paragraph.\n"
            "- For the temporal/date, write the date exactly as provided (e.g., \"2017-12-19\"). Do not spell it out.\n"
            "- Do not introduce any other locations, dates, or times; do not paraphrase the four verbatim strings.\n"
            "- No background lore beyond what the event requires.\n"
        )
    else:
        instructions_block = (
            f"=== HARD CONTENT CONSTRAINTS (PER PARAGRAPH i=1..{k_paragraphs}) ===\n"
            "- Event-specific verbatim strings are omitted from this prompt. The event details will be provided separately and must be used as-is.\n"
            "- Do not introduce any locations, dates, or entities that were not provided externally.\n"
            "- For the temporal/date, write the date exactly as provided (e.g., \"2017-12-19\"). Do not spell it out.\n"
            "- The required event elements should each appear exactly once in the corresponding paragraph.\n"
            "- No background lore beyond what the event requires.\n"
        )

    style_info = f" ({style_description})" if include_style_description else ""

    user = f"""
    Write a complete chapter in a {style} style{style_info}, consisting of exactly {k_paragraphs} numbered paragraphs (1)–({k_paragraphs}).
    Each paragraph centers on one distinct event. The story must be strictly chronological: (1) is the earliest, ({k_paragraphs}) the latest.

    {instructions_block}

    === STYLE CONSTRAINTS ===
    - Immersive, sensory descriptions and emotions are allowed.
    - Number each paragraph (1), (2), …, ({k_paragraphs}). Output the chapter ONLY (no headers, no meta-talk).
    """.strip()

    return system, user

