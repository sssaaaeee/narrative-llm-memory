from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set
import random


@dataclass(frozen=True)
class DistractorSpec:
    ratio: float = 1.0          # target_words = ratio * len(high_text.split())
    mi_token: str = "Humpty Dumpty"
    mi_half: bool = True        # MI is half length (like your original)


def _used_set(events: Sequence[dict], key: str) -> Set[str]:
    return {e[key] for e in events}


def _choice_excluding(
    rng: random.Random,
    pool: Sequence[str],
    used: Set[str],
) -> str:
    """Pick from pool excluding used if possible; fallback to any in pool."""
    candidates = [x for x in pool if x not in used]
    return rng.choice(candidates) if candidates else rng.choice(list(pool))


def _build_sentences_until(
    *,
    rng: random.Random,
    target_words: int,
    sentence_fn,
) -> str:
    """
    Append sentences until word count reaches target.
    sentence_fn must return a single sentence string.
    """
    parts: List[str] = []
    word_count = 0
    # guard: avoid pathological infinite loops if sentence_fn returns ""
    for _ in range(max(1, target_words * 4)):
        s = sentence_fn().strip()
        if not s:
            continue
        parts.append(s)
        word_count = len(" ".join(parts).split())
        if word_count >= target_words:
            break
    return " ".join(parts).strip()


def generate_mi(target_words: int, *, token: str, half: bool) -> str:
    """
    MI (Meaningless Interference): repeated token.
    Original behavior: roughly C/2 repetitions => about C words (depending on token length).
    We reproduce the intent: if half=True, aim for target_words//2 repetitions of the token.
    """
    reps = max(1, target_words // 2) if half else max(1, target_words)
    # token may contain spaces, so repetition count is in "token units", not words.
    return (" ".join([token] * reps)).strip()


def generate_ri(
    *,
    rng: random.Random,
    events: Sequence[dict],
    elements: Dict[str, Sequence[str]],
    target_words: int,
) -> str:
    """
    RI (Related Interference): share entity with the chapter events,
    but use location/temporal/content NOT used in the chapter if possible.
    """
    used_locations = _used_set(events, "location")
    used_temporals = _used_set(events, "temporal")
    used_contents = _used_set(events, "content")

    def sentence() -> str:
        shared_entity = rng.choice(events)["entity"]
        location = _choice_excluding(rng, elements["locations"], used_locations)
        temporal = _choice_excluding(rng, elements["temporals"], used_temporals)
        content = _choice_excluding(rng, elements["contents"], used_contents)
        return f"{shared_entity} was in {content} at {location} on {temporal}."

    return _build_sentences_until(rng=rng, target_words=target_words, sentence_fn=sentence)


def generate_ui(
    *,
    rng: random.Random,
    events: Sequence[dict],
    elements: Dict[str, Sequence[str]],
    target_words: int,
) -> str:
    """
    UI (Unrelated Interference): use entity/location/temporal/content NOT used in the chapter if possible.
    """
    used_entities = _used_set(events, "entity")
    used_locations = _used_set(events, "location")
    used_temporals = _used_set(events, "temporal")
    used_contents = _used_set(events, "content")

    def sentence() -> str:
        entity = _choice_excluding(rng, elements["entities"], used_entities)
        location = _choice_excluding(rng, elements["locations"], used_locations)
        temporal = _choice_excluding(rng, elements["temporals"], used_temporals)
        content = _choice_excluding(rng, elements["contents"], used_contents)
        return f"{entity} was in {content} at {location} on {temporal}."

    return _build_sentences_until(rng=rng, target_words=target_words, sentence_fn=sentence)


def generate_distractors_for_chapter(
    *,
    chapter: dict,
    elements: Dict[str, Sequence[str]],
    rng: random.Random,
    spec: DistractorSpec,
) -> dict:
    chapter_id = chapter["chapter"]
    events = chapter["events"]
    high_text = chapter["high_narrativity_text"]

    target_words = int(spec.ratio * len(high_text.split()))
    target_words = max(1, target_words)

    mi = generate_mi(target_words, token=spec.mi_token, half=spec.mi_half)
    ri = generate_ri(rng=rng, events=events, elements=elements, target_words=target_words)
    ui = generate_ui(rng=rng, events=events, elements=elements, target_words=target_words)

    return {
        "chapter": chapter_id,
        "MI": mi,
        "RI": ri,
        "UI": ui,
        "meta": {
            "target_words": target_words,
            "high_words": len(high_text.split()),
            "ratio": spec.ratio,
        },
    }


def generate_distractors_bulk(
    *,
    base_data: Sequence[dict],
    elements: Dict[str, Sequence[str]],
    seed: int = 0,
    spec: DistractorSpec = DistractorSpec(),
    verbose: bool = True,
) -> List[dict]:
    rng = random.Random(seed)
    out: List[dict] = []

    for ch in base_data:
        d = generate_distractors_for_chapter(chapter=ch, elements=elements, rng=rng, spec=spec)
        out.append(d)

        if verbose:
            mi_wc = len(d["MI"].split())
            ri_wc = len(d["RI"].split())
            ui_wc = len(d["UI"].split())
            print(
                f"[chapter {d['chapter']}] target={d['meta']['target_words']} "
                f"MI={mi_wc} RI={ri_wc} UI={ui_wc}"
            )

    return out
