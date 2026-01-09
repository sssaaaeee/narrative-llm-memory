from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random


@dataclass(frozen=True)
class QASpec:
    """
    - topics: which attribute to ask about
    - p_true: probability of generating a True question (else False)
    """
    topics: Tuple[str, ...] = ("entity", "location", "temporal", "content")
    p_true: float = 0.5


def _pick_other_event(rng: random.Random, events: Sequence[dict], idx: int) -> dict:
    """Pick a different event from the same chapter if possible."""
    if len(events) <= 1:
        return events[idx]
    j = idx
    while j == idx:
        j = rng.randrange(len(events))
    return events[j]


def _replace_with_false_value(
    *,
    rng: random.Random,
    events: Sequence[dict],
    idx: int,
    key: str,
    true_event: dict,
) -> str:
    """
    Replace only the asked field with another event's value to make it False.
    If we accidentally pick same value, retry a few times; then accept.
    """
    original = true_event[key]
    for _ in range(10):
        other = _pick_other_event(rng, events, idx)
        cand = other[key]
        if cand != original:
            return cand
    return original  # fallback (rare): may not flip truth if duplicates exist


def build_question(event: dict, topic: str, value: str) -> str:
    """
    Natural-ish but simple templates. Keep all fields explicit to avoid coreference issues.
    """
    e = event["entity"]
    loc = event["location"]
    tmp = event["temporal"]
    cont = event["content"]

    if topic == "entity":
        return (
            f"Is it true that who was in {cont} at {loc} on {tmp} is {value}?"
        )
    if topic == "location":
        return (
            f"Is it true that where {e} was in {cont} on {tmp} is {value}?"
        )
    if topic == "temporal":
        return (
            f"Is it true that when {e} was in {cont} at {loc} is {value}?"
        )
    if topic == "content":
        return (
            f"Is it true that what {e} did at {loc} on {tmp} is {value}?"
        )

    raise ValueError(f"Unknown topic: {topic}")


def generate_qa_for_chapter(
    *,
    chapter: dict,
    rng: random.Random,
    spec: QASpec,
) -> Dict[str, object]:
    events = chapter["events"]

    questions: List[str] = []
    answers: List[str] = []
    topics: List[str] = []

    for idx, ev in enumerate(events):
        topic = rng.choice(spec.topics)
        make_true = rng.random() < spec.p_true

        if make_true:
            value = ev[topic]
            ans = "True"
        else:
            value = _replace_with_false_value(
                rng=rng,
                events=events,
                idx=idx,
                key=topic,
                true_event=ev,
            )
            # もし同値が返ってきてしまう（重複が多い）と True になり得るので、厳密に判定
            ans = "False" if value != ev[topic] else "True"

        q = build_question(ev, topic, value)

        questions.append(q)
        answers.append(ans)
        topics.append(topic)

    # chapter id は input に合わせて chapter["chapter"] を優先
    chapter_id = chapter.get("chapter", None)
    return {
        "chapter": chapter_id,
        "questions": questions,
        "answers": answers,
        "topics": topics,
    }


def generate_qa_bulk(
    *,
    base_data: Sequence[dict],
    seed: int = 0,
    spec: QASpec = QASpec(),
    verbose: bool = False,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)

    out: List[Dict[str, object]] = []
    for i, ch in enumerate(base_data):
        qa = generate_qa_for_chapter(chapter=ch, rng=rng, spec=spec)
        # chapter が None の場合は連番を入れる（入力の互換性対策）
        if qa["chapter"] is None:
            qa["chapter"] = i
        out.append(qa)

        if verbose:
            print(f"[chapter {qa['chapter']}] questions={len(qa['questions'])}")

    return out
