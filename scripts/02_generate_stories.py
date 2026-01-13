from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Any, List

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from src.dataio import read_json, write_json
from src.dataset import (
    sample_events,
    events_to_dicts,
    build_high_narrativity_prompts,
    verify_verbatim_inclusion,
    Event,
)


def chat_generate(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_low_narrativity_from_template(events: List[Event]) -> str:
    """
    Generate low-narrativity text using a simple template.
    Format: [Entity] was in [Content] at [Location] on [Temporal].
    """
    lines = []
    for event in events:
        line = f"- {event.entity} was in {event.content} at {event.location} on {event.temporal}."
        lines.append(line)
    return "\n".join(lines)


def generate_with_retries(
    *,
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    events: list,
    max_retries: int,
    label: str,
) -> Dict[str, Any]:
    """
    Generate text and retry if constraints are not satisfied.
    Returns:
      {
        "text": ...,
        "ok": bool,
        "missing": [..]   # missing constraints if not ok
        "tries": int      # number of generations performed
      }
    """
    tries = 0
    text = ""
    ok = False
    missing: List[str] = []

    for _ in range(max_retries + 1):  # first try + retries
        tries += 1
        text = chat_generate(client, model, system, user)
        ok, missing = verify_verbatim_inclusion(events, text)
        if ok:
            break

    if not ok:
        # keep it as warning; we still return the last text for transparency/debuggability
        print(f"[WARN] {label}: constraints not satisfied after {tries} tries.")
        for m in missing[:8]:
            print("  -", m)
        if len(missing) > 8:
            print(f"  ... and {len(missing)-8} more")

    return {"text": text, "ok": ok, "missing": missing, "tries": tries}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--elements", type=str, default="data/elements/common_elements.json")
    p.add_argument("--out", type=str, default="data/stories/base_data.json")
    p.add_argument("--n_chapters", type=int, default=100)
    p.add_argument("--k_events", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="gpt-3.5-turbo")
    p.add_argument("--max_retries", type=int, default=2)
    return p.parse_args()

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env")

    args = parse_args()
    rng = random.Random(args.seed)

    client = OpenAI(api_key=api_key)
    elements = read_json(args.elements)
    elements = elements["elements"]  # adjust for the structure

    all_data: List[Dict[str, Any]] = []

    for chapter in tqdm(range(args.n_chapters), desc="chapters"):
        events = sample_events(elements, num_events=args.k_events, rng=rng)

        # High narrativity
        hn_system, hn_user = build_high_narrativity_prompts(events, k_paragraphs=args.k_events)
        hn = generate_with_retries(
            client=client,
            model=args.model,
            system=hn_system,
            user=hn_user,
            events=events,
            max_retries=args.max_retries,
            label=f"high(chapter={chapter})",
        )

        # Low narrativity (template-based)
        ln_text = generate_low_narrativity_from_template(events)

        all_data.append(
            {
                "chapter": chapter,
                "events": events_to_dicts(events),
                "high_narrativity_text": hn["text"],
                "low_narrativity_text": ln_text
            }
        )

    write_json(all_data, args.out)
    print(f"generated {args.n_chapters} chapters: {args.out}")


if __name__ == "__main__":
    main()
