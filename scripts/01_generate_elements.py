"""
Generate common elements (location / temporal / entity / content) using OpenAI,
and save as a single JSON file.

Usage:
  python scripts/01_generate_elements.py \
    --model gpt-3.5-turbo \
    --out data/elements/common_elements.json \
    --seed 42

Requirements:
  - OPENAI_API_KEY in environment (or .env)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Prompt specs
# -----------------------------
@dataclass(frozen=True)
class ElementSpec:
    key: str
    prompts: List[str]


TEMPORAL_SPEC = ElementSpec(
    key="temporals",
    prompts=[
        (
            "Please list 200 different dates within the years 2010 to 2020 in New York.\n"
            "Each date should be specific and unique (format: YYYY-MM-DD).\n"
            "The dates should be distributed across different seasons, months, and years, and must not overlap.\n"
            "Examples: '2012-04-15', '2014-11-03', '2018-07-22', '2016-01-01', '2020-09-10'."
        ),
        "Keep only neutral and positive time points, removing major holidays and widely recognized negative events.",
        "Are those time points all distinct and spread across different dates within 2010 to 2020?",
        "Discard less distinct or odd ones, keeping only 120 different time points.",
        "Are those time points all located at different dates within 2010 to 2020?",
    ],
)

ENTITY_SPEC = ElementSpec(
    key="entities",
    prompts=[
        (
            "Please list 200 different combinations of first names and last names commonly used in the United States.\n"
            "Each combination should be unique, realistic, and natural-sounding.\n"
            "The names should reflect a diversity of backgrounds.\n"
            "Examples: 'Emily Carter', 'Michael Johnson', 'Sophia Lee', 'James Anderson', 'Ava Martinez'."
        ),
        "Keep only neutral and positive-sounding names, removing company names, celebrity names, and historically significant figures.",
        "Are those name combinations all unique and diverse?",
        "Discard less distinct or odd ones, keeping only 120 different names.",
        "Are those names all unique and natural-sounding in the United States?",
    ],
)

LOCATION_SPEC = ElementSpec(
    key="locations",
    prompts=[
        (
            "Please list 200 different locations in New York and surrounding areas.\n"
            "Each location should correspond to a specific (longitude, latitude) point and must not overlap.\n"
            "Examples: 'Empire State Building', 'Statue of Liberty', 'Museum of Modern Art (MoMA)', "
            "'Chrysler Building', 'Fort Greene Park'."
        ),
        "Keep only neutral and positive locations, removing any location whose name specifies a company name.",
        "Are those locations all at different (longitude, latitude) points in New York?",
        "Discard less distinct or odd ones, keeping only 120 different locations.",
        "Are those locations all at different (longitude, latitude) points in New York?",
    ],
)

CONTENT_SPEC = ElementSpec(
    key="contents",
    prompts=[
        (
            "Please list 200 different types of events or happenings that could take place in New York and surrounding areas.\n"
            'Each event should be concrete and specific, described as a short phrase or sentence (e.g., "a food festival", '
            '"a marriage proposal", "children playing chess in the park").\n'
            "The events should be diverse and should not overlap in content."
        ),
        "Keep only neutral or positive events, removing any event that includes a company name or refers to commercial brands.",
        "Are all of these events clearly distinct and feasible in New York or the surrounding area?",
        "Discard less distinct, less feasible, or odd events, keeping only 120 clearly distinct, realistic, concrete events.",
        "Are these 120 events all clearly distinct and realistically could take place at different locations and times in New York?",
    ],
)

SPECS = [LOCATION_SPEC, TEMPORAL_SPEC, ENTITY_SPEC, CONTENT_SPEC]


JSON_ONLY_INSTRUCTION = (
    "Output ONLY one JSON array.\n"
    'Example: ["item1", "item2", "item3"]\n'
    "No extra explanation, no markdown, no surrounding text."
)


# -----------------------------
# Utilities
# -----------------------------
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def extract_json_array(text: str) -> list:
    """
    Extract the first JSON array from model output.
    We still instruct JSON-only, but this makes it robust.
    """
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        raise ValueError("No JSON array found in the model output.")
    return json.loads(m.group(0))


def dedup_keep_order(items: list) -> list:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def call_openai_list(
    client: OpenAI,
    model: str,
    prompts: List[str],
    *,
    max_items: int = 100,
) -> list:
    messages = [{"role": "user", "content": p} for p in prompts]
    messages.append({"role": "user", "content": JSON_ONLY_INSTRUCTION})

    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""

    items = extract_json_array(text)

    # Normalize: strings only, strip spaces, drop empties, dedup, then cut.
    items = [str(x).strip() for x in items if str(x).strip()]
    items = dedup_keep_order(items)

    if len(items) < max_items:
        # Not fatal, but makes failures visible.
        raise ValueError(f"Got only {len(items)} items (< {max_items}). Output:\n{text}")

    return items[:max_items]


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--out", default="data/elements/common_elements.json")
    p.add_argument("--seed", type=int, default=42)  # kept for reproducibility logs
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in env or .env.")

    client = OpenAI(api_key=api_key)

    results = {}
    for spec in SPECS:
        print(f"[INFO] Generating {spec.key} ...")
        results[spec.key] = call_openai_list(client, args.model, spec.prompts, max_items=100)
        print(f"[INFO]  -> {len(results[spec.key])} items")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "model": args.model,
            "seed": args.seed,
            "note": "Generated by scripts/01_generate_elements.py",
        },
        "elements": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved: {out_path}")


if __name__ == "__main__":
    main()
