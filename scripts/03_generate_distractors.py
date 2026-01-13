from __future__ import annotations

import argparse

from src.dataio import read_json, write_json
from src.distractors import generate_distractors_bulk, DistractorSpec


ELEMENTS_PATH = "data/elements/common_elements.json"
BASE_DATA_PATH = "data/stories/base_data.json"
OUT_PATH = "data/distractors/distractor.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratio", type=float, default=1.0)  # r
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    elements = read_json(ELEMENTS_PATH)
    elements = elements["elements"]  # adjust for the structure
    base_data = read_json(BASE_DATA_PATH)

    spec = DistractorSpec(ratio=args.ratio)

    distractors = generate_distractors_bulk(
        base_data=base_data,
        elements=elements,
        seed=args.seed,
        spec=spec,
        verbose=not args.quiet,
    )

    write_json(distractors, OUT_PATH)
    print(f"saved: {OUT_PATH} (chapters={len(distractors)})")


if __name__ == "__main__":
    main()
