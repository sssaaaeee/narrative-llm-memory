from __future__ import annotations

import argparse

from src.dataio import read_json, write_json
from src.qa import generate_qa_bulk, QASpec


BASE_DATA_PATH = "data/stories/base_data.json"
OUT_PATH = "data/qa/qa.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p_true", type=float, default=0.5)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_data = read_json(BASE_DATA_PATH)

    spec = QASpec(p_true=args.p_true)

    qa = generate_qa_bulk(
        base_data=base_data,
        seed=args.seed,
        spec=spec,
        verbose=not args.quiet,
    )

    write_json(qa, OUT_PATH)
    print(f"saved: {OUT_PATH} (chapters={len(qa)})")


if __name__ == "__main__":
    main()
