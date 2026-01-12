from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.dataio import read_json, write_json
from src.models import load_model, generate_text, extract_true_false_list, GenerationConfig
from src.prompts import build_all_prompts_for_chapter
from src.metrics import evaluate_conditions


BASE_PATH = "data/stories/base_data.json"
DIST_PATH = "data/distractors/distractor.json"
QA_PATH = "data/qa/qa.json"

RESP_DIR = Path("outputs/responses")
MET_DIR = Path("outputs/metrics")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id (e.g., meta-llama/Llama-2-13b-chat-hf or Qwen/Qwen2.5-14B-Instruct)",
    )
    p.add_argument("--model_tag", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _tag_from_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def index_by_chapter(items: List[dict]) -> Dict[int, dict]:
    return {int(x["chapter"]): x for x in items}


def main() -> None:
    args = parse_args()
    model_tag = args.model_tag or _tag_from_model_id(args.model)

    RESP_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)

    resp_path = RESP_DIR / f"{model_tag}_responses.json"
    met_path = MET_DIR / f"{model_tag}_metrics.json"

    if resp_path.exists() and met_path.exists() and not args.overwrite:
        print(f"exists: {resp_path} and {met_path} (use --overwrite to regenerate)")
        return

    base = read_json(BASE_PATH)
    dist = read_json(DIST_PATH)
    qa = read_json(QA_PATH)

    if args.limit is not None:
        base = base[: args.limit]
        qa = qa[: args.limit]

    dist_by = index_by_chapter(dist)
    qa_by = index_by_chapter(qa)

    lm = load_model(args.model, offload_dir=f"offload_{model_tag}")

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    conditions = ["h_NI", "l_NI", "h_MI", "l_MI", "h_RI", "l_RI", "h_UI", "l_UI"]
    responses: Dict[str, List[List[str]]] = {c: [] for c in conditions}
    gold_answers: List[List[str]] = []
    raw_generations: List[dict] = []

    for ch in tqdm(base, desc="chapters"):
        chapter_id = int(ch["chapter"])
        d = dist_by[chapter_id]
        q = qa_by[chapter_id]

        prompts = build_all_prompts_for_chapter(
            base_chapter=ch,
            distractor_chapter=d,
            qa_chapter=q,
        )

        n_q = len(q["questions"])

        chapter_pred: Dict[str, List[str]] = {}
        chapter_raw: Dict[str, str] = {}

        for cond in conditions:
            out_text = generate_text(lm, prompts[cond], gen=gen_cfg)
            chapter_raw[cond] = out_text
            chapter_pred[cond] = extract_true_false_list(out_text, n=n_q)

        for cond in conditions:
            responses[cond].append(chapter_pred[cond])

        gold_answers.append(q["answers"])

        raw_generations.append(
            {
                "chapter": chapter_id,
                "pred": chapter_pred,
                "raw": chapter_raw,
            }
        )

    # Save responses first
    write_json(
        {
            "model": args.model,
            "model_tag": model_tag,
            "conditions": conditions,
            "responses": responses,
            "raw_generations": raw_generations,
        },
        resp_path,
    )
    print(f"saved responses: {resp_path}")

    # Then evaluate and save metrics
    metrics = evaluate_conditions(responses=responses, gold_answers=gold_answers)
    write_json(
        {
            "model": args.model,
            "model_tag": model_tag,
            "metrics": metrics,
            "note": "Metrics exclude NA predictions from n_valid.",
        },
        met_path,
    )
    print(f"saved metrics: {met_path}")


if __name__ == "__main__":
    main()
