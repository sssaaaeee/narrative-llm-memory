from __future__ import annotations

import argparse
from pathlib import Path

from src.dataio import read_json
from src.viz import compute_topic_accuracy_matrices, plot_topic_accuracy_heatmap


QA_PATH_DEFAULT = "data/qa/qa.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--responses", type=str, required=False,
                   help="e.g., outputs/responses/<model_tag>_responses.json")
    p.add_argument("--model", type=str, required=False,
                   help="Model name (llama or qwen) - will auto-detect response file")
    p.add_argument("--qa", type=str, default=QA_PATH_DEFAULT)
    p.add_argument("--outdir", type=str, default="outputs/figures")
    p.add_argument("--model_label", type=str, default=None)
    
    # These are passed by run_all.py but not used directly
    p.add_argument("--exp", type=str, default=None)
    p.add_argument("--paths", type=str, default=None)

    # Optional manual override
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    return p.parse_args()


def _default_range_from_model_name(name: str) -> tuple[float, float]:
    # your requested defaults:
    # llama: 0.4-0.6, qwen: 0.7-1.0
    low = name.lower()
    if "qwen" in low:
        return 0.7, 1.0
    return 0.4, 0.6


def main() -> None:
    args = parse_args()
    
    # Auto-detect response file from --model if --responses not provided
    if not args.responses and args.model:
        response_dir = Path("outputs/responses")
        model_key = args.model.lower()
        
        # Find matching response file
        candidates = []
        if response_dir.exists():
            for f in response_dir.glob("*_responses.json"):
                if model_key in f.name.lower():
                    candidates.append(f)
        
        if not candidates:
            raise FileNotFoundError(f"No response file found for model '{args.model}' in {response_dir}")
        
        args.responses = str(candidates[0])
        print(f"Auto-detected response file: {args.responses}")
    
    if not args.responses:
        raise ValueError("Either --responses or --model must be provided")

    qa_data = read_json(args.qa)
    resp_obj = read_json(args.responses)

    responses = resp_obj["responses"]  # condition -> chapterwise list[list[str]]
    model_label = args.model_label or resp_obj.get("model_tag") or resp_obj.get("model") or "model"

    dvmin, dvmax = _default_range_from_model_name(str(model_label))
    vmin = args.vmin if args.vmin is not None else dvmin
    vmax = args.vmax if args.vmax is not None else dvmax

    result = compute_topic_accuracy_matrices(
        qa_data=qa_data,
        responses_by_condition=responses,
    )

    outdir = Path(args.outdir)
    out_png = outdir / f"heatmap_{model_label}.png"
    out_pdf = outdir / f"heatmap_{model_label}.pdf"

    plot_topic_accuracy_heatmap(
        result,
        model_label=str(model_label),
        out_png=out_png,
        out_pdf=out_pdf,
        vmin=vmin,
        vmax=vmax,
        topic_labels=["s", "t", "ent", "c"],
        distractor_labels=["NI", "MI", "RI", "UI"],
        font_size=22,
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # (optional) print counts used per cell (chapter-average aggregation)
    print("\n=== Chapter counts used per topic × distractor (High) ===")
    for t, counts in result.chapter_counts_high.items():
        print(f"{t}: {counts}")
    print("\n=== Chapter counts used per topic × distractor (Low) ===")
    for t, counts in result.chapter_counts_low.items():
        print(f"{t}: {counts}")


if __name__ == "__main__":
    main()
