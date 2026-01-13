from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.dataio import read_json, write_json
from src.prompts import build_all_prompts_for_chapter
from src.attention import (
    load_backbone_for_attention,
    load_elements_for_matching,
    analyze_prompt_attention,
    aggregate_prompt_summaries,
    REGIONS,
    LABELS,
)

DEFAULT_BASE = "data/stories/base_data.json"
DEFAULT_DIST = "data/distractors/distractor.json"
DEFAULT_QA = "data/qa/qa.json"
DEFAULT_ELEM = "data/elements/common_elements.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="meta-llama/Llama-2-13b-chat-hf or Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--base", type=str, default=DEFAULT_BASE)
    p.add_argument("--distractor", type=str, default=DEFAULT_DIST)
    p.add_argument("--qa", type=str, default=DEFAULT_QA)
    p.add_argument("--elements", type=str, default=DEFAULT_ELEM)

    p.add_argument("--outdir", type=str, default="outputs/attention")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def index_by_chapter(items: List[dict]) -> Dict[int, dict]:
    return {int(x["chapter"]): x for x in items}


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir) / args.model.replace("/", "__")
    outdir.mkdir(parents=True, exist_ok=True)

    base = read_json(args.base)
    dist = read_json(args.distractor)
    qa = read_json(args.qa)
    elements_json = read_json(args.elements)

    if args.limit is not None:
        base = base[: args.limit]
        qa = qa[: args.limit]

    dist_by = index_by_chapter(dist)
    qa_by = index_by_chapter(qa)

    elements = load_elements_for_matching(elements_json)

    backbone = load_backbone_for_attention(args.model)

    conditions = ["h_NI", "l_NI", "h_MI", "l_MI", "h_RI", "l_RI", "h_UI", "l_UI"]
    total_chapters = len(base)
    bs = args.batch_size
    num_batches = (total_chapters + bs - 1) // bs

    # accumulate all per-chapter summaries in memory (lightweight)
    all_prompt_summaries: Dict[str, List[dict]] = {c: [] for c in conditions}

    for batch_idx in range(num_batches):
        start = batch_idx * bs
        end = min(start + bs, total_chapters)

        batch_path = outdir / f"batch_{batch_idx}_ch{start}-{end-1}.json"
        if batch_path.exists() and not args.overwrite:
            # load and also extend all_prompt_summaries
            batch_obj = read_json(batch_path)
            for c in conditions:
                all_prompt_summaries[c].extend(batch_obj.get(c, []))
            print(f"loaded: {batch_path}")
            continue

        batch_obj = {c: [] for c in conditions}

        for local_idx in tqdm(range(start, end), desc=f"batch {batch_idx} ch{start}-{end-1}"):
            ch = base[local_idx]
            chapter_id = int(ch["chapter"])
            d = dist_by[chapter_id]
            q = qa_by[chapter_id]

            prompts = build_all_prompts_for_chapter(
                base_chapter=ch,
                distractor_chapter=d,
                qa_chapter=q,
            )

            for cond in conditions:
                try:
                    summary = analyze_prompt_attention(
                        backbone,
                        prompts[cond],
                        elements=elements,
                    )
                except Exception as e:
                    print(f"⚠️  Error processing chapter {chapter_id}, condition {cond}: {e}")
                    # Create a dummy summary to continue
                    summary = analyze_prompt_attention(
                        backbone,
                        prompts[cond][:1000],  # Truncate if needed
                        elements=elements,
                    )

                # save per-chapter minimal stats
                batch_obj[cond].append({
                    "chapter": chapter_id,
                    "region_sums": summary.region_sums,
                    "region_ratios": summary.region_ratios,
                    "label_sums": summary.label_sums,
                    "label_ratios": summary.label_ratios,
                    "total_sum": summary.total_sum,
                    "text_sum": summary.text_sum,
                })

        write_json(batch_obj, batch_path)
        print(f"saved: {batch_path}")

        for c in conditions:
            all_prompt_summaries[c].extend(batch_obj[c])
        
        # Clear GPU cache after each batch
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU cache cleared after batch {batch_idx}")

    # aggregate over chapters for each condition
    final = {}
    for cond in conditions:
        per_ch = []
        for item in all_prompt_summaries[cond]:
            # reconstruct a lightweight PromptAttentionSummary-like dict
            # aggregate_prompt_summaries expects dataclass objects, so we'll just compute here:
            # easiest: wrap as small objects via a tiny local class
            class _S:
                def __init__(self, d):
                    self.region_sums = d["region_sums"]
                    self.region_ratios = d["region_ratios"]
                    self.label_sums = d.get("label_sums")
                    self.label_ratios = d.get("label_ratios")
                    self.total_sum = d["total_sum"]
                    self.text_sum = d["text_sum"]
            per_ch.append(_S(item))

        final[cond] = aggregate_prompt_summaries(per_ch)

    results_path = outdir / "results_attention.json"
    results_obj = {
        "model": args.model,
        "conditions": conditions,
        "regions": REGIONS,
        "labels": LABELS,
        "results": final,
    }
    write_json(results_obj, results_path)
    print(f"saved: {results_path}")
    
    # Generate attention visualization plots
    from src.viz import plot_element_attention_2x2, plot_element_attention_1x4
    
    # Extract model name for display (e.g., "Llama" or "Qwen")
    model_display_name = args.model.split("/")[-1].split("-")[0]
    
    print(f"\nGenerating attention plots for {model_display_name}...")
    plot_element_attention_2x2(
        results_obj,
        output_dir=outdir,
        model_name=model_display_name,
        save_format="pdf",
        show_values=False
    )
    plot_element_attention_1x4(
        results_obj,
        output_dir=outdir,
        model_name=model_display_name,
        save_format="pdf",
        show_values=False
    )
    print(f"Attention plots saved in: {outdir}")


if __name__ == "__main__":
    main()
