from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir_from_paths(paths_yaml: dict) -> None:
    for key in ["data", "experiments", "logs", "cache"]:
        root = paths_yaml.get(key, {}).get("root")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)

    # Create specific subdirectories
    # data/*
    for p in [
        "data/elements",
        "data/stories",
        "data/distractors",
        "data/qa",
        # results/*
        "results/llama",
        "results/qwen",
        "results/llama/heatmap",
        "results/qwen/heatmap",
        "results/llama/attention/batches",
        "results/qwen/attention/batches",
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--gen", default="configs/gen.yaml")
    ap.add_argument("--exp", default="configs/exp.yaml")

    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--skip_infer", action="store_true")
    ap.add_argument("--skip_heatmap", action="store_true")
    ap.add_argument("--skip_attention", action="store_true")

    ap.add_argument("--model", choices=["llama", "qwen", "both"], default="both")
    args = ap.parse_args()

    paths_cfg = load_yaml(args.paths)
    mkdir_from_paths(paths_cfg)

    # -------- generation --------
    if not args.skip_gen:
        # Load config values
        gen_cfg = load_yaml(args.gen)
        paths_cfg = load_yaml(args.paths)
        
        run([sys.executable, "scripts/01_generate_elements.py", 
            "--model", "gpt-3.5-turbo", 
            "--out", "data/elements/common_elements.json", 
            "--seed", "42"])
        
        run([sys.executable, "scripts/02_generate_stories.py",
            "--elements", "data/elements/common_elements.json",
            "--out", "data/stories/base_data.json",
            "--n_chapters", str(gen_cfg.get("stories", {}).get("n_chapters", 2)),
            "--k_events", str(gen_cfg.get("stories", {}).get("k_events", 10)),
            "--model", gen_cfg.get("stories", {}).get("openai", {}).get("model", "gpt-3.5-turbo"),
            "--seed", str(gen_cfg.get("project", {}).get("seed", 42))])

        run([sys.executable, "scripts/03_generate_distractors.py",
            "--seed", str(gen_cfg.get("project", {}).get("seed", 42)),
            "--ratio", "1.0"])

        run([sys.executable, "scripts/04_generate_qa.py",
            "--seed", str(gen_cfg.get("project", {}).get("seed", 42)),
            "--p_true", "0.5"])

    # -------- inference + evaluation (Acc/F1) --------
    if not args.skip_infer:
        exp_cfg = load_yaml(args.exp)
        
        if args.model in ["llama", "both"]:
            llama_model = exp_cfg.get("models", {}).get("llama", {}).get("name", "meta-llama/Llama-2-13b-chat-hf")
            max_tokens = exp_cfg.get("inference", {}).get("generation", {}).get("max_new_tokens", 80)
            temp = exp_cfg.get("inference", {}).get("generation", {}).get("temperature", 0.0)
            top_p = exp_cfg.get("inference", {}).get("generation", {}).get("top_p", 1.0)
            run([sys.executable, "scripts/11_run_inference.py", 
                 "--model", llama_model,
                 "--max_new_tokens", str(max_tokens),
                 "--temperature", str(temp),
                 "--top_p", str(top_p)])
        
        if args.model in ["qwen", "both"]:
            qwen_model = exp_cfg.get("models", {}).get("qwen", {}).get("name", "Qwen/Qwen2.5-14B-Instruct")
            max_tokens = exp_cfg.get("inference", {}).get("generation", {}).get("max_new_tokens", 80)
            temp = exp_cfg.get("inference", {}).get("generation", {}).get("temperature", 0.0)
            top_p = exp_cfg.get("inference", {}).get("generation", {}).get("top_p", 1.0)
            run([sys.executable, "scripts/11_run_inference.py", 
                 "--model", qwen_model,
                 "--max_new_tokens", str(max_tokens),
                 "--temperature", str(temp),
                 "--top_p", str(top_p)])

    # -------- heatmap --------
    if not args.skip_heatmap:
        if args.model in ["llama", "both"]:
            run([sys.executable, "scripts/12_eval_heatmap.py", "--exp", args.exp, "--paths", args.paths, "--model", "llama"])
        if args.model in ["qwen", "both"]:
            run([sys.executable, "scripts/12_eval_heatmap.py", "--exp", args.exp, "--paths", args.paths, "--model", "qwen"])

    # -------- attention --------
    if not args.skip_attention:
        if args.model in ["llama", "both"]:
            run([sys.executable, "scripts/13_attention_analysis.py", "--exp", args.exp, "--paths", args.paths, "--model", "llama"])
        if args.model in ["qwen", "both"]:
            run([sys.executable, "scripts/13_attention_analysis.py", "--exp", args.exp, "--paths", args.paths, "--model", "qwen"])


if __name__ == "__main__":
    main()
