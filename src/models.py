from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TF_PATTERN = re.compile(r"\b(true|false)\b", flags=re.IGNORECASE)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class LoadedModel:
    model_id: str
    tokenizer: Any
    model: Any


def _get_max_memory() -> Optional[Dict[Any, str]]:
    """
    Optional GPU/CPU memory cap via env vars:
      GPU_MEMORY="24GiB", CPU_MEMORY="64GiB"
    """
    gpu_mem = os.getenv("GPU_MEMORY")
    cpu_mem = os.getenv("CPU_MEMORY")
    if not gpu_mem and not cpu_mem:
        return None
    return {
        0: gpu_mem or "24GiB",
        "cpu": cpu_mem or "64GiB",
    }


def load_model(model_id: str, *, offload_dir: str = "offload") -> LoadedModel:
    """
    Load a HF causal LM with sane defaults for large models.
    Works for both:
      - meta-llama/Llama-2-13b-chat-hf
      - Qwen/Qwen2.5-14B-Instruct  (or your exact repo id)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    max_memory = _get_max_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
    ).eval()

    return LoadedModel(model_id=model_id, tokenizer=tokenizer, model=model)


def generate_text(
    lm: LoadedModel,
    prompt: str,
    *,
    gen: GenerationConfig,
) -> str:
    tok = lm.tokenizer(prompt, return_tensors="pt")
    tok = {k: v.to(lm.model.device) for k, v in tok.items()}

    with torch.no_grad():
        out = lm.model.generate(
            **tok,
            max_new_tokens=gen.max_new_tokens,
            do_sample=(gen.temperature > 0),
            temperature=gen.temperature if gen.temperature > 0 else None,
            top_p=gen.top_p,
        )

    generated = out[0][tok["input_ids"].shape[-1]:]
    text = lm.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def extract_true_false_list(text: str, n: int) -> List[str]:
    """
    Extract up to n occurrences of True/False from model output.
    If fewer than n, pad with "NA". If more than n, truncate.
    """
    matches = [m.group(1).lower() for m in TF_PATTERN.finditer(text)]
    norm = ["True" if x == "true" else "False" for x in matches]
    if len(norm) < n:
        norm += ["NA"] * (n - len(norm))
    return norm[:n]
