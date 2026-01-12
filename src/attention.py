# src/attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


REGIONS = ["text", "distractor", "question_other"]
LABELS = ["temporal", "location", "entity", "content", "other"]


# -----------------------------
# model / attention extraction
# -----------------------------
@dataclass
class HFBackbone:
    model_name: str
    tokenizer: AutoTokenizer
    model: AutoModel
    device: torch.device


def load_backbone_for_attention(
    model_name: str,
    *,
    hf_token_env: str = "HF_TOKEN",
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> HFBackbone:
    """
    Load backbone model with output_attentions=True usage in forward.
    Important: attn_implementation='eager' so attention tensors exist.
    """
    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)

    use_cuda = torch.cuda.is_available()
    if device is None:
        device = "cuda" if use_cuda else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,  # keep simple; user can change later
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    dev = torch.device(device)
    model.to(dev)

    return HFBackbone(model_name=model_name, tokenizer=tok, model=model, device=dev)


@torch.no_grad()
def get_last_token_last_layer_attention(
    backbone: HFBackbone,
    text: str,
) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      tokens: list[str] length = seq_len (token strings)
      vectors: Tensor shape [heads, seq_len] attention from last-token query to all keys (last layer)
    """
    inputs = backbone.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(backbone.device) for k, v in inputs.items()}

    tokens = backbone.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    out = backbone.model(**inputs, output_attentions=True)
    attns = out.attentions  # list[num_layers] of [B, H, Q, K]
    last_layer = attns[-1][0]  # [H, Q, K]
    seq_len = inputs["input_ids"].shape[1]
    last_q = seq_len - 1
    vectors = last_layer[:, last_q, :]  # [H, K==seq_len]

    return tokens, vectors


# -----------------------------
# token -> word merge
# -----------------------------
def merge_tokens_to_words_llama(tokens: List[str], vectors: torch.Tensor) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Llama tokenizer: word starts with '▁'
    vectors: [heads, seq_len]
    returns:
      words: list[str]
      attn_vecs: list[Tensor[heads]] aligned with words
    """
    words: List[str] = []
    attn: List[torch.Tensor] = []

    buf_word = ""
    buf_vecs: List[torch.Tensor] = []

    def flush():
        nonlocal buf_word, buf_vecs
        if buf_word != "":
            words.append(buf_word.lower())
            stacked = torch.stack(buf_vecs, dim=0)  # [n_sub, heads]
            attn.append(stacked.mean(dim=0))         # [heads]
            buf_word = ""
            buf_vecs = []

    for i, t in enumerate(tokens):
        v = vectors[:, i]  # [heads]

        if t in ["<s>", "</s>", "<0x0A>"]:
            flush()
            words.append(t)
            attn.append(v)
            continue

        if t.startswith("▁"):
            flush()
            buf_word = t[1:]
            buf_vecs = [v]
            continue

        if t in [",", ".", "!", "?"]:
            flush()
            words.append(t)
            attn.append(v)
            continue

        # continuation
        buf_word += t
        buf_vecs.append(v)

    flush()
    return words, attn


def merge_tokens_to_words_qwen(tokens: List[str], vectors: torch.Tensor) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Qwen2.5 tokenizer often uses 'Ġ' to indicate new word.
    Also has newline-ish tokens like 'ċ', 'ċċ', 'Ċ' etc in some vocab.
    """
    words: List[str] = []
    attn: List[torch.Tensor] = []

    buf_word = ""
    buf_vecs: List[torch.Tensor] = []

    def flush():
        nonlocal buf_word, buf_vecs
        if buf_word != "":
            words.append(buf_word.lower())
            stacked = torch.stack(buf_vecs, dim=0)  # [n_sub, heads]
            attn.append(stacked.mean(dim=0))
            buf_word = ""
            buf_vecs = []

    for i, t in enumerate(tokens):
        v = vectors[:, i]

        # treat some special tokens as standalone
        if t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "Ċ"]:
            flush()
            words.append(t)
            attn.append(v)
            continue

        if t.startswith("Ġ"):
            flush()
            buf_word = t[1:]
            buf_vecs = [v]
            continue

        if t.startswith((",", ".", "!", "?", "ċ", "ċċ")):
            flush()
            words.append(t)
            attn.append(v)
            continue

        buf_word += t
        buf_vecs.append(v)

    flush()
    return words, attn


def merge_tokens_to_words(
    model_name: str,
    tokens: List[str],
    vectors: torch.Tensor,
) -> Tuple[List[str], List[torch.Tensor]]:
    lower = model_name.lower()
    if "qwen" in lower:
        return merge_tokens_to_words_qwen(tokens, vectors)
    return merge_tokens_to_words_llama(tokens, vectors)


# -----------------------------
# region split: text / distractor / question_other
# -----------------------------
def find_question_start_llama(words: List[str]) -> int:
    # your old logic was brittle; simplest robust:
    for i, w in enumerate(words):
        if "question:" in w.lower() or w.lower() == "question":
            return i
    return -1


def find_question_start_qwen(words: List[str]) -> int:
    for i, w in enumerate(words):
        if "question:" in w.lower():
            return i
    return -1


def find_distractor_start(words: List[str]) -> int:
    for i, w in enumerate(words):
        if "~~" in w:
            return i
    return -1


def classify_regions(model_name: str, words: List[str]) -> List[str]:
    q = find_question_start_qwen(words) if "qwen" in model_name.lower() else find_question_start_llama(words)
    d = find_distractor_start(words)

    regions: List[str] = []
    for i in range(len(words)):
        if d != -1 and i >= d:
            if q != -1 and i >= q:
                regions.append("question_other")
            else:
                regions.append("distractor")
        elif q != -1 and i >= q:
            regions.append("question_other")
        else:
            regions.append("text")
    return regions


# -----------------------------
# label text words: temporal/location/entity/content/other
# -----------------------------
FUNCTION_WORDS = {
    # articles
    "a", "an", "the",
    # conjunctions
    "and", "or", "but", "so", "yet", "for", "nor", "because", "since", "as",
    "if", "when", "while", "although", "though", "unless", "until", "after", "before",
    # prepositions
    "in", "on", "at", "to", "from", "with", "by", "about", "of", "up", "down",
    "into", "over", "under", "between", "through", "during", "without", "within",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
    # be / aux
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # misc
    "not", "no", "yes", "all", "some", "any", "many", "much", "few", "little",
    "more", "most", "less", "least", "very", "too", "also", "just", "only",
    "there", "here", "where", "how", "why", "then", "now",
}


def load_elements_for_matching(elements_json: dict) -> Dict[str, set]:
    # expected keys: temporal/location/entity/content
    return {
        "temporal": set([x.lower() for x in elements_json["temporal"]]),
        "location": set([x.lower() for x in elements_json["location"]]),
        "entity": set([x.lower() for x in elements_json["entity"]]),
        "content": set([x.lower() for x in elements_json["content"]]),
    }


def classify_word_label(word: str, elements: Dict[str, set]) -> str:
    w = word.lower()
    if w in FUNCTION_WORDS:
        return "other"

    # partial match, with priority temporal > location > entity > content
    for e in elements["temporal"]:
        if e in w or w in e:
            return "temporal"
    for e in elements["location"]:
        if e in w or w in e:
            return "location"
    for e in elements["entity"]:
        if e in w or w in e:
            return "entity"
    for e in elements["content"]:
        if e in w or w in e:
            return "content"
    return "other"


# -----------------------------
# per-prompt analysis + aggregation
# -----------------------------
@dataclass
class PromptAttentionSummary:
    region_sums: Dict[str, float]           # sum of mean(attn_vec) over words per region
    region_ratios: Dict[str, float]         # region_sums / total
    label_sums: Optional[Dict[str, float]]  # for text region only
    label_ratios: Optional[Dict[str, float]]# label_sums / text_sum
    total_sum: float
    text_sum: float


def analyze_prompt_attention(
    backbone: HFBackbone,
    prompt_text: str,
    *,
    elements: Optional[Dict[str, set]] = None,
) -> PromptAttentionSummary:
    tokens, vectors = get_last_token_last_layer_attention(backbone, prompt_text)
    words, attn_vecs = merge_tokens_to_words(backbone.model_name, tokens, vectors)
    regions = classify_regions(backbone.model_name, words)

    # region sums
    region_sums = {r: 0.0 for r in REGIONS}
    for r, vec in zip(regions, attn_vecs):
        region_sums[r] += float(vec.mean().item())

    total_sum = float(sum(region_sums.values()))
    region_ratios = {r: (region_sums[r] / total_sum if total_sum != 0 else 0.0) for r in REGIONS}
    text_sum = float(region_sums["text"])

    label_sums = None
    label_ratios = None
    if elements is not None:
        label_sums = {l: 0.0 for l in LABELS}
        for w, r, vec in zip(words, regions, attn_vecs):
            if r != "text":
                continue
            lab = classify_word_label(w, elements)
            label_sums[lab] += float(vec.mean().item())

        label_ratios = {l: (label_sums[l] / text_sum if text_sum != 0 else 0.0) for l in LABELS}

    return PromptAttentionSummary(
        region_sums=region_sums,
        region_ratios=region_ratios,
        label_sums=label_sums,
        label_ratios=label_ratios,
        total_sum=total_sum,
        text_sum=text_sum,
    )


def aggregate_prompt_summaries(per_chapter: List[PromptAttentionSummary]) -> dict:
    """
    Aggregate by:
      - mean/std over chapters for region_ratios and label_ratios
      - also keep mean/std for sums (optional reference)
    """
    import numpy as np

    # ratios
    region_ratio_mat = {r: [] for r in REGIONS}
    label_ratio_mat = {l: [] for l in LABELS}
    region_sum_mat = {r: [] for r in REGIONS}
    label_sum_mat = {l: [] for l in LABELS}
    total_sums = []
    text_sums = []

    for s in per_chapter:
        for r in REGIONS:
            region_ratio_mat[r].append(s.region_ratios[r])
            region_sum_mat[r].append(s.region_sums[r])
        total_sums.append(s.total_sum)
        text_sums.append(s.text_sum)
        if s.label_ratios is not None and s.label_sums is not None:
            for l in LABELS:
                label_ratio_mat[l].append(s.label_ratios[l])
                label_sum_mat[l].append(s.label_sums[l])

    def mean_std(xs: List[float]) -> Tuple[float, float]:
        if len(xs) == 0:
            return 0.0, 0.0
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    out = {
        "num_chapters": len(per_chapter),
        "region_ratios_mean": {r: mean_std(region_ratio_mat[r])[0] for r in REGIONS},
        "region_ratios_std": {r: mean_std(region_ratio_mat[r])[1] for r in REGIONS},
        "label_ratios_mean": {l: mean_std(label_ratio_mat[l])[0] for l in LABELS},
        "label_ratios_std": {l: mean_std(label_ratio_mat[l])[1] for l in LABELS},
        # reference (sums)
        "region_sums_mean": {r: mean_std(region_sum_mat[r])[0] for r in REGIONS},
        "region_sums_std": {r: mean_std(region_sum_mat[r])[1] for r in REGIONS},
        "label_sums_mean": {l: mean_std(label_sum_mat[l])[0] for l in LABELS},
        "label_sums_std": {l: mean_std(label_sum_mat[l])[1] for l in LABELS},
        "total_sum_mean": mean_std(total_sums)[0],
        "total_sum_std": mean_std(total_sums)[1],
        "text_sum_mean": mean_std(text_sums)[0],
        "text_sum_std": mean_std(text_sums)[1],
    }
    return out
