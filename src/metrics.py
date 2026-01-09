from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple


@dataclass
class BinaryMetrics:
    accuracy: float
    recall_micro: float
    precision_micro: float
    f1_micro: float
    recall_macro: float
    precision_macro: float
    f1_macro: float
    n_total: int
    n_valid: int
    n_na: int


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def compute_binary_metrics(
    preds: Sequence[str],
    golds: Sequence[str],
) -> BinaryMetrics:
    """
    preds/golds are flat lists with values in {"True","False"}.
    preds may include "NA" (missing output) which is excluded from n_valid.
    """
    assert len(preds) == len(golds)

    tp = tn = fp = fn = 0
    n_na = 0

    for p, g in zip(preds, golds):
        if p not in ("True", "False"):
            n_na += 1
            continue

        if p == "True" and g == "True":
            tp += 1
        elif p == "False" and g == "False":
            tn += 1
        elif p == "True" and g == "False":
            fp += 1
        elif p == "False" and g == "True":
            fn += 1

    n_total = len(golds)
    n_valid = n_total - n_na

    accuracy = _safe_div(tp + tn, n_valid) if n_valid else 0.0

    recall_micro = _safe_div(tp, tp + fn)
    precision_micro = _safe_div(tp, tp + fp)
    f1_micro = _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)

    # Macro for two classes (True=positive, False=negative)
    recall_true = _safe_div(tp, tp + fn)
    precision_true = _safe_div(tp, tp + fp)
    f1_true = _safe_div(2 * precision_true * recall_true, precision_true + recall_true)

    recall_false = _safe_div(tn, tn + fp)
    precision_false = _safe_div(tn, tn + fn)
    f1_false = _safe_div(2 * precision_false * recall_false, precision_false + recall_false)

    recall_macro = (recall_true + recall_false) / 2
    precision_macro = (precision_true + precision_false) / 2
    f1_macro = (f1_true + f1_false) / 2

    return BinaryMetrics(
        accuracy=accuracy,
        recall_micro=recall_micro,
        precision_micro=precision_micro,
        f1_micro=f1_micro,
        recall_macro=recall_macro,
        precision_macro=precision_macro,
        f1_macro=f1_macro,
        n_total=n_total,
        n_valid=n_valid,
        n_na=n_na,
    )


def flatten_chapterwise(labels: List[List[str]]) -> List[str]:
    return [x for row in labels for x in row]


def evaluate_conditions(
    *,
    responses: Dict[str, List[List[str]]],
    gold_answers: List[List[str]],
) -> Dict[str, Dict]:
    """
    responses: {"h_NI": [[...10], ...], ...}  chapterwise
    gold_answers: [[...10], ...]             chapterwise
    """
    gold_flat = flatten_chapterwise(gold_answers)

    out: Dict[str, Dict] = {}
    for cond, pred_chapterwise in responses.items():
        pred_flat = flatten_chapterwise(pred_chapterwise)
        m = compute_binary_metrics(pred_flat, gold_flat)
        out[cond] = asdict(m)
    return out
