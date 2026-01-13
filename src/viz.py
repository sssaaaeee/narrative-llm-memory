from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


TOPICS = ["location", "temporal", "entity", "content"]
DISTRACTORS = ["NI", "MI", "RI", "UI"]


@dataclass
class HeatmapResult:
    acc_high: np.ndarray  # shape: (len(TOPICS), 4)
    acc_low: np.ndarray   # shape: (len(TOPICS), 4)
    chapter_counts_high: Dict[str, List[int]]  # topic -> [count per dist]
    chapter_counts_low: Dict[str, List[int]]   # topic -> [count per dist]


def _to_bool(x: str) -> Optional[bool]:
    if x == "True":
        return True
    if x == "False":
        return False
    return None  # NA or unknown


def compute_topic_accuracy_matrices(
    qa_data: List[dict],
    responses_by_condition: Dict[str, List[List[str]]],
    *,
    topics: List[str] = TOPICS,
    distractors: List[str] = DISTRACTORS,
) -> HeatmapResult:
    """
    qa_data: list of chapters, each has:
      - answers: list[str] in {"True","False"}
      - topics: list[str] in topics
    responses_by_condition: dict condition -> chapterwise predictions
      e.g., responses_by_condition["h_NI"][chapter_idx] = ["True","False",...]
    Returns:
      acc_high/topics×distractors, acc_low/topics×distractors
      where each cell is the mean over chapters of that chapter's per-topic accuracy.
    """
    # per level/topic/dist -> list of chapter accuracies
    chapter_acc = {
        "high": {t: [[] for _ in distractors] for t in topics},
        "low": {t: [[] for _ in distractors] for t in topics},
    }

    # mapping dist index
    dist_to_idx = {d: i for i, d in enumerate(distractors)}

    # condition keys expected: h_NI, l_NI, ...
    for chapter_idx, qa_ch in enumerate(qa_data):
        gold = qa_ch["answers"]
        q_topics = qa_ch["topics"]

        # for each distractor, handle high & low separately
        for d in distractors:
            for level, prefix in [("high", "h_"), ("low", "l_")]:
                cond = f"{prefix}{d}"
                preds = responses_by_condition.get(cond, [])
                if chapter_idx >= len(preds):
                    continue
                pred_list = preds[chapter_idx]

                # compute per-topic counts within this chapter/condition
                correct = {t: 0 for t in topics}
                total = {t: 0 for t in topics}

                n = min(len(gold), len(pred_list))
                for i in range(n):
                    t = q_topics[i]
                    if t not in correct:
                        continue

                    g = _to_bool(gold[i])
                    p = _to_bool(pred_list[i])
                    if g is None or p is None:
                        continue  # skip NA / invalid

                    total[t] += 1
                    if p == g:
                        correct[t] += 1

                di = dist_to_idx[d]
                for t in topics:
                    if total[t] > 0:
                        chapter_acc[level][t][di].append(correct[t] / total[t])

    # aggregate to matrices (topic x distractor)
    acc_high = np.zeros((len(topics), len(distractors)), dtype=float)
    acc_low = np.zeros((len(topics), len(distractors)), dtype=float)

    counts_high = {t: [len(chapter_acc["high"][t][i]) for i in range(len(distractors))] for t in topics}
    counts_low = {t: [len(chapter_acc["low"][t][i]) for i in range(len(distractors))] for t in topics}

    for ti, t in enumerate(topics):
        for di in range(len(distractors)):
            acc_high[ti, di] = float(np.mean(chapter_acc["high"][t][di])) if chapter_acc["high"][t][di] else 0.0
            acc_low[ti, di] = float(np.mean(chapter_acc["low"][t][di])) if chapter_acc["low"][t][di] else 0.0

    return HeatmapResult(
        acc_high=acc_high,
        acc_low=acc_low,
        chapter_counts_high=counts_high,
        chapter_counts_low=counts_low,
    )


def plot_topic_accuracy_heatmap(
    result: HeatmapResult,
    *,
    model_label: str,
    out_png: str | Path,
    out_pdf: str | Path,
    vmin: float,
    vmax: float,
    topic_labels: Optional[List[str]] = None,
    distractor_labels: Optional[List[str]] = None,
    font_size: int = 22,
) -> None:
    """
    Matplotlib-only heatmap (no seaborn dependency).
    Produces a 1x2 figure: High | Low.
    """
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    topic_labels = topic_labels or ["s", "t", "ent", "c"]
    distractor_labels = distractor_labels or DISTRACTORS

    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["axes.labelsize"] = font_size
    mpl.rcParams["xtick.labelsize"] = font_size
    mpl.rcParams["ytick.labelsize"] = font_size

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.25)
    ax_h = fig.add_subplot(gs[0])
    ax_l = fig.add_subplot(gs[1])

    def draw(ax, mat, title, show_y: bool, show_cbar: bool):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Distractor Conditions")
        ax.set_xticks(range(len(distractor_labels)))
        ax.set_xticklabels(distractor_labels)

        if show_y:
            ax.set_ylabel("Topics")
            ax.set_yticks(range(len(topic_labels)))
            ax.set_yticklabels(topic_labels)
        else:
            ax.set_yticks(range(len(topic_labels)))
            ax.set_yticklabels([])

        # annotate
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")

        if show_cbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Accuracy")

    draw(ax_h, result.acc_high, f"High ( {model_label} )", show_y=True, show_cbar=False)
    draw(ax_l, result.acc_low, f"Low ( {model_label} )", show_y=False, show_cbar=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Attention Visualization
# -----------------------------

def plot_element_attention_2x2(
    results: dict,
    output_dir: str | Path,
    model_name: str = "Model",
    save_format: str = "pdf",
    show_values: bool = False,
) -> None:
    """
    4つの干渉条件を 2×2 の1枚図にまとめて、
    temporal/location/entity/content の attention 平均を High vs Low で比較。

    Args:
        results: results_attention.jsonから読み込んだ結果データ
        output_dir: 出力ディレクトリ
        model_name: 図タイトル等に使うモデル名
        save_format: "png" or "pdf"
        show_values: バー上に数値を出すか
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition mapping: h_NI, l_NI, h_MI, l_MI, h_RI, l_RI, h_UI, l_UI
    conditions = [
        {'name': 'No Interference (NI)', 'hn_key': 'h_NI', 'ln_key': 'l_NI'},
        {'name': 'Meaningless Interference (MI)', 'hn_key': 'h_MI', 'ln_key': 'l_MI'},
        {'name': 'Related Interference (RI)', 'hn_key': 'h_RI', 'ln_key': 'l_RI'},
        {'name': 'Unrelated Interference (UI)', 'hn_key': 'h_UI', 'ln_key': 'l_UI'}
    ]

    categories = ['temporal', 'location', 'entity', 'content']
    category_labels = ['Temporal', 'Location', 'Entity', 'Content']

    # 2×2 サブプロット
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharey=True)
    axes = axes.flatten()

    hn_color = '#2E86AB'
    ln_color = '#A23B72'
    width = 0.36
    x = np.arange(len(categories))

    # y上限を決定
    all_vals = []
    results_data = results.get('results', {})
    for c in conditions:
        if c['hn_key'] in results_data:
            hn_data = results_data[c['hn_key']]['label_ratios_mean']
            ln_data = results_data[c['ln_key']]['label_ratios_mean']
            all_vals.extend([hn_data[k] for k in categories])
            all_vals.extend([ln_data[k] for k in categories])
    ymax = max(all_vals) * 1.15 if len(all_vals) else 1.0

    legend_handles = None

    for i, (ax, condition) in enumerate(zip(axes, conditions)):
        if condition['hn_key'] not in results_data:
            continue
            
        hn_data = results_data[condition['hn_key']]['label_ratios_mean']
        ln_data = results_data[condition['ln_key']]['label_ratios_mean']
        
        hn_std = results_data[condition['hn_key']].get('label_ratios_std', {})
        ln_std = results_data[condition['ln_key']].get('label_ratios_std', {})
        n_chapters = results_data[condition['hn_key']].get('num_chapters', 100)

        hn_values = [hn_data[cat] for cat in categories]
        ln_values = [ln_data[cat] for cat in categories]
        
        # 標準誤差
        hn_se = [hn_std.get(cat, 0) / np.sqrt(n_chapters) for cat in categories]
        ln_se = [ln_std.get(cat, 0) / np.sqrt(n_chapters) for cat in categories]

        bars1 = ax.bar(x - width/2, hn_values, width, label='High Narrativity',
                       color=hn_color, alpha=0.85, edgecolor='black', linewidth=0.8,
                       yerr=hn_se, capsize=4, error_kw={'linewidth': 1.2, 'elinewidth': 1.2})
        bars2 = ax.bar(x + width/2, ln_values, width, label='Low Narrativity',
                       color=ln_color, alpha=0.85, edgecolor='black', linewidth=0.8,
                       yerr=ln_se, capsize=4, error_kw={'linewidth': 1.2, 'elinewidth': 1.2})

        if legend_handles is None:
            legend_handles = (bars1[0], bars2[0])

        ax.set_title(condition['name'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, fontsize=10)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.set_ylim(0, ymax)

        if i % 2 == 0:
            ax.set_ylabel('Attention Ratio (of Text Region)', fontsize=10)

        if show_values:
            for bar in list(bars1) + list(bars2):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'{model_name}: Element-wise Attention Allocation (Final Layer, Head-Averaged)',
                 fontsize=12, fontweight='bold', y=1.00)

    fig.legend(legend_handles, ['High Narrativity', 'Low Narrativity'],
               loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2,
               frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = output_dir / f"{model_name.lower().replace('/', '_')}_element_attention_2x2.{save_format}"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_element_attention_1x4(
    results: dict,
    output_dir: str | Path,
    model_name: str = "Model",
    save_format: str = "pdf",
    show_values: bool = False,
) -> None:
    """
    縦方向に4つのサブプロットを並べたレイアウト (1x4)。
    各行が1条件を表し，2x2図と同じ指標を示す。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        {'name': 'No Interference (NI)', 'hn_key': 'h_NI', 'ln_key': 'l_NI'},
        {'name': 'Meaningless Interference (MI)', 'hn_key': 'h_MI', 'ln_key': 'l_MI'},
        {'name': 'Related Interference (RI)', 'hn_key': 'h_RI', 'ln_key': 'l_RI'},
        {'name': 'Unrelated Interference (UI)', 'hn_key': 'h_UI', 'ln_key': 'l_UI'}
    ]

    categories = ['temporal', 'location', 'entity', 'content']
    category_labels = ['Temporal', 'Location', 'Entity', 'Content']

    # 1x4 の縦配置
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 12), sharey=True)

    hn_color = '#2E86AB'
    ln_color = '#A23B72'
    width = 0.36
    x = np.arange(len(categories))

    # y上限を決定
    all_vals = []
    results_data = results.get('results', {})
    for c in conditions:
        if c['hn_key'] in results_data:
            hn_data = results_data[c['hn_key']]['label_ratios_mean']
            ln_data = results_data[c['ln_key']]['label_ratios_mean']
            all_vals.extend([hn_data[k] for k in categories])
            all_vals.extend([ln_data[k] for k in categories])
    ymax = max(all_vals) * 1.15 if len(all_vals) else 1.0

    legend_handles = None

    for i, (ax, condition) in enumerate(zip(axes, conditions)):
        if condition['hn_key'] not in results_data:
            continue
            
        hn_data = results_data[condition['hn_key']]['label_ratios_mean']
        ln_data = results_data[condition['ln_key']]['label_ratios_mean']

        hn_std = results_data[condition['hn_key']].get('label_ratios_std', {})
        ln_std = results_data[condition['ln_key']].get('label_ratios_std', {})
        n_chapters = results_data[condition['hn_key']].get('num_chapters', 100)

        hn_values = [hn_data[cat] for cat in categories]
        ln_values = [ln_data[cat] for cat in categories]

        hn_se = [hn_std.get(cat, 0) / np.sqrt(n_chapters) for cat in categories]
        ln_se = [ln_std.get(cat, 0) / np.sqrt(n_chapters) for cat in categories]

        bars1 = ax.bar(x - width/2, hn_values, width, label='High Narrativity',
                       color=hn_color, alpha=0.85, edgecolor='black', linewidth=0.8,
                       yerr=hn_se, capsize=4, error_kw={'linewidth': 1.2, 'elinewidth': 1.2})
        bars2 = ax.bar(x + width/2, ln_values, width, label='Low Narrativity',
                       color=ln_color, alpha=0.85, edgecolor='black', linewidth=0.8,
                       yerr=ln_se, capsize=4, error_kw={'linewidth': 1.2, 'elinewidth': 1.2})

        if legend_handles is None:
            legend_handles = (bars1[0], bars2[0])

        ax.set_title(condition['name'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        
        if i == len(conditions) - 1:
            ax.set_xticklabels(category_labels, fontsize=10)
        else:
            ax.set_xticklabels([''] * len(category_labels))

        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.set_ylim(0, ymax)

        if i == 0:
            ax.set_ylabel('Attention Ratio (of Text Region)', fontsize=10)

        if show_values:
            for bar in list(bars1) + list(bars2):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'{model_name}: Element-wise Attention Allocation (Final Layer, Head-Averaged)',
                 fontsize=12, fontweight='bold', y=0.995)

    fig.legend(legend_handles, ['High Narrativity', 'Low Narrativity'],
               loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=2,
               frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"{model_name.lower().replace('/', '_')}_element_attention_1x4.{save_format}"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
