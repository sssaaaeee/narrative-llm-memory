# src/prompts.py
from __future__ import annotations

from typing import Dict, List, Optional


def _format_block(n: int) -> str:
    return f"""
    There are exactly {n} questions.
    You must answer ALL of them.
    You must output exactly {n} lines.

    Format:
    1. True or False
    2. True or False
    3. True or False
    4. True or False
    5. True or False

    Do not stop early.
    Do not output anything else.
    """.strip()


def _question_block(questions: List[str]) -> str:
    return "\n".join(questions).strip()


def build_prompt(
    *,
    story_text: str,
    questions: List[str],
    distractor_text: Optional[str] = None,
) -> str:
    """
    Matches the user's original create_prompt() style:
      text: {story}
      ~~{distractor}   (only if provided)
      question: {questions}
      {format_prompt}
      answer(True / False):
    """
    q = _question_block(questions)
    fmt = _format_block(len(questions))

    parts: List[str] = []
    parts.append(f"text: {story_text}".rstrip())

    if distractor_text:
        parts.append("")  # blank line
        parts.append(f"~~{distractor_text}".rstrip())

    parts.append("")  # blank line
    parts.append("question: " + q)
    parts.append("")
    parts.append(fmt)
    parts.append("")
    parts.append("answer(True / Flase):")
    return "\n".join(parts)


def build_all_prompts_for_chapter(
    *,
    base_chapter: Dict,
    distractor_chapter: Dict,
    qa_chapter: Dict,
) -> Dict[str, str]:
    """
    Returns prompts for 8 conditions:
      h_NI, l_NI, h_MI, l_MI, h_RI, l_RI, h_UI, l_UI
    Assumes:
      base_chapter: has high_narrativity_text / low_narrativity_text
      distractor_chapter: has MI/RI/UI
      qa_chapter: has questions
    """
    h_text = base_chapter["high_narrativity_text"]
    l_text = base_chapter["low_narrativity_text"]
    questions = qa_chapter["questions"]

    return {
        "h_NI": build_prompt(story_text=h_text, questions=questions, distractor_text=None),
        "l_NI": build_prompt(story_text=l_text, questions=questions, distractor_text=None),
        "h_MI": build_prompt(story_text=h_text, questions=questions, distractor_text=distractor_chapter["MI"]),
        "l_MI": build_prompt(story_text=l_text, questions=questions, distractor_text=distractor_chapter["MI"]),
        "h_RI": build_prompt(story_text=h_text, questions=questions, distractor_text=distractor_chapter["RI"]),
        "l_RI": build_prompt(story_text=l_text, questions=questions, distractor_text=distractor_chapter["RI"]),
        "h_UI": build_prompt(story_text=h_text, questions=questions, distractor_text=distractor_chapter["UI"]),
        "l_UI": build_prompt(story_text=l_text, questions=questions, distractor_text=distractor_chapter["UI"]),
    }
