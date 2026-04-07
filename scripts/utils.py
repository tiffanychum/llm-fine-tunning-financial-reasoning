"""
Shared scoring utilities — used by all scripts.
"""

import re
from typing import Optional


def extract_number(text: str) -> Optional[float]:
    """Extract the first numeric value from a string."""
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    # Handle M/B suffixes
    mult = 1.0
    if cleaned.endswith(("M", "m")):
        mult, cleaned = 1e6, cleaned[:-1]
    elif cleaned.endswith(("B", "b")):
        mult, cleaned = 1e9, cleaned[:-1]
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        try:
            return float(match.group()) * mult
        except ValueError:
            return None
    return None


def answers_match(predicted: Optional[float], ground_truth: str, tolerance: float = 0.01) -> bool:
    """
    Returns True if the predicted value is within `tolerance` (default 1%) of ground truth.
    Falls back to string comparison for non-numeric answers.
    """
    if predicted is None:
        return False
    true_val = extract_number(ground_truth)
    if true_val is None:
        # String fallback
        return str(predicted).strip().lower() == ground_truth.strip().lower()
    if abs(true_val) < 1e-9:
        return abs(predicted) < 1e-9
    return abs(predicted - true_val) / abs(true_val) < tolerance


def extract_answer_from_tags(text: str) -> Optional[str]:
    """Pull the content from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def score_completion(completion: str, ground_truth: str) -> dict:
    """
    Returns a breakdown dict:
      has_think  — model used <think> block
      has_answer — model used <answer> tag
      correct    — final answer is numerically correct (within 1%)
      score      — combined float 0.0–1.0
    """
    has_think  = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.+?</answer>", completion, re.DOTALL))
    answer_str = extract_answer_from_tags(completion)
    predicted  = extract_number(answer_str) if answer_str else None
    correct    = answers_match(predicted, ground_truth)

    if has_think and has_answer and correct:
        score = 1.0
    elif has_answer and correct:
        score = 0.8   # right answer but skipped think block
    elif has_think and has_answer:
        score = 0.2   # right format, wrong answer
    elif has_answer:
        score = 0.1
    else:
        score = 0.0

    return {
        "has_think":    has_think,
        "has_answer":   has_answer,
        "correct":      correct,
        "score":        score,
        "predicted":    str(predicted) if predicted is not None else answer_str,
    }
