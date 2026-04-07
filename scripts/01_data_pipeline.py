"""
Week 1 — Data Pipeline
Loads FinCoT (which already has CoT reasoning traces) → filters → formats → splits.

NO API CALLS NEEDED. FinCoT includes Reasoning_process for every example.
Runs in ~2-5 minutes locally.

Usage:
    python scripts/01_data_pipeline.py

Outputs (in data/processed/):
    sft_train_v1.jsonl   — training set (~1,800+ examples)
    sft_val_v1.jsonl     — validation set (~200+ examples)
    rl_raw.jsonl         — raw RL split for GRPO (week 3-4)

Data flow:
    FinCoT SFT (7,686)
      → difficulty filter        → ~5,500  (drop trivial lookup questions)
      → answer extraction        → ~5,000  (drop non-numeric answers)
      → quality filter           → ~3,000  (drop low-quality traces)
      → 90/10 split              → train ~2,700 / val ~300
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_number, answers_match

load_dotenv()

WANDB_PROJECT = "financial-reasoning-ft"
PROCESSED_DIR = Path("data/processed")


# ─────────────────────────────────────────────────────────────────
# STEP 1: Load and parse FinCoT
# ─────────────────────────────────────────────────────────────────

def extract_final_answer(final_response: str) -> Optional[str]:
    """
    Extract the key numeric answer from FinCoT's narrative Final_response.

    Examples handled:
      "The free cash flow increased by approximately 7.44% from 2014 to 2015." → "7.44"
      "Approximately 80.64% of the total contractual obligations..."           → "80.64"
      "The total assets for the year 2019 are $5,765.2 million."               → "5765.2"
      "Approximately 8,816 towers were leased or subleased in 2004."           → "8816"
    """
    # Match numbers: optional $, integer with commas, optional decimal, optional %
    raw = re.findall(r'\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*%?', final_response)

    candidates = []
    for m in raw:
        clean = m.replace(",", "")
        try:
            val = float(clean)
            # Skip years (1900–2100) and plain zero
            if 1900 <= val <= 2100 or val == 0:
                continue
            candidates.append(clean)
        except ValueError:
            continue

    if not candidates:
        return None

    # Prefer decimal numbers (computed results) over integers
    decimals = [c for c in candidates if "." in c]
    return decimals[0] if decimals else candidates[0]


def parse_question_field(raw: str) -> tuple[str, str]:
    """
    Parse FinCoT's Question field into (context, question).

    Format:
        Please answer the given financial question based on the context.
        Context: [financial tables/text]
        Question: [the actual question]
        Answer:
    """
    ctx_match = re.search(r"Context:(.*?)Question:", raw, re.DOTALL)
    q_match   = re.search(r"Question:(.*?)(?:Answer:|$)", raw, re.DOTALL)

    context  = ctx_match.group(1).strip() if ctx_match else ""
    question = q_match.group(1).strip()   if q_match  else raw.strip()

    return context, question


def load_fincot() -> tuple[list[dict], list[dict]]:
    """
    Load and parse FinCoT SFT + RL splits.

    FinCoT columns:
      Question           — context + question bundled together
      Reasoning_process  — step-by-step CoT (what we use as the <think> block)
      Final_response     — narrative answer (we extract the number from here)
      Negative_*         — bad reasoning examples (not used here)

    Returns: (sft_examples, rl_examples)
    """
    print("\n[Step 1] Loading TheFinAI/FinCoT...")
    sft_ds = load_dataset("TheFinAI/FinCoT", split="SFT")
    rl_ds  = load_dataset("TheFinAI/FinCoT", split="RL")
    print(f"  SFT split: {len(sft_ds)} examples")
    print(f"  RL split:  {len(rl_ds)} examples")

    def parse_example(ex: dict, source: str) -> Optional[dict]:
        context, question = parse_question_field(ex["Question"])
        reasoning         = ex["Reasoning_process"].strip()
        final_response    = ex["Final_response"].strip()

        if not question or not reasoning:
            return None

        # Extract the numeric answer from the narrative Final_response
        answer_str = extract_final_answer(final_response)
        if answer_str is None:
            return None

        # Build the formatted trace: wrap reasoning in <think> tags
        trace = f"<think>\n{reasoning}\n</think>\n<answer>{answer_str}</answer>"

        return {
            "source":         source,
            "question":       question,
            "context":        context,
            "answer":         answer_str,
            "reasoning_trace": trace,
        }

    sft_examples = []
    for ex in tqdm(sft_ds, desc="  Parsing SFT"):
        parsed = parse_example(ex, "fincot_sft")
        if parsed:
            sft_examples.append(parsed)

    rl_examples = []
    for ex in tqdm(rl_ds, desc="  Parsing RL"):
        parsed = parse_example(ex, "fincot_rl")
        if parsed:
            rl_examples.append(parsed)

    print(f"  Parsed SFT: {len(sft_examples)} / {len(sft_ds)}")
    print(f"  Parsed RL:  {len(rl_examples)} / {len(rl_ds)}")
    return sft_examples, rl_examples


# ─────────────────────────────────────────────────────────────────
# STEP 2: Difficulty filter
# ─────────────────────────────────────────────────────────────────

def is_too_easy(example: dict) -> bool:
    """
    Filter out trivially easy questions that require no calculation.
    Easy = single lookup, yes/no, name/date questions.
    """
    q = example["question"].lower()

    calc_keywords = [
        "percent", "change", "growth", "ratio", "margin", "increase",
        "decrease", "compare", "average", "total", "difference", "rate",
        "return", "yield", "cagr", "how much", "calculate", "by how",
    ]
    has_calc = any(kw in q for kw in calc_keywords)

    trivial_patterns = [
        r"what (is|was|were) the (?:name|date|location)",
        r"(name|identify|list|which) the",
        r"(true|false):",
        r"yes or no",
    ]
    is_trivial = any(re.search(p, q) for p in trivial_patterns)

    return is_trivial or not has_calc


def apply_difficulty_filter(examples: list[dict]) -> list[dict]:
    print("\n[Step 2] Applying difficulty filter...")
    before = len(examples)
    hard   = [ex for ex in examples if not is_too_easy(ex)]
    print(f"  {before} → {len(hard)} ({len(hard)/before:.0%} kept)")
    return hard


# ─────────────────────────────────────────────────────────────────
# STEP 3: Quality filter
# ─────────────────────────────────────────────────────────────────

def passes_quality_filter(example: dict) -> tuple[bool, dict]:
    """
    Returns (passes, scores) for each example.
    Checks 6 criteria; requires 5/6 to pass.
    """
    trace   = example.get("reasoning_trace", "")
    context = example.get("context", "")
    answer  = str(example.get("answer", ""))

    think_match = re.search(r"<think>(.*?)</think>", trace, re.DOTALL)
    if not think_match:
        return False, {"fail": "no_think_block"}
    reasoning = think_match.group(1)

    scores = {
        # Must have structured tags
        "has_think_block":      bool(think_match),
        "has_answer_tag":       bool(re.search(r"<answer>.+?</answer>", trace)),

        # Reasoning must show arithmetic work
        "has_calc_steps":       len(re.findall(
            r"(Step \d|[\+\-×÷=]|subtract|divide|multiply|add|calculat|percent|ratio)",
            reasoning, re.IGNORECASE
        )) >= 3,

        # Reasoning must reference numbers from the financial context
        "uses_context_numbers": any(
            num in reasoning
            for num in re.findall(r"\b\d[\d,\.]+\b", context)[:8]
        ),

        # Must be substantive (not a one-liner)
        "sufficient_length":    len(reasoning.split()) >= 40,

        # Final answer must match ground truth
        "answer_correct":       answers_match(
            extract_number(
                re.search(r"<answer>(.*?)</answer>", trace, re.DOTALL).group(1)
                if re.search(r"<answer>(.*?)</answer>", trace, re.DOTALL) else ""
            ),
            answer,
        ),
    }

    passed = sum(scores.values()) >= 5
    return passed, scores


def apply_quality_filter(examples: list[dict]) -> list[dict]:
    print("\n[Step 3] Applying quality filter...")
    before  = len(examples)
    passed  = []
    reasons = {}

    for ex in tqdm(examples, desc="  Filtering"):
        ok, scores = passes_quality_filter(ex)
        if ok:
            passed.append(ex)
        else:
            # Track which check is failing most
            for k, v in scores.items():
                if not v:
                    reasons[k] = reasons.get(k, 0) + 1

    print(f"  {before} → {len(passed)} ({len(passed)/before:.0%} kept)")
    if reasons:
        print("  Top failure reasons:")
        for k, n in sorted(reasons.items(), key=lambda x: -x[1])[:3]:
            print(f"    {k}: {n} examples")
    return passed


# ─────────────────────────────────────────────────────────────────
# STEP 4: Save
# ─────────────────────────────────────────────────────────────────

def hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def save_and_split(sft_examples: list[dict], rl_examples: list[dict]) -> None:
    print("\n[Step 4] Splitting and saving...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(sft_examples)

    split_idx = int(len(sft_examples) * 0.90)
    sft_train = sft_examples[:split_idx]
    sft_val   = sft_examples[split_idx:]

    files = [
        ("sft_train_v1", sft_train),
        ("sft_val_v1",   sft_val),
        ("rl_raw",       rl_examples),
    ]

    for name, data in files:
        path = PROCESSED_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        h = hash_file(str(path))
        print(f"  {name}: {len(data)} examples  |  hash: {h[:8]}...")
        wandb.config.update({f"data/{name}_hash": h})

    print(f"\n  ✅ Pipeline complete")
    print(f"     SFT train : {len(sft_train)}")
    print(f"     SFT val   : {len(sft_val)}")
    print(f"     RL (GRPO) : {len(rl_examples)}")
    print(f"\n  Files saved to: {PROCESSED_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    run = wandb.init(
        project=WANDB_PROJECT,
        name="data-pipeline-v1",
        tags=["data"],
    )

    sft_raw, rl_raw = load_fincot()

    sft_hard    = apply_difficulty_filter(sft_raw)
    sft_quality = apply_quality_filter(sft_hard)

    # RL examples: no quality filter needed (used for GRPO reward, not SFT format)
    rl_quality  = apply_difficulty_filter(rl_raw)

    wandb.log({
        "data/sft_raw":         len(sft_raw),
        "data/sft_after_diff":  len(sft_hard),
        "data/sft_after_qual":  len(sft_quality),
        "data/rl_raw":          len(rl_raw),
        "data/rl_after_diff":   len(rl_quality),
    })

    if len(sft_quality) < 1800:
        print(f"\n⚠️  Only {len(sft_quality)} examples — below 1,800 threshold.")
        print("   Consider: lowering quality filter from 5/6 → 4/6")
    else:
        print(f"\n✅ Gate passed: {len(sft_quality)} high-quality SFT examples")

    save_and_split(sft_quality, rl_quality)
    wandb.finish()


if __name__ == "__main__":
    main()
