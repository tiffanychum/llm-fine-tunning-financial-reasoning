"""
Week 0 — Local Baseline Evaluation (no GPU required)
Supports two backends:
  --backend openrouter  (default) — calls Qwen2.5-7B directly, the TRUE baseline
  --backend poe                   — calls a Poe bot (Claude Haiku, GPT-4o-mini, etc.)

Why OpenRouter is preferred:
  OpenRouter gives API access to Qwen/Qwen2.5-7B-Instruct — the EXACT model
  you will fine-tune. So the baseline score is directly comparable to your
  SFT and GRPO results. Poe doesn't offer Qwen models.

Usage:
    # True Qwen2.5-7B baseline via OpenRouter (default)
    python scripts/00_setup_eval_local.py --limit 3

    # Poe fallback (Claude Haiku, GPT-4o-mini, etc.)
    python scripts/00_setup_eval_local.py --backend poe --bot Claude-3-5-Haiku --limit 3

    # Full run (~100 numeric questions)
    python scripts/00_setup_eval_local.py

Output:
    results/baseline_local/baseline_scores.json
    W&B run tagged "baseline-local"
"""

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

import fastapi_poe as fp
import nest_asyncio
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

nest_asyncio.apply()

sys.path.insert(0, str(Path(__file__).parent))
from utils import score_completion, extract_number

load_dotenv()

WANDB_PROJECT        = "financial-reasoning-ft"
RESULTS_DIR          = Path("results/baseline_local")
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL     = "qwen/qwen-2.5-7b-instruct"   # the model we fine-tune

SYSTEM_PROMPT = """You are a financial analyst expert.
Always solve financial problems step-by-step, showing every calculation.
Respond strictly in this format:
<think>
[step-by-step reasoning with all arithmetic shown]
</think>
<answer>[numeric answer only — no units, no currency symbols]</answer>"""


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

def load_financebench(limit: int | None = None) -> list[dict]:
    """
    PatronusAI/financebench — 150 real financial QA examples from 10-K filings.
    Filtered to numeric-answer questions only (~100 examples).

    Replaces ibm-research/finqa locally (that dataset uses a legacy loading
    script dropped in datasets>=4.0). Colab uses the real FinQA via lm-eval.
    """
    print("[Data] Loading PatronusAI/financebench...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    examples = []
    for ex in ds:
        evidence_texts = [
            e.get("evidence_text", "") for e in (ex.get("evidence") or [])
        ]
        context    = "\n\n".join(t for t in evidence_texts if t)
        raw_answer = str(ex.get("answer", "")).strip()

        # Keep only short numeric answers (skip narrative responses)
        if extract_number(raw_answer) is None or len(raw_answer) > 30:
            continue

        examples.append({
            "source":   "financebench",
            "company":  ex.get("company", ""),
            "question": ex["question"],
            "context":  context,
            "answer":   raw_answer,
        })

    if limit:
        examples = examples[:limit]
        print(f"  Limited to {limit} examples")

    print(f"  Loaded {len(examples)} numeric financial QA examples")
    return examples


# ─────────────────────────────────────────────────────────────────
# OpenRouter backend  (Qwen2.5-7B — the TRUE baseline)
# ─────────────────────────────────────────────────────────────────

async def call_openrouter(
    question: str,
    context: str,
    client: AsyncOpenAI,
    model: str = OPENROUTER_MODEL,
) -> str:
    """Call Qwen2.5-7B via OpenRouter (OpenAI-compatible API)."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Financial Context:\n{context[:3000]}\n\nQuestion: {question}"},
            ],
            temperature=0.1,   # low temp for eval (deterministic)
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"\n  OpenRouter error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────
# Poe backend  (fallback — Claude Haiku, GPT-4o-mini, etc.)
# ─────────────────────────────────────────────────────────────────

async def call_poe(question: str, context: str, bot: str) -> str:
    """Call a Poe bot. Bot name examples: Claude-3-5-Haiku, GPT-4o-mini."""
    messages = [
        fp.ProtocolMessage(role="system", content=SYSTEM_PROMPT),
        fp.ProtocolMessage(role="user",   content=f"Financial Context:\n{context[:2000]}\n\nQuestion: {question}"),
    ]
    response = ""
    try:
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name=bot,
            api_key=os.environ["POE_API_KEY"],
        ):
            response += partial.text
    except Exception as e:
        print(f"\n  Poe error: {e}")
    return response


# ─────────────────────────────────────────────────────────────────
# Evaluation runner (backend-agnostic)
# ─────────────────────────────────────────────────────────────────

async def run_eval(
    examples: list[dict],
    backend: str,
    bot_or_model: str,
    max_concurrent: int,
    save_path: Path,
) -> list[dict]:
    """Run inference on all examples, save incrementally (resume-safe)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-completed questions
    done = set()
    if save_path.exists():
        with open(save_path) as f:
            for line in f:
                done.add(json.loads(line).get("question", ""))
        if done:
            print(f"  Resuming: {len(done)} already evaluated")

    to_run = [ex for ex in examples if ex["question"] not in done]
    print(f"  Evaluating {len(to_run)} remaining examples via {backend}...")

    # Shared OpenAI client for OpenRouter (created once, reused)
    or_client = None
    if backend == "openrouter":
        or_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_one(ex: dict) -> dict:
        async with semaphore:
            if backend == "openrouter":
                completion = await call_openrouter(ex["question"], ex["context"], or_client, bot_or_model)
            else:
                completion = await call_poe(ex["question"], ex["context"], bot_or_model)

            result = {**ex, "completion": completion, **score_completion(completion, ex["answer"])}
            with open(save_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            return result

    tasks   = [eval_one(ex) for ex in to_run]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        results.append(await coro)

    # Merge with cached results
    if done:
        with open(save_path) as f:
            cached = [json.loads(l) for l in f if json.loads(l)["question"] in done]
        results = cached + results

    return results


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "n":                 n,
        "accuracy":          sum(1 for r in results if r["correct"]) / n,
        "format_think_pct":  sum(1 for r in results if r["has_think"]) / n,
        "format_answer_pct": sum(1 for r in results if r["has_answer"]) / n,
        "mean_score":        sum(r["score"] for r in results) / n,
    }


def print_report(metrics: dict, label: str, results: list[dict]) -> None:
    print(f"\n{'='*58}")
    print(f"  LOCAL BASELINE — {label}")
    print(f"{'='*58}")
    print(f"  Examples evaluated : {metrics['n']}")
    print(f"  Accuracy           : {metrics['accuracy']:.1%}   ← beat this after fine-tuning")
    print(f"  Mean score         : {metrics['mean_score']:.3f}")
    print(f"  Has <think> tag    : {metrics['format_think_pct']:.1%}")
    print(f"  Has <answer> tag   : {metrics['format_answer_pct']:.1%}")
    print(f"{'='*58}")

    failures = [r for r in results if not r["correct"]][:3]
    if failures:
        print("\n  Sample failures (first 3):")
        for r in failures:
            print(f"\n  Q: {r['question'][:80]}")
            print(f"  Expected : {r['answer']}")
            print(f"  Got      : {r.get('predicted', 'N/A')}")
            print(f"  <think>  : {r['has_think']}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main(backend: str, bot: str, model: str, limit: int | None, max_concurrent: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bot_or_model = model if backend == "openrouter" else bot
    label        = f"{backend}/{bot_or_model}"

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"baseline-local-{backend}",
        tags=["baseline", "baseline-local", "no-gpu", backend],
        config={"backend": backend, "model": bot_or_model, "limit": limit, "dataset": "financebench"},
    )

    examples = load_financebench(limit=limit)

    results = await run_eval(
        examples,
        backend=backend,
        bot_or_model=bot_or_model,
        max_concurrent=max_concurrent,
        save_path=RESULTS_DIR / f"samples_{backend}.jsonl",
    )

    metrics = compute_metrics(results)
    print_report(metrics, label, results)

    scores_file = RESULTS_DIR / f"baseline_scores_{backend}.json"
    with open(scores_file, "w") as f:
        json.dump({"backend": backend, "model": bot_or_model, "metrics": metrics}, f, indent=2)

    wandb.log({f"baseline_local/{k}": v for k, v in metrics.items()})
    wandb.summary["baseline_local/accuracy"] = metrics["accuracy"]
    print(f"\n✅ Results saved to {scores_file}")
    wandb.finish()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*event loop.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*async generator.*")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default="openrouter", choices=["openrouter", "poe"],
        help="Inference backend: openrouter (Qwen2.5-7B, true baseline) or poe (Claude/GPT)"
    )
    parser.add_argument(
        "--model", default=OPENROUTER_MODEL,
        help=f"OpenRouter model (default: {OPENROUTER_MODEL})"
    )
    parser.add_argument(
        "--bot", default="Claude-3-5-Haiku",
        help="Poe bot name — only used when --backend poe (default: Claude-3-5-Haiku)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of examples to evaluate (omit for full ~100 numeric questions)"
    )
    parser.add_argument(
        "--concurrent", type=int, default=5,
        help="Max concurrent API calls (default: 5)"
    )
    args = parser.parse_args()
    asyncio.run(main(
        backend=args.backend,
        bot=args.bot,
        model=args.model,
        limit=args.limit,
        max_concurrent=args.concurrent,
    ))
