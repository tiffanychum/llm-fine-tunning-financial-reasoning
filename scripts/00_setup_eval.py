"""
Week 0 — Baseline Evaluation (Colab GPU)
Loads Qwen2.5-7B locally and evaluates it on our validation set BEFORE any fine-tuning.
This gives the "before" score that SFT and GRPO must beat.

Usage (on Colab):
    python scripts/00_setup_eval.py

    # Limit to 100 examples for a quick check
    python scripts/00_setup_eval.py --limit 100

Output:
    results/baseline/baseline_scores.json
    W&B run tagged "baseline"

Why not lm-eval?
    ibm-research/finqa uses a legacy loading script dropped in datasets>=4.0.
    We evaluate on our own sft_val_v1.jsonl instead — same distribution as training,
    same scoring function as GRPO rewards, directly comparable across all phases.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import score_completion

load_dotenv()

WANDB_PROJECT = "financial-reasoning-ft"
RESULTS_DIR   = Path("results/baseline")
VAL_DATA_PATH = Path("data/processed/sft_val_v1.jsonl")

SYSTEM_PROMPT = """You are a financial analyst expert.
Always solve financial problems step-by-step, showing every calculation.
Respond strictly in this format:
<think>
[step-by-step reasoning with all arithmetic shown]
</think>
<answer>[numeric answer only — no units, no currency symbols]</answer>"""


def load_val_data(path: Path, limit: int | None = None) -> list[dict]:
    """Load validation examples from our processed dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Val data not found at {path}\n"
            "Run scripts/01_data_pipeline.py first (locally or on Colab)."
        )
    with open(path) as f:
        examples = [json.loads(l) for l in f]

    if limit:
        examples = examples[:limit]
        print(f"  Limited to {limit} examples")

    print(f"  Loaded {len(examples)} validation examples from {path}")
    return examples


def build_prompt(example: dict) -> str:
    """Build the inference prompt — same format the model will see during training."""
    context  = example.get("context", "")
    question = example.get("question", "")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Financial Context:\n{context[:3000]}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_inference(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Run batch inference on all examples."""
    results = []

    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[i : i + batch_size]
        prompts = [build_prompt(ex) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,    # low temp for deterministic eval
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, (ex, output) in enumerate(zip(batch, outputs)):
            prompt_len  = inputs["input_ids"][j].shape[0]
            completion  = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            scored      = score_completion(completion, ex["answer"])
            results.append({**ex, "completion": completion, **scored})

    return results


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


def print_report(metrics: dict, model_name: str, results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  BASELINE — {model_name}")
    print(f"{'='*60}")
    print(f"  Examples evaluated : {metrics['n']}")
    print(f"  Accuracy           : {metrics['accuracy']:.1%}  ← beat this after SFT")
    print(f"  Mean score         : {metrics['mean_score']:.3f}")
    print(f"  Has <think> tag    : {metrics['format_think_pct']:.1%}")
    print(f"  Has <answer> tag   : {metrics['format_answer_pct']:.1%}")
    print(f"{'='*60}")

    # Show 3 failure examples for insight
    failures = [r for r in results if not r["correct"]][:3]
    if failures:
        print("\n  Sample failures:")
        for r in failures:
            print(f"\n  Q: {r['question'][:80]}")
            print(f"  Expected  : {r['answer']}")
            print(f"  Predicted : {r.get('predicted', 'N/A')}")
            print(f"  Has think : {r['has_think']}")


def main(model_name: str, limit: int | None, batch_size: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"baseline-{model_name.split('/')[-1]}",
        tags=["baseline", "no-finetune"],
        config={"model": model_name, "limit": limit, "eval_data": str(VAL_DATA_PATH)},
    )

    # Load data
    examples = load_val_data(VAL_DATA_PATH, limit=limit)

    # Load model
    print(f"\n[Model] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"   # left-pad for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Loaded on {next(model.parameters()).device}")

    # Run inference
    print(f"\n[Eval] Running inference (batch_size={batch_size})...")
    results = run_inference(model, tokenizer, examples, batch_size=batch_size)

    # Score and report
    metrics = compute_metrics(results)
    print_report(metrics, model_name, results)

    # Save
    scores_file = RESULTS_DIR / "baseline_scores.json"
    with open(scores_file, "w") as f:
        json.dump({
            "model":   model_name,
            "eval_data": str(VAL_DATA_PATH),
            "metrics": metrics,
        }, f, indent=2)

    # Save all samples for error analysis
    samples_file = RESULTS_DIR / "samples.jsonl"
    with open(samples_file, "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "context"}) + "\n")

    wandb.log({f"baseline/{k}": v for k, v in metrics.items()})
    wandb.summary["baseline/accuracy"] = metrics["accuracy"]

    print(f"\n✅ Baseline saved to {scores_file}")
    print("   This is your comparison point — SFT and GRPO must beat this.")
    wandb.finish()

    # Return metrics so the Colab notebook can use them
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model to evaluate (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of val examples (omit for full val set)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Inference batch size (default: 4)"
    )
    args = parser.parse_args()
    main(model_name=args.model, limit=args.limit, batch_size=args.batch_size)
