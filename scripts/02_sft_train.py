"""
Week 2 — SFT Training
Teaches the model the FORMAT of financial reasoning: <think>...</think><answer>X</answer>

Usage:
    python scripts/02_sft_train.py --config config/sft_v1.yaml

Outputs:
    checkpoints/sft/best/     — best checkpoint (highest eval_loss)
    results/sft/              — lm-eval results for all 3 epoch checkpoints
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
import wandb
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

load_dotenv()

SYSTEM_PROMPT = """You are a financial analyst expert.
Always solve financial problems step-by-step, showing every calculation.
Use the <think> and <answer> tags in your response."""


# ─────────────────────────────────────────────────────────────────
# Data formatting
# ─────────────────────────────────────────────────────────────────

def format_example(example: dict) -> dict | None:
    """Wrap a training example into the chat template."""
    context = example.get("context", example.get("table", ""))
    trace   = example.get("reasoning_trace", "")

    think_match  = re.search(r"<think>.*?</think>", trace, re.DOTALL)
    answer_match = re.search(r"<answer>.*?</answer>", trace, re.DOTALL)

    if not think_match or not answer_match:
        return None

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFinancial Context:\n{context}\n\nQuestion: {example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_match.group(0)}\n{answer_match.group(0)}<|im_end|>"
    )
    return {"text": text}


def load_and_format(path: str) -> Dataset:
    with open(path) as f:
        raw = [json.loads(l) for l in f]
    formatted = [format_example(ex) for ex in raw]
    formatted = [ex for ex in formatted if ex is not None]
    return Dataset.from_list(formatted)


# ─────────────────────────────────────────────────────────────────
# Evaluate a checkpoint with lm-eval
# ─────────────────────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path: str, output_dir: str, task: str = "finqa") -> float:
    """Run lm-eval on a checkpoint and return FinQA accuracy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},dtype=float16",
        "--tasks", task,
        "--batch_size", "4",
        "--output_path", output_dir,
    ], capture_output=True, text=True)

    results_file = Path(output_dir) / "results.json"
    if not results_file.exists():
        print(f"  lm_eval failed: {result.stderr[:200]}")
        return 0.0

    with open(results_file) as f:
        data = json.load(f)
    return data["results"][task].get("exact_match,none", 0.0)


# ─────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    run = wandb.init(
        project="financial-reasoning-ft",
        name=cfg["experiment"]["name"],
        tags=cfg["experiment"]["tags"],
        config=cfg,
    )

    # Load model
    print(f"\n[Setup] Loading {cfg['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=torch.float16 if cfg["model"]["load_in_4bit"] else torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    tokenizer.eos_token = "<|im_end|>"   # Qwen2.5 chat-end token; Unsloth sets a placeholder

    # Load data
    print("\n[Data] Loading and formatting datasets...")
    train_ds = load_and_format(cfg["data"]["train_path"])
    val_ds   = load_and_format(cfg["data"]["val_path"])
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Train
    t = cfg["training"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            # SFT-specific
            max_length=cfg["model"]["max_seq_length"],
            packing=t["packing"],
            dataset_text_field="text",
            eos_token="<|im_end|>",      # Qwen2.5 chat-end token; TRL default <EOS_TOKEN> is invalid
            # Training
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_ratio=t["warmup_ratio"],
            lr_scheduler_type=t["scheduler"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            max_grad_norm=t["max_grad_norm"],
            eval_strategy="steps",
            eval_steps=t["save_steps"],
            save_strategy="steps",
            save_steps=t["save_steps"],
            save_total_limit=t["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=t["logging_steps"],
            output_dir=checkpoint_dir,
            report_to="wandb",
        ),
    )

    print("\n[Train] Starting SFT...")
    trainer.train()

    best_dir = str(Path(checkpoint_dir) / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n✅ Best SFT model saved to {best_dir}")

    # Evaluate all epoch checkpoints
    print("\n[Eval] Evaluating checkpoints on FinQA...")
    results_dir = cfg["output"]["results_dir"]
    checkpoint_scores = {}

    for subdir in sorted(Path(checkpoint_dir).glob("checkpoint-*")):
        name = subdir.name
        score = evaluate_checkpoint(str(subdir), f"{results_dir}/{name}")
        checkpoint_scores[name] = score
        print(f"  {name}: {score:.1%}")
        wandb.log({f"sft/finqa_{name}": score})

    best_name  = max(checkpoint_scores, key=checkpoint_scores.get)
    best_score = checkpoint_scores[best_name]
    print(f"\n  Best checkpoint: {best_name} ({best_score:.1%})")
    wandb.summary["sft/best_finqa"] = best_score

    # Gate check
    if best_score < 0.65:
        print(f"\n⚠️  WARNING: SFT FinQA {best_score:.1%} < 65% threshold.")
        print("   Check: data quality, learning rate, number of training examples.")
    else:
        print(f"\n✅ Gate passed: SFT FinQA {best_score:.1%} ≥ 65%")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sft_v1.yaml")
    args = parser.parse_args()
    main(args.config)
