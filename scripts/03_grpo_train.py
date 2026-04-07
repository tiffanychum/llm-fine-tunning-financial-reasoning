"""
Week 3-4 — GRPO Training
Teaches the model to get answers CORRECT through trial-and-error (8 attempts/question).
Always starts from the SFT checkpoint.

Usage:
    python scripts/03_grpo_train.py --config config/grpo_v1.yaml

Outputs:
    checkpoints/grpo/best/    — best GRPO checkpoint
    results/grpo/             — lm-eval comparison vs SFT
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import torch
import wandb
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

load_dotenv()

SYSTEM_PROMPT = """You are a financial analyst expert.
Always solve financial problems step-by-step, showing every calculation.
Use the <think> and <answer> tags in your response."""


# ─────────────────────────────────────────────────────────────────
# Reward functions — tested before any training
# ─────────────────────────────────────────────────────────────────

def extract_answer_str(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def normalize_number(value: str) -> Optional[float]:
    """Handle $, %, M, B, commas."""
    if not value:
        return None
    v = value.replace(",", "").replace("$", "").replace("%", "").strip()
    mult = 1.0
    if v.endswith(("M", "m")):
        mult, v = 1e6, v[:-1]
    elif v.endswith(("B", "b")):
        mult, v = 1e9, v[:-1]
    try:
        return float(v) * mult
    except ValueError:
        return None


def format_reward(completions: list, **kwargs) -> list:
    """Binary format reward: has both <think> and <answer> tags?"""
    rewards = []
    for c in completions:
        has_think  = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.+?</answer>", c, re.DOTALL))
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions: list, ground_truths: list, **kwargs) -> list:
    """Graded accuracy reward with tolerance for rounding errors."""
    cfg = kwargs.get("reward_cfg", {
        "tolerance_exact": 0.01,
        "tolerance_close": 0.05,
        "tolerance_partial": 0.15,
    })
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred_str = extract_answer_str(completion)
        if pred_str is None:
            rewards.append(0.0)
            continue

        pred_val = normalize_number(pred_str)
        true_val = normalize_number(str(gt))

        if pred_val is None or true_val is None:
            rewards.append(1.0 if pred_str.lower() == str(gt).lower() else 0.0)
        elif abs(true_val) < 1e-9:
            rewards.append(1.0 if abs(pred_val) < 1e-9 else 0.0)
        else:
            rel_err = abs(pred_val - true_val) / abs(true_val)
            if rel_err < cfg["tolerance_exact"]:
                rewards.append(1.0)
            elif rel_err < cfg["tolerance_close"]:
                rewards.append(0.5)
            elif rel_err < cfg["tolerance_partial"]:
                rewards.append(0.2)
            else:
                rewards.append(0.0)
    return rewards


def combined_reward(completions: list, ground_truths: list, **kwargs) -> list:
    fmt_w = kwargs.get("format_weight", 0.2)
    acc_w = kwargs.get("accuracy_weight", 0.8)
    fmt = format_reward(completions, **kwargs)
    acc = accuracy_reward(completions, ground_truths, **kwargs)
    return [fmt_w * f + acc_w * a for f, a in zip(fmt, acc)]


def run_reward_tests() -> bool:
    """Must pass 100% before GRPO training begins."""
    TEST_CASES = [
        ("<think>calc</think><answer>17.14%</answer>", "17.14%",  1.0),
        ("<think>calc</think><answer>17.14</answer>",  "17.14%",  1.0),
        ("<think>calc</think><answer>17.50%</answer>", "17.14%",  0.4),
        ("no tags, 17.14",                             "17.14%",  0.24),
        ("",                                           "17.14%",  0.0),
    ]
    all_pass = True
    print("\n[Reward Tests]")
    for completion, gt, expected in TEST_CASES:
        actual = combined_reward([completion], [gt])[0]
        ok = abs(actual - expected) <= 0.15
        print(f"  {'✅' if ok else '❌'} expected≈{expected:.2f} got {actual:.2f}  |  {completion[:40]!r}")
        if not ok:
            all_pass = False
    return all_pass


# ─────────────────────────────────────────────────────────────────
# RL dataset: filter to solvable questions only
# ─────────────────────────────────────────────────────────────────

def filter_solvable(rl_path: str, model, tokenizer, n_attempts: int = 8) -> list[dict]:
    """
    Keep only questions where SFT model gets ≥1 correct out of 8 attempts.
    GRPO cannot learn from questions where all 8 attempts score reward=0.
    """
    with open(rl_path) as f:
        rl_raw = [json.loads(l) for l in f]

    print(f"\n[RL Filter] Testing {len(rl_raw)} questions ({n_attempts} attempts each)...")
    solvable = []

    for example in tqdm(rl_raw):
        context  = example.get("context", "")
        prompt   = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nFinancial Context:\n{context}\n\n"
            f"Question: {example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        any_correct = False
        for _ in range(n_attempts):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            rewards = accuracy_reward([generated], [example["answer"]])
            if rewards[0] > 0:
                any_correct = True
                break

        if any_correct:
            solvable.append(example)

    print(f"  Solvable: {len(solvable)} / {len(rl_raw)} ({len(solvable)/len(rl_raw):.0%})")
    return solvable


# ─────────────────────────────────────────────────────────────────
# GRPO training
# ─────────────────────────────────────────────────────────────────

def format_rl_example(example: dict) -> dict:
    context = example.get("context", "")
    return {
        "prompt": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nFinancial Context:\n{context}\n\n"
            f"Question: {example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "ground_truth": str(example["answer"]),
    }


def evaluate_checkpoint(checkpoint_path: str, output_dir: str) -> float:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},dtype=float16",
        "--tasks", "finqa",
        "--batch_size", "4",
        "--output_path", output_dir,
    ], capture_output=True, text=True)
    results_file = Path(output_dir) / "results.json"
    if not results_file.exists():
        return 0.0
    with open(results_file) as f:
        data = json.load(f)
    return data["results"]["finqa"].get("exact_match,none", 0.0)


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    # Reward tests MUST pass before training
    assert run_reward_tests(), "❌ Reward function tests failed — fix before training."
    print("✅ All reward tests passed.")

    run = wandb.init(
        project="financial-reasoning-ft",
        name=cfg["experiment"]["name"],
        tags=cfg["experiment"]["tags"],
        config=cfg,
    )

    # W&B alerts for collapse detection
    wandb.alert(title="entropy collapse",         text="Entropy < 0.15 — increase kl_coef")
    wandb.alert(title="KL spike",                 text="KL > 0.4 — potential collapse")
    wandb.alert(title="format rate drop",         text="Format rate < 0.7 — reward hacking?")

    # Load SFT checkpoint
    print(f"\n[Setup] Loading SFT checkpoint: {cfg['model']['sft_checkpoint']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["sft_checkpoint"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Filter RL dataset
    solvable_path = "data/processed/rl_solvable.jsonl"
    if not Path(solvable_path).exists():
        solvable = filter_solvable(cfg["data"]["rl_path"], model, tokenizer)
        with open(solvable_path, "w") as f:
            for ex in solvable:
                f.write(json.dumps(ex) + "\n")
    else:
        with open(solvable_path) as f:
            solvable = [json.loads(l) for l in f]
        print(f"\n[RL Filter] Loaded {len(solvable)} solvable questions from cache.")

    rl_dataset = Dataset.from_list([format_rl_example(ex) for ex in solvable])

    # Train
    t = cfg["training"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]

    grpo_config = GRPOConfig(
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=1,
        num_generations=t["num_generations"],
        max_new_tokens=t["max_new_tokens"],
        temperature=t["temperature"],
        kl_coef=t["kl_coef"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        output_dir=checkpoint_dir,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[combined_reward],
        args=grpo_config,
        train_dataset=rl_dataset,
    )

    print("\n[Train] Starting GRPO...")
    trainer.train()

    best_dir = str(Path(checkpoint_dir) / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n✅ Best GRPO model saved to {best_dir}")

    # Compare GRPO vs SFT
    print("\n[Eval] Comparing GRPO vs SFT on FinQA...")
    results_dir = cfg["output"]["results_dir"]

    grpo_score = evaluate_checkpoint(best_dir, f"{results_dir}/grpo_best")
    sft_score  = evaluate_checkpoint(cfg["model"]["sft_checkpoint"], f"{results_dir}/sft_reference")

    print(f"\n  SFT score:  {sft_score:.1%}")
    print(f"  GRPO score: {grpo_score:.1%}")
    print(f"  GRPO gain:  {(grpo_score - sft_score)*100:+.1f} pp")

    wandb.summary["grpo/finqa"]      = grpo_score
    wandb.summary["sft/finqa"]       = sft_score
    wandb.summary["grpo/gain_over_sft"] = grpo_score - sft_score

    if grpo_score >= sft_score + 0.03:
        print("✅ Gate passed: GRPO gained ≥ +3 pp over SFT")
    elif grpo_score >= sft_score:
        print("⚠️  Marginal improvement — use GRPO checkpoint, note the small gain")
    else:
        print("❌ GRPO did not help — use SFT checkpoint as final model")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/grpo_v1.yaml")
    args = parser.parse_args()
    main(args.config)
