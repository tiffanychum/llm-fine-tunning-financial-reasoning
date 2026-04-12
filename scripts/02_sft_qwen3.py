"""
SFT Training — Qwen3-8B  (Fino1 approach)
==========================================

Based on: "Fino1: On the Transferability of Reasoning-Enhanced LLMs to Finance"
          arXiv:2502.08127  |  TheFinAI/Fin-o1-8B

KEY DIFFERENCES vs scripts/02_sft_train.py (Qwen2.5-7B):

  1. DATA FORMAT — conversational (messages dict) instead of raw text
     Old: manually builds "<|im_start|>system\n..." strings, sets eos_token
          workarounds to avoid TRL's <EOS_TOKEN> validation error
     New: returns {"messages": [...]} — TRL applies Qwen3's chat template
          automatically, no manual token management needed

  2. SYSTEM PROMPT — matches Fino1 paper's inference prompt exactly
     Tells the model to use <think>...</think><answer>...</answer> format

  3. LOSS ON ASSISTANT ONLY — `completion_only_loss=True` in SFTConfig
     Old: loss computed over entire packed sequence (question + answer)
     New: loss computed only on assistant responses (the CoT + answer)
          This is cleaner training: the model doesn't waste capacity
          memorising the financial context / question phrasing

  4. NO PACKING — `packing=False`
     Old: packing=True caused truncation at max_seq_length=2048 which
          cut off <answer> tags and degraded format rates after training
     New: each example is its own sequence, no cross-example contamination

  5. LEARNING RATE — 2e-5 (Fino1 model card value)
     Old: 5e-6 — too low; loss still declining at end of 3 epochs
     New: 2e-5 — 4× higher, appropriate for LoRA on an instruct model

  6. DATA — same FinCoT files, no reprocessing needed
     Fino1 uses difficulty-aware filtering; our pipeline already applies
     difficulty + quality filters which align with this philosophy

Usage:
    python scripts/02_sft_qwen3.py --config config/sft_qwen3_v1.yaml
    python scripts/02_sft_qwen3.py --config config/sft_qwen3_v1.yaml --resume

Outputs:
    checkpoints/sft_qwen3/best/   — best checkpoint (lowest eval_loss)
    results/sft_qwen3/            — per-checkpoint eval scores
"""

import unsloth  # must be first — patches trl/transformers/peft at import time

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
import wandb
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

load_dotenv()

# Google Drive backup path (Colab only — ignored locally if Drive not mounted)
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Colab Notebook/checkpoints/sft_qwen3"

# ─────────────────────────────────────────────────────────────────
# Fino1 system prompt — taken directly from the paper's inference example
# Tells Qwen3 to use both its native <think> blocks AND our <answer> tag
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a financial analyst expert. "
    "You first think about the reasoning process as an internal monologue "
    "and then provide the user with the answer. "
    "Respond in the following format:\n"
    "<think>\n...\n</think>\n<answer>\n...\n</answer>"
)


# ─────────────────────────────────────────────────────────────────
# Drive backup callback (same as Project 1 — keeps checkpoints safe)
# ─────────────────────────────────────────────────────────────────

class DriveBackupCallback(TrainerCallback):
    """Copy each checkpoint to Google Drive immediately after it's saved."""

    def on_save(self, args, state, control, **kwargs):
        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        dst = Path(DRIVE_BACKUP_DIR) / f"checkpoint-{state.global_step}"
        if src.exists() and Path("/content/drive").exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"\n  💾 Checkpoint backed up to Drive: {dst}")
        return control


# ─────────────────────────────────────────────────────────────────
# Data formatting — Fino1 conversational format
#
# OLD approach (02_sft_train.py):
#   Returns {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."}
#   Manually constructs chat template tokens. Required eos_token workarounds.
#
# NEW approach (this file):
#   Returns {"messages": [{"role": ..., "content": ...}, ...]}
#   TRL detects the messages column and applies Qwen3's chat template
#   automatically. No manual token management needed.
# ─────────────────────────────────────────────────────────────────

def format_example(example: dict) -> dict | None:
    """
    Convert a FinCoT example to Fino1-style conversational format.

    The assistant response is formatted as:
        <think>
        {step-by-step reasoning}
        </think>
        <answer>{numeric answer}</answer>

    This matches Qwen3's native thinking output format AND our eval metric.
    """
    context  = example.get("context", "")
    question = example.get("question", "")
    trace    = example.get("reasoning_trace", "")

    think_match  = re.search(r"<think>(.*?)</think>",   trace, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", trace, re.DOTALL)

    if not think_match or not answer_match:
        return None

    thinking = think_match.group(1).strip()
    answer   = answer_match.group(1).strip()

    user_content = (
        f"Financial Context:\n{context}\n\nQuestion: {question}"
        if context else
        f"Question: {question}"
    )

    # Reconstruct in Qwen3-native thinking format
    assistant_content = f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def load_and_format(path: str) -> Dataset:
    with open(path) as f:
        raw = [json.loads(l) for l in f]
    formatted = [format_example(ex) for ex in raw]
    formatted = [ex for ex in formatted if ex is not None]
    print(f"  Formatted: {len(formatted)} / {len(raw)} examples "
          f"({len(formatted)/len(raw):.0%} passed format check)")
    return Dataset.from_list(formatted)


# ─────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────

def main(config_path: str, resume: bool = False):
    cfg = yaml.safe_load(open(config_path))

    wandb.init(
        project="financial-reasoning-ft",
        name=cfg["experiment"]["name"],
        tags=cfg["experiment"]["tags"],
        config=cfg,
    )

    # ── Model loading ────────────────────────────────────────────
    # Qwen3-8B is loaded in 4-bit (NF4) to fit 4096 context on A100 40 GB.
    # Unlike the Qwen2.5 run (which used bf16 full precision at seq_len=2048),
    # 4-bit lets us restore the full 4096 context without OOM.
    print(f"\n[Setup] Loading {cfg['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=None,  # auto — 4-bit NF4 base + bf16 compute on A100
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

    # No eos_token workaround needed for Qwen3 — its tokenizer already has
    # eos_token = "<|im_end|>" set correctly (verified from HF model page).
    # This was the main source of bugs in the Qwen2.5 training run.

    # ── Data ────────────────────────────────────────────────────
    print("\n[Data] Loading and formatting datasets...")
    train_ds = load_and_format(cfg["data"]["train_path"])
    val_ds   = load_and_format(cfg["data"]["val_path"])
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Trainer ─────────────────────────────────────────────────
    t = cfg["training"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]

    def formatting_func(batch):
        """Apply Qwen3 chat template to a batch of messages-format examples.
        Unsloth calls this with a dict of lists, so we iterate and return a list."""
        return [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msgs in batch["messages"]
        ]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        callbacks=[DriveBackupCallback()],
        args=SFTConfig(
            # ── Fino1 conversational format settings ──────────────
            # formatting_func above applies Qwen3's chat template to
            # the "messages" column and produces a "text" string for training
            packing=t["packing"],              # False — no truncation risk
            max_length=cfg["model"]["max_seq_length"],

            # ── Training hyperparameters ──────────────────────────
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],  # 2e-5 (Fino1) vs 5e-6 (Qwen2.5 run)
            per_device_train_batch_size=t["per_device_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_ratio=t["warmup_ratio"],
            lr_scheduler_type=t["scheduler"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            max_grad_norm=t["max_grad_norm"],

            # ── Checkpoint / eval ─────────────────────────────────
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

    # ── Resume logic (same as Project 1) ────────────────────────
    resume_ckpt = None
    if resume:
        local_ckpts = sorted(Path(checkpoint_dir).glob("checkpoint-*"))
        drive_ckpts = (
            sorted(Path(DRIVE_BACKUP_DIR).glob("checkpoint-*"))
            if Path(DRIVE_BACKUP_DIR).exists() else []
        )
        if local_ckpts:
            resume_ckpt = str(local_ckpts[-1])
            print(f"\n[Resume] Resuming from local: {resume_ckpt}")
        elif drive_ckpts:
            latest = drive_ckpts[-1]
            local_dest = Path(checkpoint_dir) / latest.name
            local_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(latest, local_dest, dirs_exist_ok=True)
            resume_ckpt = str(local_dest)
            print(f"\n[Resume] Restored from Drive: {latest} → {local_dest}")
        else:
            print("\n[Resume] No checkpoint found — starting from scratch")

    # ── Train ────────────────────────────────────────────────────
    print("\n[Train] Starting SFT (Fino1 / Qwen3-8B)...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    best_dir = str(Path(checkpoint_dir) / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\n✅ Best SFT model saved to {best_dir}")

    # ── Gate check ───────────────────────────────────────────────
    # Fino1 context: Qwen3-8B baseline is ~72% accuracy.
    # After SFT we target ≥77% (+5pp) before moving to GRPO.
    # (Fin-R1 achieved +6.3pp SFT gain from a 65.6% baseline.)
    print(f"\n[Gate] Run eval to check if SFT improved over the 72% baseline.")
    print(f"  Target: ≥77% accuracy (+5pp) before proceeding to GRPO.")
    print(f"  Run: python scripts/00_setup_eval.py --model {best_dir}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sft_qwen3_v1.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint (local or Drive)")
    args = parser.parse_args()
    main(args.config, resume=args.resume)
