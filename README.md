# Financial Reasoning — LLM Fine-tuning (SFT + GRPO)

Fine-tunes Qwen2.5-7B on financial numerical reasoning using a two-stage pipeline:
SFT teaches the reasoning **format**, GRPO teaches getting the **answer correct**.

Target: outperform GPT-4o on FinQA benchmarks at a fraction of the inference cost.

---

## Project Structure

```
├── config/
│   ├── sft_v1.yaml          # SFT hyperparameters (edit to create v2, v3...)
│   └── grpo_v1.yaml         # GRPO hyperparameters
├── data/
│   ├── raw/                 # downloaded datasets (gitignored)
│   └── processed/           # filtered/formatted datasets (gitignored)
├── scripts/
│   ├── 00_setup_eval.py     # Week 0: baseline evaluation
│   ├── 01_data_pipeline.py  # Week 1: data download → filter → CoT gen → split
│   ├── 02_sft_train.py      # Week 2: supervised fine-tuning
│   └── 03_grpo_train.py     # Week 3-4: GRPO reinforcement learning
├── results/                 # lm-eval output (committed when ready)
├── checkpoints/             # model checkpoints (gitignored — too large)
├── .env.example             # copy to .env and fill in API keys
└── requirements.txt
```

---

## Quickstart (Google Colab Pro)

```python
# 1. Clone and install
!git clone <your-repo>
%cd llm-fine-tunning-financial-reasoning
!pip install -r requirements.txt

# 2. Set up environment variables
import os
os.environ["POE_API_KEY"]  = "your_key"
os.environ["WANDB_API_KEY"] = "your_key"
os.environ["HF_TOKEN"]     = "your_token"

# 3. Run the pipeline in order
!python scripts/00_setup_eval.py       # ~30 min — establishes baseline
!python scripts/01_data_pipeline.py    # ~4-6h  — generates CoT traces via Poe
!python scripts/02_sft_train.py        # ~4-6h  — SFT on T4, 8-12h on A100 for 14B
!python scripts/03_grpo_train.py       # ~6-8h  — GRPO from SFT checkpoint
```

---

## Phase-by-Phase Gate Criteria

| Phase | Exit Criteria | Action if missed |
|---|---|---|
| Data (Week 1) | ≥1,800 verified examples | Lower quality filter 5/6 → 4/6 |
| SFT (Week 2) | FinQA ≥65%, format compliance ≥90% | Check data quality, try epoch 2 checkpoint |
| GRPO (Week 3-4) | GRPO gain ≥+3pp over SFT | Check entropy/reward in W&B; use SFT as final if GRPO doesn't help |

---

## Creating a New Experiment

1. Copy the config: `cp config/sft_v1.yaml config/sft_v2.yaml`
2. Edit ONE hyperparameter (e.g. learning rate, LoRA rank)
3. Run: `python scripts/02_sft_train.py --config config/sft_v2.yaml`
4. Compare runs in W&B project `financial-reasoning-ft`

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Strongest math pre-training at 7B |
| CoT teacher | Claude 3.5 Haiku (Poe) | Best quality/cost; ~$5 for 3,500 traces |
| Difficulty filter | Heuristic rules | Free; removes trivially easy examples |
| SFT before GRPO | Yes | GRPO-only (without SFT warmup) is worse (Fin-R1 ablation) |
| RL algorithm | GRPO (not DPO) | Financial answers are verifiable — no preference pairs needed |
| RL dataset | Solvable questions only | GRPO can't learn from all-zero reward groups |

---

## Expected Results

| Benchmark | Base Model | After SFT | After SFT+GRPO | GPT-4o ref |
|---|---|---|---|---|
| FinQA | ~57% | ~71% | ~74% | 71.4% |
| ConvFinQA | ~43% | ~63% | ~67% | 69.8% |

---

## Cost Summary

| Item | Cost |
|---|---|
| CoT generation (Poe, Claude 3.5 Haiku) | ~$5–8 |
| Google Colab Pro (1 month) | $10 |
| **Total** | **~$15–18** |
