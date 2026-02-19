# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is an open-source project for training ultra-lightweight language models (25M–108M params) from scratch using pure PyTorch. All core algorithms (attention, RoPE, MoE, DPO, PPO, GRPO, etc.) are implemented from scratch without relying on abstracted third-party framework interfaces. The project is primarily in Chinese with English translations available.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training (run from the `trainer/` directory)
```bash
# Pretraining (Phase 1)
python train_pretrain.py --hidden_size 512 --num_hidden_layers 8 --data_path ../dataset/pretrain_hq.jsonl

# Full SFT (Phase 2)
python train_full_sft.py --hidden_size 512 --data_path ../dataset/sft_mini_512.jsonl --from_weight pretrain

# LoRA fine-tuning (Phase 3)
python train_lora.py --hidden_size 512 --data_path ../dataset/lora_identity.jsonl

# DPO (Phase 4)
python train_dpo.py --hidden_size 512 --data_path ../dataset/dpo.jsonl

# RLAIF variants (Phase 5)
python train_grpo.py --hidden_size 512 --data_path ../dataset/rlaif-mini.jsonl
python train_ppo.py --hidden_size 512
python train_spo.py --hidden_size 512

# Reasoning model (R1-style)
python train_reason.py --hidden_size 512

# Model distillation
python train_distillation.py --hidden_size 512
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node N trainer/train_pretrain.py
deepspeed --master_port 29500 --num_gpus=N trainer/train_pretrain.py
```

### Inference & Evaluation
```bash
# Evaluation on benchmarks (C-Eval, C-MMLU)
python eval_llm.py --load_from model --weight full_sft --hidden_size 512

# OpenAI-compatible API server
python scripts/serve_openai_api.py --port 8000

# Web demo
streamlit run scripts/web_demo.py
```

### Resume from Checkpoint
Any training script supports `--from_resume 1` to auto-detect and resume from the last checkpoint.

## Architecture

### Model (`model/model_minimind.py`)
Single-file transformer implementation with:
- **MiniMindConfig**: All model hyperparameters (extends HuggingFace `PretrainedConfig`)
- **RMSNorm**: Root Mean Square layer normalization
- **RoPE** with YaRN scaling for long-context extrapolation (up to 32k positions)
- **Grouped Query Attention (GQA)**: 8 attention heads, 2 KV heads, with Flash Attention support
- **SwiGLU feedforward** (gate-linear-up with SiLU activation)
- **Mixture of Experts (MoE)**: Optional sparse routing with shared experts and auxiliary load-balancing loss
- **Weight tying**: Token embeddings shared with LM head

Model variants are controlled by `--hidden_size` (512=small, 640=MoE, 768=base) and `--num_hidden_layers` (8 or 16). MoE is toggled with `--use_moe 1`.

### Training Pipeline (`trainer/`)
Each training phase is a standalone script sharing utilities from `trainer_utils.py`:
- **trainer_utils.py**: Learning rate schedule (cosine annealing), model init, checkpoint save/resume, DDP setup, `SkipBatchSampler` for resuming mid-epoch
- All scripts use the same pattern: argparse config → init distributed → init model → training loop with gradient accumulation and mixed precision (bfloat16/float16)
- Checkpoints saved to `out/` (model weights as `.pth`) and `checkpoints/` (full training state for resume)

### Datasets (`dataset/lm_dataset.py`)
Four dataset classes, all reading from JSONL files:
- **PretrainDataset**: Plain text with BOS/EOS, padded to max_length
- **SFTDataset**: Chat conversations using `apply_chat_template()` with `<|im_start|>` tokens; loss masked to assistant responses only; 50% random system prompt inclusion
- **DPODataset**: Chosen/rejected response pairs with per-token loss masks
- **RLAIFDataset**: Prompt/answer extraction for RL training

### Tokenizer
Custom 6400-token vocabulary stored in `model/tokenizer.json`. Loaded via HuggingFace `AutoTokenizer`.

### Key Conventions
- Training scripts use `__package__ = "trainer"` and `sys.path.append` to handle relative imports — they must be run from the `trainer/` directory or via `torchrun`/`deepspeed`
- Model weights are saved as half-precision CPU tensors: `{k: v.half().cpu() for k, v in state_dict.items()}`
- The `--from_weight` arg loads a pretrained `.pth` from `out/` by name prefix (e.g., `pretrain` loads `out/pretrain_512.pth`)
- DDP ignores RoPE buffers: `model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}`
- Experiment tracking uses SwanLab (imported as `wandb`) when `--use_wandb` is set
