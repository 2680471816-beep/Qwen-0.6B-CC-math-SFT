#!/usr/bin/env python3
"""SFT training wrapper for Qwen3-0.6B-Base math reasoning.

Usage:
    python src/training/train_sft.py \
        --train-data outputs/coldstart_llm/coldstart_llm_A_sft.jsonl \
        --output-dir outputs/experiments/coldstart_round1 \
        --max-epochs 4 \
        --learning-rate 2e-5 \
        --wandb-run-name my_experiment
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LLM_SCRIPTS = PROJECT_ROOT.parent / "llm" / "scripts"
SFT_PIPELINE = LLM_SCRIPTS / "sft_async_pipeline" / "run_openrlhf_sft_train_eval_best.py"
CHAT_TEMPLATE = PROJECT_ROOT.parent / "models" / "Qwen3-0.6B-Base" / "chat_template.jinja"
MODEL_PATH = "/home/ubuntu/models/Qwen3-0.6B-Base"


def train_sft(
    train_data: str,
    output_dir: str,
    max_epochs: int = 4,
    learning_rate: float = 2e-5,
    max_len: int = 4096,
    micro_batch_size: int = 4,
    gradient_accumulation: int = 4,
    train_gpu: str = "0",
    zero_stage: int = 2,
    attn_impl: str = "flash_attention_2",
    wandb_run_name: str = None,
    lora_rank: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    target_modules: list = None,
    model_path: str = None,
):
    """Run OpenRLHF SFT training with wandb monitoring."""
    # WandB 配置
    import wandb as _wandb
    wandb_api_key = _wandb.api.api_key or ""
    wandb_project = os.environ.get("WANDB_PROJECT", "midterm_math_sft")

    cmd = [
        sys.executable,
        str(SFT_PIPELINE),
        "--model-path", model_path or MODEL_PATH,
        "--train-dataset", train_data,
        "--validation-input", str(PROJECT_ROOT / "valid_1000.jsonl"),
        "--chat-template-file", str(CHAT_TEMPLATE),
        "--run-root", output_dir,
        "--max-epochs", str(max_epochs),
        "--max-len", str(max_len),
        "--micro-train-batch-size", str(micro_batch_size),
        "--gradient-accumulation", str(gradient_accumulation),
        "--learning-rate", str(learning_rate),
        "--train-gpu", train_gpu,
        "--eval-device-id", train_gpu,
        "--zero-stage", str(zero_stage),
        "--attn-implementation", attn_impl,
        "--tb-run-name", wandb_run_name or f"sft_{max_epochs}ep",
        "--no-adam-offload",
    ]
    if wandb_api_key:
        cmd += [
            "--use_wandb", wandb_api_key,
            "--wandb_project", wandb_project,
        ]
    # LoRA parameters
    if lora_rank > 0:
        cmd += [
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--lora_dropout", str(lora_dropout),
        ]
        if target_modules:
            cmd += ["--target_modules"] + target_modules
    print(f"Running: {' '.join(cmd)}")
    print(f"WandB Project: {os.environ.get('WANDB_PROJECT')}")
    print(f"WandB Run Name: {os.environ.get('WANDB_RUN_NAME', 'default')}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="SFT training for Qwen3-0.6B-Base")
    parser.add_argument("--train-data", required=True, help="Path to training JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--max-epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--train-gpu", default="0", help="GPU ID for training")
    parser.add_argument("--wandb-run-name", default=None, help="WandB run name")
    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank, 0 means full fine-tuning")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0, help="LoRA dropout")
    parser.add_argument("--target-modules", type=str, nargs="*", default=None, help="Target modules for LoRA")
    parser.add_argument("--model-path", default=None, help="Base model path, default is Qwen3-0.6B-Base")
    args = parser.parse_args()

    train_sft(
        train_data=args.train_data,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        train_gpu=args.train_gpu,
        wandb_run_name=args.wandb_run_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()