#!/usr/bin/env python3
"""Phase 3 Dual-GPU bfloat16 LoRA Supervised Fine-Tuning (SFT) for Gemma 4 E4B IT / Gemma 2 9B IT.

Usage:
    ./scripts/gpu-mode.zsh train
    CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY .venv/bin/python train/sft_lora.py --tier smoke --max-steps 15
    ./scripts/gpu-mode.zsh inference
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Enforce dual-GPU visibility and lazy CUDA kernel loading before importing torch/bitsandbytes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_dataset import format_gemma2_chat_and_mask_labels
from train.model_loader import (
    DEFAULT_MODEL_ID,
    load_sharded_lora_model_and_tokenizer,
    query_dual_gpu_vram,
)


class ParquetSftDataset(Dataset):
    """In-memory Arrow/Parquet SFT dataset with dynamic tokenizer alignment and max_seq_length truncation."""

    def __init__(
        self,
        parquet_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 4096,
    ) -> None:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing SFT Parquet file: {parquet_path}")
        raw_rows = pq.read_table(parquet_path).to_pylist()
        expected_sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))

        self.samples: list[dict[str, list[int]]] = []
        for row in raw_rows:
            text_val = str(row.get("text", ""))
            if not text_val.startswith(expected_sot) and "instruction" in row and "response" in row:
                tok_fields = format_gemma2_chat_and_mask_labels(
                    row["instruction"],
                    row["response"],
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_length,
                )
                input_ids = list(tok_fields["input_ids"])
                attention_mask = list(tok_fields["attention_mask"])
                labels = list(tok_fields["labels"])
            else:
                input_ids = list(row["input_ids"])[:max_seq_length]
                attention_mask = list(row["attention_mask"])[:max_seq_length]
                labels = list(row["labels"])[:max_seq_length]

            # Guarantee at least 1 unmasked completion token so cross-entropy never divides by 0 (NaN)
            if all(lbl == -100 for lbl in labels) and len(input_ids) > 0:
                labels[-1] = input_ids[-1]

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


@dataclass
class SftCodingDataCollator:
    """Dynamic batch collator that right-pads sequences to a multiple of 8 for Tensor Core MMA alignment."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1 and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            pad_len = max_len - len(feat["input_ids"])
            batch_input_ids.append(feat["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(feat["attention_mask"] + [0] * pad_len)
            batch_labels.append(feat["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class DualGpuTelemetryCallback(TrainerCallback):
    """Records step-by-step loss, perplexity, grad_norm, LR, and dual-GPU VRAM to Rich stdout, TensorBoard, and JSON."""

    def __init__(self, console: Console, tb_writer: SummaryWriter) -> None:
        self.console = console
        self.tb_writer = tb_writer
        self.step_history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        vram = query_dual_gpu_vram()
        cuda0_peak = float(vram.get("cuda:0", {}).get("peak_allocated_mib", 0.0))
        cuda1_peak = float(vram.get("cuda:1", {}).get("peak_allocated_mib", 0.0))
        cuda0_drv = float(vram.get("cuda:0", {}).get("driver_used_mib", 0.0))
        cuda1_drv = float(vram.get("cuda:1", {}).get("driver_used_mib", 0.0))

        if "loss" in logs:
            loss_val = float(logs["loss"])
            ppl_val = math.exp(min(loss_val, 20.0))
            grad_norm_val = float(logs.get("grad_norm", 0.0))
            lr_val = float(logs.get("learning_rate", 0.0))
            step = int(state.global_step)

            self.tb_writer.add_scalar("train/perplexity", ppl_val, step)
            self.tb_writer.add_scalar("vram/cuda0_peak_mib", cuda0_peak, step)
            self.tb_writer.add_scalar("vram/cuda1_peak_mib", cuda1_peak, step)

            entry = {
                "step": step,
                "epoch": round(float(logs.get("epoch", 0.0)), 4),
                "loss": round(loss_val, 4),
                "perplexity": round(ppl_val, 4),
                "grad_norm": round(grad_norm_val, 4),
                "learning_rate": lr_val,
                "cuda0_peak_mib": cuda0_peak,
                "cuda1_peak_mib": cuda1_peak,
                "cuda0_driver_mib": cuda0_drv,
                "cuda1_driver_mib": cuda1_drv,
            }
            self.step_history.append(entry)
            self.console.print(
                f"[cyan][Step {step:02d}][/cyan] loss=[bold green]{loss_val:.4f}[/bold green] "
                f"(PPL={ppl_val:.2f}) | grad_norm={grad_norm_val:.4f} | lr={lr_val:.2e} | "
                f"VRAM peak cuda:0={cuda0_peak:.0f} MiB, cuda:1={cuda1_peak:.0f} MiB"
            )

        if "eval_loss" in logs:
            eval_loss = float(logs["eval_loss"])
            eval_ppl = math.exp(min(eval_loss, 20.0))
            step = int(state.global_step)
            self.tb_writer.add_scalar("eval/perplexity", eval_ppl, step)
            self.eval_history.append(
                {
                    "step": step,
                    "eval_loss": round(eval_loss, 4),
                    "eval_perplexity": round(eval_ppl, 4),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Dual-GPU bfloat16/4-bit LoRA SFT Trainer.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke", help="Dataset scale tier.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Base Hugging Face model checkpoint.")
    parser.add_argument("--data-dir", default=None, help="Processed Parquet directory (defaults to data/processed/<tier>).")
    parser.add_argument("--adapter-name", default=None, help="Optional domain adapter suffix (e.g., dotnet, react).")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit NF4 QLoRA loading (for 26B-A4B MoE).")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum context sequence length.")
    parser.add_argument("--max-steps", type=int, default=15, help="Max optimizer steps (-1 for full epochs).")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of training epochs when max-steps=-1.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Micro-batch size per pass.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--eval-steps", type=int, default=5, help="Evaluation frequency in global steps.")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank r.")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA scaling alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to checkpoints/sft-<tier>).")
    parser.add_argument("--logging-dir", default=None, help="TensorBoard log directory (defaults to runs/sft-<tier>).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    t0 = time.time()

    suffix = f"{args.tier}-{args.adapter_name}" if args.adapter_name else args.tier
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "checkpoints" / f"sft-{suffix}")
    logging_dir = Path(args.logging_dir) if args.logging_dir else (REPO_ROOT / "runs" / f"sft-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    console.rule(f"[bold cyan]Phase 3 Dual-GPU LoRA SFT — Tier: {args.tier.upper()} ({args.model_id})")
    model, tokenizer, load_meta = load_sharded_lora_model_and_tokenizer(
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        console=console,
    )

    data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "data" / "processed" / args.tier)
    train_dataset = ParquetSftDataset(data_dir / "sft_train.parquet", tokenizer, max_seq_length=args.max_seq_length)
    eval_dataset = ParquetSftDataset(data_dir / "sft_eval.parquet", tokenizer, max_seq_length=args.max_seq_length)
    collator = SftCodingDataCollator(pad_token_id=int(tokenizer.pad_token_id or 0), pad_to_multiple_of=8)

    console.print(
        f"[bold]Loaded Parquet splits:[/bold] train=[green]{len(train_dataset)}[/green] rows, "
        f"eval=[green]{len(eval_dataset)}[/green] rows (max_seq_length={args.max_seq_length})"
    )

    tb_writer = SummaryWriter(log_dir=str(logging_dir))
    telemetry_cb = DualGpuTelemetryCallback(console=console, tb_writer=tb_writer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        prediction_loss_only=True,
        save_strategy="no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[telemetry_cb],
    )

    console.print("[bold yellow]Running baseline pre-train evaluation on held-out eval split...[/bold yellow]")
    pre_eval = trainer.evaluate()
    pre_eval_loss = float(pre_eval["eval_loss"])
    pre_eval_ppl = math.exp(min(pre_eval_loss, 20.0))
    console.print(f"    Pre-train eval_loss = [bold]{pre_eval_loss:.4f}[/bold] (perplexity = {pre_eval_ppl:.2f})")

    console.print("[bold green]Starting dual-GPU bfloat16 LoRA SFT training loop...[/bold green]")
    train_result = trainer.train()

    console.print("[bold yellow]Running post-train evaluation on held-out eval split...[/bold yellow]")
    post_eval = trainer.evaluate()
    post_eval_loss = float(post_eval["eval_loss"])
    post_eval_ppl = math.exp(min(post_eval_loss, 20.0))
    console.print(f"    Post-train eval_loss = [bold green]{post_eval_loss:.4f}[/bold green] (perplexity = {post_eval_ppl:.2f})")

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    tb_writer.flush()
    tb_writer.close()

    final_vram = query_dual_gpu_vram()
    elapsed_sec = round(time.time() - t0, 2)

    summary_payload = {
        "tier": args.tier,
        "elapsed_sec": elapsed_sec,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "model_meta": load_meta,
        "pre_train_eval": {"eval_loss": round(pre_eval_loss, 4), "eval_perplexity": round(pre_eval_ppl, 4)},
        "post_train_eval": {"eval_loss": round(post_eval_loss, 4), "eval_perplexity": round(post_eval_ppl, 4)},
        "eval_loss_delta": round(post_eval_loss - pre_eval_loss, 4),
        "train_runtime_metrics": train_result.metrics,
        "step_telemetry": telemetry_cb.step_history,
        "eval_telemetry": telemetry_cb.eval_history,
        "final_vram": final_vram,
        "final_adapter_dir": str(final_adapter_dir),
        "tensorboard_logging_dir": str(logging_dir),
    }
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    tbl = Table(title=f"Phase 3 SFT Step-by-Step Telemetry ({args.tier.upper()})")
    tbl.add_column("Step", justify="right", style="cyan")
    tbl.add_column("Train Loss", justify="right", style="green")
    tbl.add_column("Perplexity", justify="right")
    tbl.add_column("Grad Norm", justify="right")
    tbl.add_column("LR", justify="right")
    tbl.add_column("cuda:0 Peak (MiB)", justify="right", style="magenta")
    tbl.add_column("cuda:1 Peak (MiB)", justify="right", style="yellow")
    for row in telemetry_cb.step_history:
        tbl.add_row(
            str(row["step"]),
            f"{row['loss']:.4f}",
            f"{row['perplexity']:.2f}",
            f"{row['grad_norm']:.4f}",
            f"{row['learning_rate']:.2e}",
            f"{row['cuda0_peak_mib']:.1f}",
            f"{row['cuda1_peak_mib']:.1f}",
        )
    console.print(tbl)
    console.print(
        f"[bold green]Saved PEFT LoRA adapter:[/bold green] {final_adapter_dir} | "
        f"[bold green]Metrics JSON:[/bold green] {metrics_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
