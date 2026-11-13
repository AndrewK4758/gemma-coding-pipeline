#!/usr/bin/env python3
"""Phase 4A Dual-GPU bfloat16 Compiler-Verified Direct Preference Optimization (DPO).

Loads the Phase 3 SFT LoRA adapter (`checkpoints/sft-<tier>/final_adapter`) atop the
42-layer asymmetrically sharded `google/gemma-4-E4B-it` backbone (`cuda:1` 16 GB +
`cuda:0` 12 GB) and aligns the policy on compiler-verified `(prompt, chosen, rejected)`
preference pairs using 256-token chunked sequence log-probabilities (`compute_chunked_sequence_logps`).

Usage:
    ./scripts/gpu-mode.zsh train
    CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY .venv/bin/python train/dpo_compiler.py --tier smoke --max-steps 15
    ./scripts/gpu-mode.zsh inference
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bitsandbytes as bnb
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_dataset import format_gemma2_chat_and_mask_labels, format_gemma_messages_and_mask_labels
from train.model_loader import (
    DEFAULT_MODEL_ID,
    load_sharded_lora_model_and_tokenizer,
    query_dual_gpu_vram,
)


def _pad_to_multiple(seq: list[int], pad_val: int, multiple: int = 8) -> list[int]:
    if multiple <= 1 or len(seq) % multiple == 0:
        return seq
    return seq + [pad_val] * (multiple - (len(seq) % multiple))


def _encode_padded_tensors(
    prompt: str | list[dict[str, str]], response: str, tokenizer: PreTrainedTokenizerBase, max_seq_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = int(tokenizer.pad_token_id or 0)
    if isinstance(prompt, list):
        tok = format_gemma_messages_and_mask_labels(
            [*prompt, {"role": "assistant", "content": response}], tokenizer=tokenizer, max_seq_len=max_seq_length
        )
    else:
        tok = format_gemma2_chat_and_mask_labels(prompt, response, tokenizer=tokenizer, max_seq_len=max_seq_length)
    ids = _pad_to_multiple(list(tok["input_ids"]), pad_id, 8)
    mask = _pad_to_multiple(list(tok["attention_mask"]), 0, 8)
    lbls = _pad_to_multiple(list(tok["labels"]), -100, 8)
    if all(x == -100 for x in lbls):
        lbls[-1] = ids[-1]
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
        torch.tensor([lbls], dtype=torch.long),
    )


def compute_chunked_sequence_logps(
    model: PeftModel | nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact sequence log-probabilities over completion tokens (labels != -100) in 256-token chunks."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_cfg = causal_lm.config.get_text_config() if hasattr(causal_lm.config, "get_text_config") else causal_lm.config
    softcap: float | None = getattr(text_cfg, "final_logit_softcapping", None)

    def _chunk_logp_sum(lm_head: nn.Module, h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        logits_chunk = lm_head(h_chunk)
        if softcap is not None:
            logits_chunk = torch.tanh(logits_chunk / softcap) * softcap
        return -F.cross_entropy(logits_chunk.float(), y_chunk, reduction="sum")

    first_dev = next(causal_lm.model.parameters()).device
    outputs = causal_lm.model(
        input_ids=input_ids.to(first_dev),
        attention_mask=attention_mask.to(first_dev),
        use_cache=False,
        return_dict=True,
    )
    hidden_states: torch.Tensor = outputs.last_hidden_state
    head_dev = hidden_states.device

    batch_logps: list[torch.Tensor] = []
    batch_counts: list[torch.Tensor] = []
    for b_idx in range(hidden_states.size(0)):
        shift_h = hidden_states[b_idx, :-1, :].contiguous()
        shift_y = labels[b_idx, 1:].to(head_dev).contiguous()
        valid_mask = shift_y != -100
        active_h = shift_h[valid_mask]
        active_y = shift_y[valid_mask]
        tok_count = torch.tensor(float(max(active_y.numel(), 1)), device=head_dev, dtype=torch.float32)

        if active_y.numel() == 0:
            seq_logp = (hidden_states[b_idx, :1, :].sum() * 0.0).float()
        else:
            seq_logp = torch.zeros((), device=head_dev, dtype=torch.float32)
            for start in range(0, active_h.size(0), chunk_size):
                h_slice = active_h[start : start + chunk_size]
                y_slice = active_y[start : start + chunk_size]
                if torch.is_grad_enabled() and h_slice.requires_grad:
                    chunk_lp = torch.utils.checkpoint.checkpoint(
                        _chunk_logp_sum, causal_lm.lm_head, h_slice, y_slice, use_reentrant=False
                    )
                else:
                    chunk_lp = _chunk_logp_sum(causal_lm.lm_head, h_slice, y_slice)
                seq_logp = seq_logp + chunk_lp
        batch_logps.append(seq_logp)
        batch_counts.append(tok_count)

    return torch.stack(batch_logps, dim=0), torch.stack(batch_counts, dim=0)


def load_dpo_split(parquet_path: Path, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 4096) -> list[dict[str, Any]]:
    """Load DPO preference pairs from Parquet and tokenize (prompt, chosen) and (prompt, rejected)."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing DPO Parquet file: {parquet_path}")
    samples: list[dict[str, Any]] = []
    for row in pq.read_table(parquet_path).to_pylist():
        c_ids, c_mask, c_lbls = _encode_padded_tensors(row["prompt"], row["chosen"], tokenizer, max_seq_length)
        r_ids, r_mask, r_lbls = _encode_padded_tensors(row["prompt"], row["rejected"], tokenizer, max_seq_length)
        samples.append(
            {
                "id": row["id"],
                "language": row["language"],
                "chosen_input_ids": c_ids,
                "chosen_attention_mask": c_mask,
                "chosen_labels": c_lbls,
                "rejected_input_ids": r_ids,
                "rejected_attention_mask": r_mask,
                "rejected_labels": r_lbls,
                "ref_chosen_logp": None,
                "ref_rejected_logp": None,
            }
        )
    return samples


@torch.no_grad()
def precompute_reference_logps(
    model: PeftModel, samples: list[dict[str, Any]], ref_mode: str, desc: str, console: Console
) -> None:
    """Pre-compute reference log-probs in torch.no_grad() using either the initial SFT adapter or disable_adapter()."""
    model.eval()
    t_start = time.time()
    for s in samples:
        if ref_mode == "disable_adapter":
            with model.disable_adapter():
                c_lp, _ = compute_chunked_sequence_logps(model, s["chosen_input_ids"], s["chosen_attention_mask"], s["chosen_labels"])
                r_lp, _ = compute_chunked_sequence_logps(model, s["rejected_input_ids"], s["rejected_attention_mask"], s["rejected_labels"])
        else:
            c_lp, _ = compute_chunked_sequence_logps(model, s["chosen_input_ids"], s["chosen_attention_mask"], s["chosen_labels"])
            r_lp, _ = compute_chunked_sequence_logps(model, s["rejected_input_ids"], s["rejected_attention_mask"], s["rejected_labels"])
        s["ref_chosen_logp"] = float(c_lp.item())
        s["ref_rejected_logp"] = float(r_lp.item())
    console.print(f"    Cached reference log-probs for [green]{len(samples)}[/green] {desc} pairs (mode={ref_mode}, {time.time() - t_start:.1f}s)")


@torch.no_grad()
def evaluate_dpo(
    model: PeftModel, eval_samples: list[dict[str, Any]], beta: float, step: int, tb_writer: SummaryWriter
) -> dict[str, float]:
    """Evaluate DPO loss, implicit rewards, reward margins, and preference accuracy on held-out eval split."""
    model.eval()
    losses, c_rews, r_rews, margins, accs, c_logps, r_logps, logp_accs = [], [], [], [], [], [], [], []
    for s in eval_samples:
        pol_c_lp, c_cnt = compute_chunked_sequence_logps(model, s["chosen_input_ids"], s["chosen_attention_mask"], s["chosen_labels"])
        pol_r_lp, r_cnt = compute_chunked_sequence_logps(model, s["rejected_input_ids"], s["rejected_attention_mask"], s["rejected_labels"])
        pol_c, pol_r = float(pol_c_lp.item()), float(pol_r_lp.item())
        c_rew = beta * (pol_c - float(s["ref_chosen_logp"]))
        r_rew = beta * (pol_r - float(s["ref_rejected_logp"]))
        margin = c_rew - r_rew
        losses.append(float(-F.logsigmoid(torch.tensor(margin, dtype=torch.float32)).item()))
        c_rews.append(c_rew)
        r_rews.append(r_rew)
        margins.append(margin)
        accs.append(1.0 if margin > 1e-7 else 0.0)
        c_logps.append(pol_c)
        r_logps.append(pol_r)
        logp_accs.append(1.0 if (pol_c / c_cnt.item()) > (pol_r / r_cnt.item()) else 0.0)

    n = max(len(eval_samples), 1)
    metrics = {
        "step": step,
        "eval_loss": round(sum(losses) / n, 4),
        "rewards/chosen": round(sum(c_rews) / n, 4),
        "rewards/rejected": round(sum(r_rews) / n, 4),
        "rewards/margins": round(sum(margins) / n, 4),
        "rewards/accuracies": round(sum(accs) / n, 4),
        "logps/chosen": round(sum(c_logps) / n, 4),
        "logps/rejected": round(sum(r_logps) / n, 4),
        "logps/accuracies": round(sum(logp_accs) / n, 4),
    }
    for k, v in metrics.items():
        if k != "step":
            tb_writer.add_scalar(f"eval/{k.replace('/', '_')}", v, step)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4A Dual-GPU Compiler-Verified DPO Trainer.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--ref-mode", choices=("precompute_sft", "disable_adapter"), default="precompute_sft")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console, t0 = Console(), time.time()
    random.seed(42)
    torch.manual_seed(42)

    suffix = f"{args.tier}-{args.adapter_name}" if args.adapter_name else args.tier
    sft_adapter_dir = Path(args.sft_adapter) if args.sft_adapter else (REPO_ROOT / "checkpoints" / f"sft-{suffix}" / "final_adapter")
    if not sft_adapter_dir.exists():
        raise FileNotFoundError(f"Phase 3 SFT adapter not found at {sft_adapter_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "checkpoints" / f"dpo-{suffix}")
    logging_dir = Path(args.logging_dir) if args.logging_dir else (REPO_ROOT / "runs" / f"dpo-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]Phase 4A DPO — Tier: {args.tier.upper()} ({args.model_id}, beta={args.beta})")
    model, tokenizer, load_meta = load_sharded_lora_model_and_tokenizer(
        model_id=args.model_id, adapter_path=sft_adapter_dir, load_in_4bit=args.load_in_4bit, console=console
    )

    data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "data" / "processed" / args.tier)
    all_train = load_dpo_split(data_dir / "dpo_train.parquet", tokenizer, max_seq_length=args.max_seq_length)
    eval_samples = load_dpo_split(data_dir / "dpo_eval.parquet", tokenizer, max_seq_length=args.max_seq_length)
    random.Random(42).shuffle(all_train)
    needed = min(len(all_train), args.max_steps * args.gradient_accumulation_steps) if args.max_steps > 0 else len(all_train)
    active_train = all_train[:needed]

    console.print(f"[bold yellow]Pre-computing zero-VRAM reference policy log-probs (mode={args.ref_mode})...[/bold yellow]")
    precompute_reference_logps(model, eval_samples, args.ref_mode, "eval", console)
    precompute_reference_logps(model, active_train, args.ref_mode, "active train", console)

    tb_writer = SummaryWriter(log_dir=str(logging_dir))
    pre_eval = evaluate_dpo(model, eval_samples, args.beta, step=0, tb_writer=tb_writer)
    console.print(
        f"    Pre-DPO eval_loss=[bold]{pre_eval['eval_loss']:.4f}[/bold] | "
        f"margin={pre_eval['rewards/margins']:.4f} | rew_acc={pre_eval['rewards/accuracies']:.4f}"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, len(active_train) // args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(round(0.1 * total_steps))), num_training_steps=total_steps
    )

    step_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = [pre_eval]
    sample_idx = 0

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        s_loss, s_cr, s_rr, s_m, s_acc, s_clp, s_rlp = [], [], [], [], [], [], []

        for _ in range(args.gradient_accumulation_steps):
            s = active_train[sample_idx % len(active_train)]
            sample_idx += 1
            pol_c_lp, _ = compute_chunked_sequence_logps(model, s["chosen_input_ids"], s["chosen_attention_mask"], s["chosen_labels"])
            pol_r_lp, _ = compute_chunked_sequence_logps(model, s["rejected_input_ids"], s["rejected_attention_mask"], s["rejected_labels"])
            ref_c = torch.tensor([s["ref_chosen_logp"]], device=pol_c_lp.device, dtype=torch.float32)
            ref_r = torch.tensor([s["ref_rejected_logp"]], device=pol_r_lp.device, dtype=torch.float32)

            c_rew, r_rew = args.beta * (pol_c_lp - ref_c), args.beta * (pol_r_lp - ref_r)
            margin = c_rew - r_rew
            pair_loss = -F.logsigmoid(margin).mean()
            (pair_loss / args.gradient_accumulation_steps).backward()

            s_loss.append(float(pair_loss.item()))
            s_cr.append(float(c_rew.item()))
            s_rr.append(float(r_rew.item()))
            s_m.append(float(margin.item()))
            s_acc.append(1.0 if float(margin.item()) > 0.0 else 0.0)
            s_clp.append(float(pol_c_lp.item()))
            s_rlp.append(float(pol_r_lp.item()))

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item())
        lr_now = float(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

        vram = query_dual_gpu_vram()
        c0_peak = float(vram.get("cuda:0", {}).get("peak_allocated_mib", 0.0))
        c1_peak = float(vram.get("cuda:1", {}).get("peak_allocated_mib", 0.0))
        k = args.gradient_accumulation_steps
        entry = {
            "step": step,
            "loss": round(sum(s_loss) / k, 4),
            "rewards/chosen": round(sum(s_cr) / k, 4),
            "rewards/rejected": round(sum(s_rr) / k, 4),
            "rewards/margins": round(sum(s_m) / k, 4),
            "rewards/accuracies": round(sum(s_acc) / k, 4),
            "logps/chosen": round(sum(s_clp) / k, 4),
            "logps/rejected": round(sum(s_rlp) / k, 4),
            "grad_norm": round(grad_norm, 4),
            "learning_rate": lr_now,
            "cuda0_peak_mib": c0_peak,
            "cuda1_peak_mib": c1_peak,
        }
        step_history.append(entry)
        for mk in ("loss", "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "grad_norm"):
            tb_writer.add_scalar(f"train/{mk.replace('/', '_')}", entry[mk], step)
        tb_writer.add_scalar("train/learning_rate", lr_now, step)
        tb_writer.add_scalar("vram/cuda0_peak_mib", c0_peak, step)
        tb_writer.add_scalar("vram/cuda1_peak_mib", c1_peak, step)

        console.print(
            f"[cyan][DPO Step {step:02d}/{total_steps:02d}][/cyan] loss=[bold green]{entry['loss']:.4f}[/bold green] | "
            f"rew_c={entry['rewards/chosen']:+.3f}, rew_r={entry['rewards/rejected']:+.3f} | "
            f"margin=[bold]{entry['rewards/margins']:+.3f}[/bold] | acc={entry['rewards/accuracies']:.2f} | "
            f"grad_norm={grad_norm:.3f} | VRAM c0={c0_peak:.0f}M c1={c1_peak:.0f}M"
        )
        if step % args.eval_steps == 0 or step == total_steps:
            ev = evaluate_dpo(model, eval_samples, args.beta, step=step, tb_writer=tb_writer)
            eval_history.append(ev)
            console.print(
                f"    [yellow][Eval @ Step {step:02d}][/yellow] eval_loss=[bold green]{ev['eval_loss']:.4f}[/bold green] | "
                f"margin=[bold green]{ev['rewards/margins']:+.4f}[/bold green] | acc=[bold green]{ev['rewards/accuracies']:.4f}[/bold green]"
            )

    post_eval = eval_history[-1]
    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    tb_writer.flush()
    tb_writer.close()

    payload = {
        "tier": args.tier, "elapsed_sec": round(time.time() - t0, 2), "beta": args.beta, "ref_mode": args.ref_mode,
        "sft_adapter_dir": str(sft_adapter_dir), "final_adapter_dir": str(final_adapter_dir),
        "max_steps": total_steps, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "pre_dpo_eval": pre_eval, "post_dpo_eval": post_eval,
        "eval_loss_delta": round(post_eval["eval_loss"] - pre_eval["eval_loss"], 4),
        "reward_margin_delta": round(post_eval["rewards/margins"] - pre_eval["rewards/margins"], 4),
        "reward_accuracy_delta": round(post_eval["rewards/accuracies"] - pre_eval["rewards/accuracies"], 4),
        "step_telemetry": step_history, "eval_telemetry": eval_history,
        "final_vram": query_dual_gpu_vram(), "model_meta": load_meta,
    }
    metrics_path = output_dir / "dpo_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved DPO LoRA adapter:[/bold green] {final_adapter_dir} | [bold green]Metrics:[/bold green] {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
