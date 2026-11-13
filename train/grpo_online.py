#!/usr/bin/env python3
"""Phase 4B Modular Dual-GPU Online Group Relative Policy Optimization (GRPO) with Compiler Verifier.

Samples G candidate code completions per prompt (--num-generations 4), grades each completion
in the real TypeScript 6.0.3 / Go 1.24 / .NET 10 Roslyn compiler sandbox, computes group-normalized
advantages A_i = (r_i - mean(r)) / max(std(r), 1e-4), and performs a KL-regularized policy-gradient
LoRA update using 256-token chunked sequence log-probabilities.

Usage:
    ./scripts/gpu-mode.zsh train
    CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY .venv/bin/python train/grpo_online.py --tier smoke --max-steps 2 --num-generations 4
    ./scripts/gpu-mode.zsh inference
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
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
from peft import PeftModel
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_dataset import SYSTEM_PREAMBLE
from data.compiler_sandbox import SandboxResult, extract_diff_block, verify_code_sample, verify_oss_worktree_patch
from data.ingest_sources import extract_fenced_code
from train.dpo_compiler import _encode_padded_tensors, compute_chunked_sequence_logps
from train.model_loader import (
    DEFAULT_MODEL_ID,
    get_input_device,
    load_sharded_lora_model_and_tokenizer,
    query_dual_gpu_vram,
)


def grade_completion_rubric(
    completion_text: str, language: str, task: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deterministic RLVR reward rubric supporting snippet compilation and live OSS worktree tests."""
    if task and task.get("record_type") == "oss_worktree" and task.get("repo_path") and task.get("parent_sha"):
        diff_body = extract_diff_block(completion_text)
        if not diff_body:
            return {
                "reward": -1.0, "code_extracted": False, "compiler_passed": False,
                "toolchain": "oss-worktree", "diagnostics": "Missing valid unified diff",
            }
        lowered = completion_text.lower()
        if "rest of implementation" in lowered or "... existing code" in lowered:
            return {
                "reward": -1.0, "code_extracted": True, "compiler_passed": False,
                "toolchain": "oss-worktree", "diagnostics": "Rejected truncated placeholder diff",
            }
        eco = str(task.get("ecosystem", "react"))
        passed, diag = verify_oss_worktree_patch(
            repo_path=str(task["repo_path"]),
            parent_sha=str(task["parent_sha"]),
            diff_text=completion_text,
            ecosystem=eco,
            test_files=list(task.get("test_files") or []),
        )
        reward = 3.5 if passed else (-0.5 if "git apply failed" in diag else 0.25)
        return {
            "reward": round(float(reward), 4),
            "code_extracted": True,
            "compiler_passed": passed,
            "toolchain": f"oss-{eco}",
            "diagnostics": diag[:200],
        }

    code = extract_fenced_code(completion_text)
    if not code:
        return {"reward": -1.0, "code_extracted": False, "compiler_passed": False, "toolchain": "none", "diagnostics": "Missing fenced code"}
    res: SandboxResult = verify_code_sample(completion_text, language)
    if res.passed:
        clean_bonus = 0.5 if (res.diagnostics == "OK" and len(code.strip().splitlines()) >= 2) else 0.0
        reward = 1.0 + 2.0 + clean_bonus
    else:
        reward = 1.0 - 1.0
    return {
        "reward": round(float(reward), 4),
        "code_extracted": True,
        "compiler_passed": res.passed,
        "toolchain": res.toolchain,
        "diagnostics": res.diagnostics[:200],
    }


@torch.no_grad()
def generate_group_completions(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    num_generations: int = 4,
    max_new_tokens: int = 224,
) -> list[str]:
    """Sample G candidate code completions for a single coding prompt with KV cache enabled."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))
    chat_prompt = f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{prompt_text.strip()}{eot}\n{sot}model\n"

    first_dev = get_input_device(causal_lm)
    enc = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].repeat(num_generations, 1).to(first_dev)
    prompt_mask = enc["attention_mask"].repeat(num_generations, 1).to(first_dev)
    prompt_len = prompt_ids.size(1)

    model.eval()
    causal_lm.gradient_checkpointing_disable()
    causal_lm.config.use_cache = True
    if hasattr(causal_lm.config, "text_config"):
        causal_lm.config.text_config.use_cache = True

    eot_id = tokenizer.convert_tokens_to_ids(eot)
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    if isinstance(eot_id, int) and eot_id > 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    try:
        out_ids = causal_lm.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=int(tokenizer.pad_token_id or 0),
            eos_token_id=eos_ids if eos_ids else None,
            use_cache=True,
        )
    finally:
        causal_lm.config.use_cache = False
        if hasattr(causal_lm.config, "text_config"):
            causal_lm.config.text_config.use_cache = False
        causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return [tokenizer.decode(out_ids[i, prompt_len:], skip_special_tokens=True).strip() for i in range(num_generations)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4B Dual-GPU Online GRPO Trainer with Compiler Sandbox.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 2)
    console = Console()
    t0 = time.time()
    torch.manual_seed(42)

    suffix = f"{args.tier}-{args.adapter_name}" if args.adapter_name else args.tier
    dpo_cand = REPO_ROOT / "checkpoints" / f"dpo-{suffix}" / "final_adapter"
    sft_cand = REPO_ROOT / "checkpoints" / f"sft-{suffix}" / "final_adapter"
    adapter_dir = Path(args.adapter_path) if args.adapter_path else (dpo_cand if dpo_cand.exists() else sft_cand)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Initial adapter checkpoint not found at {adapter_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "checkpoints" / f"grpo-{suffix}")
    logging_dir = Path(args.logging_dir) if args.logging_dir else (REPO_ROOT / "runs" / f"grpo-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]Phase 4B Online GRPO — Tier: {args.tier.upper()} (G={args.num_generations}, kl={args.kl_beta})")
    model, tokenizer, load_meta = load_sharded_lora_model_and_tokenizer(
        model_id=args.model_id, adapter_path=adapter_dir, load_in_4bit=args.load_in_4bit, console=console
    )

    data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "data" / "processed" / args.tier)
    grpo_parquet = data_dir / "grpo_tasks.parquet"
    grpo_rows = pq.read_table(grpo_parquet).to_pylist() if grpo_parquet.exists() else []
    if grpo_rows:
        prompt_pool = grpo_rows
    else:
        raw_rows = pq.read_table(data_dir / "dpo_train.parquet").to_pylist()
        prompt_pool = sorted(raw_rows, key=lambda r: len(r["chosen"]))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    tb_writer = SummaryWriter(log_dir=str(logging_dir))
    step_history: list[dict[str, Any]] = []

    for step in range(1, args.max_steps + 1):
        rec = prompt_pool[(step - 1) % len(prompt_pool)]
        lang, prompt_text = str(rec["language"]), str(rec["prompt"])
        step_t0 = time.time()

        completions = generate_group_completions(
            model, tokenizer, prompt_text, num_generations=args.num_generations, max_new_tokens=args.max_new_tokens
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.num_generations, 4)) as pool:
            grades = list(pool.map(lambda c: grade_completion_rubric(c, lang, task=rec), completions))

        live_rewards = [float(g["reward"]) for g in grades]
        live_pass_rate = round(sum(1.0 for g in grades if g["compiler_passed"]) / max(len(grades), 1), 4)
        if len(set(live_rewards)) == 1 and args.num_generations >= 2:
            pos_ref = rec.get("chosen") or f"```diff\n{rec.get('reference_diff', '')}\n```"
            neg_ref = rec.get("rejected") or "// ... rest of implementation remains the same"
            contrast_comp = str(neg_ref) if grades[0]["compiler_passed"] else str(pos_ref)
            completions[-1] = contrast_comp
            grades[-1] = grade_completion_rubric(contrast_comp, lang, task=rec)

        rewards = [float(g["reward"]) for g in grades]
        mean_r, std_r = statistics.mean(rewards), statistics.pstdev(rewards)
        advantages = [(r - mean_r) / max(std_r, 1e-4) for r in rewards]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        group_losses, group_pg, group_kl = [], [], []

        for idx in range(args.num_generations):
            ids_t, mask_t, lbls_t = _encode_padded_tensors(prompt_text, completions[idx], tokenizer, args.max_seq_length)
            with torch.no_grad(), model.disable_adapter():
                ref_seq_lp, tok_cnt = compute_chunked_sequence_logps(model, ids_t, mask_t, lbls_t)

            pol_seq_lp, tok_cnt = compute_chunked_sequence_logps(model, ids_t, mask_t, lbls_t)
            pol_tok_lp, ref_tok_lp = pol_seq_lp / tok_cnt, ref_seq_lp.detach() / tok_cnt
            log_ratio = ref_tok_lp - pol_tok_lp
            kl_est = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0)) - log_ratio - 1.0
            adv_t = torch.tensor([advantages[idx]], device=pol_tok_lp.device, dtype=torch.float32)

            pg_loss_i = -(adv_t * pol_tok_lp).mean()
            kl_loss_i = (args.kl_beta * kl_est).mean()
            total_loss_i = pg_loss_i + kl_loss_i
            (total_loss_i / args.num_generations).backward()

            group_losses.append(float(total_loss_i.item()))
            group_pg.append(float(pg_loss_i.item()))
            group_kl.append(float(kl_est.mean().item()))

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item())
        optimizer.step()

        vram = query_dual_gpu_vram()
        c0_peak = float(vram.get("cuda:0", {}).get("peak_allocated_mib", 0.0))
        c1_peak = float(vram.get("cuda:1", {}).get("peak_allocated_mib", 0.0))
        entry = {
            "step": step,
            "prompt_id": rec["id"],
            "language": lang,
            "loss": round(statistics.mean(group_losses), 4),
            "pg_loss": round(statistics.mean(group_pg), 4),
            "kl_div": round(statistics.mean(group_kl), 4),
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "live_pass_rate": live_pass_rate,
            "rewards": [round(r, 2) for r in rewards],
            "advantages": [round(a, 4) for a in advantages],
            "grad_norm": round(grad_norm, 4),
            "elapsed_sec": round(time.time() - step_t0, 2),
            "cuda0_peak_mib": c0_peak,
            "cuda1_peak_mib": c1_peak,
        }
        step_history.append(entry)
        for mk in ("loss", "mean_reward", "live_pass_rate", "kl_div"):
            tb_writer.add_scalar(f"grpo/{mk}", entry[mk], step)
        console.print(
            f"[cyan][GRPO Step {step:02d}/{args.max_steps:02d}][/cyan] ({lang}) "
            f"loss=[bold green]{entry['loss']:.4f}[/bold green] | "
            f"rewards={entry['rewards']} (mean={mean_r:+.2f}, live_pass={live_pass_rate*100:.0f}%) | "
            f"A_i={[round(a, 2) for a in advantages]} | kl={entry['kl_div']:.4f} | "
            f"grad_norm={grad_norm:.3f} | VRAM c0={c0_peak:.0f}M c1={c1_peak:.0f}M"
        )

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    tb_writer.flush()
    tb_writer.close()

    metrics_path = output_dir / "grpo_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "tier": args.tier,
                "elapsed_sec": round(time.time() - t0, 2),
                "initial_adapter_dir": str(adapter_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "num_generations": args.num_generations,
                "kl_beta": args.kl_beta,
                "step_telemetry": step_history,
                "final_vram": query_dual_gpu_vram(),
                "model_meta": load_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[bold green]Saved GRPO LoRA adapter:[/bold green] {final_adapter_dir} | [bold green]Metrics:[/bold green] {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
