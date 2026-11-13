#!/usr/bin/env python3
"""Top-Level End-to-End Orchestrator for the Gemma 4 E4B / Gemma 2 9B Coding Fine-Tuning Pipeline.

Orchestrates all pipeline stages (`data`, `sft`, `dpo`, `grpo`, `eval`, `export`) across scale tiers
(`smoke`, `dev`, `full`) with automatic Dual-GPU mode switching (`./scripts/gpu-mode.zsh`) and a
guaranteed `try...finally` restoration of any optional `$LOCAL_OFFLOAD_SERVICE` endpoint.

Usage:
    .venv/bin/python run_pipeline.py --tier smoke --stages eval,export
    .venv/bin/python run_pipeline.py --tier dev --stages data,sft,dpo,eval,export
    .venv/bin/python run_pipeline.py --tier full --stages data,sft,dpo,eval,export
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent
GPU_MODE_SCRIPT = REPO_ROOT / "scripts" / "gpu-mode.zsh"
HEALTH_URL = os.environ.get("INFERENCE_HEALTH_URL", "http://127.0.0.1:11434/health")

VALID_STAGES = ("data", "sft", "dpo", "grpo", "eval", "export")
GPU_STAGES = {"sft", "dpo", "grpo", "eval", "export"}

TIER_DEFAULTS: dict[str, dict[str, int]] = {
    "smoke": {"sft_steps": 15, "sft_eval_steps": 5, "dpo_steps": 15, "dpo_eval_steps": 5, "grpo_steps": 2, "max_seq_len": 4096},
    "dev": {"sft_steps": 180, "sft_eval_steps": 50, "dpo_steps": 120, "dpo_eval_steps": 30, "grpo_steps": 20, "max_seq_len": 4096},
    "full": {"sft_steps": -1, "sft_eval_steps": 250, "dpo_steps": 600, "dpo_eval_steps": 150, "grpo_steps": 50, "max_seq_len": 8192},
}


def _resolve_eval_steps(eval_override: int | None, tier_default: int, step_override: int | None) -> int:
    """Return a positive validation cadence, scaling down when max_steps is overridden below tier defaults."""
    if eval_override is not None:
        return max(1, eval_override)
    if step_override is not None and step_override > 0:
        return max(1, min(tier_default, max(1, step_override // 4)))
    return tier_default


def switch_gpu_mode(mode: str, console: Console) -> None:
    """Switch Dual-GPU mode between 'train' and 'inference' via ./scripts/gpu-mode.zsh."""
    console.print(f"[bold magenta][GPU-Mode] Switching to '{mode}' mode via {GPU_MODE_SCRIPT.name}...[/bold magenta]")
    subprocess.run([str(GPU_MODE_SCRIPT), mode], cwd=str(REPO_ROOT), check=True)


def verify_offload_health() -> bool:
    """Verify http://127.0.0.1:11434/health responds HTTP 200 OK."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3.0) as resp:
            body = resp.read().decode("utf-8", errors="ignore").strip()
            return resp.status == 200 and ("OK" in body.upper() or "HEALTH" in body.upper() or len(body) >= 0)
    except Exception:
        return False


def run_stage_command(stage: str, cmd: list[str], console: Console, dry_run: bool = False) -> dict[str, Any]:
    """Execute a single pipeline stage command with live console streaming."""
    console.rule(f"[bold cyan]Stage: {stage.upper()} — {' '.join(cmd)}")
    t0 = time.time()
    if dry_run:
        console.print("[yellow]Dry-run enabled; skipping execution.[/yellow]")
        return {"stage": stage, "command": cmd, "status": "DRY_RUN", "elapsed_sec": 0.0}

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL)
    elapsed = round(time.time() - t0, 2)
    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline stage '{stage}' failed with exit code {proc.returncode}")
    return {"stage": stage, "command": cmd, "status": "OK", "elapsed_sec": elapsed}


def build_stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Construct ordered stage command invocations for the requested tier, domain, and model."""
    from data.build_dataset import resolve_processed_dir

    py = sys.executable
    t_cfg = TIER_DEFAULTS[args.tier]
    is_26b = "26b" in args.model_id.lower()
    use_4bit = bool(getattr(args, "load_in_4bit", False) or is_26b)
    domain = getattr(args, "domain", "all")
    oss_only = bool(getattr(args, "oss_only", False))
    adapter_name = getattr(args, "adapter_name", None) or (
        domain if domain in ("dotnet", "react", "go", "cpp") else ("oss" if oss_only or domain == "oss" else None)
    )
    data_dir = str(resolve_processed_dir(args.tier, domain=domain, oss_only=oss_only))

    sft_steps = args.sft_steps if args.sft_steps is not None else t_cfg["sft_steps"]
    sft_eval_steps = _resolve_eval_steps(getattr(args, "sft_eval_steps", None), t_cfg["sft_eval_steps"], args.sft_steps)
    dpo_steps = args.dpo_steps if args.dpo_steps is not None else t_cfg["dpo_steps"]
    dpo_eval_steps = _resolve_eval_steps(getattr(args, "dpo_eval_steps", None), t_cfg["dpo_eval_steps"], args.dpo_steps)
    grpo_steps = args.grpo_steps if args.grpo_steps is not None else t_cfg["grpo_steps"]
    seq_len = args.max_seq_length if args.max_seq_length is not None else (
        min(4096, t_cfg["max_seq_len"]) if is_26b else t_cfg["max_seq_len"]
    )
    sft_lr, dpo_lr = ("1e-4", "2e-5") if use_4bit else ("2e-4", "5e-5")
    grad_accum = "16" if use_4bit else "8"

    shared_train: list[str] = ["--tier", args.tier, "--model-id", args.model_id, "--data-dir", data_dir]
    if adapter_name:
        shared_train.extend(["--adapter-name", adapter_name])
    if use_4bit:
        shared_train.append("--load-in-4bit")

    requested = [s.strip().lower() for s in args.stages.split(",") if s.strip()]
    for s in requested:
        if s not in VALID_STAGES:
            raise ValueError(f"Unknown stage {s!r}. Valid stages: {VALID_STAGES}")

    plan: list[tuple[str, list[str]]] = []
    for stage in requested:
        if stage == "data":
            cmd = [py, "data/build_dataset.py", "--tier", args.tier, "--domain", domain, "--output-dir", data_dir]
            if oss_only:
                cmd.append("--oss-only")
            if getattr(args, "oss_corpus_root", None):
                cmd.extend(["--oss-corpus-root", args.oss_corpus_root])
            plan.append(("data", cmd))
        elif stage == "sft":
            plan.append(
                (
                    "sft",
                    [
                        py, "train/sft_lora.py", *shared_train,
                        "--max-seq-length", str(seq_len), "--max-steps", str(sft_steps),
                        "--eval-steps", str(sft_eval_steps),
                        "--learning-rate", sft_lr, "--gradient-accumulation-steps", grad_accum,
                    ],
                )
            )
        elif stage == "dpo":
            plan.append(
                (
                    "dpo",
                    [
                        py, "train/dpo_compiler.py", *shared_train,
                        "--max-seq-length", str(seq_len), "--max-steps", str(dpo_steps),
                        "--eval-steps", str(dpo_eval_steps),
                        "--learning-rate", dpo_lr,
                    ],
                )
            )
        elif stage == "grpo":
            plan.append(
                (
                    "grpo",
                    [py, "train/grpo_online.py", *shared_train, "--max-seq-length", str(seq_len), "--max-steps", str(grpo_steps)],
                )
            )
        elif stage == "eval":
            plan.append(("eval", [py, "eval/benchmark_coding.py", *shared_train]))
        elif stage == "export":
            exp_cmd = [py, "export/merge_and_quantize_gguf.py", "--tier", args.tier, "--model-id", args.model_id, "--data-dir", data_dir]
            if adapter_name:
                exp_cmd.extend(["--adapter-name", adapter_name])
            plan.append(("export", exp_cmd))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-End Gemma Coding Fine-Tuning Pipeline Orchestrator.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke", help="Dataset & training scale tier.")
    parser.add_argument(
        "--stages", "--stage", dest="stages", default="data,sft,dpo,eval,export",
        help="Comma-separated stages to execute (subset of: data,sft,dpo,grpo,eval,export).",
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it", help="Base Hugging Face model ID.")
    parser.add_argument("--domain", choices=("all", "dotnet", "react", "go", "cpp", "general", "oss"), default="all")
    parser.add_argument("--adapter-name", default=None, help="Named domain LoRA suffix (e.g. dotnet, react, go, cpp).")
    parser.add_argument("--oss-only", action="store_true", help="Train solely on mined OSS repos.")
    parser.add_argument(
        "--oss-corpus-root",
        default=os.environ.get("OSS_CORPUS_DIR", str(Path.home() / "corpus" / "oss_repos")),
        help="Root of cloned OSS corpus.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit NF4 QLoRA loading (auto on 26B-A4B).")
    parser.add_argument("--sft-steps", type=int, default=None, help="Override SFT max_steps.")
    parser.add_argument("--sft-eval-steps", type=int, default=None, help="Override SFT validation cadence in global steps.")
    parser.add_argument("--dpo-steps", type=int, default=None, help="Override DPO max_steps.")
    parser.add_argument("--dpo-eval-steps", type=int, default=None, help="Override DPO validation cadence in global steps.")
    parser.add_argument("--grpo-steps", type=int, default=None, help="Override GRPO max_steps.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override max sequence length.")
    parser.add_argument("--dry-run", action="store_true", help="Print stage execution plan without running.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    console = Console()
    t0 = time.time()
    plan = build_stage_commands(args)
    stage_results: list[dict[str, Any]] = []
    current_gpu_mode: str | None = None

    console.rule(f"[bold green]Gemma Coding Pipeline Orchestrator — Tier: {args.tier.upper()} | Stages: {args.stages}")
    try:
        for stage, cmd in plan:
            if not args.dry_run:
                required_mode = "inference" if stage == "data" else ("train" if stage in GPU_STAGES else None)
                if required_mode and required_mode != current_gpu_mode:
                    switch_gpu_mode(required_mode, console)
                    current_gpu_mode = required_mode
            res = run_stage_command(stage, cmd, console, dry_run=args.dry_run)
            stage_results.append(res)
    finally:
        if not args.dry_run and current_gpu_mode == "train":
            switch_gpu_mode("inference", console)

    healthy = verify_offload_health() if not args.dry_run else True
    tbl = Table(title=f"Pipeline Execution Summary ({args.tier.upper()})")
    tbl.add_column("Stage", style="cyan")
    tbl.add_column("Status", style="green")
    tbl.add_column("Elapsed (s)", justify="right")
    tbl.add_column("Command")
    for r in stage_results:
        tbl.add_row(r["stage"], r["status"], f"{r['elapsed_sec']:.2f}s", " ".join(r["command"]))
    console.print(tbl)

    summary_path = REPO_ROOT / "outputs" / f"pipeline_run_{args.tier}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "tier": args.tier,
                "stages": [s for s, _ in plan],
                "total_elapsed_sec": round(time.time() - t0, 2),
                "offload_service_healthy": healthy,
                "stage_results": stage_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(
        f"[bold green]Pipeline finished in {time.time() - t0:.1f}s[/bold green] | "
        f"inference_endpoint={'OK' if healthy else 'DOWN'} | Summary: {summary_path}"
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
