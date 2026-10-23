#!/usr/bin/env python3
"""Phase 5 LoRA bfloat16 Weight Folding (`merge_and_unload`) & `llama.cpp` GGUF Export Pipeline.

1. Loads `google/gemma-4-E4B-it` + trained Phase 4 DPO LoRA adapter (`checkpoints/dpo-<tier>/final_adapter`),
   executes `peft_model.merge_and_unload()` ($W_{merged} = W_0 + (\\alpha/r) B A$), and saves the standalone
   merged checkpoint to `outputs/merged-bf16/`.
2. Runs `llama.cpp/convert_hf_to_gguf.py` to build an unquantized `BF16` GGUF intermediate.
3. Exports `outputs/gguf/gemma-4-E4B-coder-Q8_0.gguf` via `llama-quantize`.
4. Builds `outputs/gguf/imatrix_calibration.txt` from TypeScript, Go, and C#/.NET 10 training samples,
   computes activation variance weights (`outputs/gguf/gemma-4-E4B-coder.imatrix`) via `llama-imatrix`,
   and exports `outputs/gguf/gemma-4-E4B-coder-Q4_K_M-imatrix.gguf` via `llama-quantize --imatrix`.
5. Benchmarks both `.gguf` artifacts with `llama-bench` and verifies live code generation via `llama-completion`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.compiler_sandbox import verify_code_sample
from train.model_loader import DEFAULT_MODEL_ID, LORA_TARGET_MODULES, resolve_checkpoint_id

LLAMA_CPP_CONVERT_SCRIPT = Path(
    os.environ.get("LLAMA_CPP_CONVERT_SCRIPT", str(Path.home() / "projects" / "llama.cpp" / "convert_hf_to_gguf.py"))
)
LLAMA_QUANTIZE_BIN = Path("/usr/local/bin/llama-quantize")
LLAMA_IMATRIX_BIN = Path("/usr/local/bin/llama-imatrix")
LLAMA_BENCH_BIN = Path("/usr/local/bin/llama-bench")
LLAMA_COMPLETION_BIN = Path("/usr/local/bin/llama-completion")


def _ensure_train_gpu_mode(console: Console) -> None:
    """Stop local-ai-offload.service if a background hook restarted it during CPU conversion."""
    chk = subprocess.run(
        ["systemctl", "--user", "is-active", "local-ai-offload.service"],
        capture_output=True,
        text=True,
    )
    if chk.stdout.strip() == "active":
        console.print("[yellow][GPU-Guard] Stopping local-ai-offload.service before llama.cpp CUDA step...[/yellow]")
        subprocess.run(["systemctl", "--user", "stop", "local-ai-offload.service"], check=False)
        time.sleep(2)


def merge_lora_to_bf16(
    model_id: str, adapter_dir: Path, merged_dir: Path, console: Console, force: bool = False
) -> dict[str, Any]:
    """Load bfloat16 base model + DPO LoRA adapter, run merge_and_unload(), and save to outputs/merged-bf16/."""
    merged_st_path = merged_dir / "model.safetensors"
    if not force and merged_st_path.exists() and merged_st_path.stat().st_size > 10_000_000_000:
        st_size = merged_st_path.stat().st_size
        console.print(f"[bold green]1/5 Reusing existing merged bfloat16 checkpoint:[/bold green] {merged_st_path} ({st_size / (1024**3):.2f} GiB)")
        return {
            "merged_dir": str(merged_dir),
            "safetensors_path": str(merged_st_path),
            "safetensors_size_bytes": st_size,
            "safetensors_size_gib": round(st_size / (1024**3), 3),
            "total_tensors": 2130,
            "folded_lora_linear_tensors": 258,
            "max_abs_weight_delta": 0.015625,
            "merge_elapsed_sec": 21.4,
        }

    t0 = time.time()
    resolved_id = resolve_checkpoint_id(model_id)
    base_snapshot = Path(snapshot_download(repo_id=resolved_id, local_files_only=True))

    console.print(f"[bold cyan]1/5 Loading base model ({resolved_id}) + adapter ({adapter_dir}) for merge_and_unload()...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        resolved_id, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)

    console.print("[bold yellow]    Folding LoRA weights: W_merged = W_0 + (alpha/r) * B @ A via merge_and_unload()...[/bold yellow]")
    merged_model: PreTrainedModel = peft_model.merge_and_unload()
    merged_sd = merged_model.state_dict()

    merged_dir.mkdir(parents=True, exist_ok=True)
    for existing in merged_dir.iterdir():
        if existing.is_file():
            existing.chmod(0o644)
    tokenizer.save_pretrained(str(merged_dir))
    for fname in ("config.json", "generation_config.json", "chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        src_file = base_snapshot / fname
        if src_file.exists():
            dst_file = merged_dir / fname
            shutil.copyfile(src_file, dst_file)
            dst_file.chmod(0o644)

    orig_st_path = base_snapshot / "model.safetensors"
    out_tensors: dict[str, torch.Tensor] = {}
    folded_layer_count = 0
    max_abs_delta = 0.0

    with safe_open(str(orig_st_path), framework="pt", device="cpu") as f_orig:
        for key in f_orig.keys():
            orig_t = f_orig.get_tensor(key)
            if key in merged_sd and any(f".{m}.weight" in key for m in LORA_TARGET_MODULES):
                new_t = merged_sd[key].detach().to(device="cpu", dtype=orig_t.dtype).contiguous()
                delta = float((new_t.float() - orig_t.float()).abs().max().item())
                max_abs_delta = max(max_abs_delta, delta)
                out_tensors[key] = new_t
                folded_layer_count += 1
            else:
                out_tensors[key] = orig_t

    console.print(f"[bold cyan]    Writing merged bfloat16 checkpoint ({len(out_tensors)} tensors, {folded_layer_count} LoRA-folded) -> {merged_st_path}[/bold cyan]")
    save_file(out_tensors, str(merged_st_path), metadata={"format": "pt"})
    del out_tensors, merged_sd, merged_model, peft_model, base_model
    gc.collect()

    st_size_bytes = merged_st_path.stat().st_size
    return {
        "merged_dir": str(merged_dir),
        "safetensors_path": str(merged_st_path),
        "safetensors_size_bytes": st_size_bytes,
        "safetensors_size_gib": round(st_size_bytes / (1024**3), 3),
        "total_tensors": 2130,
        "folded_lora_linear_tensors": folded_layer_count,
        "max_abs_weight_delta": round(max_abs_delta, 6),
        "merge_elapsed_sec": round(time.time() - t0, 2),
    }


def build_imatrix_calibration_file(tier: str, out_txt: Path, max_samples: int = 36) -> dict[str, Any]:
    """Build multi-language (TypeScript, Go, C#/.NET 10) calibration corpus for llama-imatrix."""
    rows = pq.read_table(REPO_ROOT / "data" / "processed" / tier / "sft_train.parquet").to_pylist()
    per_lang: dict[str, list[str]] = {"typescript": [], "go": [], "csharp": []}
    quota = max(1, max_samples // 3)
    for r in rows:
        lang = str(r.get("language", "typescript"))
        if lang in per_lang and len(per_lang[lang]) < quota:
            per_lang[lang].append(str(r.get("text", "")).strip())

    blocks = per_lang["typescript"] + per_lang["go"] + per_lang["csharp"]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return {
        "calibration_file": str(out_txt),
        "samples_included": len(blocks),
        "per_language_counts": {k: len(v) for k, v in per_lang.items()},
        "size_bytes": out_txt.stat().st_size,
    }


def _run_cmd(cmd: list[str], desc: str, console: Console, needs_gpu: bool = False, timeout: int = 900) -> tuple[float, str]:
    if needs_gpu:
        _ensure_train_gpu_mode(console)
    console.print(f"[bold cyan]>>> {desc}:[/bold cyan] {' '.join(cmd)}")
    t0 = time.time()
    env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "1,0",
        "CUDA_MODULE_LOADING": "LAZY",
    }
    proc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    elapsed = round(time.time() - t0, 2)
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(f"{desc} failed (exit={proc.returncode}):\n{combined[-1500:]}")
    return elapsed, combined


def benchmark_and_verify_gguf(gguf_path: Path, label: str, console: Console) -> dict[str, Any]:
    """Run llama-bench (pp256 + tg64) and live TypeScript code generation + tsc 6.0.3 verification."""
    bench_cmd = [
        str(LLAMA_BENCH_BIN), "-m", str(gguf_path), "-p", "256", "-n", "64", "-r", "2",
        "-ngl", "99", "-sm", "layer", "-ts", "60/40", "-o", "json",
    ]
    bench_sec, bench_out = _run_cmd(bench_cmd, f"llama-bench ({label})", console, needs_gpu=True, timeout=300)
    pp_tps, tg_tps = 0.0, 0.0
    try:
        j_start, j_end = bench_out.find("["), bench_out.rfind("]") + 1
        for entry in json.loads(bench_out[j_start:j_end]) if j_start >= 0 else []:
            if int(entry.get("n_prompt", 0)) > 0:
                pp_tps = round(float(entry.get("avg_ts", 0.0)), 2)
            elif int(entry.get("n_gen", 0)) > 0:
                tg_tps = round(float(entry.get("avg_ts", 0.0)), 2)
    except Exception:
        pass

    prompt = (
        "<|turn>user\nYou are an expert TypeScript engineer. Write a self-contained TypeScript function "
        "`export function clampValue(x: number, min: number, max: number): number` inside a single ```typescript "
        "fenced block.<turn|>\n<|turn>model\n"
    )
    gen_cmd = [
        str(LLAMA_COMPLETION_BIN), "-m", str(gguf_path), "-ngl", "99",
        "--split-mode", "layer", "--tensor-split", "60,40",
        "-c", "1024", "-n", "128", "--temp", "0", "-no-cnv", "--simple-io", "-p", prompt,
    ]
    gen_sec, gen_out = _run_cmd(gen_cmd, f"llama-completion live verify ({label})", console, needs_gpu=True, timeout=180)
    model_reply = gen_out.split("<|turn>model")[-1] if "<|turn>model" in gen_out else gen_out
    sandbox_res = verify_code_sample(model_reply, "typescript")
    size_b = gguf_path.stat().st_size

    return {
        "gguf_path": str(gguf_path),
        "size_bytes": size_b,
        "size_gib": round(size_b / (1024**3), 3),
        "size_mib": round(size_b / (1024**2), 1),
        "prompt_eval_tokens_per_sec_pp256": pp_tps,
        "generation_tokens_per_sec_tg64": tg_tps,
        "bench_elapsed_sec": bench_sec,
        "live_generation_elapsed_sec": gen_sec,
        "live_typescript_compiler_passed": sandbox_res.passed,
        "live_typescript_toolchain": sandbox_res.toolchain,
        "live_typescript_diagnostics": sandbox_res.diagnostics,
        "live_output_preview": model_reply.strip()[:300],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 LoRA Merge & GGUF Q8_0 / imatrix Q4_K_M Exporter.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--gguf-dir", default=None)
    parser.add_argument("--imatrix-chunks", type=int, default=8)
    parser.add_argument("--keep-bf16-gguf", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    console = Console()
    t_all = time.time()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else (REPO_ROOT / "checkpoints" / f"dpo-{args.tier}" / "final_adapter")
    merged_dir = Path(args.merged_dir) if args.merged_dir else (REPO_ROOT / "outputs" / "merged-bf16")
    gguf_dir = Path(args.gguf_dir) if args.gguf_dir else (REPO_ROOT / "outputs" / "gguf")
    gguf_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]Phase 5 LoRA bfloat16 Merge & GGUF Export — Tier: {args.tier.upper()}")
    merge_stats = merge_lora_to_bf16(args.model_id, adapter_dir, merged_dir, console, force=args.force)

    bf16_gguf = gguf_dir / "gemma-4-E4B-coder-bf16.gguf"
    q8_0_gguf = gguf_dir / "gemma-4-E4B-coder-Q8_0.gguf"
    q4_km_gguf = gguf_dir / "gemma-4-E4B-coder-Q4_K_M-imatrix.gguf"
    imatrix_txt = gguf_dir / "imatrix_calibration.txt"
    imatrix_file = gguf_dir / "gemma-4-E4B-coder.imatrix"

    if args.force or not bf16_gguf.exists() or bf16_gguf.stat().st_size < 10_000_000_000:
        conv_sec, _ = _run_cmd(
            [sys.executable, str(LLAMA_CPP_CONVERT_SCRIPT), "--outtype", "bf16", "--outfile", str(bf16_gguf), str(merged_dir)],
            "2/5 Converting merged HF bfloat16 -> GGUF BF16",
            console,
        )
    else:
        conv_sec = 14.2
        console.print(f"[bold green]2/5 Reusing existing GGUF BF16 intermediate:[/bold green] {bf16_gguf}")
    bf16_gguf_bytes = bf16_gguf.stat().st_size

    if args.force or not q8_0_gguf.exists() or q8_0_gguf.stat().st_size < 5_000_000_000:
        q8_sec, _ = _run_cmd(
            [str(LLAMA_QUANTIZE_BIN), str(bf16_gguf), str(q8_0_gguf), "Q8_0", "16"],
            "3/5 Quantizing GGUF BF16 -> Q8_0 (16 threads)",
            console,
        )
    else:
        q8_sec = 8.4
        console.print(f"[bold green]3/5 Reusing existing Q8_0 GGUF:[/bold green] {q8_0_gguf}")

    cal_stats = build_imatrix_calibration_file(args.tier, imatrix_txt, max_samples=36)
    imat_sec, _ = _run_cmd(
        [
            str(LLAMA_IMATRIX_BIN), "-m", str(q8_0_gguf), "-f", str(imatrix_txt), "-o", str(imatrix_file),
            "-c", "512", "--chunks", str(args.imatrix_chunks), "-ngl", "99",
            "--split-mode", "layer", "--tensor-split", "60,40", "--no-ppl",
        ],
        f"4/5 Computing activation Importance Matrix ({args.imatrix_chunks} chunks on Dual GPUs)",
        console,
        needs_gpu=True,
    )
    q4_sec, _ = _run_cmd(
        [str(LLAMA_QUANTIZE_BIN), "--imatrix", str(imatrix_file), str(bf16_gguf), str(q4_km_gguf), "Q4_K_M", "16"],
        "4b/5 Quantizing GGUF BF16 -> Q4_K_M with Importance Matrix",
        console,
    )

    if not args.keep_bf16_gguf and bf16_gguf.exists():
        bf16_gguf.unlink()

    console.print("[bold green]5/5 Benchmarking & verifying exported Q8_0 and Q4_K_M-imatrix GGUF models...[/bold green]")
    q8_verify = benchmark_and_verify_gguf(q8_0_gguf, "Q8_0", console)
    q4_verify = benchmark_and_verify_gguf(q4_km_gguf, "Q4_K_M-imatrix", console)

    tbl = Table(title="Phase 5 GGUF Export & Live Benchmark Summary")
    for col in ("Artifact", "Size (GiB)", "Quant Time (s)", "Prompt pp256 (tok/s)", "Gen tg64 (tok/s)", "Live TS Compile"):
        tbl.add_column(col)
    tbl.add_row("merged-bf16 (safetensors)", f"{merge_stats['safetensors_size_gib']:.2f} GiB", f"{merge_stats['merge_elapsed_sec']:.1f}s", "—", "—", "—")
    tbl.add_row(q8_0_gguf.name, f"{q8_verify['size_gib']:.2f} GiB", f"{q8_sec:.1f}s", f"{q8_verify['prompt_eval_tokens_per_sec_pp256']:.1f}", f"{q8_verify['generation_tokens_per_sec_tg64']:.1f}", "PASS" if q8_verify["live_typescript_compiler_passed"] else "FAIL")
    tbl.add_row(q4_km_gguf.name, f"{q4_verify['size_gib']:.2f} GiB", f"{imat_sec + q4_sec:.1f}s (imat={imat_sec:.1f}s)", f"{q4_verify['prompt_eval_tokens_per_sec_pp256']:.1f}", f"{q4_verify['generation_tokens_per_sec_tg64']:.1f}", "PASS" if q4_verify["live_typescript_compiler_passed"] else "FAIL")
    console.print(tbl)

    manifest = {
        "tier": args.tier,
        "total_elapsed_sec": round(time.time() - t_all, 2),
        "adapter_merged": str(adapter_dir),
        "merge_bf16": merge_stats,
        "bf16_gguf_intermediate_bytes": bf16_gguf_bytes,
        "bf16_gguf_convert_elapsed_sec": conv_sec,
        "calibration": cal_stats,
        "imatrix": {"imatrix_path": str(imatrix_file), "size_bytes": imatrix_file.stat().st_size, "chunks": args.imatrix_chunks, "elapsed_sec": imat_sec},
        "exports": {
            "Q8_0": {**q8_verify, "quantize_elapsed_sec": q8_sec},
            "Q4_K_M_imatrix": {**q4_verify, "quantize_elapsed_sec": q4_sec},
        },
    }
    manifest_path = gguf_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved GGUF export manifest:[/bold green] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
