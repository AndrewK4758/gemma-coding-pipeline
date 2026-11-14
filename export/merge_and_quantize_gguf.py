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
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_dataset import resolve_processed_dir
from data.compiler_sandbox import verify_code_sample
from export.multi_engine import (  # noqa: F401
    export_lora_gguf,
    merge_lora_weights_into_shards,
    resolve_llama_cpp_bin,
    resolve_llama_cpp_dir,
    resolve_llama_cpp_script,
    write_ollama_modelfile,
    write_vllm_serve_script,
)
from train.model_loader import DEFAULT_MODEL_ID, resolve_checkpoint_id

LLAMA_CPP_DIR = resolve_llama_cpp_dir()
LLAMA_CPP_CONVERT_SCRIPT = resolve_llama_cpp_script("convert_hf_to_gguf.py", "LLAMA_CPP_CONVERT_SCRIPT")
LLAMA_QUANTIZE_BIN = resolve_llama_cpp_bin("llama-quantize", "LLAMA_QUANTIZE_BIN")
LLAMA_IMATRIX_BIN = resolve_llama_cpp_bin("llama-imatrix", "LLAMA_IMATRIX_BIN")
LLAMA_BENCH_BIN = resolve_llama_cpp_bin("llama-bench", "LLAMA_BENCH_BIN")
LLAMA_COMPLETION_BIN = resolve_llama_cpp_bin("llama-completion", "LLAMA_COMPLETION_BIN")
DEFAULT_TENSOR_SPLIT = os.environ.get("LLAMA_TENSOR_SPLIT", "65,35")


def _ensure_train_gpu_mode(console: Console) -> None:
    """Stop optional external local inference service ($LOCAL_OFFLOAD_SERVICE) before CUDA steps."""
    svc_name = os.environ.get("LOCAL_OFFLOAD_SERVICE", "").strip()
    if not svc_name:
        return
    chk = subprocess.run(
        ["systemctl", "--user", "is-active", svc_name],
        capture_output=True,
        text=True,
    )
    if chk.stdout.strip() == "active":
        console.print(f"[yellow][GPU-Guard] Stopping {svc_name} before llama.cpp CUDA step...[/yellow]")
        subprocess.run(["systemctl", "--user", "stop", svc_name], check=False)
        time.sleep(2)


def resolve_base_snapshot_dir(resolved_id: str) -> Path:
    """Resolve local HF snapshot directory containing base weight shards without failing on missing docs/eval files."""
    direct = Path(resolved_id)
    if direct.is_dir():
        return direct
    for marker in ("model.safetensors.index.json", "model.safetensors"):
        try:
            return Path(cached_file(resolved_id, marker, local_files_only=True)).parent
        except Exception:
            pass
    try:
        snaps_dir = Path(cached_file(resolved_id, "config.json", local_files_only=True)).parent.parent
        candidates = sorted((d for d in snaps_dir.iterdir() if d.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in candidates:
            if (snap / "config.json").exists() and (
                (snap / "model.safetensors.index.json").exists() or (snap / "model.safetensors").exists()
            ):
                return snap
    except Exception:
        pass
    return Path(snapshot_download(repo_id=resolved_id, local_files_only=True, ignore_patterns=[".eval_results/*", "*.md", ".git*"]))


def merge_lora_to_bf16(
    model_id: str, adapter_dir: Path, merged_dir: Path, console: Console, force: bool = False
) -> tuple[dict[str, Any], Path]:
    """Stream base shards on CPU and fold LoRA weights into merged bfloat16 safetensors."""
    resolved_id = resolve_checkpoint_id(model_id)
    base_snapshot = resolve_base_snapshot_dir(resolved_id)
    existing_shards = list(merged_dir.glob("*.safetensors")) if merged_dir.exists() else []
    total_existing = sum(p.stat().st_size for p in existing_shards)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_mtime = adapter_weights.stat().st_mtime if adapter_weights.exists() else 0.0
    shards_fresh = bool(existing_shards) and min(p.stat().st_mtime for p in existing_shards) >= adapter_mtime
    if not force and total_existing > 10_000_000_000 and shards_fresh:
        console.print(f"[bold green]1/5 Reusing merged bfloat16 checkpoint:[/bold green] {merged_dir} ({total_existing / (1024**3):.2f} GiB)")
        return {
            "merged_dir": str(merged_dir),
            "safetensors_path": str(existing_shards[0]),
            "safetensors_size_bytes": total_existing,
            "safetensors_size_gib": round(total_existing / (1024**3), 3),
            "shard_count": len(existing_shards),
            "total_tensors": 0,
            "folded_lora_linear_tensors": 0,
            "max_abs_weight_delta": 0.0,
            "merge_elapsed_sec": 0.0,
        }, base_snapshot

    console.print(f"[bold cyan]1/5 Streaming shard-by-shard CPU LoRA merge ({resolved_id} + {adapter_dir})...[/bold cyan]")
    stats = merge_lora_weights_into_shards(base_snapshot, adapter_dir, merged_dir)
    console.print(
        f"[bold cyan]    Merged {stats['shard_count']} shard(s), {stats['total_tensors']} tensors "
        f"({stats['folded_lora_linear_tensors']} LoRA-folded, max_delta={stats['max_abs_weight_delta']})[/bold cyan]"
    )
    return stats, base_snapshot


def build_imatrix_calibration_file(
    tier: str, out_txt: Path, max_samples: int = 36, data_dir: Path | None = None
) -> dict[str, Any]:
    """Build multi-language calibration corpus for llama-imatrix from the active Parquet dataset."""
    src_dir = data_dir if data_dir else (REPO_ROOT / "data" / "processed" / tier)
    rows = pq.read_table(src_dir / "sft_train.parquet").to_pylist()
    per_lang: dict[str, list[str]] = {"typescript": [], "go": [], "csharp": [], "cpp": []}
    present_langs = {str(r.get("language", "typescript")) for r in rows} & set(per_lang.keys())
    quota = max(1, max_samples // max(len(present_langs), 1))
    for r in rows:
        lang = str(r.get("language", "typescript"))
        if lang in per_lang and len(per_lang[lang]) < quota:
            per_lang[lang].append(str(r.get("text", "")).strip())

    blocks = [txt for lang in ("typescript", "go", "csharp", "cpp") for txt in per_lang[lang]]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return {
        "calibration_file": str(out_txt),
        "samples_included": len(blocks),
        "per_language_counts": {k: len(v) for k, v in per_lang.items()},
        "size_bytes": out_txt.stat().st_size,
    }


def _run_cmd(cmd: list[str], desc: str, console: Console, needs_gpu: bool = False, timeout: int = 1800) -> tuple[float, str]:
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


def benchmark_and_verify_gguf(
    gguf_path: Path,
    label: str,
    console: Console,
    extra_gpu_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run llama-bench (pp256 + tg64) and live TypeScript code generation + tsc 6.0.3 verification."""
    gpu_extra = list(extra_gpu_args or [])
    bench_cmd = [
        str(LLAMA_BENCH_BIN), "-m", str(gguf_path), "-p", "256", "-n", "64", "-r", "2",
        "-ngl", "99", *gpu_extra, "-sm", "layer", "-ts", DEFAULT_TENSOR_SPLIT.replace(",", "/"), "-o", "json",
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
        str(LLAMA_COMPLETION_BIN), "-m", str(gguf_path), "-ngl", "99", *gpu_extra,
        "--split-mode", "layer", "--tensor-split", DEFAULT_TENSOR_SPLIT,
        "-c", "1024", "-n", "384", "--temp", "0", "-no-cnv", "--simple-io", "-p", prompt,
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


def _resolve_adapter_dir(repo_root: Path, tier: str, adapter_name: str | None) -> Path:
    suffix = f"{tier}-{adapter_name}" if adapter_name else tier
    for stage in ("grpo", "dpo", "sft"):
        cand = repo_root / "checkpoints" / f"{stage}-{suffix}" / "final_adapter"
        if cand.exists():
            return cand
    return repo_root / "checkpoints" / f"dpo-{suffix}" / "final_adapter"


def resolve_export_paths(
    repo_root: Path,
    model_id: str,
    adapter_name: str | None = None,
    merged_dir: str | None = None,
    gguf_dir: str | None = None,
    tier: str = "smoke",
    adapter_dir: str | None = None,
) -> dict[str, Any]:
    """Compute isolated model tags, merged BF16 directories, GGUF paths, and MoE Q8 offload flags."""
    is_moe = "26b" in model_id.lower()
    base_tag = "gemma-4-26B-A4B-coder" if is_moe else "gemma-4-E4B-coder"
    model_tag = f"{base_tag}-{adapter_name}" if adapter_name else base_tag
    if merged_dir:
        resolved_merged = Path(merged_dir)
    elif is_moe or adapter_name:
        resolved_merged = repo_root / "outputs" / f"merged-bf16-{model_tag}"
    else:
        resolved_merged = repo_root / "outputs" / "merged-bf16"
    resolved_gguf = Path(gguf_dir) if gguf_dir else (repo_root / "outputs" / "gguf")
    resolved_adapter = Path(adapter_dir) if adapter_dir else _resolve_adapter_dir(repo_root, tier, adapter_name)
    return {
        "is_moe": is_moe,
        "model_tag": model_tag,
        "adapter_dir": resolved_adapter,
        "merged_dir": resolved_merged,
        "gguf_dir": resolved_gguf,
        "lora_gguf_out": resolved_gguf / f"lora-{model_tag}.gguf",
        "q8_extra_gpu_args": ["-ncmoe", "16"] if is_moe else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 Multi-Engine Exporter (llama.cpp GGUF/LoRA, Ollama, vLLM).")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--gguf-dir", default=None)
    parser.add_argument("--imatrix-chunks", type=int, default=8)
    parser.add_argument("--keep-bf16-gguf", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    console, t_all = Console(), time.time()
    paths = resolve_export_paths(
        REPO_ROOT, args.model_id, args.adapter_name, args.merged_dir, args.gguf_dir, args.tier, args.adapter_dir
    )
    is_moe: bool = paths["is_moe"]
    model_tag: str = paths["model_tag"]
    adapter_dir: Path = paths["adapter_dir"]
    merged_dir: Path = paths["merged_dir"]
    gguf_dir: Path = paths["gguf_dir"]
    q8_extra_gpu_args: list[str] = paths["q8_extra_gpu_args"]
    gguf_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else resolve_processed_dir(
        args.tier, domain=args.adapter_name or "all"
    )
    if not data_dir.exists():
        data_dir = REPO_ROOT / "data" / "processed" / args.tier

    console.rule(f"[bold cyan]Phase 5 Multi-Engine Export — {model_tag} ({adapter_dir.parent.name})")
    merge_stats, base_snapshot = merge_lora_to_bf16(args.model_id, adapter_dir, merged_dir, console, force=args.force)
    fresh_merge = bool(merge_stats.get("merge_elapsed_sec", 0.0) > 0.0)

    lora_gguf_out: Path = paths["lora_gguf_out"]
    lora_gguf = export_lora_gguf(adapter_dir, lora_gguf_out, base_dir=base_snapshot)

    bf16_gguf = gguf_dir / f"{model_tag}-bf16.gguf"
    q8_0_gguf = gguf_dir / f"{model_tag}-Q8_0.gguf"
    q4_km_gguf = gguf_dir / f"{model_tag}-Q4_K_M-imatrix.gguf"
    imatrix_txt = gguf_dir / f"imatrix_calibration_{args.adapter_name or 'all'}.txt"
    imatrix_file = gguf_dir / f"{model_tag}.imatrix"

    if args.force or fresh_merge or not bf16_gguf.exists() or bf16_gguf.stat().st_size < 10_000_000_000:
        conv_sec, _ = _run_cmd(
            [sys.executable, str(LLAMA_CPP_CONVERT_SCRIPT), "--outtype", "bf16", "--outfile", str(bf16_gguf), str(merged_dir)],
            "2/5 Converting merged HF bfloat16 -> GGUF BF16",
            console,
        )
    else:
        conv_sec = 0.0
        console.print(f"[bold green]2/5 Reusing existing GGUF BF16 intermediate:[/bold green] {bf16_gguf}")
    bf16_gguf_bytes = bf16_gguf.stat().st_size

    if args.force or fresh_merge or not q8_0_gguf.exists() or q8_0_gguf.stat().st_size < 5_000_000_000:
        q8_sec, _ = _run_cmd(
            [str(LLAMA_QUANTIZE_BIN), str(bf16_gguf), str(q8_0_gguf), "Q8_0", "16"],
            "3/5 Quantizing GGUF BF16 -> Q8_0 (16 threads)",
            console,
        )
    else:
        q8_sec = 0.0
        console.print(f"[bold green]3/5 Reusing existing Q8_0 GGUF:[/bold green] {q8_0_gguf}")

    cal_stats = build_imatrix_calibration_file(args.tier, imatrix_txt, max_samples=36, data_dir=data_dir)
    imat_sec, _ = _run_cmd(
        [
            str(LLAMA_IMATRIX_BIN), "-m", str(q8_0_gguf), "-f", str(imatrix_txt), "-o", str(imatrix_file),
            "-c", "512", "--chunks", str(args.imatrix_chunks), "-ngl", "99", *q8_extra_gpu_args,
            "--split-mode", "layer", "--tensor-split", DEFAULT_TENSOR_SPLIT, "--no-ppl",
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

    modelfile_path = write_ollama_modelfile(gguf_dir / f"Modelfile.{model_tag}", q4_km_gguf.name)
    write_ollama_modelfile(gguf_dir / "Modelfile", q4_km_gguf.name)
    vllm_script = write_vllm_serve_script(
        gguf_dir / f"serve_vllm_{model_tag}.sh", merged_dir, model_tag.lower(), is_moe=is_moe
    )
    write_vllm_serve_script(gguf_dir / "serve_vllm.sh", merged_dir, model_tag.lower(), is_moe=is_moe)

    console.print("[bold green]5/5 Benchmarking & verifying exported Q8_0 and Q4_K_M-imatrix GGUF models...[/bold green]")
    q8_verify = benchmark_and_verify_gguf(q8_0_gguf, "Q8_0", console, extra_gpu_args=q8_extra_gpu_args)
    q4_verify = benchmark_and_verify_gguf(q4_km_gguf, "Q4_K_M-imatrix", console)

    manifest = {
        "tier": args.tier,
        "model_tag": model_tag,
        "total_elapsed_sec": round(time.time() - t_all, 2),
        "adapter_merged": str(adapter_dir),
        "lora_gguf": str(lora_gguf) if lora_gguf else None,
        "ollama_modelfile": str(modelfile_path),
        "vllm_serve_script": str(vllm_script),
        "merge_bf16": merge_stats,
        "bf16_gguf_intermediate_bytes": bf16_gguf_bytes,
        "bf16_gguf_convert_elapsed_sec": conv_sec,
        "calibration": cal_stats,
        "imatrix": {"imatrix_path": str(imatrix_file), "chunks": args.imatrix_chunks, "elapsed_sec": imat_sec},
        "exports": {
            "Q8_0": {**q8_verify, "quantize_elapsed_sec": q8_sec},
            "Q4_K_M_imatrix": {**q4_verify, "quantize_elapsed_sec": q4_sec},
        },
    }
    manifest_payload = json.dumps(manifest, indent=2)
    manifest_path = gguf_dir / "export_manifest.json"
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    (gguf_dir / f"export_manifest_{model_tag}.json").write_text(manifest_payload, encoding="utf-8")
    console.print(f"[bold green]Saved multi-engine export manifest:[/bold green] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
