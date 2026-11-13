"""Streaming CPU LoRA shard merger and multi-engine (llama.cpp, Ollama, vLLM) export helpers."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software engineer specializing in .NET/C#, React/TypeScript, Go, and C/C++. "
    "Output concise reasoning and complete, strictly typed, compiler-verified code or minimal diffs."
)

LLAMA_CPP_LORA_SCRIPT = Path(
    os.environ.get("LLAMA_CPP_LORA_SCRIPT", str(Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp"))) / "convert_lora_to_gguf.py"))
)


def _match_pattern(pattern_map: dict[str, Any], mod_name: str, default: float) -> float:
    if not pattern_map:
        return default
    if mod_name in pattern_map:
        return float(pattern_map[mod_name])
    for pat, val in pattern_map.items():
        if re.search(pat, mod_name):
            return float(val)
    return default


def _load_lora_deltas(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict[str, Any]]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    default_r = float(cfg.get("r", 32))
    default_alpha = float(cfg.get("lora_alpha", 64))
    use_rslora = bool(cfg.get("use_rslora", False))
    rank_pattern = cfg.get("rank_pattern") or {}
    alpha_pattern = cfg.get("alpha_pattern") or {}

    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Expected PEFT safetensors adapter at {st_path}")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as fh:
        for key in fh.keys():
            if ".lora_A." in key or key.endswith(".lora_A.weight"):
                mod_key = re.sub(r"\.lora_A(?:\.[^.]+)?\.weight$", "", key)
                pairs.setdefault(mod_key, {})["A"] = fh.get_tensor(key)
            elif ".lora_B." in key or key.endswith(".lora_B.weight"):
                mod_key = re.sub(r"\.lora_B(?:\.[^.]+)?\.weight$", "", key)
                pairs.setdefault(mod_key, {})["B"] = fh.get_tensor(key)

    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for mod_key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        clean_mod = re.sub(r"^base_model\.model\.", "", mod_key)
        lora_a, lora_b = ab["A"], ab["B"]
        r_val = float(lora_a.shape[0]) or _match_pattern(rank_pattern, clean_mod, default_r)
        alpha_val = _match_pattern(alpha_pattern, clean_mod, default_alpha)
        scale = alpha_val / math.sqrt(r_val) if use_rslora else alpha_val / r_val
        deltas[f"{clean_mod}.weight"] = (lora_a, lora_b, scale)
    return deltas, cfg


def _copy_non_weight_assets(base_snapshot: Path, adapter_dir: Path, merged_dir: Path) -> None:
    copy_names = (
        "config.json", "generation_config.json", "chat_template.jinja",
        "processor_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    )
    for fname in copy_names:
        src = (adapter_dir / fname) if (adapter_dir / fname).exists() else (base_snapshot / fname)
        if src.exists():
            dst = merged_dir / fname
            shutil.copyfile(src, dst)
            dst.chmod(0o644)


def _lookup_lora_delta(
    tensor_key: str, deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    if tensor_key in deltas:
        return deltas[tensor_key]
    if f"model.{tensor_key}" in deltas:
        return deltas[f"model.{tensor_key}"]
    if tensor_key.startswith("model.") and tensor_key[len("model.") :] in deltas:
        return deltas[tensor_key[len("model.") :]]
    return None


def merge_lora_weights_into_shards(base_snapshot: Path, adapter_dir: Path, merged_dir: Path) -> dict[str, Any]:
    """Stream base safetensors shard-by-shard on CPU (<5 GiB RAM) and fold W + scale*(B@A)."""
    t0 = time.time()
    base_snapshot, adapter_dir, merged_dir = Path(base_snapshot), Path(adapter_dir), Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    for existing in list(merged_dir.iterdir()):
        if existing.is_file():
            existing.chmod(0o644)
            if existing.suffix == ".safetensors" or existing.name == "model.safetensors.index.json":
                existing.unlink()

    deltas, _ = _load_lora_deltas(adapter_dir)
    _copy_non_weight_assets(base_snapshot, adapter_dir, merged_dir)

    index_path = base_snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]
        index_data = None

    total_tensors = 0
    folded_count = 0
    max_delta = 0.0
    total_bytes = 0
    primary_out = merged_dir / shard_names[0]

    for shard_name in shard_names:
        in_shard = base_snapshot / shard_name
        out_shard = merged_dir / shard_name
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(in_shard), framework="pt", device="cpu") as f_in:
            for key in f_in.keys():
                orig_t = f_in.get_tensor(key)
                total_tensors += 1
                hit = _lookup_lora_delta(key, deltas)
                if hit is not None:
                    lora_a, lora_b, scale = hit
                    delta_w = (lora_b.float() @ lora_a.float()) * scale
                    new_t = (orig_t.float() + delta_w).to(dtype=orig_t.dtype).contiguous()
                    max_delta = max(max_delta, float((new_t.float() - orig_t.float()).abs().max().item()))
                    out_tensors[key] = new_t
                    folded_count += 1
                else:
                    out_tensors[key] = orig_t
        save_file(out_tensors, str(out_shard), metadata={"format": "pt"})
        total_bytes += out_shard.stat().st_size
        del out_tensors
        gc.collect()

    if index_data is not None:
        index_data.setdefault("metadata", {})["total_size"] = total_bytes
        (merged_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_data, indent=2), encoding="utf-8"
        )

    return {
        "merged_dir": str(merged_dir),
        "safetensors_path": str(primary_out),
        "safetensors_size_bytes": total_bytes,
        "safetensors_size_gib": round(total_bytes / (1024**3), 3),
        "shard_count": len(shard_names),
        "total_tensors": total_tensors,
        "folded_lora_linear_tensors": folded_count,
        "max_abs_weight_delta": round(max_delta, 6),
        "merge_elapsed_sec": round(time.time() - t0, 2),
    }


def write_ollama_modelfile(
    out_path: Path,
    gguf_filename: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    num_ctx: int = 65536,
) -> Path:
    """Write a native Gemma 4 Ollama Modelfile referencing a quantized GGUF artifact."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"FROM ./{gguf_filename}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_p 0.95\n"
        'PARAMETER stop "<turn|>"\n'
        'PARAMETER stop "<eos>"\n'
        f'SYSTEM """{system_prompt}"""\n'
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path


def write_vllm_serve_script(
    out_path: Path,
    model_dir: Path,
    served_name: str,
    is_moe: bool = False,
    lora_modules: dict[str, Path] | None = None,
) -> Path:
    """Write an executable vLLM serving launcher with native Gemma 4 reasoning & optional LoRAs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tp = 2 if is_moe else 1
    extra_flags = [
        f"--tensor-parallel-size {tp}",
        "--kv-cache-dtype fp8",
        "--max-model-len 65536",
        "--gpu-memory-utilization 0.90",
        "--reasoning-parser gemma4",
        "--enable-auto-tool-choice",
        "--tool-call-parser gemma4",
    ]
    if lora_modules:
        mods = " ".join(f"{name}={path}" for name, path in lora_modules.items())
        extra_flags.extend(["--enable-lora", "--max-lora-rank 64", f"--lora-modules {mods}"])

    body = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'exec vllm serve "{model_dir}" \\\n'
        f'  --served-model-name "{served_name}" \\\n  '
        + " \\\n  ".join(extra_flags)
        + "\n"
    )
    out_path.write_text(body, encoding="utf-8")
    out_path.chmod(0o755)
    return out_path


def export_lora_gguf(adapter_dir: Path, out_gguf: Path, base_dir: Path) -> Path | None:
    """Convert a PEFT LoRA adapter into a standalone hot-swappable GGUF overlay."""
    if not LLAMA_CPP_LORA_SCRIPT.exists():
        return None
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(LLAMA_CPP_LORA_SCRIPT),
        "--base",
        str(base_dir),
        "--outtype",
        "bf16",
        "--outfile",
        str(out_gguf),
        str(adapter_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out_gguf if proc.returncode == 0 and out_gguf.exists() else None
