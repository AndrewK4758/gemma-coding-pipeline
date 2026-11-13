#!/usr/bin/env python3
"""Dual-GPU (`cuda:1` 16 GB + `cuda:0` 12 GB) bfloat16 & 4-bit QLoRA model loader.

Supports both:
- `google/gemma-4-E4B-it` (default: 8.0B total parameters, 42 layers, Per-Layer Embeddings + KV sharing)
- `google/gemma-2-9b-it` (with automatic fallback to `unsloth/gemma-2-9b-it` if gated without HF_TOKEN)
"""

from __future__ import annotations

import os
import types
from typing import Any

# Ensure lazy CUDA module loading and dual-GPU visibility before importing torch/bitsandbytes
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from train.moe_quant import quantize_gemma4_moe_experts_in_place

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
GEMMA2_GATED_ID = "google/gemma-2-9b-it"
GEMMA2_MIRROR_ID = "unsloth/gemma-2-9b-it"

GEMMA_NUM_LAYERS = 42
PRIMARY_GPU_MAX_LAYER = 26  # Layers 0..26 (27 layers) on cuda:1 (16 GB); Layers 27..41 (15 layers) on cuda:0

LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
MULTIMODAL_EXCLUDE_REGEX = r".*(vision_tower|audio_tower|embed_vision|embed_audio).*"


def resolve_checkpoint_id(requested_model_id: str = DEFAULT_MODEL_ID) -> str:
    """Resolve model checkpoint ID, falling back to ungated mirror if google/gemma-2-9b-it returns 401/403."""
    candidates = [requested_model_id]
    if requested_model_id == GEMMA2_GATED_ID:
        candidates.append(GEMMA2_MIRROR_ID)

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            AutoConfig.from_pretrained(candidate)
            return candidate
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to resolve checkpoint {requested_model_id!r}: {last_err}") from last_err


def build_asymmetric_device_map(
    model_type: str = "gemma4",
    num_layers: int = GEMMA_NUM_LAYERS,
    primary_gpu_max_layer: int = PRIMARY_GPU_MAX_LAYER,
    offload_multimodal: bool = False,
    offload_experts_to_cpu: bool = False,
) -> dict[str, int | str]:
    """Construct static asymmetric device_map across cuda:1 (16 GB) and cuda:0 (12 GB)."""
    if model_type == "gemma4":
        mm_dev: int | str = "cpu" if offload_multimodal else 1
        device_map: dict[str, int | str] = {
            "model.vision_tower": mm_dev,
            "model.audio_tower": mm_dev,
            "model.embed_vision": mm_dev,
            "model.embed_audio": mm_dev,
            "model.language_model.embed_tokens": 0,
            "model.language_model.embed_tokens_per_layer": 1,
            "model.language_model.per_layer_model_projection": 1,
            "model.language_model.per_layer_projection_norm": 1,
        }
        for layer_idx in range(num_layers):
            target_gpu = 1 if layer_idx <= primary_gpu_max_layer else 0
            device_map[f"model.language_model.layers.{layer_idx}"] = target_gpu
            if offload_experts_to_cpu:
                device_map[f"model.language_model.layers.{layer_idx}.experts"] = "cpu"
        device_map["model.language_model.norm"] = 0
        device_map["model.language_model.rotary_emb"] = 0
        device_map["lm_head"] = 0
        return device_map

    device_map = {"model.embed_tokens": 0}
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = 1 if layer_idx <= primary_gpu_max_layer else 0
    device_map["model.norm"] = 0
    device_map["model.rotary_emb"] = 0
    device_map["lm_head"] = 0
    return device_map


def install_chunked_causal_lm_loss(base_model: PreTrainedModel, chunk_size: int = 256) -> None:
    """Replace full-sequence 262k-vocab logit materialization (~6.8 GiB) with 256-token chunked cross-entropy (~250 MiB)."""
    orig_forward = base_model.forward
    text_cfg = base_model.config.get_text_config() if hasattr(base_model.config, "get_text_config") else base_model.config
    softcap: float | None = getattr(text_cfg, "final_logit_softcapping", None)

    def _chunk_nll(lm_head: nn.Module, h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        logits_chunk = lm_head(h_chunk)
        if softcap is not None:
            logits_chunk = torch.tanh(logits_chunk / softcap) * softcap
        return F.cross_entropy(logits_chunk.float(), y_chunk, reduction="sum")

    def _chunked_forward(
        self: PreTrainedModel,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        num_items_in_batch: int | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if labels is None:
            return orig_forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)

        backbone_kwargs = {k: v for k, v in kwargs.items() if k not in ("logits_to_keep", "return_dict")}
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **backbone_kwargs,
        )
        hidden_states: torch.Tensor = outputs.last_hidden_state
        shift_h = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1))
        shift_y = labels[:, 1:].to(shift_h.device).contiguous().view(-1)
        valid_mask = shift_y != -100
        active_h = shift_h[valid_mask]
        active_y = shift_y[valid_mask]

        if active_y.numel() == 0:
            loss = (hidden_states[:, :1, :].sum() * 0.0).float()
        else:
            total_nll = torch.zeros((), device=active_h.device, dtype=torch.float32)
            for start in range(0, active_h.size(0), chunk_size):
                h_slice = active_h[start : start + chunk_size]
                y_slice = active_y[start : start + chunk_size]
                if torch.is_grad_enabled() and h_slice.requires_grad:
                    chunk_loss = torch.utils.checkpoint.checkpoint(
                        _chunk_nll, self.lm_head, h_slice, y_slice, use_reentrant=False
                    )
                else:
                    chunk_loss = _chunk_nll(self.lm_head, h_slice, y_slice)
                total_nll = total_nll + chunk_loss

            denom = float(num_items_in_batch) if num_items_in_batch is not None else float(active_y.numel())
            loss = total_nll / max(denom, 1.0)

        # Return a lightweight CausalLM output without keeping a 1.72 GiB [B, S, 262144] logits tensor alive
        dummy_logits = hidden_states[:, :1, :1]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=dummy_logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    base_model.forward = types.MethodType(_chunked_forward, base_model)


def query_dual_gpu_vram() -> dict[str, dict[str, float | str]]:
    """Return current allocated, reserved, and total VRAM (in MiB and GiB) for cuda:0 and cuda:1."""
    stats: dict[str, dict[str, float | str]] = {}
    if not torch.cuda.is_available():
        return stats
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        alloc_mib = torch.cuda.memory_allocated(idx) / (1024**2)
        res_mib = torch.cuda.memory_reserved(idx) / (1024**2)
        peak_mib = torch.cuda.max_memory_allocated(idx) / (1024**2)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        used_driver_mib = (total_b - free_b) / (1024**2)
        total_mib = total_b / (1024**2)
        stats[f"cuda:{idx}"] = {
            "name": props.name,
            "allocated_mib": round(alloc_mib, 1),
            "reserved_mib": round(res_mib, 1),
            "peak_allocated_mib": round(peak_mib, 1),
            "driver_used_mib": round(used_driver_mib, 1),
            "total_mib": round(total_mib, 1),
            "allocated_gib": round(alloc_mib / 1024.0, 2),
            "driver_used_gib": round(used_driver_mib / 1024.0, 2),
            "total_gib": round(total_mib / 1024.0, 2),
        }
    return stats


def count_model_parameters(model: nn.Module) -> dict[str, int | float]:
    """Compute trainable, frozen, and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_params": total,
        "trainable_pct": round(pct, 4),
    }


def load_sharded_lora_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    attn_implementation: str = "sdpa",
    adapter_path: str | os.PathLike[str] | None = None,
    load_in_4bit: bool = False,
    console: Console | None = None,
) -> tuple[PeftModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Load base Gemma checkpoint (bfloat16 or QLoRA 4-bit NF4) across cuda:1 + cuda:0 with PEFT LoRA."""
    console = console or Console()
    resolved_id = resolve_checkpoint_id(model_id)
    cfg = AutoConfig.from_pretrained(resolved_id)
    model_type = getattr(cfg, "model_type", "gemma4")
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    num_layers = getattr(text_cfg, "num_hidden_layers", GEMMA_NUM_LAYERS)
    is_moe = bool(getattr(text_cfg, "enable_moe_block", False) or getattr(text_cfg, "num_experts", 0) > 1)
    split_layer = 21 if num_layers <= 30 else PRIMARY_GPU_MAX_LAYER

    device_map = build_asymmetric_device_map(
        model_type=model_type,
        num_layers=num_layers,
        primary_gpu_max_layer=split_layer,
        offload_multimodal=bool(load_in_4bit or is_moe),
        offload_experts_to_cpu=bool(load_in_4bit and is_moe),
    )
    attn_arg: str | dict[str, str] = (
        {"text_config": attn_implementation, "vision_config": attn_implementation, "audio_config": "eager"}
        if model_type == "gemma4" and attn_implementation == "sdpa"
        else attn_implementation
    )
    quant_label = "4-bit NF4 QLoRA" if load_in_4bit else "bfloat16"
    console.print(
        f"[bold cyan]Loading base model:[/bold cyan] [green]{resolved_id}[/green] "
        f"(model_type={model_type}, layers={num_layers}, moe={is_moe}, precision={quant_label}, attn={attn_arg})"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path) if adapter_path else resolved_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "attn_implementation": attn_arg,
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(resolved_id, **load_kwargs)
    if load_in_4bit and is_moe:
        layer_devs = {i: (1 if i <= split_layer else 0) for i in range(num_layers)}
        replaced = quantize_gemma4_moe_experts_in_place(base_model, layer_device_map=layer_devs)
        console.print(f"[cyan]Quantized {replaced} Gemma 4 MoE expert layers to 4-bit NF4 on target GPUs.[/cyan]")

    for param in base_model.parameters():
        param.requires_grad_(False)

    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config"):
        base_model.config.text_config.use_cache = False

    install_chunked_causal_lm_loss(base_model, chunk_size=256)
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    if adapter_path is not None:
        console.print(f"[bold cyan]Loading existing trainable PEFT LoRA adapter:[/bold cyan] [green]{adapter_path}[/green]")
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
        lora_desc = f"pretrained={adapter_path}"
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
            exclude_modules=MULTIMODAL_EXCLUDE_REGEX,
        )
        peft_model = get_peft_model(base_model, lora_config)
        lora_desc = f"r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={LORA_TARGET_MODULES}"

    for param in peft_model.parameters():
        if param.requires_grad and param.dtype != torch.bfloat16:
            param.data = param.data.to(dtype=torch.bfloat16)

    param_stats = count_model_parameters(peft_model)
    vram_stats = query_dual_gpu_vram()

    tbl = Table(title=f"Sharded Model & LoRA Summary ({resolved_id})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", style="green")
    tbl.add_row("Resolved Model ID", f"{resolved_id} ({model_type}, {quant_label})")
    tbl.add_row("Attention Backend", f"{attn_arg} + 256-Token Chunked Vocab Cross-Entropy")
    tbl.add_row("LoRA Config", lora_desc)
    tbl.add_row(
        "Parameters (Trainable / Frozen / Total)",
        f"{param_stats['trainable_params']:,} / {param_stats['frozen_params']:,} / "
        f"{param_stats['total_params']:,} ({param_stats['trainable_pct']}%)",
    )
    for dev_key, dstat in vram_stats.items():
        tbl.add_row(
            f"{dev_key} ({dstat['name']})",
            f"torch_alloc={dstat['allocated_mib']} MiB ({dstat['allocated_gib']} GiB) | "
            f"driver_used={dstat['driver_used_mib']} / {dstat['total_mib']} MiB",
        )
    console.print(tbl)

    load_meta = {
        "requested_model_id": model_id,
        "resolved_model_id": resolved_id,
        "model_type": model_type,
        "is_moe": is_moe,
        "load_in_4bit": load_in_4bit,
        "attn_implementation": attn_arg,
        "device_map": device_map,
        "param_stats": param_stats,
        "post_load_vram": vram_stats,
    }
    return peft_model, tokenizer, load_meta
