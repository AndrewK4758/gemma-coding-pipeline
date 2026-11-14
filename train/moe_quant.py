"""4-bit NF4 double-quantized expert wrapper for Gemma 4 26B-A4B MoE layers.

`bitsandbytes`' built-in `Bnb4BitHfQuantizer` skips 3D `nn.Parameter` tensors
stored inside `Gemma4TextExperts` (`gate_up_proj` and `down_proj`, ~22.8B params).
This module quantizes each routed expert slice on demand to 4-bit NF4 packed
buffers and dequantizes only the `expert_hit` experts active per forward/backward pass.
"""

from __future__ import annotations

import gc
from typing import Any

import bitsandbytes.functional as F_bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_target_device(target_device: torch.device | str | int | None) -> torch.device | None:
    if target_device is None:
        return None
    if isinstance(target_device, int):
        return torch.device(f"cuda:{target_device}")
    return torch.device(target_device)


def _extract_expert_tensor(experts_mod: nn.Module, attr_name: str) -> torch.Tensor:
    """Return concrete expert weights, resolving Accelerate CPU-offloaded `weights_map` when on meta."""
    tensor = getattr(experts_mod, attr_name).data
    if tensor.device.type != "meta":
        return tensor
    hook = getattr(experts_mod, "_hf_hook", None)
    hooks = getattr(hook, "hooks", [hook]) if hook is not None else []
    for h in hooks:
        wmap = getattr(h, "weights_map", None)
        if wmap is not None and attr_name in wmap:
            return wmap[attr_name]
    raise RuntimeError(
        f"Expert parameter {attr_name!r} is on meta device without a backing accelerate weights_map."
    )


def _release_offload_hooks(experts_mod: nn.Module) -> None:
    """Release CPU-offloaded state tensors and Accelerate hooks once an expert layer is quantized."""
    hook = getattr(experts_mod, "_hf_hook", None)
    hooks = getattr(hook, "hooks", [hook]) if hook is not None else []
    for h in hooks:
        wmap = getattr(h, "weights_map", None)
        if isinstance(wmap, dict):
            wmap.pop("gate_up_proj", None)
            wmap.pop("down_proj", None)
    if hook is not None:
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(experts_mod)
        except Exception:
            pass


def _grouped_experts_forward_core(
    hidden_states: torch.Tensor,
    top_k_weights: torch.Tensor,
    order: torch.Tensor,
    tok_order: torch.Tensor,
    sorted_valid: torch.Tensor,
    offs: torch.Tensor,
    gate_up_q: torch.Tensor,
    gate_up_state: Any,
    down_q: torch.Tensor,
    down_state: Any,
    act_fn: nn.Module,
    compute_dtype: torch.dtype,
    return_gu_out: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run NaN-guarded CUDA grouped-GEMM forward over frozen 4-bit expert buffers."""
    sorted_in = hidden_states[tok_order].to(dtype=compute_dtype).contiguous()
    w_gu = F_bnb.dequantize_4bit(gate_up_q, gate_up_state).to(dtype=compute_dtype)
    gu_out = torch._grouped_mm(sorted_in, w_gu.transpose(-2, -1), offs=offs)
    del sorted_in, w_gu
    gu_out = torch.where(sorted_valid[:, None], gu_out, gu_out.new_zeros(()))

    gate, up = gu_out.chunk(2, dim=-1)
    act = (act_fn(gate) * up).contiguous()
    if not return_gu_out:
        del gu_out, gate, up
        saved_gu = None
    else:
        saved_gu = gu_out

    w_dn = F_bnb.dequantize_4bit(down_q, down_state).to(dtype=compute_dtype)
    routed_out = torch._grouped_mm(act, w_dn.transpose(-2, -1), offs=offs)
    del act, w_dn

    sorted_weights = torch.where(
        sorted_valid,
        top_k_weights.reshape(-1)[order].to(dtype=routed_out.dtype),
        routed_out.new_zeros(()),
    )
    weighted = torch.where(
        sorted_valid[:, None],
        routed_out * sorted_weights[:, None],
        routed_out.new_zeros(()),
    )
    del routed_out

    final_hidden_states = torch.zeros_like(hidden_states)
    final_hidden_states.index_add_(0, tok_order, weighted.to(dtype=final_hidden_states.dtype))
    return final_hidden_states, saved_gu


class _GroupedExpertsFn(torch.autograd.Function):
    """Exact CUDA grouped-GEMM forward/backward over frozen 4-bit quantized expert tensors.

    Avoids saving full dequantized ``(E, 2I, H)`` / ``(E, H, I)`` weights or
    recomputing ``gate_up`` GEMM during backward by saving only ``gu_out``
    (``(M, 2I)``, <90 MiB at 4096 tokens) and freeing ``w_dn`` before
    dequantizing ``w_gu``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        order: torch.Tensor,
        tok_order: torch.Tensor,
        sorted_valid: torch.Tensor,
        offs: torch.Tensor,
        gate_up_q: torch.Tensor,
        gate_up_state: Any,
        down_q: torch.Tensor,
        down_state: Any,
        act_fn: nn.Module,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        final_hidden_states, gu_out = _grouped_experts_forward_core(
            hidden_states,
            top_k_weights,
            order,
            tok_order,
            sorted_valid,
            offs,
            gate_up_q,
            gate_up_state,
            down_q,
            down_state,
            act_fn,
            compute_dtype,
            return_gu_out=True,
        )
        ctx.save_for_backward(gu_out, top_k_weights, order, tok_order, sorted_valid, offs)
        ctx.gate_up_q = gate_up_q
        ctx.gate_up_state = gate_up_state
        ctx.down_q = down_q
        ctx.down_state = down_state
        ctx.act_fn = act_fn
        ctx.compute_dtype = compute_dtype
        return final_hidden_states

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        grad_final = grad_outputs[0]
        gu_out, top_k_weights, order, tok_order, sorted_valid, offs = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype

        grad_weighted = grad_final[tok_order].to(dtype=compute_dtype)
        gate, up = gu_out.chunk(2, dim=-1)
        with torch.enable_grad():
            gate_v = gate.detach().requires_grad_(True)
            up_v = up.detach().requires_grad_(True)
            act_v = ctx.act_fn(gate_v) * up_v

        w_dn = F_bnb.dequantize_4bit(ctx.down_q, ctx.down_state).to(dtype=compute_dtype)

        grad_top_k_weights: torch.Tensor | None = None
        if ctx.needs_input_grad[1]:
            with torch.no_grad():
                routed_out = torch._grouped_mm(act_v.detach().contiguous(), w_dn.transpose(-2, -1), offs=offs)
                grad_sorted_w = (grad_weighted * routed_out).sum(dim=-1)
                grad_sorted_w = torch.where(sorted_valid, grad_sorted_w, grad_sorted_w.new_zeros(()))
                grad_flat_w = torch.zeros(
                    top_k_weights.numel(), device=top_k_weights.device, dtype=top_k_weights.dtype
                )
                grad_flat_w.scatter_(0, order, grad_sorted_w.to(dtype=top_k_weights.dtype))
                grad_top_k_weights = grad_flat_w.view_as(top_k_weights)
                del routed_out

        grad_hidden: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            sorted_w = torch.where(
                sorted_valid,
                top_k_weights.reshape(-1)[order].to(dtype=compute_dtype),
                grad_weighted.new_zeros(()),
            )
            grad_routed = (grad_weighted * sorted_w[:, None]).contiguous()
            grad_act = torch._grouped_mm(grad_routed, w_dn, offs=offs)
            del w_dn, grad_routed, grad_weighted

            grad_gate, grad_up = torch.autograd.grad(act_v, (gate_v, up_v), grad_outputs=grad_act)
            grad_gu = torch.cat([grad_gate, grad_up], dim=-1).contiguous()
            del act_v, gate_v, up_v, grad_gate, grad_up, grad_act

            w_gu = F_bnb.dequantize_4bit(ctx.gate_up_q, ctx.gate_up_state).to(dtype=compute_dtype)
            grad_sorted_in = torch._grouped_mm(grad_gu, w_gu, offs=offs)
            del w_gu, grad_gu
            grad_sorted_in = torch.where(sorted_valid[:, None], grad_sorted_in, grad_sorted_in.new_zeros(()))
            grad_hidden = torch.zeros_like(grad_final)
            grad_hidden.index_add_(0, tok_order, grad_sorted_in.to(dtype=grad_final.dtype))
        else:
            del w_dn, act_v, gate_v, up_v, grad_weighted

        return (grad_hidden, grad_top_k_weights) + (None,) * 10


class QuantizedGemma4Experts4bit(nn.Module):
    """Memory-frugal 4-bit NF4 drop-in replacement for `Gemma4TextExperts`."""

    def __init__(
        self,
        num_experts: int,
        act_fn: nn.Module,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        target_device: torch.device | str | int | None = None,
        quant_type: str = "nf4",
        compress_statistics: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.act_fn = act_fn
        self.quant_type = quant_type
        self.compress_statistics = compress_statistics
        self.compute_dtype = compute_dtype
        self.quantized_param_count = int(gate_up_weight.numel() + down_weight.numel())
        dev = _resolve_target_device(target_device)

        gu_src = gate_up_weight.detach()
        if dev is not None and gu_src.device != dev:
            gu_src = gu_src.to(device=dev, dtype=compute_dtype).contiguous()
        else:
            gu_src = gu_src.to(dtype=compute_dtype).contiguous()
        gu_q, gu_state = F_bnb.quantize_4bit(
            gu_src, quant_type=quant_type, compress_statistics=compress_statistics
        )
        self.register_buffer("gate_up_q", gu_q)
        self._gate_up_state = gu_state
        del gu_src

        dn_src = down_weight.detach()
        if dev is not None and dn_src.device != dev:
            dn_src = dn_src.to(device=dev, dtype=compute_dtype).contiguous()
        else:
            dn_src = dn_src.to(dtype=compute_dtype).contiguous()
        dn_q, dn_state = F_bnb.quantize_4bit(
            dn_src, quant_type=quant_type, compress_statistics=compress_statistics
        )
        self.register_buffer("down_q", dn_q)
        self._down_state = dn_state
        del dn_src

    @property
    def gate_up_q_0(self) -> torch.Tensor:
        return self.gate_up_q

    @property
    def down_q_0(self) -> torch.Tensor:
        return self.down_q

    @classmethod
    def from_float_experts(
        cls,
        experts_mod: nn.Module,
        target_device: torch.device | str | int | None = None,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "QuantizedGemma4Experts4bit":
        """Quantize a `Gemma4TextExperts` module onto target_device in stacked 4-bit NF4 format."""
        gate_up = _extract_expert_tensor(experts_mod, "gate_up_proj")
        down = _extract_expert_tensor(experts_mod, "down_proj")
        return cls(
            num_experts=experts_mod.num_experts,
            act_fn=experts_mod.act_fn,
            gate_up_weight=gate_up,
            down_weight=down,
            target_device=target_device,
            quant_type=quant_type,
            compress_statistics=True,
            compute_dtype=compute_dtype,
        )

    def _sync_state_device(self, target_device: torch.device) -> None:
        with torch.no_grad():
            if getattr(self._gate_up_state.absmax, "device", None) != target_device:
                self._gate_up_state.to(target_device)
            if getattr(self._down_state.absmax, "device", None) != target_device:
                self._down_state.to(target_device)

    def _apply(self, fn):  # type: ignore[override]
        out = super()._apply(fn)
        self._sync_state_device(self.gate_up_q.device)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.numel() == 0 or top_k_index.numel() == 0:
            return torch.zeros_like(hidden_states)

        self._sync_state_device(self.gate_up_q.device)
        T = hidden_states.size(0)
        K = top_k_index.size(1)

        with torch.no_grad():
            flat_experts = top_k_index.reshape(-1)
            valid_mask = (flat_experts >= 0) & (flat_experts < self.num_experts)
            clamped = torch.where(valid_mask, flat_experts, self.num_experts)
            order = torch.argsort(clamped)
            counts = torch.bincount(clamped, minlength=self.num_experts + 1)[: self.num_experts]
            sorted_valid = valid_mask[order]
            slot_tok = torch.arange(T, device=hidden_states.device).unsqueeze(1).expand(T, K).reshape(-1)
            tok_order = slot_tok[order]

        use_grouped = (
            hidden_states.device.type == "cuda"
            and hasattr(torch, "_grouped_mm")
            and self.compute_dtype in (torch.bfloat16, torch.float16)
        )
        if use_grouped:
            offs = counts.cumsum(0, dtype=torch.int32)
            if torch.is_grad_enabled() and (hidden_states.requires_grad or top_k_weights.requires_grad):
                return _GroupedExpertsFn.apply(
                    hidden_states,
                    top_k_weights,
                    order,
                    tok_order,
                    sorted_valid,
                    offs,
                    self.gate_up_q,
                    self._gate_up_state,
                    self.down_q,
                    self._down_state,
                    self.act_fn,
                    self.compute_dtype,
                )
            with torch.no_grad():
                final_hidden_states, _ = _grouped_experts_forward_core(
                    hidden_states,
                    top_k_weights,
                    order,
                    tok_order,
                    sorted_valid,
                    offs,
                    self.gate_up_q,
                    self._gate_up_state,
                    self.down_q,
                    self._down_state,
                    self.act_fn,
                    self.compute_dtype,
                    return_gu_out=False,
                )
                return final_hidden_states

        # CPU / fallback path: single dequantize per tensor + contiguous slice loop
        w_gu = F_bnb.dequantize_4bit(self.gate_up_q, self._gate_up_state).to(dtype=self.compute_dtype)
        w_dn = F_bnb.dequantize_4bit(self.down_q, self._down_state).to(dtype=self.compute_dtype)
        sorted_in = hidden_states[tok_order].to(dtype=self.compute_dtype)
        sorted_weights = torch.where(
            sorted_valid,
            top_k_weights.reshape(-1)[order].to(dtype=self.compute_dtype),
            sorted_in.new_zeros(()),
        )
        final_hidden_states = torch.zeros_like(hidden_states)
        counts_list = counts.tolist()
        start = 0
        for idx, count in enumerate(counts_list):
            if count == 0:
                continue
            end = start + count
            gate, up = F.linear(sorted_in[start:end], w_gu[idx]).chunk(2, dim=-1)
            routed_out = F.linear(self.act_fn(gate) * up, w_dn[idx]) * sorted_weights[start:end, None]
            final_hidden_states.index_add_(0, tok_order[start:end], routed_out.to(dtype=final_hidden_states.dtype))
            start = end
        return final_hidden_states


def quantize_gemma4_moe_experts_in_place(
    model: nn.Module,
    layer_device_map: dict[int, int | str] | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Replace 3D `Gemma4TextExperts` modules layer-by-layer with 4-bit NF4 wrappers."""
    lang_model = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
    layers = getattr(lang_model, "layers", [])
    replaced = 0
    for idx, layer in enumerate(layers):
        experts = getattr(layer, "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
            continue
        target_dev = layer_device_map.get(idx) if layer_device_map is not None else None
        q_experts = QuantizedGemma4Experts4bit.from_float_experts(
            experts, target_device=target_dev, compute_dtype=compute_dtype
        )
        layer.experts = q_experts
        _release_offload_hooks(experts)
        if target_dev is not None and isinstance(getattr(model, "hf_device_map", None), dict):
            model.hf_device_map[f"model.language_model.layers.{idx}.experts"] = target_dev
        del experts
        replaced += 1
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    return replaced


def finalize_disk_offloaded_moe_layers(
    model: nn.Module,
    layer_device_map: dict[int, int | str],
    prefix: str = "model.language_model.layers",
) -> None:
    """Install entry device hooks on decoder layers and compact `hf_device_map` after disk-expert replacement."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    lang_model = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
    layers = getattr(lang_model, "layers", [])
    hmap = getattr(model, "hf_device_map", None)
    for idx in range(len(layers)):
        target_dev = layer_device_map.get(idx)
        if target_dev is None:
            continue
        if isinstance(target_dev, int) and not torch.cuda.is_available():
            target_dev = "cpu"
        if not hasattr(layers[idx], "_hf_hook"):
            add_hook_to_module(layers[idx], AlignDevicesHook(execution_device=target_dev, place_submodules=False))
        if isinstance(hmap, dict):
            layer_prefix = f"{prefix}.{idx}"
            for key in [k for k in hmap if k.startswith(f"{layer_prefix}.")]:
                del hmap[key]
            hmap[layer_prefix] = target_dev
