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
        self._gate_up_states: list[Any] = []
        self._down_states: list[Any] = []
        self.quantized_param_count = int(gate_up_weight.numel() + down_weight.numel())
        dev = _resolve_target_device(target_device)

        for idx in range(self.num_experts):
            gu_slice = gate_up_weight[idx].detach()
            if dev is not None and gu_slice.device != dev:
                gu_slice = gu_slice.to(device=dev, dtype=compute_dtype).contiguous()
            else:
                gu_slice = gu_slice.to(dtype=compute_dtype).contiguous()
            gu_q, gu_state = F_bnb.quantize_4bit(
                gu_slice, quant_type=quant_type, compress_statistics=compress_statistics
            )
            self.register_buffer(f"gate_up_q_{idx}", gu_q)
            self._gate_up_states.append(gu_state)
            del gu_slice

            dn_slice = down_weight[idx].detach()
            if dev is not None and dn_slice.device != dev:
                dn_slice = dn_slice.to(device=dev, dtype=compute_dtype).contiguous()
            else:
                dn_slice = dn_slice.to(dtype=compute_dtype).contiguous()
            dn_q, dn_state = F_bnb.quantize_4bit(
                dn_slice, quant_type=quant_type, compress_statistics=compress_statistics
            )
            self.register_buffer(f"down_q_{idx}", dn_q)
            self._down_states.append(dn_state)
            del dn_slice

    @classmethod
    def from_float_experts(
        cls,
        experts_mod: nn.Module,
        target_device: torch.device | str | int | None = None,
        quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> "QuantizedGemma4Experts4bit":
        """Quantize a `Gemma4TextExperts` module slice-by-slice onto target_device."""
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

    def _sync_state_device(self, idx: int, target_device: torch.device) -> None:
        with torch.no_grad():
            gu_state = self._gate_up_states[idx]
            if getattr(gu_state.absmax, "device", None) != target_device:
                gu_state.to(target_device)
            dn_state = self._down_states[idx]
            if getattr(dn_state.absmax, "device", None) != target_device:
                dn_state.to(target_device)

    def _apply(self, fn):  # type: ignore[override]
        out = super()._apply(fn)
        for idx in range(self.num_experts):
            buf = getattr(self, f"gate_up_q_{idx}")
            self._sync_state_device(idx, buf.device)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for entry in expert_hit:
            idx = int(entry[0].item())
            if idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[idx])
            current_state = hidden_states[token_idx]

            gu_q = getattr(self, f"gate_up_q_{idx}")
            dn_q = getattr(self, f"down_q_{idx}")
            self._sync_state_device(idx, gu_q.device)

            w_gu = F_bnb.dequantize_4bit(gu_q, self._gate_up_states[idx]).to(dtype=current_state.dtype)
            gate, up = F.linear(current_state, w_gu).chunk(2, dim=-1)
            del w_gu
            act = self.act_fn(gate) * up

            w_dn = F_bnb.dequantize_4bit(dn_q, self._down_states[idx]).to(dtype=current_state.dtype)
            routed_out = F.linear(act, w_dn)
            del w_dn

            routed_out = routed_out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, routed_out.to(final_hidden_states.dtype))

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
