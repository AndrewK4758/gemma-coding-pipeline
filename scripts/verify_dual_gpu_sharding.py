#!/usr/bin/env python3
"""Verify dual-GPU CUDA environment, 42-layer device_map, and cross-GPU LoRA + 8-bit PagedAdamW."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Keep CUDA context and cuBLAS workspace footprint minimal (<2 MiB) so verification
# runs cleanly alongside any local inference server without disturbing loaded models.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

GEMMA2_9B_NUM_LAYERS = 42
PRIMARY_GPU_MAX_LAYER = 26  # Layers 0..26 (27 layers) on cuda:1 (16 GB)


def build_gemma2_9b_device_map() -> dict[str, int]:
    """Construct static 42-layer device_map across cuda:1 (16 GB) and cuda:0 (12 GB)."""
    device_map: dict[str, int] = {"model.embed_tokens": 0}
    for layer_idx in range(GEMMA2_9B_NUM_LAYERS):
        device_map[f"model.layers.{layer_idx}"] = 1 if layer_idx <= PRIMARY_GPU_MAX_LAYER else 0
    device_map["model.norm"] = 0
    device_map["lm_head"] = 0
    return device_map


class DualGpuShardedToyModel(nn.Module):
    """Minimal 2-stage bfloat16 model spanning cuda:1 -> cuda:0 for non-disruptive PCIe + LoRA verification."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.stage_cuda1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        ).to(device="cuda:1", dtype=torch.bfloat16)
        self.stage_cuda0 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        ).to(device="cuda:0", dtype=torch.bfloat16)

    def forward(self, x_cuda1: torch.Tensor) -> torch.Tensor:
        h_cuda1 = self.stage_cuda1(x_cuda1)
        h_cuda0 = h_cuda1.to("cuda:0", non_blocking=False)
        return self.stage_cuda0(h_cuda0)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    tool_versions_path = repo_root / ".tool-versions"
    pinned_py = tool_versions_path.read_text(encoding="utf-8").strip() if tool_versions_path.exists() else "missing"
    runtime_py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    print("=== 1. Python & Package Environment ===")
    print(f".tool-versions : {pinned_py}")
    print(f"Runtime Python : {runtime_py} ({sys.executable})")
    print(f"PyTorch        : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"bitsandbytes   : {bnb.__version__}")
    arch_list = torch.cuda.get_arch_list()
    print(f"CUDA Arch List : {arch_list}")

    print("\n=== 2. Dual-GPU CUDA Hardware Verification ===")
    device_count = torch.cuda.device_count()
    assert device_count >= 2, f"Expected >=2 GPUs with CUDA_VISIBLE_DEVICES=0,1, found {device_count}"
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        bf16_ok = torch.cuda.is_bf16_supported()
        p2p_other = torch.cuda.can_device_access_peer(idx, 1 - idx) if idx < 2 else False
        print(
            f"cuda:{idx} -> {props.name} | capability={cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]}) | "
            f"bf16={bf16_ok} | p2p_to_cuda:{1 - idx}={p2p_other} | "
            f"VRAM free={free_b / (1024**2):.1f} MiB / total={total_b / (1024**2):.1f} MiB"
        )
        assert cap[0] >= 8, f"Expected Ampere+ compute capability >=8.0 on cuda:{idx}, got {cap}"
        assert bf16_ok, f"bfloat16 not supported on cuda:{idx}"

    print("\n=== 3. Static 42-Layer Device Map (bfloat16) ===")
    device_map = build_gemma2_9b_device_map()
    cuda1_keys = [k for k, v in device_map.items() if v == 1]
    cuda0_keys = [k for k, v in device_map.items() if v == 0]
    print(f"cuda:1 (Primary 16 GB GPU)   : {len(cuda1_keys)} modules (model.layers.0 .. model.layers.26)")
    print(f"cuda:0 (Secondary 12 GB GPU) : {len(cuda0_keys)} modules (embed_tokens, model.layers.27 .. 41, norm, lm_head)")
    print(json.dumps(device_map, indent=2))

    print("\n=== 4. Synthetic 2-Device bfloat16 Cross-GPU Activation + LoRA + PagedAdamW8bit Verification ===")
    torch.manual_seed(42)
    model = DualGpuShardedToyModel(hidden_size=256)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["0", "2"],
        lora_dropout=0.0,
        bias="none",
    )
    model = inject_adapter_in_model(lora_cfg, model)

    # Ensure LoRA parameters match the bfloat16 dtype of their parent modules on cuda:1 and cuda:0
    trainable_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch.bfloat16)
            trainable_params.append(param)

    pre_step_snapshot = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=1e-3)

    x = torch.randn(4, 32, 256, device="cuda:1", dtype=torch.bfloat16)
    target = torch.randn(4, 32, 256, device="cuda:0", dtype=torch.bfloat16)

    # Run 2 optimization steps: step 1 moves zero-initialized lora_B off 0; step 2 updates both lora_A and lora_B
    loss = torch.tensor(0.0, device="cuda:0")
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out_cuda0 = model(x)
        assert out_cuda0.device == torch.device("cuda:0") and out_cuda0.dtype == torch.bfloat16
        loss = nn.functional.mse_loss(out_cuda0, target)
        loss.backward()
        optimizer.step()

    grad_norms = {
        name: float(p.grad.detach().float().norm().cpu())
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    cuda1_grads = [v for k, v in grad_norms.items() if "stage_cuda1" in k and v > 0.0]
    cuda0_grads = [v for k, v in grad_norms.items() if "stage_cuda0" in k and v > 0.0]
    assert len(cuda1_grads) == 4 and len(cuda0_grads) == 4, f"Missing non-zero gradients across devices: {grad_norms}"

    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")

    deltas = {
        name: float((p.detach() - pre_step_snapshot[name]).float().norm().cpu())
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    assert all(d > 0.0 for d in deltas.values()), f"Expected all LoRA weights to update, got deltas={deltas}"

    print(f"Forward/Backward Loss (cuda:1 -> cuda:0) : {float(loss.detach().cpu()):.6f}")
    print(f"Non-zero LoRA grad tensors               : cuda:1={len(cuda1_grads)}, cuda:0={len(cuda0_grads)}")
    print(f"PagedAdamW8bit max parameter update norm : {max(deltas.values()):.6f}")
    print("STATUS: PASS (Dual-GPU bfloat16 sharding + cross-PCIe autograd + PagedAdamW8bit verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
