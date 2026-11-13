"""Regression tests for E4B num_experts=None handling, non-CPU input device resolution, and 26B export isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeVisionFirstGemma(nn.Module):
    """Mimics Gemma 4 parameter order where CPU-offloaded vision_tower precedes language_model."""

    def __init__(self, embed_dev: str = "cpu") -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vision_tower = nn.Linear(4, 4, bias=False)
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(8, 4)
        self._embed_dev = embed_dev

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens


class TestModelLoadingAndExport(unittest.TestCase):
    def test_e4b_num_experts_none_topology_does_not_raise(self) -> None:
        from train.model_loader import resolve_model_topology

        e4b_text_cfg = SimpleNamespace(num_hidden_layers=42, enable_moe_block=False, num_experts=None)
        fake_cfg = SimpleNamespace(model_type="gemma4", get_text_config=lambda: e4b_text_cfg)
        with (
            mock.patch("train.model_loader.resolve_checkpoint_id", return_value="google/gemma-4-E4B-it"),
            mock.patch("train.model_loader.AutoConfig.from_pretrained", return_value=fake_cfg),
        ):
            topo = resolve_model_topology("google/gemma-4-E4B-it", load_in_4bit=False)
        self.assertFalse(topo["is_moe"])
        self.assertEqual(topo["num_layers"], 42)
        self.assertEqual(topo["split_layer"], 26)
        self.assertEqual(topo["device_map"]["model.vision_tower"], 1)

    def test_26b_topology_omits_absent_audio_and_ple_keys(self) -> None:
        from train.model_loader import resolve_model_topology

        moe_text_cfg = SimpleNamespace(
            num_hidden_layers=30, enable_moe_block=True, num_experts=128, hidden_size_per_layer_input=0
        )
        fake_cfg = SimpleNamespace(model_type="gemma4", audio_config=None, get_text_config=lambda: moe_text_cfg)
        with (
            mock.patch("train.model_loader.resolve_checkpoint_id", return_value="google/gemma-4-26B-A4B-it"),
            mock.patch("train.model_loader.AutoConfig.from_pretrained", return_value=fake_cfg),
        ):
            topo = resolve_model_topology("google/gemma-4-26B-A4B-it", load_in_4bit=True)
        self.assertTrue(topo["is_moe"])
        self.assertNotIn("model.audio_tower", topo["device_map"])
        self.assertNotIn("model.language_model.embed_tokens_per_layer", topo["device_map"])
        self.assertNotIn("model.language_model.layers.0", topo["device_map"])
        self.assertEqual(topo["device_map"]["model.language_model.layers.0.self_attn"], 1)
        self.assertEqual(topo["device_map"]["model.language_model.layers.22.self_attn"], 0)
        self.assertEqual(topo["device_map"]["model.language_model.layers.0.experts"], "disk")

    def test_quantize_moe_experts_reads_meta_offloaded_weights_map(self) -> None:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        from train.moe_quant import QuantizedGemma4Experts4bit, quantize_gemma4_moe_experts_in_place

        cfg = Gemma4TextConfig(
            hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
            vocab_size=128, enable_moe_block=True, num_experts=4, top_k_experts=2,
            hidden_size_per_layer_input=0, num_kv_shared_layers=0, layer_types=["full_attention"],
        )
        model = Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16)
        exp = model.model.layers[0].experts
        add_hook_to_module(exp, AlignDevicesHook(execution_device="cpu", offload=True))
        self.assertEqual(exp.gate_up_proj.device.type, "meta")
        hook = exp._hf_hook

        replaced = quantize_gemma4_moe_experts_in_place(model, layer_device_map={0: "cpu"})
        self.assertEqual(replaced, 1)
        q_exp = model.model.layers[0].experts
        self.assertIsInstance(q_exp, QuantizedGemma4Experts4bit)
        self.assertEqual(q_exp.gate_up_q_0.device.type, "cpu")
        self.assertEqual(len(hook.weights_map), 0)

        ids = torch.randint(0, 128, (1, 4))
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()

    def test_get_input_device_skips_cpu_offloaded_vision_tower(self) -> None:
        from train.model_loader import get_input_device

        fake_model = _FakeVisionFirstGemma()
        with mock.patch.object(fake_model.model.vision_tower, "parameters", return_value=iter([torch.zeros(1)])):
            dev = get_input_device(fake_model)
        self.assertEqual(dev, fake_model.get_input_embeddings().weight.device)

        fake_emb = mock.MagicMock()
        fake_emb.weight = SimpleNamespace(device=torch.device("cuda", 0))
        fake_model.get_input_embeddings = lambda: fake_emb
        self.assertEqual(get_input_device(fake_model), torch.device("cuda", 0))

    def test_export_merged_dir_isolates_26b_and_sets_q8_ncmoe(self) -> None:
        from export.merge_and_quantize_gguf import resolve_export_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e4b_default = resolve_export_paths(root, model_id="google/gemma-4-E4B-it", adapter_name=None)
            moe_default = resolve_export_paths(root, model_id="google/gemma-4-26B-A4B-it", adapter_name=None)
            self.assertEqual(e4b_default["merged_dir"], root / "outputs" / "merged-bf16")
            self.assertEqual(moe_default["merged_dir"], root / "outputs" / "merged-bf16-gemma-4-26B-A4B-coder")
            self.assertNotEqual(e4b_default["merged_dir"], moe_default["merged_dir"])
            self.assertEqual(moe_default["q8_extra_gpu_args"], ["-ncmoe", "10"])
            self.assertEqual(e4b_default["q8_extra_gpu_args"], [])

    def test_train_26b_all_languages_script_dry_run(self) -> None:
        script = REPO_ROOT / "scripts" / "train_26b_all_languages.zsh"
        self.assertTrue(script.is_file(), f"Missing runner script: {script}")
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "training_data"
            for eco in ("dotnet", "react", "go", "cpp"):
                (corpus / eco).mkdir(parents=True)
            env = {**os.environ, "OSS_CORPUS_DIR": str(corpus)}
            proc = subprocess.run(
                ["zsh", str(script), "--tier", "smoke", "--mode", "per-domain", "--dry-run"],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
        combined = proc.stdout + "\n" + proc.stderr
        for domain in ("dotnet", "react", "go", "cpp"):
            self.assertIn(domain, combined)
        self.assertIn("google/gemma-4-26B-A4B-it", combined)

    def test_multigpu_rng_context_fn_preserves_and_restores_all_device_states(self) -> None:
        from train.model_loader import make_multigpu_rng_context_fn

        fake_states = {0: torch.tensor([10], dtype=torch.uint8), 1: torch.tensor([20], dtype=torch.uint8)}

        def _get(dev="cuda"):
            return fake_states[int(dev)].clone()

        def _set(st, dev="cuda"):
            fake_states[int(dev)] = st.clone()

        with (
            mock.patch("train.model_loader.torch.cuda.is_available", return_value=True),
            mock.patch("train.model_loader.torch.cuda.device_count", return_value=2),
            mock.patch("train.model_loader.torch.cuda.get_rng_state", side_effect=_get),
            mock.patch("train.model_loader.torch.cuda.set_rng_state", side_effect=_set),
        ):
            ctx_fn = make_multigpu_rng_context_fn()
            fwd_ctx, rec_ctx = ctx_fn()
            with fwd_ctx:
                fake_states[0] = torch.tensor([99], dtype=torch.uint8)
                fake_states[1] = torch.tensor([88], dtype=torch.uint8)
            with rec_ctx:
                self.assertEqual(int(fake_states[0].item()), 10)
                self.assertEqual(int(fake_states[1].item()), 20)
            self.assertEqual(int(fake_states[0].item()), 99)
            self.assertEqual(int(fake_states[1].item()), 88)

    def test_disk_offload_moe_finalize_hooks_and_clean_hf_device_map(self) -> None:
        from transformers import AutoModelForCausalLM, Gemma4ForCausalLM, Gemma4TextConfig
        from train.moe_quant import (
            _extract_expert_tensor,
            _release_offload_hooks,
            finalize_disk_offloaded_moe_layers,
        )

        cfg = Gemma4TextConfig(
            hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
            vocab_size=128, enable_moe_block=True, num_experts=4, top_k_experts=2,
            hidden_size_per_layer_input=0, num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"],
        )
        m0 = Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as ckpt, tempfile.TemporaryDirectory() as off:
            m0.save_pretrained(ckpt, safe_serialization=True)
            dmap = {"model.embed_tokens": "cpu", "model.norm": "cpu", "model.rotary_emb": "cpu", "lm_head": "cpu"}
            for i in range(2):
                dmap[f"model.layers.{i}.layer_scalar"] = "cpu"
                for name, _ in m0.model.layers[i].named_children():
                    dmap[f"model.layers.{i}.{name}"] = "disk" if name == "experts" else "cpu"
            loaded = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map=dmap, offload_folder=off)
            self.assertEqual(list(Path(off).iterdir()), [])
            for i in range(2):
                gu = _extract_expert_tensor(loaded.model.layers[i].experts, "gate_up_proj")
                self.assertEqual(gu.shape, (4, 128, 64))
                _release_offload_hooks(loaded.model.layers[i].experts)
            finalize_disk_offloaded_moe_layers(loaded, layer_device_map={0: "cpu", 1: "cpu"}, prefix="model.layers")
            self.assertEqual(loaded.hf_device_map["model.layers.0"], "cpu")
            self.assertNotIn("model.layers.0.layer_scalar", loaded.hf_device_map)
            self.assertTrue(hasattr(loaded.model.layers[0], "_hf_hook"))

    def test_resolve_base_snapshot_dir_handles_partial_main_revision(self) -> None:
        from export.merge_and_quantize_gguf import resolve_base_snapshot_dir

        with tempfile.TemporaryDirectory() as tmp:
            snaps = Path(tmp) / "snapshots"
            partial = snaps / "rev_partial"
            complete = snaps / "rev_complete"
            partial.mkdir(parents=True)
            complete.mkdir(parents=True)
            (partial / "config.json").write_text("{}")
            (complete / "config.json").write_text("{}")
            (complete / "model.safetensors").write_text("weights")

            def _fake_cached(repo_id: str, filename: str, local_files_only: bool = False) -> str:
                if filename == "config.json":
                    return str(partial / "config.json")
                raise FileNotFoundError(filename)

            with (
                mock.patch("export.merge_and_quantize_gguf.cached_file", side_effect=_fake_cached),
                mock.patch("export.merge_and_quantize_gguf.snapshot_download", side_effect=AssertionError("unexpected call")),
            ):
                resolved = resolve_base_snapshot_dir("google/gemma-4-E4B-it")
            self.assertEqual(resolved, complete)

            index_file = complete / "model.safetensors.index.json"
            index_file.write_text("{}")
            with (
                mock.patch("export.merge_and_quantize_gguf.cached_file", return_value=str(index_file)),
                mock.patch("export.merge_and_quantize_gguf.snapshot_download", side_effect=AssertionError("unexpected call")),
            ):
                self.assertEqual(resolve_base_snapshot_dir("google/gemma-4-26B-A4B-it"), complete)


if __name__ == "__main__":
    unittest.main()
