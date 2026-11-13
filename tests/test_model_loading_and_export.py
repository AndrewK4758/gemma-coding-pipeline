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


if __name__ == "__main__":
    unittest.main()
