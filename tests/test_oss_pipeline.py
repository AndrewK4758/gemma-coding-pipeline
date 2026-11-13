"""Unit and integration smoke tests for .NET & React OSS mining, sandbox verification, 4-bit MoE, and multi-engine export."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _init_git_repo(repo_dir: Path, files: dict[str, str], msg: str = "init") -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "Test User")
    for rel, content in files.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-m", msg)
    return _git(repo_dir, "rev-parse", "HEAD")


def _commit_files(repo_dir: Path, files: dict[str, str], msg: str) -> str:
    for rel, content in files.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-m", msg)
    return _git(repo_dir, "rev-parse", "HEAD")


class TestOssPipeline(unittest.TestCase):
    def test_mine_oss_repos_extracts_atomic_commits_and_splits_domains(self) -> None:
        from data.mine_oss_repos import mine_all_oss_repos

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "oss_repos"
            out_dir = Path(tmp) / "raw"

            cs_repo = corpus / "dotnet" / "CleanOrder"
            _init_git_repo(
                cs_repo,
                {
                    "src/Order.cs": "namespace App;\npublic class Order { public int Total { get; set; } }\n",
                    "tests/OrderTests.cs": "namespace App.Tests;\npublic class OrderTests {}\n",
                },
            )
            # Valid atomic commit (2 files, impl + test, >=10 diff lines)
            _commit_files(
                cs_repo,
                {
                    "src/Order.cs": (
                        "namespace App;\n"
                        "public sealed class Order\n{\n"
                        "    public int Total { get; private set; }\n"
                        "    public void AddLine(int cents)\n    {\n"
                        "        if (cents <= 0) throw new ArgumentOutOfRangeException(nameof(cents));\n"
                        "        Total += cents;\n    }\n}\n"
                    ),
                    "tests/OrderTests.cs": (
                        "namespace App.Tests;\n"
                        "public sealed class OrderTests\n{\n"
                        "    public void AddLine_RejectsNegative()\n    {\n"
                        "        var o = new App.Order();\n"
                        "        o.AddLine(100);\n    }\n}\n"
                    ),
                },
                "Enforce positive line item invariant on Order aggregate",
            )
            # Commit touching 4 files -> should be ignored
            _commit_files(
                cs_repo,
                {
                    "src/A.cs": "class A {}\n",
                    "src/B.cs": "class B {}\n",
                    "src/C.cs": "class C {}\n",
                    "tests/DTests.cs": "class DTests {}\n",
                },
                "Too many files",
            )

            ts_repo = corpus / "react" / "UiKit"
            _init_git_repo(
                ts_repo,
                {
                    "src/Button.tsx": "export function Button() { return null; }\n",
                    "src/Button.test.tsx": "export const t = 1;\n",
                },
            )
            _commit_files(
                ts_repo,
                {
                    "src/Button.tsx": (
                        "import React from 'react';\n"
                        "export interface ButtonProps {\n  label: string;\n  disabled?: boolean;\n}\n"
                        "export function Button({ label, disabled = false }: ButtonProps) {\n"
                        "  return <button type=\"button\" disabled={disabled}>{label}</button>;\n}\n"
                    ),
                    "src/Button.test.tsx": (
                        "import { Button } from './Button';\n"
                        "export function testButton() {\n"
                        "  return Button({ label: 'Save', disabled: false });\n}\n"
                    ),
                },
                "Add strict ButtonProps and disabled state",
            )

            counts = mine_all_oss_repos(corpus_root=corpus, output_dir=out_dir, domain="all")
            self.assertEqual(counts["sft"], 2)
            self.assertEqual(counts["dpo"], 2)
            self.assertEqual(counts["grpo"], 2)
            self.assertTrue((out_dir / "oss_dotnet_sft.jsonl").exists())
            self.assertTrue((out_dir / "oss_react_sft.jsonl").exists())

            dotnet_rows = [json.loads(ln) for ln in (out_dir / "oss_dotnet_sft.jsonl").read_text().splitlines()]
            self.assertEqual(len(dotnet_rows), 1)
            self.assertEqual(dotnet_rows[0]["language"], "csharp")
            self.assertEqual(dotnet_rows[0]["domain"], "dotnet")
            self.assertIn("<|channel>thought", dotnet_rows[0]["messages"][-1]["content"])
            self.assertIn("```diff", dotnet_rows[0]["messages"][-1]["content"])

            dpo_rows = [json.loads(ln) for ln in (out_dir / "oss_dpo.jsonl").read_text().splitlines()]
            self.assertIn("// ... rest of implementation remains the same", dpo_rows[0]["rejected"])

    def test_verify_oss_worktree_patch_and_cleanup(self) -> None:
        from data.compiler_sandbox import verify_oss_worktree_patch

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ts_proj"
            parent_sha = _init_git_repo(
                repo,
                {
                    "tsconfig.json": json.dumps({"compilerOptions": {"strict": True, "noEmit": True}}),
                    "src/math.ts": "export function add(a: number, b: number): number { return a + b; }\n",
                },
                "initial",
            )
            good_diff = (
                "```diff\n"
                "diff --git a/src/math.ts b/src/math.ts\n"
                "--- a/src/math.ts\n"
                "+++ b/src/math.ts\n"
                "@@ -1 +1,2 @@\n"
                " export function add(a: number, b: number): number { return a + b; }\n"
                "+export function mul(a: number, b: number): number { return a * b; }\n"
                "```"
            )
            ok, out = verify_oss_worktree_patch(str(repo), parent_sha, good_diff, ecosystem="react", timeout_sec=30)
            self.assertTrue(ok, msg=out)
            wt_list = _git(repo, "worktree", "list")
            self.assertEqual(len(wt_list.splitlines()), 1)

            bad_ok, bad_out = verify_oss_worktree_patch(str(repo), parent_sha, "not a diff", ecosystem="react", timeout_sec=15)
            self.assertFalse(bad_ok)
            self.assertIn("git apply failed", bad_out)

    def test_streaming_lora_merge_matches_peft_merge_and_unload(self) -> None:
        from export.merge_and_quantize_gguf import merge_lora_weights_into_shards

        cfg = Gemma4TextConfig(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            vocab_size=128,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            layer_types=["sliding_attention", "full_attention"],
            tie_word_embeddings=True,
        )
        torch.manual_seed(7)
        base_model = Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            snap_dir = tmp_p / "base_snap"
            snap_dir.mkdir()
            sd = base_model.state_dict()
            # Split across 2 shards to verify sharded index merging
            keys = list(sd.keys())
            mid = len(keys) // 2
            s1 = {k: sd[k].clone() for k in keys[:mid]}
            s2 = {k: sd[k].clone() for k in keys[mid:]}
            save_file(s1, str(snap_dir / "model-00001-of-00002.safetensors"))
            save_file(s2, str(snap_dir / "model-00002-of-00002.safetensors"))
            weight_map = {k: "model-00001-of-00002.safetensors" for k in s1}
            weight_map.update({k: "model-00002-of-00002.safetensors" for k in s2})
            (snap_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
            )

            peft_m = get_peft_model(
                base_model,
                LoraConfig(
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            # Populate non-zero lora_B weights so merge delta is non-trivial
            with torch.no_grad():
                for n, p in peft_m.named_parameters():
                    if "lora_B" in n:
                        p.normal_(mean=0.0, std=0.05)

            adapter_dir = tmp_p / "adapter"
            peft_m.save_pretrained(str(adapter_dir))
            ref_merged = peft_m.merge_and_unload().state_dict()

            out_dir = tmp_p / "merged"
            merge_stats = merge_lora_weights_into_shards(snap_dir, adapter_dir, out_dir)
            self.assertGreater(merge_stats["folded_lora_linear_tensors"], 0)
            self.assertTrue((out_dir / "model.safetensors.index.json").exists())

            loaded = {}
            for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
                with safe_open(str(out_dir / shard_name), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        loaded[k] = f.get_tensor(k)

            for k, ref_t in ref_merged.items():
                if k in loaded:
                    diff = float((loaded[k].float() - ref_t.float()).abs().max().item())
                    self.assertLess(diff, 2e-2, msg=f"Mismatch on {k}: {diff}")

    def test_device_map_and_export_helpers_and_cli(self) -> None:
        from export.merge_and_quantize_gguf import write_ollama_modelfile, write_vllm_serve_script
        from run_pipeline import build_stage_commands, parse_args
        from train.model_loader import build_asymmetric_device_map

        dmap_26b = build_asymmetric_device_map(
            model_type="gemma4", num_layers=30, primary_gpu_max_layer=21, offload_multimodal=True
        )
        self.assertEqual(dmap_26b["model.vision_tower"], "cpu")
        self.assertEqual(dmap_26b["model.language_model.layers.21"], 1)
        self.assertEqual(dmap_26b["model.language_model.layers.22"], 0)
        self.assertEqual(dmap_26b["lm_head"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            mf = write_ollama_modelfile(tmp_p / "Modelfile", "gemma-4-E4B-coder-Q4_K_M-imatrix.gguf")
            self.assertIn("FROM ./gemma-4-E4B-coder-Q4_K_M-imatrix.gguf", mf.read_text())
            vs = write_vllm_serve_script(tmp_p / "serve_vllm.sh", tmp_p / "merged", "gemma-4-e4b-coder", is_moe=False)
            self.assertIn("--reasoning-parser gemma4", vs.read_text())

        args = parse_args(["--tier", "smoke", "--stage", "data", "--domain", "dotnet", "--adapter-name", "dotnet", "--oss-only"])
        plan = build_stage_commands(args)
        self.assertEqual(plan[0][0], "data")
        self.assertIn("--domain", plan[0][1])
        self.assertIn("dotnet", plan[0][1])

    def _make_mock_tokenizer(self):
        class _MockTok:
            special_tokens_map = {"sot_token": "<|turn>", "eot_token": "<turn|>"}
            pad_token_id = 0
            vocab: dict[str, int] = {"<pad>": 0}
            inv: dict[int, str] = {0: "<pad>"}

            def encode(self, text: str, add_special_tokens: bool = True, truncation: bool = False, max_length: int = 4096):
                ids = [1] if add_special_tokens else []
                for ch in text:
                    if ch not in self.vocab:
                        idx = len(self.vocab) + 1
                        self.vocab[ch] = idx
                        self.inv[idx] = ch
                    ids.append(self.vocab[ch])
                return ids[:max_length] if truncation else ids

            def decode(self, ids):
                return "".join(self.inv.get(i, "") for i in ids)

        return _MockTok()

    def test_multiturn_assistant_only_loss_masking(self) -> None:
        from data.build_dataset import format_gemma_messages_and_mask_labels

        tok = self._make_mock_tokenizer()
        out = format_gemma_messages_and_mask_labels(
            [
                {"role": "user", "content": "Inspect Order.cs"},
                {"role": "assistant", "content": "REPLY_ONE"},
                {"role": "user", "content": "Now write tests"},
                {"role": "assistant", "content": "REPLY_TWO"},
            ],
            tokenizer=tok,
            max_seq_len=1024,
        )
        self.assertEqual(len(out["input_ids"]), len(out["labels"]))
        unmasked = tok.decode([i for i, l in zip(out["input_ids"], out["labels"]) if l != -100])
        self.assertIn("REPLY_ONE", unmasked)
        self.assertIn("REPLY_TWO", unmasked)
        self.assertNotIn("Inspect Order.cs", unmasked)
        self.assertNotIn("Now write tests", unmasked)

    def test_quantized_gemma4_experts_4bit_forward_and_backward(self) -> None:
        from train.moe_quant import QuantizedGemma4Experts4bit, quantize_gemma4_moe_experts_in_place

        cfg = Gemma4TextConfig(
            hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
            vocab_size=128, enable_moe_block=True, num_experts=4, top_k_experts=2,
            hidden_size_per_layer_input=0, num_kv_shared_layers=0, layer_types=["full_attention"],
        )
        torch.manual_seed(3)
        model = Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16)
        replaced = quantize_gemma4_moe_experts_in_place(model)
        self.assertEqual(replaced, 1)
        self.assertIsInstance(model.model.layers[0].experts, QuantizedGemma4Experts4bit)

        ids = torch.randint(0, 128, (1, 6))
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        grad = model.model.layers[0].router.proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_grpo_online_worktree_rubric_grades_diffs_and_placeholders(self) -> None:
        from train.grpo_online import grade_completion_rubric

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ts_grpo"
            sha = _init_git_repo(
                repo,
                {"tsconfig.json": json.dumps({"compilerOptions": {"strict": True, "noEmit": True}}),
                 "src/calc.ts": "export function sub(a: number, b: number): number { return a - b; }\n"},
            )
            task = {"record_type": "oss_worktree", "repo_path": str(repo), "parent_sha": sha, "ecosystem": "react", "test_files": []}
            self.assertEqual(grade_completion_rubric("no diff here", "typescript", task=task)["reward"], -1.0)
            lazy = "```diff\ndiff --git a/src/calc.ts b/src/calc.ts\n--- a/src/calc.ts\n+++ b/src/calc.ts\n@@ -1 +1 @@\n+// ... rest of implementation remains the same\n```"
            self.assertEqual(grade_completion_rubric(lazy, "typescript", task=task)["reward"], -1.0)
            good = "```diff\ndiff --git a/src/calc.ts b/src/calc.ts\n--- a/src/calc.ts\n+++ b/src/calc.ts\n@@ -1 +1,2 @@\n export function sub(a: number, b: number): number { return a - b; }\n+export function inc(x: number): number { return x + 1; }\n```"
            res = grade_completion_rubric(good, "typescript", task=task)
            self.assertTrue(res["compiler_passed"], msg=res["diagnostics"])
            self.assertEqual(res["reward"], 3.5)

    def test_build_dataset_oss_only_integration(self) -> None:
        from data.build_dataset import build_dataset_for_tier

        tok = self._make_mock_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            corpus = tmp_p / "oss_repos"
            cs_repo = corpus / "dotnet" / "SampleSvc"
            _init_git_repo(cs_repo, {"src/Svc.cs": "class Svc {}\n", "tests/SvcTests.cs": "class SvcTests {}\n"})
            _commit_files(
                cs_repo,
                {"src/Svc.cs": "namespace S;\npublic sealed class Svc\n{\n    public int Add(int a, int b)\n    {\n        return a + b;\n    }\n}\n",
                 "tests/SvcTests.cs": "namespace S.Tests;\npublic sealed class SvcTests\n{\n    public int Check()\n    {\n        return new S.Svc().Add(1, 2);\n    }\n}\n"},
                "Implement Svc.Add with test coverage",
            )
            out_dir = tmp_p / "processed"
            with (
                mock.patch("data.build_dataset.load_gemma2_tokenizer", return_value=tok),
                mock.patch(
                    "data.dedup_and_filter._fetch_embeddings_batch",
                    side_effect=lambda texts, **_kw: [[1.0 if i == j else 0.0 for j in range(max(len(texts), 4))] for i in range(len(texts))],
                ),
            ):
                manifest = build_dataset_for_tier(
                    "smoke", domain="dotnet", oss_only=True, oss_corpus_root=str(corpus), output_dir=out_dir
                )
            self.assertEqual(manifest["sft_total_rows"], 1)
            self.assertEqual(manifest["grpo_tasks"], 1)
            self.assertEqual(pq.read_table(out_dir / "grpo_tasks.parquet").num_rows, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
