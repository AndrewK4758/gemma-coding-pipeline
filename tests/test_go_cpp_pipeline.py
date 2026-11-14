"""Unit and integration tests for Go and C/C++ OSS mining, sandboxes, DPO, and CLI wiring."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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


class TestGoCppPipeline(unittest.TestCase):
    def test_mine_oss_repos_extracts_go_and_cpp_atomic_commits(self) -> None:
        from data.mine_oss_repos import load_oss_records, mine_all_oss_repos

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "oss_repos"
            out_dir = Path(tmp) / "raw"

            go_repo = corpus / "go" / "gin_demo"
            _init_git_repo(
                go_repo,
                {
                    "go.mod": "module example.com/gindemo\n\ngo 1.24\n",
                    "calc/sum.go": "package calc\n\nfunc Sum(a, b int) int { return a + b }\n",
                    "calc/sum_test.go": "package calc\n\nimport \"testing\"\n\nfunc TestSum(t *testing.T) {}\n",
                },
            )
            _commit_files(
                go_repo,
                {
                    "calc/sum.go": (
                        "package calc\n\n"
                        "func Sum(a, b int) int { return a + b }\n\n"
                        "func ClampPositive(v, hi int) int {\n"
                        "    if v < 0 {\n        return 0\n    }\n"
                        "    if v > hi {\n        return hi\n    }\n"
                        "    return v\n}\n"
                    ),
                    "calc/sum_test.go": (
                        "package calc\n\nimport \"testing\"\n\n"
                        "func TestClampPositive(t *testing.T) {\n"
                        "    if got := ClampPositive(-3, 10); got != 0 {\n"
                        "        t.Fatalf(\"want 0, got %d\", got)\n    }\n"
                        "    if got := ClampPositive(20, 10); got != 10 {\n"
                        "        t.Fatalf(\"want 10, got %d\", got)\n    }\n}\n"
                    ),
                },
                "Add bounded ClampPositive helper and unit test",
            )

            cpp_repo = corpus / "cpp" / "fmt_demo"
            _init_git_repo(
                cpp_repo,
                {
                    "src/clamp.cpp": "int clamp_val(int x, int lo, int hi) { return x; }\n",
                    "tests/clamp_test.cpp": "int run_test() { return 0; }\n",
                },
            )
            _commit_files(
                cpp_repo,
                {
                    "src/clamp.cpp": (
                        "int clamp_val(int x, int lo, int hi) {\n"
                        "    if (x < lo) return lo;\n"
                        "    if (x > hi) return hi;\n"
                        "    return x;\n}\n\n"
                        "int abs_diff(int a, int b) {\n"
                        "    return a > b ? (a - b) : (b - a);\n}\n"
                    ),
                    "tests/clamp_test.cpp": (
                        "#include <cassert>\n\n"
                        "int clamp_val(int x, int lo, int hi);\n"
                        "int main() {\n"
                        "    assert(clamp_val(-5, 0, 10) == 0);\n"
                        "    assert(clamp_val(25, 0, 10) == 10);\n"
                        "    assert(clamp_val(4, 0, 10) == 4);\n"
                        "    return 0;\n}\n"
                    ),
                },
                "Fix clamp_val bounds handling and add regression assertions",
            )

            counts = mine_all_oss_repos(corpus_root=corpus, output_dir=out_dir, domain="all")
            self.assertEqual(counts["go_sft"], 1)
            self.assertEqual(counts["cpp_sft"], 1)
            self.assertEqual(counts["sft"], 2)

            go_recs = load_oss_records("sft", raw_dir=out_dir, corpus_root=corpus, domain="go")
            cpp_recs = load_oss_records("sft", raw_dir=out_dir, corpus_root=corpus, domain="cpp")
            self.assertEqual(len(go_recs), 1)
            self.assertEqual(len(cpp_recs), 1)
            self.assertEqual(go_recs[0]["language"], "go")
            self.assertEqual(cpp_recs[0]["language"], "cpp")
            self.assertIn("<|channel>thought", cpp_recs[0]["response"])

    def test_verify_oss_worktree_go_patch_pass_and_fail(self) -> None:
        from data.oss_sandbox import verify_oss_worktree_patch

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "go_proj"
            parent_sha = _init_git_repo(
                repo,
                {
                    "go.mod": "module example.com/gopatch\n\ngo 1.24\n",
                    "mathx/add.go": "package mathx\n\nfunc Add(a, b int) int { return a + b }\n",
                    "mathx/add_test.go": (
                        "package mathx\n\nimport \"testing\"\n\n"
                        "func TestAdd(t *testing.T) {\n"
                        "    if Add(2, 3) != 5 { t.Fatal(\"bad\") }\n}\n"
                    ),
                },
            )
            good_diff = (
                "```diff\n"
                "diff --git a/mathx/add.go b/mathx/add.go\n"
                "--- a/mathx/add.go\n"
                "+++ b/mathx/add.go\n"
                "@@ -1,3 +1,5 @@\n"
                " package mathx\n"
                " \n"
                " func Add(a, b int) int { return a + b }\n"
                "+\n"
                "+func Mul(a, b int) int { return a * b }\n"
                "diff --git a/mathx/add_test.go b/mathx/add_test.go\n"
                "--- a/mathx/add_test.go\n"
                "+++ b/mathx/add_test.go\n"
                "@@ -4,3 +4,7 @@ import \"testing\"\n"
                " func TestAdd(t *testing.T) {\n"
                "     if Add(2, 3) != 5 { t.Fatal(\"bad\") }\n"
                " }\n"
                "+\n"
                "+func TestMul(t *testing.T) {\n"
                "+    if Mul(3, 4) != 12 { t.Fatal(\"bad mul\") }\n"
                "+}\n"
                "```"
            )
            ok, out = verify_oss_worktree_patch(
                str(repo), parent_sha, good_diff, ecosystem="go", test_files=["mathx/add_test.go"], timeout_sec=30
            )
            self.assertTrue(ok, msg=out)
            self.assertEqual(len(_git(repo, "worktree", "list").splitlines()), 1)

            bad_diff = good_diff.replace("return a * b", "return a + b")
            bad_ok, _ = verify_oss_worktree_patch(
                str(repo), parent_sha, bad_diff, ecosystem="go", test_files=["mathx/add_test.go"], timeout_sec=30
            )
            self.assertFalse(bad_ok)
            self.assertEqual(len(_git(repo, "worktree", "list").splitlines()), 1)

    def test_verify_oss_worktree_cpp_cmake_and_standalone(self) -> None:
        from data.oss_sandbox import verify_oss_worktree_patch

        with tempfile.TemporaryDirectory() as tmp:
            standalone = Path(tmp) / "cpp_standalone"
            p_sha = _init_git_repo(
                standalone,
                {
                    "src/lib.cpp": "int triple(int x) { return x * 3; }\n",
                    "tests/main.cpp": "#include <cassert>\nint triple(int x);\nint main() { assert(triple(2) == 6); return 0; }\n",
                },
            )
            pass_diff = (
                "diff --git a/src/lib.cpp b/src/lib.cpp\n"
                "--- a/src/lib.cpp\n"
                "+++ b/src/lib.cpp\n"
                "@@ -1 +1,2 @@\n"
                " int triple(int x) { return x * 3; }\n"
                "+int quadruple(int x) { return x * 4; }\n"
            )
            ok, out = verify_oss_worktree_patch(str(standalone), p_sha, pass_diff, ecosystem="cpp", timeout_sec=30)
            self.assertTrue(ok, msg=out)

            fail_diff = pass_diff.replace("x * 4", "x +")
            bad_ok, _ = verify_oss_worktree_patch(str(standalone), p_sha, fail_diff, ecosystem="cpp", timeout_sec=30)
            self.assertFalse(bad_ok)

            cmake_repo = Path(tmp) / "cpp_cmake"
            c_sha = _init_git_repo(
                cmake_repo,
                {
                    "CMakeLists.txt": (
                        "cmake_minimum_required(VERSION 3.16)\n"
                        "project(mini_ctest CXX)\n"
                        "set(CMAKE_CXX_STANDARD 20)\n"
                        "enable_testing()\n"
                        "add_executable(mini_test main.cpp)\n"
                        "add_test(NAME mini_test COMMAND mini_test)\n"
                    ),
                    "main.cpp": "int add(int a, int b) { return a + b; }\nint main() { return add(2, 2) == 4 ? 0 : 1; }\n",
                },
            )
            cmake_ok_diff = (
                "diff --git a/main.cpp b/main.cpp\n"
                "--- a/main.cpp\n"
                "+++ b/main.cpp\n"
                "@@ -1,2 +1,3 @@\n"
                " int add(int a, int b) { return a + b; }\n"
                "+int sub(int a, int b) { return a - b; }\n"
                " int main() { return add(2, 2) == 4 ? 0 : 1; }\n"
            )
            ok_cm, out_cm = verify_oss_worktree_patch(str(cmake_repo), c_sha, cmake_ok_diff, ecosystem="cpp", timeout_sec=45)
            self.assertTrue(ok_cm, msg=out_cm)
            self.assertEqual(len(_git(cmake_repo, "worktree", "list").splitlines()), 1)

    def test_verify_c_cpp_snippets_and_dpo_pairs(self) -> None:
        from data.compiler_sandbox import build_dpo_preference_pairs, verify_code_sample

        c_ok = verify_code_sample("```c\n#include <stdint.h>\nint32_t add_i32(int32_t a, int32_t b) { return a + b; }\n```", language="c")
        self.assertTrue(c_ok.passed, msg=c_ok.diagnostics)
        self.assertEqual(c_ok.toolchain, "gcc-c17")

        c_bad = verify_code_sample("```c\nint broken(void) { return \"bad\"; }\n```", language="c")
        self.assertFalse(c_bad.passed)

        cpp_code = (
            "```cpp\n"
            "#include <numeric>\n"
            "#include <span>\n"
            "long long sum_pos(std::span<const int> xs) {\n"
            "    long long s = 0;\n"
            "    for (int v : xs) { if (v > 0) s += v; }\n"
            "    return s;\n}\n```"
        )
        cpp_ok = verify_code_sample(cpp_code, language="cpp")
        self.assertTrue(cpp_ok.passed, msg=cpp_ok.diagnostics)
        self.assertEqual(cpp_ok.toolchain, "g++-cpp20")

        cpp_bad = verify_code_sample("```cpp\nint bad_fn(int x) { return x + ; }\n```", language="cpp")
        self.assertFalse(cpp_bad.passed)

        pairs = build_dpo_preference_pairs(
            [{"id": "cpp_1", "instruction": "Write sum_pos", "response": cpp_code, "language": "cpp", "source": "oss_commits"}],
            target_dpo_pairs=1,
            max_workers=2,
        )
        self.assertEqual(len(pairs), 1)
        self.assertIn("cpp", pairs[0]["language"])
        self.assertTrue(pairs[0]["compiler_feedback"])

    def test_run_pipeline_and_imatrix_go_cpp_wiring(self) -> None:
        from export.merge_and_quantize_gguf import build_imatrix_calibration_file
        from run_pipeline import build_stage_commands, parse_args

        for domain in ("go", "cpp"):
            args = parse_args(["--tier", "smoke", "--domain", domain, "--oss-only", "--stages", "data,sft,dpo,eval,export"])
            plan = build_stage_commands(args)
            by_name = dict(plan)
            self.assertIn("--adapter-name", by_name["sft"])
            self.assertIn(domain, by_name["sft"])
            self.assertIn(f"smoke-{domain}", by_name["data"][by_name["data"].index("--output-dir") + 1])

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "proc"
            data_dir.mkdir(parents=True)
            rows = [
                {"language": "typescript", "text": "const x: number = 1;"},
                {"language": "go", "text": "package main\nfunc main() {}"},
                {"language": "csharp", "text": "public class C {}"},
                {"language": "cpp", "text": "int add(int a, int b) { return a + b; }"},
            ]
            pq.write_table(pa.Table.from_pylist(rows), data_dir / "sft_train.parquet")
            meta = build_imatrix_calibration_file("smoke", Path(tmp) / "calib.txt", max_samples=16, data_dir=data_dir)
            self.assertEqual(meta["samples_included"], 4)
            self.assertEqual(meta["per_language_counts"]["cpp"], 1)

    def test_dpo_length_normalization_and_hard_oss_negative(self) -> None:
        from types import SimpleNamespace
        from unittest import mock
        import torch
        from data.mine_oss_repos import to_dpo_record
        from train.dpo_compiler import evaluate_dpo, precompute_reference_logps

        commit = {
            "ecosystem": "go", "language": "go", "repo_name": "acme/pkg", "repo_path": "/tmp/r",
            "sha": "abc123456789", "parent_sha": "def987654321", "subject": "Fix bounds check",
            "impl_files": ["math.go"], "test_files": ["math_test.go"], "parent_excerpt": "",
            "diff": "diff --git a/math.go b/math.go\n--- a/math.go\n+++ b/math.go\n@@ -1,4 +1,6 @@\n func Clamp(x int) int {\n+\tif x < 0 { return 0 }\n+\treturn x\n }\n",
        }
        rec = to_dpo_record(commit)
        self.assertIn("<|channel>thought", rec["rejected"])
        self.assertNotIn("Modifying math.go quickly.", rec["rejected"])
        self.assertGreater(len(rec["rejected"]), int(0.5 * len(rec["chosen"])))

        fake_sample = {
            "chosen_input_ids": torch.ones(1, 10, dtype=torch.long),
            "chosen_attention_mask": torch.ones(1, 10, dtype=torch.long),
            "chosen_labels": torch.ones(1, 10, dtype=torch.long),
            "rejected_input_ids": torch.ones(1, 4, dtype=torch.long),
            "rejected_attention_mask": torch.ones(1, 4, dtype=torch.long),
            "rejected_labels": torch.ones(1, 4, dtype=torch.long),
            "ref_chosen_logp": None,
            "ref_rejected_logp": None,
        }

        class _FakeModel:
            def eval(self):
                return self

        writer = SimpleNamespace(add_scalar=lambda *_a, **_k: None)
        side = [(torch.tensor([-1000.0]), torch.tensor(1000.0)), (torch.tensor([-100.0]), torch.tensor(100.0))] * 2
        with mock.patch("train.dpo_compiler.compute_chunked_sequence_logps", side_effect=side):
            precompute_reference_logps(_FakeModel(), [fake_sample], "precompute_sft", "eval", SimpleNamespace(print=lambda *_a: None))
            self.assertAlmostEqual(fake_sample["ref_chosen_logp"], -1.0, places=5)
            self.assertAlmostEqual(fake_sample["ref_rejected_logp"], -1.0, places=5)
            ev = evaluate_dpo(_FakeModel(), [fake_sample], beta=2.0, step=0, tb_writer=writer)
            self.assertAlmostEqual(ev["logps/chosen"], -1.0, places=5)
            self.assertAlmostEqual(ev["rewards/margins"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
