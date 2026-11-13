"""Mine atomic implementation+test commits from local .NET and React OSS repositories.

Produces grounded SFT (with Gemma 4 native <|channel>thought reasoning and
unified diffs), DPO (chosen clean patch vs. lazy/placeholder rejected patch), and
GRPO (git worktree verification tasks keyed by parent_sha and test_files) datasets,
both combined and split by domain (`dotnet`, `react`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ECOSYSTEM_CONFIG: dict[str, dict[str, Any]] = {
    "dotnet": {
        "language": "csharp",
        "label": ".NET / C#",
        "extensions": (".cs",),
        "ignore_suffixes": (".g.cs", ".designer.cs", "assemblyinfo.cs"),
    },
    "react": {
        "language": "typescript",
        "label": "React / TypeScript",
        "extensions": (".ts", ".tsx", ".js", ".jsx"),
        "ignore_suffixes": (".d.ts", ".min.js"),
    },
    "go": {
        "language": "go",
        "label": "Go",
        "extensions": (".go",),
        "ignore_suffixes": (".pb.go", "_mock.go", "mock_gen.go"),
    },
    "cpp": {
        "language": "cpp",
        "label": "C / C++",
        "extensions": (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"),
        "ignore_suffixes": (".pb.cc", ".pb.h", ".generated.h"),
    },
}
SUPPORTED_ECOSYSTEMS = tuple(ECOSYSTEM_CONFIG.keys())

DEFAULT_CORPUS_ROOT = Path(os.environ.get("OSS_CORPUS_DIR", str(Path.home() / "corpus" / "oss_repos")))
DEFAULT_OUTPUT_DIR = Path("data/raw")
LAZY_PLACEHOLDER = "// ... rest of implementation remains the same"


def _run_git(repo_path: Path, *args: str, timeout: int = 30) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_path}: {proc.stderr.strip()}")
    return proc.stdout


def _is_test_file(rel_path: str, ecosystem: str) -> bool:
    norm = rel_path.replace("\\", "/").lower()
    name = Path(norm).name
    if ecosystem == "dotnet":
        return (
            "tests/" in f"/{norm}"
            or "test/" in f"/{norm}"
            or name.endswith(("tests.cs", "test.cs", "spec.cs", "specs.cs"))
            or ".tests/" in norm
        )
    if ecosystem == "go":
        return name.endswith("_test.go")
    if ecosystem == "cpp":
        return (
            "/tests/" in f"/{norm}"
            or "/test/" in f"/{norm}"
            or "/unit/" in f"/{norm}"
            or "/unittest/" in f"/{norm}"
            or "/unittests/" in f"/{norm}"
            or name.startswith("test_")
            or any(s in name for s in ("_test.", "_unittest.", ".test."))
        )
    return (
        "__tests__/" in norm
        or "/tests/" in f"/{norm}"
        or "/test/" in f"/{norm}"
        or any(s in name for s in (".test.", ".spec.", ".stories."))
    )


def _is_relevant_code_file(rel_path: str, ecosystem: str) -> bool:
    cfg = ECOSYSTEM_CONFIG[ecosystem]
    norm = rel_path.replace("\\", "/").lower()
    if not norm.endswith(cfg["extensions"]):
        return False
    return not norm.endswith(cfg["ignore_suffixes"])


def _read_parent_excerpt(repo_path: Path, parent_sha: str, rel_paths: list[str], max_lines: int = 80) -> str:
    snippets: list[str] = []
    for rel in rel_paths[:2]:
        proc = subprocess.run(
            ["git", "-C", str(repo_path), "show", f"{parent_sha}:{rel}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        lines = proc.stdout.splitlines()
        body = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            body += "\n// ... (excerpt truncated for prompt context)"
        snippets.append(f"### `{rel}`\n```\n{body}\n```")
    return "\n\n".join(snippets)


def find_atomic_test_commits(
    repo_path: Path,
    ecosystem: str,
    max_commits: int = 300,
    min_changed_lines: int = 10,
    max_diff_lines: int = 220,
) -> list[dict[str, Any]]:
    """Extract atomic commits touching 2-3 code files (impl + test) from git history."""
    if not (repo_path / ".git").exists():
        return []
    raw = _run_git(
        repo_path,
        "log",
        "--no-merges",
        "--numstat",
        f"--max-count={max_commits}",
        "--format=__COMMIT__%H%x00%P%x00%s",
    )
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw.splitlines():
        if line.startswith("__COMMIT__"):
            if current is not None:
                entries.append(current)
            parts = line[len("__COMMIT__") :].split("\x00", 2)
            sha, parents, subject = (parts + ["", "", ""])[:3]
            current = {"sha": sha.strip(), "parents": parents.split(), "subject": subject.strip(), "stats": []}
        elif current is not None and "\t" in line:
            cols = line.split("\t", 2)
            if len(cols) == 3:
                current["stats"].append(cols)
    if current is not None:
        entries.append(current)

    mined: list[dict[str, Any]] = []
    for item in entries:
        if len(item["parents"]) != 1 or not item["stats"]:
            continue
        if not (2 <= len(item["stats"]) <= 3):
            continue
        parent_sha = item["parents"][0]
        changed_lines = 0
        impl_files: list[str] = []
        test_files: list[str] = []
        binary_or_irrelevant = False
        for ins_s, del_s, rel_path in item["stats"]:
            if ins_s == "-" or del_s == "-" or not _is_relevant_code_file(rel_path, ecosystem):
                binary_or_irrelevant = True
                break
            changed_lines += int(ins_s) + int(del_s)
            if _is_test_file(rel_path, ecosystem):
                test_files.append(rel_path)
            else:
                impl_files.append(rel_path)
        if binary_or_irrelevant or not impl_files or not test_files or changed_lines < min_changed_lines:
            continue

        patch = _run_git(repo_path, "diff", parent_sha, item["sha"], "--", *(impl_files + test_files)).strip()
        if not patch or len(patch.splitlines()) > max_diff_lines:
            continue
        mined.append(
            {
                "repo_name": repo_path.name,
                "repo_path": str(repo_path.resolve()),
                "ecosystem": ecosystem,
                "language": ECOSYSTEM_CONFIG[ecosystem]["language"],
                "sha": item["sha"],
                "parent_sha": parent_sha,
                "subject": item["subject"] or "Update implementation and tests",
                "impl_files": impl_files,
                "test_files": test_files,
                "diff": patch,
                "parent_excerpt": _read_parent_excerpt(repo_path, parent_sha, impl_files),
            }
        )
    return mined


def _build_prompt(commit: dict[str, Any]) -> str:
    stack = ECOSYSTEM_CONFIG[commit["ecosystem"]]["label"]
    targets = ", ".join(commit["impl_files"] + commit["test_files"])
    parts = [
        f"Repository: `{commit['repo_name']}` ({stack})",
        f"Task: {commit['subject']}",
        f"Target files: {targets}",
    ]
    if commit["parent_excerpt"]:
        parts.append(f"Current code before patch:\n{commit['parent_excerpt']}")
    parts.append(
        "Provide a minimal, compilable unified `git diff` that implements the requested change "
        "and updates the corresponding tests without eliding code or placeholders."
    )
    return "\n\n".join(parts)


def to_sft_record(commit: dict[str, Any]) -> dict[str, Any]:
    """Convert an atomic OSS commit into a Gemma 4 thinking + diff SFT record."""
    prompt = _build_prompt(commit)
    impl_list = ", ".join(commit["impl_files"])
    test_list = ", ".join(commit["test_files"])
    stack = ECOSYSTEM_CONFIG[commit["ecosystem"]]["label"]
    thought = (
        f"Let's analyze the {stack} change requested for `{commit['repo_name']}`: {commit['subject']}.\n"
        f"1. Modify implementation file(s): {impl_list} while preserving strict typing and contracts.\n"
        f"2. Update test file(s): {test_list} so compilation and assertions pass deterministically.\n"
        "3. Emit an exact unified diff with complete context and zero placeholder comments."
    )
    response = f"<|channel>thought\n{thought}\n<channel|>\n```diff\n{commit['diff']}\n```"
    rec_id = f"oss_{commit['ecosystem']}_{commit['sha'][:12]}"
    return {
        "id": rec_id,
        "source": "oss_commits",
        "record_type": "oss_diff",
        "domain": commit["ecosystem"],
        "language": commit["language"],
        "repo_name": commit["repo_name"],
        "repo_path": commit["repo_path"],
        "commit_sha": commit["sha"],
        "parent_sha": commit["parent_sha"],
        "instruction": prompt,
        "response": response,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _degrade_diff_with_placeholder(patch: str, impl_files: list[str]) -> str:
    target = impl_files[0] if impl_files else "src/Target.code"
    first_hunk = next((ln for ln in patch.splitlines() if ln.startswith("@@")), "@@ -1,3 +1,3 @@")
    return (
        f"<|channel>thought\nModifying {target} quickly.\n<channel|>\n"
        f"```diff\ndiff --git a/{target} b/{target}\n"
        f"--- a/{target}\n+++ b/{target}\n{first_hunk}\n"
        f"+{LAZY_PLACEHOLDER}\n```"
    )


def to_dpo_record(commit: dict[str, Any]) -> dict[str, Any]:
    """Convert an atomic OSS commit into a DPO preference pair rejecting truncated diffs."""
    sft = to_sft_record(commit)
    return {
        "id": f"dpo_{sft['id']}",
        "source": "oss_commits",
        "record_type": "oss_diff",
        "domain": commit["ecosystem"],
        "language": commit["language"],
        "repo_name": commit["repo_name"],
        "commit_sha": commit["sha"],
        "prompt": sft["instruction"],
        "chosen": sft["response"],
        "rejected": _degrade_diff_with_placeholder(commit["diff"], commit["impl_files"]),
        "compiler_toolchain": f"oss-{commit['ecosystem']}-diff",
        "compiler_feedback": f"Rejected truncated placeholder diff for {commit['repo_name']} ({commit['sha'][:8]})",
    }


def to_grpo_record(commit: dict[str, Any]) -> dict[str, Any]:
    """Convert an atomic OSS commit into an online GRPO git-worktree verification task."""
    return {
        "id": f"grpo_oss_{commit['ecosystem']}_{commit['sha'][:12]}",
        "source": "oss_commits",
        "record_type": "oss_worktree",
        "domain": commit["ecosystem"],
        "ecosystem": commit["ecosystem"],
        "language": commit["language"],
        "repo_name": commit["repo_name"],
        "repo_path": commit["repo_path"],
        "parent_sha": commit["parent_sha"],
        "commit_sha": commit["sha"],
        "test_files": commit["test_files"],
        "prompt": _build_prompt(commit),
        "reference_diff": commit["diff"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_repos(corpus_root: Path, domain: str) -> list[tuple[str, Path]]:
    ecosystems = SUPPORTED_ECOSYSTEMS if domain == "all" else (domain,)
    found: list[tuple[str, Path]] = []
    for eco in ecosystems:
        eco_dir = corpus_root / eco
        if not eco_dir.is_dir():
            continue
        for child in sorted(eco_dir.iterdir()):
            if child.is_dir() and (child / ".git").exists():
                found.append((eco, child))
    return found


def mine_all_oss_repos(
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    domain: str = "all",
    max_commits_per_repo: int = 300,
) -> dict[str, int]:
    """Mine configured local OSS repositories and write combined + per-domain JSONL artifacts."""
    if domain not in {"all", *SUPPORTED_ECOSYSTEMS}:
        raise ValueError(f"Unsupported domain {domain!r}; expected all|{'|'.join(SUPPORTED_ECOSYSTEMS)}")
    corpus_p = Path(corpus_root)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
        "all": {"sft": [], "dpo": [], "grpo": []},
        **{eco: {"sft": [], "dpo": [], "grpo": []} for eco in SUPPORTED_ECOSYSTEMS},
    }
    for eco, repo_dir in _discover_repos(corpus_p, domain):
        for commit in find_atomic_test_commits(repo_dir, eco, max_commits=max_commits_per_repo):
            sft_r = to_sft_record(commit)
            dpo_r = to_dpo_record(commit)
            grpo_r = to_grpo_record(commit)
            for key in ("all", eco):
                buckets[key]["sft"].append(sft_r)
                buckets[key]["dpo"].append(dpo_r)
                buckets[key]["grpo"].append(grpo_r)

    ecosystems_to_emit = SUPPORTED_ECOSYSTEMS if domain == "all" else (domain,)
    for split in ("sft", "dpo", "grpo"):
        _write_jsonl(out_p / f"oss_{split}.jsonl", buckets["all"][split])
        for eco in ecosystems_to_emit:
            _write_jsonl(out_p / f"oss_{eco}_{split}.jsonl", buckets[eco][split])

    summary: dict[str, int] = {
        "sft": len(buckets["all"]["sft"]),
        "dpo": len(buckets["all"]["dpo"]),
        "grpo": len(buckets["all"]["grpo"]),
    }
    for eco in SUPPORTED_ECOSYSTEMS:
        summary[f"{eco}_sft"] = len(buckets[eco]["sft"])
    return summary


def load_oss_records(
    split: str,
    raw_dir: Path | str = DEFAULT_OUTPUT_DIR,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    domain: str = "all",
) -> list[dict[str, Any]]:
    """Load pre-mined OSS JSONL records or mine local corpus_root on demand."""
    if split not in {"sft", "dpo", "grpo"}:
        raise ValueError(f"Unsupported OSS split: {split}")
    target_domain = "all" if domain in {"all", "oss"} else domain
    if target_domain == "general":
        return []
    raw_p = Path(raw_dir)
    filename = f"oss_{target_domain}_{split}.jsonl" if target_domain in SUPPORTED_ECOSYSTEMS else f"oss_{split}.jsonl"
    path = raw_p / filename
    if not path.exists() and Path(corpus_root).is_dir():
        mine_all_oss_repos(corpus_root=corpus_root, output_dir=raw_p, domain=target_domain)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine .NET, React, Go, and C/C++ OSS git histories for Gemma 4 training.")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domain", choices=["all", *SUPPORTED_ECOSYSTEMS], default="all")
    parser.add_argument("--max-commits-per-repo", type=int, default=300)
    args = parser.parse_args(argv)
    counts = mine_all_oss_repos(args.corpus_root, args.output_dir, args.domain, args.max_commits_per_repo)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
