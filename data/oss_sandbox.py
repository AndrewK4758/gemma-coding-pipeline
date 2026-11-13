"""Isolated git-worktree patch verification sandbox for .NET, React, Go, and C/C++ OSS GRPO tasks."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from data.dotnet_env import _parse_semver, discover_dotnet_sdks, resolve_dotnet_build_spec

DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_TIMEOUTS = {"dotnet": 90, "react": 60, "go": 60, "cpp": 90}


def extract_diff_block(text: str) -> str:
    """Extract a normalized unified diff string with a trailing newline, or empty string."""
    match = DIFF_FENCE_RE.search(text or "")
    candidate = match.group(1) if match else (text or "")
    stripped = candidate.strip()
    if not stripped:
        return ""
    # Require minimal unified-diff structure so prose/placeholders fail fast cleanly.
    has_diff_markers = ("diff --git " in stripped) or ("--- " in stripped and "+++ " in stripped and "@@" in stripped)
    if not has_diff_markers:
        return ""
    return stripped + "\n"


def _resolve_tsc_command(repo_dir: Path, wt_dir: Path) -> list[str] | None:
    for root in (wt_dir, repo_dir):
        local_tsc = root / "node_modules" / "typescript" / "bin" / "tsc"
        if local_tsc.exists():
            return ["node", str(local_tsc)]
    candidates = [os.environ.get("TSC_MODULE_PATH", ""), "/usr/lib/node_modules/typescript", "/usr/local/lib/node_modules/typescript"]
    for root_str in filter(None, [os.environ.get("LOCAL_PROJECTS_ROOT", ""), str(Path.home() / "workspace"), str(Path.home() / "projects")]):
        candidates.extend(sorted(glob.glob(f"{root_str}/*/node_modules/typescript")))
        candidates.extend(sorted(glob.glob(f"{root_str}/*/*/node_modules/typescript")))
    for cand in candidates:
        if cand:
            tsc_js = Path(cand) / "bin" / "tsc"
            if tsc_js.exists():
                return ["node", str(tsc_js)]
    which_tsc = shutil.which("tsc")
    return [which_tsc] if which_tsc else None


def _symlink_node_modules(repo_dir: Path, wt_dir: Path) -> None:
    for nm_src in [repo_dir / "node_modules", *repo_dir.glob("packages/*/node_modules"), *repo_dir.glob("apps/*/node_modules")]:
        if not nm_src.exists():
            continue
        rel = nm_src.relative_to(repo_dir)
        dst = wt_dir / rel
        if dst.parent.is_dir() and not dst.exists():
            os.symlink(nm_src, dst)


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: float,
    stdin_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=stdin_text,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout),
        check=False,
    )


def _verify_react_worktree(
    repo_dir: Path,
    wt_dir: Path,
    test_files: list[str] | None,
    test_filter: str,
    deadline: float,
) -> tuple[bool, str]:
    _symlink_node_modules(repo_dir, wt_dir)
    logs: list[str] = []

    if (wt_dir / "tsconfig.json").exists():
        tsc_cmd = _resolve_tsc_command(repo_dir, wt_dir)
        if not tsc_cmd:
            return False, "React verification failed: TypeScript compiler (tsc) not found"
        proc = _run_cmd([*tsc_cmd, "--noEmit", "--pretty", "false"], wt_dir, deadline - time.monotonic())
        logs.append((proc.stdout + "\n" + proc.stderr).strip())
        if proc.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]

    vitest_bin = wt_dir / "node_modules" / ".bin" / "vitest"
    jest_bin = wt_dir / "node_modules" / ".bin" / "jest"
    if vitest_bin.exists():
        cmd = [str(vitest_bin), "run", *(test_files or [])]
        if test_filter:
            cmd.extend(["-t", test_filter])
        proc = _run_cmd(cmd, wt_dir, deadline - time.monotonic())
        logs.append((proc.stdout + "\n" + proc.stderr).strip())
        if proc.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]
    elif jest_bin.exists():
        cmd = [str(jest_bin), "--ci", *(test_files or [])]
        if test_filter:
            cmd.extend(["-t", test_filter])
        proc = _run_cmd(cmd, wt_dir, deadline - time.monotonic())
        logs.append((proc.stdout + "\n" + proc.stderr).strip())
        if proc.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]

    summary = "\n".join(x for x in logs if x).strip() or "React tsc/test verification passed"
    return True, summary[-2000:]


def _verify_dotnet_worktree(
    wt_dir: Path,
    test_filter: str,
    deadline: float,
) -> tuple[bool, str]:
    spec = resolve_dotnet_build_spec(wt_dir)
    if not spec.dotnet_bin:
        return False, ".NET verification failed: dotnet CLI not found"

    has_assets = any(wt_dir.rglob("project.assets.json"))
    logs: list[str] = []
    if not has_assets:
        restore_cmd = [spec.dotnet_bin, "restore"]
        if spec.target:
            restore_cmd.append(spec.target)
        restore_cmd.append("--ignore-failed-sources")
        nuget_cache = Path.home() / ".nuget" / "packages"
        if nuget_cache.is_dir():
            restore_cmd.extend(["--source", str(nuget_cache)])
        rest = _run_cmd(restore_cmd, wt_dir, deadline - time.monotonic(), env=spec.env)
        logs.append((rest.stdout + "\n" + rest.stderr).strip())

    test_cmd = [spec.dotnet_bin, "test"]
    if spec.target:
        test_cmd.extend(["--solution", spec.target] if spec.use_mtp_solution_flag else [spec.target])
    test_cmd.extend(["--no-restore", "-v", "minimal"])
    if test_filter:
        test_cmd.extend(["--filter", test_filter])
    proc = _run_cmd(test_cmd, wt_dir, deadline - time.monotonic(), env=spec.env)
    logs.append((proc.stdout + "\n" + proc.stderr).strip())
    output = "\n".join(x for x in logs if x)[-2000:]
    return proc.returncode == 0, output or "dotnet test passed"


def _verify_go_worktree(
    wt_dir: Path,
    test_files: list[str] | None,
    test_filter: str,
    deadline: float,
) -> tuple[bool, str]:
    go_bin = shutil.which("go")
    if not go_bin:
        return False, "Go verification failed: go toolchain not found"
    pkg_targets: list[str] = []
    if test_files:
        pkg_targets = sorted({
            f"./{Path(tf).parent.as_posix()}" if Path(tf).parent.as_posix() != "." else "."
            for tf in test_files
        })
    cmd = [go_bin, "test", "-count=1", *(pkg_targets or ["./..."])]
    if test_filter:
        cmd.extend(["-run", test_filter])
    proc = _run_cmd(cmd, wt_dir, deadline - time.monotonic())
    output = (proc.stdout + "\n" + proc.stderr).strip()[-2000:]
    return proc.returncode == 0, output or "go test passed"


def _verify_cpp_worktree(
    wt_dir: Path,
    tmp_dir: Path,
    test_filter: str,
    deadline: float,
) -> tuple[bool, str]:
    logs: list[str] = []
    jobs = str(min(os.cpu_count() or 4, 8))

    if (wt_dir / "CMakeLists.txt").exists():
        cmake_bin = shutil.which("cmake")
        if not cmake_bin:
            return False, "C/C++ verification failed: cmake not found"
        bld_dir = tmp_dir / "cmake-build"
        cfg_cmd = [cmake_bin, "-S", str(wt_dir), "-B", str(bld_dir), "-DBUILD_TESTING=ON"]
        if shutil.which("ninja"):
            cfg_cmd.extend(["-G", "Ninja"])
        if (wt_dir / "ggml").is_dir() or "llama" in wt_dir.name.lower():
            cfg_cmd.extend(["-DGGML_CUDA=OFF", "-DLLAMA_BUILD_TESTS=ON"])
        cfg = _run_cmd(cfg_cmd, wt_dir, deadline - time.monotonic())
        logs.append((cfg.stdout + "\n" + cfg.stderr).strip())
        if cfg.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]
        bld = _run_cmd([cmake_bin, "--build", str(bld_dir), "--parallel", jobs], wt_dir, deadline - time.monotonic())
        logs.append((bld.stdout + "\n" + bld.stderr).strip())
        if bld.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]
        ctest_bin = shutil.which("ctest")
        if ctest_bin and (bld_dir / "CTestTestfile.cmake").exists():
            tst_cmd = [ctest_bin, "--test-dir", str(bld_dir), "--output-on-failure"]
            if test_filter:
                tst_cmd.extend(["-R", test_filter])
            tst = _run_cmd(tst_cmd, wt_dir, deadline - time.monotonic())
            logs.append((tst.stdout + "\n" + tst.stderr).strip())
            if tst.returncode != 0:
                return False, "\n".join(x for x in logs if x)[-2000:]
        return True, "\n".join(x for x in logs if x)[-2000:] or "cmake/ctest passed"

    makefile = next((wt_dir / n for n in ("Makefile", "makefile", "GNUmakefile") if (wt_dir / n).exists()), None)
    if makefile is not None:
        make_bin = shutil.which("make")
        if not make_bin:
            return False, "C/C++ verification failed: make not found"
        mf_text = makefile.read_text(encoding="utf-8", errors="ignore")
        bld = _run_cmd([make_bin, f"-j{jobs}"], wt_dir, deadline - time.monotonic())
        logs.append((bld.stdout + "\n" + bld.stderr).strip())
        if bld.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]
        for target in ("test", "check"):
            if re.search(rf"^{target}\s*:", mf_text, re.MULTILINE):
                tst = _run_cmd([make_bin, target], wt_dir, deadline - time.monotonic())
                logs.append((tst.stdout + "\n" + tst.stderr).strip())
                if tst.returncode != 0:
                    return False, "\n".join(x for x in logs if x)[-2000:]
                break
        return True, "\n".join(x for x in logs if x)[-2000:] or "make build/test passed"

    srcs = sorted(p for p in wt_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"})
    if not srcs:
        return False, "C/C++ verification failed: no build file or C/C++ source files found"
    use_cxx = any(p.suffix.lower() in {".cc", ".cpp", ".cxx"} for p in srcs)
    compiler = shutil.which("g++" if use_cxx else "gcc") or shutil.which("c++" if use_cxx else "cc")
    if not compiler:
        return False, "C/C++ verification failed: gcc/g++ not found"
    std_flag = "-std=c++20" if use_cxx else "-std=c17"
    has_main = any(re.search(r"\bint\s+main\s*\(", p.read_text(encoding="utf-8", errors="ignore")) for p in srcs)
    if has_main:
        out_bin = tmp_dir / "cpp_test_bin"
        comp = _run_cmd([compiler, std_flag, "-Wall", "-Wextra", "-I", str(wt_dir), *[str(p) for p in srcs], "-o", str(out_bin)], wt_dir, deadline - time.monotonic())
        logs.append((comp.stdout + "\n" + comp.stderr).strip())
        if comp.returncode != 0:
            return False, "\n".join(x for x in logs if x)[-2000:]
        run = _run_cmd([str(out_bin)], wt_dir, deadline - time.monotonic())
        logs.append((run.stdout + "\n" + run.stderr).strip())
        return run.returncode == 0, "\n".join(x for x in logs if x)[-2000:] or "standalone binary test passed"
    for idx, src in enumerate(srcs):
        comp = _run_cmd([compiler, std_flag, "-Wall", "-Wextra", "-I", str(wt_dir), "-c", str(src), "-o", str(tmp_dir / f"obj_{idx}.o")], wt_dir, deadline - time.monotonic())
        if comp.returncode != 0:
            return False, (comp.stdout + "\n" + comp.stderr).strip()[-2000:]
    return True, f"Compiled {len(srcs)} C/C++ source file(s)"


def verify_oss_worktree_patch(
    repo_path: str,
    parent_sha: str,
    diff_text: str,
    ecosystem: str,
    test_files: list[str] | None = None,
    test_filter: str = "",
    timeout_sec: int | None = None,
) -> tuple[bool, str]:
    """Checkout parent_sha in an isolated /tmp worktree, apply diff_text, and run toolchain tests."""
    if ecosystem not in DEFAULT_TIMEOUTS:
        return False, f"Unsupported ecosystem: {ecosystem}"

    patch = extract_diff_block(diff_text)
    if not patch:
        return False, "git apply failed: no valid unified diff found"

    repo_dir = Path(repo_path)
    if not (repo_dir / ".git").exists():
        return False, f"git apply failed: not a git repository ({repo_path})"

    effective_timeout = timeout_sec if timeout_sec is not None else DEFAULT_TIMEOUTS[ecosystem]
    deadline = time.monotonic() + effective_timeout
    tmp_dir = Path(tempfile.mkdtemp(prefix="oss-sandbox-"))
    wt_dir = tmp_dir / "wt"

    try:
        add_proc = _run_cmd(
            ["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(wt_dir), parent_sha],
            cwd=repo_dir,
            timeout=min(20.0, deadline - time.monotonic()),
        )
        if add_proc.returncode != 0:
            return False, f"git worktree add failed: {(add_proc.stderr or add_proc.stdout).strip()}"

        apply_proc = _run_cmd(
            ["git", "-C", str(wt_dir), "apply", "--whitespace=nowarn", "-"],
            cwd=wt_dir,
            timeout=min(10.0, deadline - time.monotonic()),
            stdin_text=patch,
        )
        if apply_proc.returncode != 0:
            msg = (apply_proc.stderr or apply_proc.stdout).strip()
            return False, f"git apply failed: {msg}"

        if ecosystem == "react":
            return _verify_react_worktree(repo_dir, wt_dir, test_files, test_filter, deadline)
        if ecosystem == "dotnet":
            return _verify_dotnet_worktree(wt_dir, test_filter, deadline)
        if ecosystem == "go":
            return _verify_go_worktree(wt_dir, test_files, test_filter, deadline)
        return _verify_cpp_worktree(wt_dir, tmp_dir, test_filter, deadline)
    except subprocess.TimeoutExpired as exc:
        return False, f"verification timed out after {effective_timeout}s: {exc}"
    finally:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "prune"],
            capture_output=True,
            text=True,
            check=False,
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
