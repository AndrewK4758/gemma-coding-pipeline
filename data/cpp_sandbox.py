"""GCC / G++ compiler verification and DPO mutation helpers for C and C++ snippets."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def check_cpp_snippet(code: str, language: str = "cpp") -> tuple[bool, str, str]:
    """Verify a C or C++ code snippet with gcc (-std=c17) or g++ (-std=c++20)."""
    lang = language.lower().strip()
    is_c = lang == "c"
    compiler_name = "gcc" if is_c else "g++"
    compiler = shutil.which(compiler_name) or shutil.which("cc" if is_c else "c++")
    toolchain = "gcc-c17" if is_c else "g++-cpp20"
    if not compiler:
        return False, f"Compiler not found: {compiler_name}", toolchain

    ext = ".c" if is_c else ".cpp"
    std_flag = "-std=c17" if is_c else "-std=c++20"
    with tempfile.TemporaryDirectory(prefix="sandbox_cpp_") as tmpdir:
        src_path = Path(tmpdir) / f"snippet{ext}"
        obj_path = Path(tmpdir) / "snippet.o"
        src_path.write_text(code.strip() + "\n", encoding="utf-8")
        proc = subprocess.run(
            [compiler, std_flag, "-Wall", "-Wextra", "-c", str(src_path), "-o", str(obj_path)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            err_lines = [line.strip() for line in proc.stderr.splitlines() if "error:" in line]
            if not err_lines:
                err_lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
            return False, "\n".join(err_lines[:4])[:400], toolchain
    return True, "OK", toolchain


def mutate_cpp_snippet(code: str, idx: int) -> str:
    """Inject a C/C++ type-mismatch or syntax defect into compiler-passing code for DPO."""
    if idx % 2 == 0:
        return code + '\n\n/* Injected type mismatch */\nstatic int __dpo_broken_count(void) { return "not_an_int"; }\n'
    return code + "\nstatic int __dpo_syntax_defect(int x) { return x + ; }\n"
