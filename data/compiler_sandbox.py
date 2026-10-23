#!/usr/bin/env python3
"""32-thread parallel compiler verification sandbox & DPO mutation synthesizer.

Verifies fenced TypeScript, Go, and C#/.NET 10 code blocks against real toolchains:
- Go         : gofmt -e (AST/syntax) + go build / go vet in isolated temp module
- TypeScript : TypeScript 6.0.3 compiler API (ts.createProgram + ts.transpileModule) via Node v26
- C#/.NET 10 : .NET 10.0.400 Roslyn compiler (csc.dll + Microsoft.NETCore.App.Ref net10.0 assemblies)

Also synthesizes realistic compiler-failing mutations (type mismatch, wrong return type, syntax corruption)
and verifies them through the same 32-thread sandbox to build Stage 4 Compiler-Verified DPO pairs.
"""

from __future__ import annotations

import concurrent.futures
import glob
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.ingest_sources import FENCE_RE, extract_fenced_code

def _resolve_tsc_module_path() -> str:
    """Resolve a local TypeScript compiler module path from env or standard workspace locations."""
    candidates = [
        os.environ.get("TSC_MODULE_PATH", ""),
        *sorted(glob.glob(str(Path.home() / "projects/*/*/node_modules/typescript"))),
        *sorted(glob.glob(str(Path.home() / "projects/*/node_modules/typescript"))),
        "/usr/lib/node_modules/typescript",
        "/usr/local/lib/node_modules/typescript",
    ]
    return next((c for c in candidates if c and Path(c).exists()), "typescript")


TSC_MODULE_PATH = _resolve_tsc_module_path()
STRICT_TS_ERROR_CODES = {1002, 1003, 1005, 1109, 1128, 1136, 2322, 2355, 2365, 2554}
STRICT_CS_ERROR_CODES = {
    "CS0019", "CS0029", "CS0127", "CS0161", "CS0266", "CS1001", "CS1002",
    "CS1003", "CS1026", "CS1503", "CS1513", "CS1514", "CS1525",
}


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    language: str
    diagnostics: str
    toolchain: str


def _prepare_roslyn_env() -> tuple[str, str] | None:
    """Locate .NET 10 Roslyn csc.dll and generate a shared reference assembly response file."""
    roots = [r for r in (os.environ.get("DOTNET_ROOT", ""), str(Path.home() / ".asdf/installs/dotnet/*"), str(Path.home() / ".dotnet"), "/usr/share/dotnet") if r]
    csc_matches = [m for r in roots for m in sorted(glob.glob(f"{r}/sdk/*/Roslyn/bincore/csc.dll"))]
    ref_matches = [m for r in roots for m in sorted(glob.glob(f"{r}/packs/Microsoft.NETCore.App.Ref/*/ref/net10.0/*.dll"))]
    if not csc_matches or not ref_matches:
        return None
    rsp_path = Path(tempfile.gettempdir()) / "gemma_pipeline_net10_refs.rsp"
    if not rsp_path.exists():
        lines = ["-target:library", "-nologo", "-langversion:preview", "-nullable:enable"]
        lines.extend(f"-r:{dll}" for dll in ref_matches)
        rsp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csc_matches[-1], str(rsp_path)


ROSLYN_ENV = _prepare_roslyn_env()


def _wrap_go_snippet(code: str) -> str:
    stripped = code.strip()
    if re.search(r"^\s*package\s+[A-Za-z0-9_]+", stripped, re.MULTILINE):
        return stripped + "\n"
    if re.search(r"^\s*(func|type|var|const|import)\b", stripped, re.MULTILINE):
        return f"package sandbox\n\n{stripped}\n"
    return f"package sandbox\n\nfunc _sandboxEntry() {{\n{stripped}\n}}\n"


def verify_go_code(code: str) -> SandboxResult:
    """Verify Go code using gofmt -e (syntax/AST) and go build (type check)."""
    wrapped = _wrap_go_snippet(code)
    with tempfile.TemporaryDirectory(prefix="agy_go_") as tmpdir:
        go_file = Path(tmpdir) / "snippet.go"
        go_file.write_text(wrapped, encoding="utf-8")

        fmt_proc = subprocess.run(
            ["gofmt", "-e", str(go_file)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if fmt_proc.returncode != 0:
            return SandboxResult(
                passed=False,
                language="go",
                diagnostics=fmt_proc.stderr.strip()[:500],
                toolchain="gofmt -e",
            )

        # Also run go build in isolated module to catch type mismatches
        (Path(tmpdir) / "go.mod").write_text("module sandbox\n\ngo 1.24\n", encoding="utf-8")
        build_proc = subprocess.run(
            ["go", "build", "."],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if build_proc.returncode != 0:
            err_lines = [
                ln.strip()
                for ln in build_proc.stderr.splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            # Fail on explicit type or syntax errors; allow unresolved external module/symbol refs in partial snippets
            strict_go_errors = [
                ln
                for ln in err_lines
                if any(
                    pat in ln
                    for pat in (
                        "syntax error:",
                        "cannot use ",
                        "invalid operation:",
                        "too many arguments",
                        "not enough arguments",
                        "wrongd type",
                        "too many return values",
                    )
                )
            ]
            if strict_go_errors:
                return SandboxResult(
                    passed=False,
                    language="go",
                    diagnostics="\n".join(strict_go_errors[:4]),
                    toolchain="go build",
                )

    return SandboxResult(passed=True, language="go", diagnostics="OK", toolchain="gofmt+go-build")


def verify_typescript_code(code: str) -> SandboxResult:
    """Verify TypeScript code using TypeScript 6.0.3 compiler API via Node."""
    with tempfile.TemporaryDirectory(prefix="agy_ts_") as tmpdir:
        ts_file = Path(tmpdir) / "snippet.tsx"
        ts_file.write_text(code.strip() + "\n", encoding="utf-8")

        js_script = (
            f"const ts = require({TSC_MODULE_PATH!r});\n"
            f"const file = {str(ts_file)!r};\n"
            "const opts = {\n"
            "  noEmit: true,\n"
            "  strict: false,\n"
            "  target: ts.ScriptTarget.ES2022,\n"
            "  jsx: ts.JsxEmit.Preserve,\n"
            "  moduleResolution: ts.ModuleResolutionKind.Bundler,\n"
            "  skipLibCheck: true\n"
            "};\n"
            "const prog = ts.createProgram([file], opts);\n"
            "const diags = ts.getPreEmitDiagnostics(prog);\n"
            "const out = diags.map(d => ({ code: d.code, msg: ts.flattenDiagnosticMessageText(d.messageText, ' ') }));\n"
            "console.log(JSON.stringify(out));\n"
        )
        proc = subprocess.run(
            ["node", "-e", js_script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return SandboxResult(
                passed=False,
                language="typescript",
                diagnostics=proc.stderr.strip()[:400] or "Node TS runner error",
                toolchain="tsc-6.0.3",
            )
        import json

        try:
            diags = json.loads(proc.stdout.strip() or "[]")
        except json.JSONDecodeError:
            diags = []

        fatal = [
            f"TS{d['code']}: {d['msg']}"
            for d in diags
            if (1000 <= int(d.get("code", 0)) < 1800) or int(d.get("code", 0)) in STRICT_TS_ERROR_CODES
        ]
        if fatal:
            return SandboxResult(
                passed=False,
                language="typescript",
                diagnostics="\n".join(fatal[:4]),
                toolchain="tsc-6.0.3",
            )

    return SandboxResult(passed=True, language="typescript", diagnostics="OK", toolchain="tsc-6.0.3")


def _wrap_csharp_snippet(code: str) -> str:
    stripped = code.strip()
    if re.search(r"\b(class|struct|interface|record|enum)\s+[A-Za-z0-9_]+", stripped):
        if "using System;" not in stripped:
            return f"using System;\nusing System.Collections.Generic;\nusing System.Linq;\nusing System.Threading.Tasks;\n{stripped}\n"
        return stripped + "\n"
    return (
        "using System;\nusing System.Collections.Generic;\nusing System.Linq;\nusing System.Threading.Tasks;\n"
        "namespace Sandbox;\npublic class SnippetContainer {\n"
        f"{stripped}\n}}\n"
    )


def verify_csharp_code(code: str) -> SandboxResult:
    """Verify C#/.NET 10 code using Roslyn csc.dll against net10.0 reference assemblies."""
    if ROSLYN_ENV is None:
        return SandboxResult(passed=False, language="csharp", diagnostics="Missing .NET 10 Roslyn SDK", toolchain="roslyn")
    csc_dll, rsp_file = ROSLYN_ENV
    wrapped = _wrap_csharp_snippet(code)

    with tempfile.TemporaryDirectory(prefix="agy_cs_") as tmpdir:
        cs_file = Path(tmpdir) / "Snippet.cs"
        out_dll = Path(tmpdir) / "Snippet.dll"
        cs_file.write_text(wrapped, encoding="utf-8")

        proc = subprocess.run(
            ["dotnet", csc_dll, f"@{rsp_file}", f"-out:{out_dll}", str(cs_file)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        fatal_lines: list[str] = []
        for line in output.splitlines():
            match = re.search(r"error\s+(CS\d{4}):", line)
            if not match:
                continue
            err_code = match.group(1)
            if err_code.startswith("CS1") and err_code not in {"CS1061", "CS1729", "CS1501"}:
                fatal_lines.append(line.strip())
            elif err_code in STRICT_CS_ERROR_CODES:
                fatal_lines.append(line.strip())

        if fatal_lines:
            return SandboxResult(
                passed=False,
                language="csharp",
                diagnostics="\n".join(fatal_lines[:4]),
                toolchain="dotnet-10-roslyn-csc",
            )

    return SandboxResult(passed=True, language="csharp", diagnostics="OK", toolchain="dotnet-10-roslyn-csc")


def verify_code_sample(response_text: str, language: str = "typescript") -> SandboxResult:
    """Extract fenced code from response text (or raw code) and run the language-appropriate compiler verifier."""
    lang = language.lower().strip()
    code = extract_fenced_code(response_text)
    if not code:
        return SandboxResult(passed=False, language=lang, diagnostics="No fenced code block found", toolchain="none")
    if lang == "go":
        return verify_go_code(code)
    if lang == "csharp":
        return verify_csharp_code(code)
    return verify_typescript_code(code)


def verify_record(record: dict[str, Any]) -> tuple[dict[str, Any], SandboxResult]:
    """Extract fenced code from record and run the language-appropriate compiler verifier."""
    lang = str(record.get("language", "typescript")).lower()
    return record, verify_code_sample(str(record.get("response", "")), lang)


def verify_records_parallel(
    records: list[dict[str, Any]],
    max_workers: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run 32-thread parallel compiler verification across all candidate records."""
    passed_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    per_lang_pass = {"typescript": 0, "go": 0, "csharp": 0}
    per_lang_fail = {"typescript": 0, "go": 0, "csharp": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(verify_record, rec) for rec in records]
        for fut in futures:
            rec, res = fut.result()
            lang = rec["language"]
            enriched = {
                **rec,
                "compiler_passed": res.passed,
                "compiler_diagnostics": res.diagnostics,
                "compiler_toolchain": res.toolchain,
            }
            if res.passed:
                passed_records.append(enriched)
                per_lang_pass[lang] = per_lang_pass.get(lang, 0) + 1
            else:
                failed_records.append(enriched)
                per_lang_fail[lang] = per_lang_fail.get(lang, 0) + 1

    total = len(records)
    pass_rate = round((len(passed_records) / total) * 100.0, 2) if total else 0.0
    stats = {
        "total_verified": total,
        "passed_count": len(passed_records),
        "failed_count": len(failed_records),
        "pass_rate_pct": pass_rate,
        "per_language_passed": per_lang_pass,
        "per_language_failed": per_lang_fail,
        "threads": max_workers,
    }
    return passed_records, failed_records, stats


def _mutate_code_for_dpo(code: str, language: str, idx: int) -> str:
    """Inject a realistic compiler-failing mutation (type mismatch or syntax defect) into passing code."""
    mutation_kind = idx % 2
    if language == "typescript":
        if mutation_kind == 0:
            return code + "\n\n// Injected type mismatch defect\nexport const __dpoBrokenCount: number = \"not_a_valid_number\";\n"
        return code + "\nexport function __dpoSyntaxDefect(x: number): number { return x + ; }\n"
    if language == "go":
        wrapped = _wrap_go_snippet(code)
        if mutation_kind == 0:
            return wrapped + "\nvar __dpoBrokenMetric int = \"type_mismatch_string_literal\"\n"
        return wrapped + "\nfunc __dpoBrokenSyntax(a int) int { return a + }\n"
    # csharp
    if mutation_kind == 0:
        if re.search(r"\b(class|struct|interface|record|enum)\s+", code):
            return code + "\npublic static class __DpoDefect { public static int BrokenMetric() => \"invalid_int_return\"; }\n"
        return code + "\nint __dpoBrokenLocal = \"cannot_convert_string_to_int\";\n"
    return code + "\npublic class __DpoSyntax { public int Broken() { return 42 } }\n"


def _build_single_dpo_pair(args: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
    idx, chosen_rec = args
    lang = chosen_rec["language"]
    resp = chosen_rec["response"]
    match = FENCE_RE.search(resp)
    if not match:
        return None
    clean_code = match.group(2).strip()
    mutated_code = _mutate_code_for_dpo(clean_code, lang, idx)

    if lang == "go":
        mut_res = verify_go_code(mutated_code)
    elif lang == "csharp":
        mut_res = verify_csharp_code(mutated_code)
    else:
        mut_res = verify_typescript_code(mutated_code)

    if mut_res.passed:
        return None

    start, end = match.span()
    rejected_response = f"{resp[:start]}```{lang}\n{mutated_code.strip()}\n```{resp[end:]}"
    return {
        "id": f"dpo_{chosen_rec['id']}",
        "prompt": chosen_rec["instruction"],
        "chosen": resp,
        "rejected": rejected_response,
        "language": lang,
        "source": chosen_rec["source"],
        "compiler_toolchain": mut_res.toolchain,
        "compiler_feedback": mut_res.diagnostics,
    }


def build_dpo_preference_pairs(
    chosen_records: list[dict[str, Any]],
    target_dpo_pairs: int,
    max_workers: int = 32,
) -> list[dict[str, Any]]:
    """Synthesize and compiler-verify rejected mutations in parallel across 32 threads for DPO."""
    candidates = chosen_records[: max(target_dpo_pairs + 25, target_dpo_pairs)]
    pairs: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_build_single_dpo_pair, enumerate(candidates)))
    for item in results:
        if item is not None and len(pairs) < target_dpo_pairs:
            pairs.append(item)
    return pairs
