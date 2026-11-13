"""Shared .NET SDK discovery, global.json / workflow version routing, and solution target detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

_SemverKey = tuple[int, int, int, int, tuple[tuple[int, int | str], ...]]


@dataclass(frozen=True)
class DotnetBuildSpec:
    """Resolved .NET SDK executable, root env, target solution, and MTP flag for one repo."""

    dotnet_bin: str
    dotnet_root: str
    target: str | None
    env: dict[str, str]
    use_mtp_solution_flag: bool = False


def _parse_semver(ver: str) -> _SemverKey:
    """Parse X.Y.Z[-pre] into a comparable tuple where GA > prerelease."""
    clean = ver.strip().lstrip("v=")
    main, _, pre = clean.partition("-")
    parts = [int(p) if p.isdigit() else 0 for p in (main.split(".") + ["0", "0", "0"])[:3]]
    if not pre:
        return (parts[0], parts[1], parts[2], 1, ())
    pre_toks = tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"[.\-]", pre) if p)
    return (parts[0], parts[1], parts[2], 0, pre_toks)


def read_tool_versions_dotnet() -> str | None:
    """Return the user's configured `dotnet` version from `~/.tool-versions`, if set."""
    tv = Path.home() / ".tool-versions"
    if not tv.is_file():
        return None
    for line in tv.read_text(encoding="utf-8", errors="replace").splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "dotnet":
            return toks[1]
    return None


def ordered_dotnet_roots() -> list[Path]:
    """Return candidate .NET installation roots ordered by active config, asdf installs, and fallbacks."""
    roots: list[Path] = []
    env_root = os.environ.get("DOTNET_ROOT", "")
    tv_ver = read_tool_versions_dotnet()
    asdf_dir = Path.home() / ".asdf" / "installs" / "dotnet"
    if env_root and not (tv_ver and env_root.startswith(str(asdf_dir))):
        roots.append(Path(env_root))
    if tv_ver and (asdf_dir / tv_ver).is_dir():
        roots.append(asdf_dir / tv_ver)
    if asdf_dir.is_dir():
        roots.extend(sorted(asdf_dir.iterdir()))
    if env_root:
        roots.append(Path(env_root))
    roots.extend([Path.home() / ".dotnet", Path("/usr/share/dotnet")])
    return list(dict.fromkeys(r for r in roots if r.is_dir()))


def discover_dotnet_sdks() -> list[tuple[_SemverKey, Path]]:
    """Discover installed `(semver_key, root_dir)` SDKs ordered newest first."""
    found: dict[tuple[str, str], tuple[_SemverKey, Path]] = {}
    for root in ordered_dotnet_roots():
        for csc in sorted(root.glob("sdk/*/Roslyn/bincore/csc.dll")):
            sdk_ver = csc.parents[2].name
            found[(str(root), sdk_ver)] = (_parse_semver(sdk_ver), root)
    return sorted(found.values(), key=lambda x: x[0], reverse=True)


def _match_sdk(sdks: list[tuple[_SemverKey, Path]], req_ver: str, roll_forward: str, allow_pre: bool) -> Path | None:
    req = _parse_semver(req_ver)
    rf = roll_forward.lower()
    ok: list[tuple[_SemverKey, Path]] = []
    for k, root in sdks:
        if not allow_pre and k[3] == 0:
            continue
        if rf == "disable":
            if k == req:
                ok.append((k, root))
        elif k >= req:
            same_patch_band = k[0] == req[0] and k[1] == req[1] and (k[2] // 100) == (req[2] // 100)
            if (
                ("patch" in rf and same_patch_band)
                or ("feature" in rf and k[0] == req[0] and k[1] == req[1])
                or ("minor" in rf and k[0] == req[0])
                or ("major" in rf)
            ):
                ok.append((k, root))
    if not ok:
        return None
    return max(ok, key=lambda x: x[0])[1] if rf.startswith("latest") else min(ok, key=lambda x: x[0])[1]


def _discover_dotnet_target(work: Path) -> str | None:
    wf_dir = work / ".github" / "workflows"
    if wf_dir.is_dir():
        for wf in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
            text = wf.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"\bdotnet\s+(?:build|restore|test)\b[^\n;|&]*?\s['\"]?([^\s\"'`]+\.sln[fx]?)['\"]?(?=\s|$)", text):
                rel = m.group(1).lstrip("./")
                if (work / rel).is_file():
                    return rel
    for pat in ("*.slnx", "*.sln", "*/*.slnx", "*/*.sln", "*/*/*.slnx", "*/*/*.sln"):
        hits = sorted(work.glob(pat))
        if hits:
            return hits[0].relative_to(work).as_posix()
    return None


def resolve_dotnet_build_spec(repo_dir: Path, wt_dir: Path | None = None) -> DotnetBuildSpec:
    """Resolve repo-asked .NET SDK root, PATH/DOTNET_ROOT env, and target without modifying repo files."""
    work = wt_dir or repo_dir
    sdks = discover_dotnet_sdks()
    selected_root: Path | None = None
    use_mtp = False

    gj = work / "global.json"
    if gj.is_file():
        try:
            data = json.loads(gj.read_text(encoding="utf-8"))
            sdk_cfg = data.get("sdk") or {}
            req_ver = str(sdk_cfg.get("version") or "").strip()
            if req_ver:
                selected_root = _match_sdk(
                    sdks, req_ver, str(sdk_cfg.get("rollForward") or "latestPatch"), bool(sdk_cfg.get("allowPrerelease", True))
                )
            use_mtp = (data.get("test") or {}).get("runner") == "Microsoft.Testing.Platform"
        except (OSError, json.JSONDecodeError):
            pass

    if selected_root is None and (work / ".github" / "workflows").is_dir():
        for wf in sorted(list((work / ".github" / "workflows").glob("*.yml")) + list((work / ".github" / "workflows").glob("*.yaml"))):
            matches = re.findall(r"dotnet-version:\s*['\"]?(\d+\.\d+)(?:\.[x0-9*]+)?['\"]?", wf.read_text(encoding="utf-8", errors="replace"))
            for mm in sorted(matches, key=_parse_semver, reverse=True):
                selected_root = _match_sdk(sdks, f"{mm}.0", "latestFeature", False)
                if selected_root is not None:
                    break
            if selected_root is not None:
                break

    if selected_root is None and sdks:
        tv_ver = read_tool_versions_dotnet()
        tv_root = Path.home() / ".asdf" / "installs" / "dotnet" / tv_ver if tv_ver else None
        selected_root = tv_root if (tv_root and tv_root.is_dir()) else sdks[0][1]

    env = dict(os.environ)
    if selected_root and (selected_root / "dotnet").is_file():
        dotnet_bin = str(selected_root / "dotnet")
        dotnet_root = str(selected_root)
        env["DOTNET_ROOT"] = dotnet_root
        asdf_dotnet_dir = str(Path.home() / ".asdf" / "installs" / "dotnet")
        path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and asdf_dotnet_dir not in p]
        env["PATH"] = os.pathsep.join([dotnet_root, *path_parts])
    else:
        dotnet_bin = shutil.which("dotnet") or ""
        dotnet_root = env.get("DOTNET_ROOT", "")
    return DotnetBuildSpec(dotnet_bin=dotnet_bin, dotnet_root=dotnet_root, target=_discover_dotnet_target(work), env=env, use_mtp_solution_flag=use_mtp)


if __name__ == "__main__":
    import sys

    repo_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    spec = resolve_dotnet_build_spec(repo_arg)
    print(f"{spec.dotnet_bin}\t{spec.dotnet_root}\t{spec.target or ''}")
