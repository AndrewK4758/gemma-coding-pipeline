"""Tests for per-repo .NET SDK routing, solution discovery, and multi-version Roslyn probing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _touch_fake_sdk(root: Path, ver: str, tfm: str) -> Path:
    csc = root / "sdk" / ver / "Roslyn" / "bincore" / "csc.dll"
    csc.parent.mkdir(parents=True, exist_ok=True)
    csc.write_bytes(b"")
    ref = root / "packs" / "Microsoft.NETCore.App.Ref" / ver / "ref" / tfm / "System.Runtime.dll"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"")
    exe = root / "dotnet"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return root


class TestDotnetSdkRouting(unittest.TestCase):
    def test_resolve_dotnet_build_spec_honors_global_json_and_workflows(self) -> None:
        from data.oss_sandbox import _parse_semver, resolve_dotnet_build_spec

        self.assertLess(_parse_semver("11.0.100-preview.7.26381.103"), _parse_semver("11.0.100"))
        self.assertLess(_parse_semver("10.0.400"), _parse_semver("10.0.403"))
        self.assertLess(_parse_semver("10.0.403"), _parse_semver("10.0.404"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            sdks = [
                (_parse_semver("8.0.424"), _touch_fake_sdk(tmp_p / "sdks" / "8.0.424", "8.0.424", "net8.0")),
                (_parse_semver("10.0.400"), _touch_fake_sdk(tmp_p / "sdks" / "10.0.400", "10.0.400", "net10.0")),
                (_parse_semver("10.0.404"), _touch_fake_sdk(tmp_p / "sdks" / "10.0.404", "10.0.404", "net10.0")),
                (
                    _parse_semver("11.0.100-preview.7.26381.103"),
                    _touch_fake_sdk(tmp_p / "sdks" / "11.0.100-preview.7.26381.103", "11.0.100-preview.7.26381.103", "net11.0"),
                ),
                (_parse_semver("11.0.100"), _touch_fake_sdk(tmp_p / "sdks" / "11.0.100", "11.0.100", "net11.0")),
            ]

            with mock.patch("data.dotnet_env.discover_dotnet_sdks", return_value=sdks):
                # 1. Explicit 11.0.100 in global.json skips 11.0.100-preview.7 and selects GA 11.0.100
                r11 = tmp_p / "r11"
                r11.mkdir()
                (r11 / "global.json").write_text(
                    json.dumps({"sdk": {"version": "11.0.100", "rollForward": "latestMajor", "allowPrerelease": True}}),
                    encoding="utf-8",
                )
                (r11 / "Clean.Architecture.slnx").write_text("", encoding="utf-8")
                spec11 = resolve_dotnet_build_spec(r11)
                self.assertEqual(Path(spec11.dotnet_root).name, "11.0.100")
                self.assertEqual(spec11.target, "Clean.Architecture.slnx")

                # 2. Explicit 10.0.403 + latestFeature chooses 10.0.404 (not 10.0.400 or 11.0.100)
                r10_403 = tmp_p / "r10_403"
                r10_403.mkdir()
                (r10_403 / "global.json").write_text(
                    json.dumps({"sdk": {"version": "10.0.403", "rollForward": "latestFeature"}}),
                    encoding="utf-8",
                )
                (r10_403 / "App.slnx").write_text("", encoding="utf-8")
                spec10 = resolve_dotnet_build_spec(r10_403)
                self.assertEqual(Path(spec10.dotnet_root).name, "10.0.404")
                self.assertEqual(spec10.target, "App.slnx")

                # 3. Workflow .slnf filter + 10.0.302 allowPrerelease=false
                reshop = tmp_p / "reshop"
                (reshop / ".github" / "workflows").mkdir(parents=True)
                (reshop / "global.json").write_text(
                    json.dumps({"sdk": {"version": "10.0.302", "rollForward": "latestFeature", "allowPrerelease": False}}),
                    encoding="utf-8",
                )
                (reshop / "eShop.slnx").write_text("", encoding="utf-8")
                (reshop / "eShop.Web.slnf").write_text("{}", encoding="utf-8")
                (reshop / ".github" / "workflows" / "pr-validation.yml").write_text(
                    "steps:\n  - run: dotnet build eShop.Web.slnf\n", encoding="utf-8"
                )
                spec_eshop = resolve_dotnet_build_spec(reshop)
                self.assertEqual(Path(spec_eshop.dotnet_root).name, "10.0.404")
                self.assertEqual(spec_eshop.target, "eShop.Web.slnf")

                # 4. No global.json, workflow dotnet-version: '8.0.x' + subdir src/*.sln
                r8 = tmp_p / "r8"
                (r8 / ".github" / "workflows").mkdir(parents=True)
                (r8 / "src").mkdir()
                (r8 / "src" / "Meetings.sln").write_text("", encoding="utf-8")
                (r8 / ".github" / "workflows" / "build.yml").write_text(
                    "with:\n  dotnet-version: '8.0.x'\n", encoding="utf-8"
                )
                spec8 = resolve_dotnet_build_spec(r8)
                self.assertEqual(Path(spec8.dotnet_root).name, "8.0.424")
                self.assertEqual(spec8.target, "src/Meetings.sln")

    def test_compiler_sandbox_selects_net11_roslyn_and_pairs_matching_root(self) -> None:
        from data.compiler_sandbox import _prepare_roslyn_env, verify_csharp_code

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            root11 = _touch_fake_sdk(tmp_p / "11.0.100", "11.0.100", "net11.0")
            root10 = _touch_fake_sdk(tmp_p / "10.0.103", "10.0.103", "net10.0")
            env_info = _prepare_roslyn_env(preferred_roots=[str(root11), str(root10)])
            self.assertIsNotNone(env_info)
            assert env_info is not None
            csc_dll, rsp_path, toolchain, dotnet_bin = env_info
            self.assertIn("11.0.100", csc_dll)
            self.assertIn("net11.0", Path(rsp_path).read_text(encoding="utf-8"))
            self.assertEqual(toolchain, "dotnet-11-roslyn-csc")
            self.assertEqual(Path(dotnet_bin), root11 / "dotnet")

        res = verify_csharp_code("public sealed class SmokeSample { public int Sum(int a, int b) => a + b; }")
        self.assertTrue(res.passed, msg=res.diagnostics)
        self.assertIn("roslyn", res.toolchain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
