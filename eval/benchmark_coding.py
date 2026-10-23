#!/usr/bin/env python3
"""Phase 5 Head-to-Head Coding Benchmark: Base vs. Phase 3 SFT vs. Phase 4 DPO.

Evaluates all 3 checkpoints in a SINGLE 8.0B model load using PEFT multi-adapter
switching (`disable_adapter()`, `set_adapter("sft")`, `set_adapter("dpo")`) across:
1. Held-out SFT Eval Perplexity & Token NLL (`sft_eval.parquet` across TS, Go, C#/.NET 10)
2. Held-out Compiler Preference Accuracy & Log-Prob Margin (`dpo_eval.parquet`)
3. Live Deterministic Greedy (`do_sample=False`) Code Generation `Compile Pass@1 %`
   across 15 held-out synthesis & bugfix prompts (5 TypeScript 6.0.3, 5 Go 1.27, 5 C#/.NET 10 Roslyn).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_dataset import SYSTEM_PREAMBLE
from data.compiler_sandbox import verify_code_sample
from train.dpo_compiler import _encode_padded_tensors, compute_chunked_sequence_logps, load_dpo_split
from train.model_loader import (
    DEFAULT_MODEL_ID,
    build_asymmetric_device_map,
    install_chunked_causal_lm_loss,
    query_dual_gpu_vram,
    resolve_checkpoint_id,
)

# 15 Curated Held-Out Coding Prompts (5 TypeScript 6.0.3, 5 Go 1.27, 5 C#/.NET 10 Roslyn)
HELD_OUT_CODING_SUITE: list[dict[str, str]] = [
    {"id": "ts_01_lru_ttl", "language": "typescript", "category": "synthesis", "prompt": "Write a self-contained TypeScript generic function `filterDefined<T>(items: ReadonlyArray<T | null | undefined>): T[]` using a type predicate `(x): x is T`. Return only a single ```typescript fenced code block."},
    {"id": "ts_02_result_map", "language": "typescript", "category": "synthesis", "prompt": "Write a TypeScript discriminated union `type Result<T, E> = { ok: true; value: T } | { ok: false; error: E }` and a generic function `mapResult<T, U, E>(r: Result<T, E>, fn: (v: T) => U): Result<U, E>`. Return only a single ```typescript fenced code block."},
    {"id": "ts_03_bugfix_reduce", "language": "typescript", "category": "bugfix", "prompt": "Fix the TypeScript type error (`TS2322`) in this function so it returns a valid `Record<string, number>`:\n```typescript\nexport function sumByKey(items: Array<{ key: string; count: number }>): Record<string, number> {\n  return items.reduce((acc, item) => { acc[item.key] = (acc[item.key] ?? 0) + item.count; return acc; }, {} as Record<string, number>);\n}\n```\nReturn only the corrected ```typescript fenced code block."},
    {"id": "ts_04_retry_backoff", "language": "typescript", "category": "synthesis", "prompt": "Write a TypeScript function `clampScore(value: number, min: number, max: number): number` that throws a `RangeError` if `min > max` and otherwise clamps `value` to `[min, max]`. Return only a single ```typescript fenced code block."},
    {"id": "ts_05_readonly_tuple", "language": "typescript", "category": "bugfix", "prompt": "Write a TypeScript function `partitionNumbers(nums: readonly number[]): [number[], number[]]` that splits even and odd integers into a 2-tuple. Return only a single ```typescript fenced code block."},
    {"id": "go_01_worker_sum", "language": "go", "category": "synthesis", "prompt": "Write a complete Go package `sandbox` with a function `SumPositive(nums []int) int` that returns the sum of all strictly positive integers in `nums`. Return only a single ```go fenced code block."},
    {"id": "go_02_generic_keys", "language": "go", "category": "synthesis", "prompt": "Write a complete Go package `sandbox` with a generic function `MapValues[K comparable, V any](m map[K]V) []V` that collects all values from `m` into a slice. Return only a single ```go fenced code block."},
    {"id": "go_03_bugfix_slice", "language": "go", "category": "bugfix", "prompt": "Write a complete Go package `sandbox` with a function `DeduplicateInts(in []int) []int` that preserves first-seen order using a `map[int]struct{}`. Return only a single ```go fenced code block."},
    {"id": "go_04_safe_div", "language": "go", "category": "synthesis", "prompt": "Write a complete Go package `sandbox` importing `errors` with `SafeDivide(a, b float64) (float64, error)` that returns an error when `b == 0`. Return only a single ```go fenced code block."},
    {"id": "go_05_ring_counter", "language": "go", "category": "synthesis", "prompt": "Write a complete Go package `sandbox` with a struct `BoundedCounter` (`Val int`, `Max int`) and method `(c *BoundedCounter) Inc() int` that increments `Val` up to `Max`. Return only a single ```go fenced code block."},
    {"id": "cs_01_record_validation", "language": "csharp", "category": "synthesis", "prompt": "Write a C# (.NET 10) `public static class MathUtils` with a method `public static int Clamp(int val, int min, int max)` that returns `val` clamped between `min` and `max`. Return only a single ```csharp fenced code block."},
    {"id": "cs_02_linq_group", "language": "csharp", "category": "synthesis", "prompt": "Write a C# (.NET 10) `public static class WordCounter` with `public static Dictionary<string, int> CountFrequencies(IEnumerable<string> words)` that counts case-insensitive occurrences. Return only a single ```csharp fenced code block."},
    {"id": "cs_03_bugfix_nullable", "language": "csharp", "category": "bugfix", "prompt": "Write a C# (.NET 10) `public static class ArrayHelpers` with `public static int SumNonNull(IEnumerable<int?> values)` that sums all non-null elements cleanly under `#nullable enable`. Return only a single ```csharp fenced code block."},
    {"id": "cs_04_generic_result", "language": "csharp", "category": "synthesis", "prompt": "Write a C# (.NET 10) generic `public readonly record struct Outcome<T>(bool IsSuccess, T? Value, string? Error)` with static factory methods `Ok(T value)` and `Fail(string error)`. Return only a single ```csharp fenced code block."},
    {"id": "cs_05_span_palindrome", "language": "csharp", "category": "synthesis", "prompt": "Write a C# (.NET 10) `public static class StringChecks` with `public static bool IsPalindrome(string input)` that checks if `input` reads the same forward and backward. Return only a single ```csharp fenced code block."},
]


def _align_peft_devices(model: PeftModel) -> None:
    """Ensure all multi-adapter LoRA parameters sit on the same GPU and bfloat16 dtype as their base_layer."""
    for module in model.modules():
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            dev = module.base_layer.weight.device
            if hasattr(module, "lora_A"):
                module.lora_A.to(device=dev, dtype=torch.bfloat16)
            if hasattr(module, "lora_B"):
                module.lora_B.to(device=dev, dtype=torch.bfloat16)


def load_multi_adapter_benchmark_model(
    model_id: str, sft_adapter_dir: Path, dpo_adapter_dir: Path, console: Console
) -> tuple[PeftModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Load base model once across cuda:1 + cuda:0 and attach both 'sft' and 'dpo' LoRA adapters."""
    t0 = time.time()
    resolved_id = resolve_checkpoint_id(model_id)
    cfg = AutoConfig.from_pretrained(resolved_id)
    model_type = getattr(cfg, "model_type", "gemma4")
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    num_layers = getattr(text_cfg, "num_hidden_layers", 42)
    device_map = build_asymmetric_device_map(model_type=model_type, num_layers=num_layers)
    attn_arg = {"text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"} if model_type == "gemma4" else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(str(dpo_adapter_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        resolved_id, dtype=torch.bfloat16, device_map=device_map, attn_implementation=attn_arg
    )
    install_chunked_causal_lm_loss(base_model, chunk_size=256)

    console.print(f"[bold cyan]Attaching Phase 3 SFT adapter ('sft'):[/bold cyan] {sft_adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir), adapter_name="sft", is_trainable=False)
    console.print(f"[bold cyan]Attaching Phase 4 DPO adapter ('dpo'):[/bold cyan] {dpo_adapter_dir}")
    peft_model.load_adapter(str(dpo_adapter_dir), adapter_name="dpo", is_trainable=False)
    _align_peft_devices(peft_model)
    peft_model.eval()

    return peft_model, tokenizer, {
        "resolved_model_id": resolved_id,
        "model_type": model_type,
        "sft_adapter_dir": str(sft_adapter_dir),
        "dpo_adapter_dir": str(dpo_adapter_dir),
        "single_load_elapsed_sec": round(time.time() - t0, 2),
        "vram_after_multi_adapter_load": query_dual_gpu_vram(),
    }


@torch.no_grad()
def evaluate_sft_perplexity(
    model: PeftModel, sft_rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 4096
) -> dict[str, Any]:
    """Compute exact token NLL and Perplexity overall and per-language on held-out sft_eval.parquet."""
    total_nll, total_tokens = 0.0, 0.0
    lang_nll: dict[str, float] = defaultdict(float)
    lang_tokens: dict[str, float] = defaultdict(float)

    for row in sft_rows:
        lang = str(row.get("language", "typescript"))
        ids_t, mask_t, lbls_t = _encode_padded_tensors(row["instruction"], row["response"], tokenizer, max_seq_length)
        seq_logp, tok_cnt = compute_chunked_sequence_logps(model, ids_t, mask_t, lbls_t)
        nll, cnt = float(-seq_logp.item()), float(tok_cnt.item())
        total_nll += nll
        total_tokens += cnt
        lang_nll[lang] += nll
        lang_tokens[lang] += cnt

    mean_nll = total_nll / max(total_tokens, 1.0)
    per_lang = {
        lang: {
            "token_nll": round(nll_sum / max(lang_tokens[lang], 1.0), 4),
            "perplexity": round(math.exp(min(nll_sum / max(lang_tokens[lang], 1.0), 20.0)), 4),
            "tokens_evaluated": int(lang_tokens[lang]),
        }
        for lang, nll_sum in sorted(lang_nll.items())
    }
    return {
        "eval_loss_token_nll": round(mean_nll, 4),
        "eval_perplexity": round(math.exp(min(mean_nll, 20.0)), 4),
        "total_tokens_evaluated": int(total_tokens),
        "per_language": per_lang,
    }


@torch.no_grad()
def evaluate_compiler_preference(
    model: PeftModel,
    dpo_samples: list[dict[str, Any]],
    ref_logps: dict[str, tuple[float, float]] | None = None,
    beta: float = 0.1,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    """Evaluate compiler preference accuracy and log-prob / implicit reward margins on held-out dpo_eval.parquet."""
    recorded_logps: dict[str, tuple[float, float]] = {}
    raw_margins, norm_margins, norm_wins, dpo_margins, dpo_wins = [], [], [], [], []
    lang_wins: dict[str, list[float]] = defaultdict(list)

    for s in dpo_samples:
        sid, lang = str(s["id"]), str(s["language"])
        c_lp_t, c_cnt_t = compute_chunked_sequence_logps(model, s["chosen_input_ids"], s["chosen_attention_mask"], s["chosen_labels"])
        r_lp_t, r_cnt_t = compute_chunked_sequence_logps(model, s["rejected_input_ids"], s["rejected_attention_mask"], s["rejected_labels"])
        c_lp, r_lp = float(c_lp_t.item()), float(r_lp_t.item())
        c_norm, r_norm = c_lp / float(c_cnt_t.item()), r_lp / float(r_cnt_t.item())
        recorded_logps[sid] = (c_lp, r_lp)

        raw_m, norm_m = c_lp - r_lp, c_norm - r_norm
        raw_margins.append(raw_m)
        norm_margins.append(norm_m)
        norm_wins.append(1.0 if norm_m > 0.0 else 0.0)

        rew_m = beta * ((c_lp - ref_logps[sid][0]) - (r_lp - ref_logps[sid][1])) if (ref_logps and sid in ref_logps) else beta * raw_m
        dpo_margins.append(rew_m)
        win_flag = 1.0 if (rew_m > 1e-7 if ref_logps is not None else norm_m > 0.0) else 0.0
        dpo_wins.append(win_flag)
        lang_wins[lang].append(win_flag)

    n = max(len(dpo_samples), 1)
    return (
        {
            "pairs_evaluated": len(dpo_samples),
            "raw_logp_margin_mean": round(sum(raw_margins) / n, 4),
            "length_norm_logp_margin_mean": round(sum(norm_margins) / n, 4),
            "length_norm_preference_acc_pct": round((sum(norm_wins) / n) * 100.0, 2),
            "dpo_reward_margin_vs_sft_ref": round(sum(dpo_margins) / n, 4),
            "dpo_preference_acc_pct": round((sum(dpo_wins) / n) * 100.0, 2),
            "per_language_preference_acc_pct": {
                k: round((sum(v) / max(len(v), 1)) * 100.0, 2) for k, v in sorted(lang_wins.items())
            },
        },
        recorded_logps,
    )


@torch.no_grad()
def evaluate_live_compile_pass_at_1(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int = 192, batch_size: int = 5
) -> dict[str, Any]:
    """Run deterministic greedy decoding (do_sample=False) across HELD_OUT_CODING_SUITE and verify in compiler sandbox."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))
    first_dev = next(causal_lm.model.parameters()).device

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    causal_lm.gradient_checkpointing_disable()
    causal_lm.config.use_cache = True
    if hasattr(causal_lm.config, "text_config"):
        causal_lm.config.text_config.use_cache = True

    eot_id = tokenizer.convert_tokens_to_ids(eot)
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    if isinstance(eot_id, int) and eot_id > 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    generations: list[str] = []
    t_gen = time.time()
    try:
        for start in range(0, len(HELD_OUT_CODING_SUITE), batch_size):
            batch_items = HELD_OUT_CODING_SUITE[start : start + batch_size]
            chat_prompts = [f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{item['prompt'].strip()}{eot}\n{sot}model\n" for item in batch_items]
            enc = tokenizer(chat_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            in_ids, in_mask = enc["input_ids"].to(first_dev), enc["attention_mask"].to(first_dev)
            out_ids = causal_lm.generate(
                input_ids=in_ids,
                attention_mask=in_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=int(tokenizer.pad_token_id or 0),
                eos_token_id=eos_ids if eos_ids else None,
                use_cache=True,
            )
            for idx in range(len(batch_items)):
                generations.append(tokenizer.decode(out_ids[idx, in_ids.size(1) :], skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = orig_padding_side
        causal_lm.config.use_cache = False
        if hasattr(causal_lm.config, "text_config"):
            causal_lm.config.text_config.use_cache = False

    gen_elapsed = round(time.time() - t_gen, 2)

    def _grade_one(idx: int) -> dict[str, Any]:
        item, resp = HELD_OUT_CODING_SUITE[idx], generations[idx]
        res = verify_code_sample(resp, item["language"])
        return {
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "passed": res.passed,
            "toolchain": res.toolchain,
            "diagnostics": res.diagnostics,
            "response_preview": resp[:240],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as pool:
        graded = list(pool.map(_grade_one, range(len(HELD_OUT_CODING_SUITE))))

    passed_total = sum(1 for g in graded if g["passed"])
    per_lang = {
        lang: {
            "passed": sum(1 for g in graded if g["language"] == lang and g["passed"]),
            "total": sum(1 for g in graded if g["language"] == lang),
            "pass_at_1_pct": round(
                (sum(1 for g in graded if g["language"] == lang and g["passed"]) / max(sum(1 for g in graded if g["language"] == lang), 1)) * 100.0, 2
            ),
        }
        for lang in ("typescript", "go", "csharp")
    }
    return {
        "total_prompts": len(graded),
        "passed_count": passed_total,
        "compile_pass_at_1_pct": round((passed_total / max(len(graded), 1)) * 100.0, 2),
        "generation_elapsed_sec": gen_elapsed,
        "per_language": per_lang,
        "samples": graded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 Multi-Adapter Coding Benchmark.")
    parser.add_argument("--tier", choices=("smoke", "dev", "full"), default="smoke")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--dpo-adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    console = Console()
    t0 = time.time()
    torch.manual_seed(42)

    sft_dir = Path(args.sft_adapter) if args.sft_adapter else (REPO_ROOT / "checkpoints" / f"sft-{args.tier}" / "final_adapter")
    dpo_dir = Path(args.dpo_adapter) if args.dpo_adapter else (REPO_ROOT / "checkpoints" / f"dpo-{args.tier}" / "final_adapter")
    out_path = Path(args.output_path) if args.output_path else (REPO_ROOT / "outputs" / "eval" / "benchmark_scorecard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold cyan]Phase 5 Coding Benchmark (Base vs. SFT vs. DPO) — Tier: {args.tier.upper()}")
    model, tokenizer, load_meta = load_multi_adapter_benchmark_model(args.model_id, sft_dir, dpo_dir, console)

    data_dir = REPO_ROOT / "data" / "processed" / args.tier
    sft_eval_rows = pq.read_table(data_dir / "sft_eval.parquet").to_pylist()
    dpo_eval_samples = load_dpo_split(data_dir / "dpo_eval.parquet", tokenizer, max_seq_length=4096)

    console.print("[bold yellow]1/3 Evaluating Phase 3 SFT Checkpoint (`set_adapter('sft')`)...[/bold yellow]")
    model.set_adapter("sft")
    sft_ppl = evaluate_sft_perplexity(model, sft_eval_rows, tokenizer)
    sft_pref, sft_ref_logps = evaluate_compiler_preference(model, dpo_eval_samples, ref_logps=None)
    sft_gen = evaluate_live_compile_pass_at_1(model, tokenizer, max_new_tokens=args.max_new_tokens)
    sft_pref["dpo_reward_margin_vs_sft_ref"] = 0.0

    console.print("[bold yellow]2/3 Evaluating Base Model (`with model.disable_adapter():`)...[/bold yellow]")
    with model.disable_adapter():
        base_ppl = evaluate_sft_perplexity(model, sft_eval_rows, tokenizer)
        base_pref, _ = evaluate_compiler_preference(model, dpo_eval_samples, ref_logps=sft_ref_logps)
        base_gen = evaluate_live_compile_pass_at_1(model, tokenizer, max_new_tokens=args.max_new_tokens)

    console.print("[bold yellow]3/3 Evaluating Phase 4 SFT+DPO Checkpoint (`set_adapter('dpo')`)...[/bold yellow]")
    model.set_adapter("dpo")
    dpo_ppl = evaluate_sft_perplexity(model, sft_eval_rows, tokenizer)
    dpo_pref, _ = evaluate_compiler_preference(model, dpo_eval_samples, ref_logps=sft_ref_logps)
    dpo_gen = evaluate_live_compile_pass_at_1(model, tokenizer, max_new_tokens=args.max_new_tokens)

    results = {
        "base": {"label": "Base (gemma-4-E4B-it)", "perplexity": base_ppl, "preference": base_pref, "live_coding": base_gen},
        "sft": {"label": f"Phase 3 SFT ({sft_dir.parent.name})", "perplexity": sft_ppl, "preference": sft_pref, "live_coding": sft_gen},
        "dpo": {"label": f"Phase 4 SFT+DPO ({dpo_dir.parent.name})", "perplexity": dpo_ppl, "preference": dpo_pref, "live_coding": dpo_gen},
    }

    tbl = Table(title="Phase 5 Head-to-Head Coding Benchmark Scorecard")
    for col in ("Checkpoint", "Eval Token NLL", "Eval Perplexity", "Compiler Pref Acc %", "DPO Margin vs SFT", "Live Compile Pass@1"):
        tbl.add_column(col)
    for key in ("base", "sft", "dpo"):
        r = results[key]
        p_acc = r["preference"]["dpo_preference_acc_pct"] if key == "dpo" else r["preference"]["length_norm_preference_acc_pct"]
        tbl.add_row(
            r["label"],
            f"{r['perplexity']['eval_loss_token_nll']:.4f}",
            f"{r['perplexity']['eval_perplexity']:.4f}",
            f"{p_acc:.2f}%",
            f"{r['preference']['dpo_reward_margin_vs_sft_ref']:+.4f}",
            f"{r['live_coding']['passed_count']}/{r['live_coding']['total_prompts']} ({r['live_coding']['compile_pass_at_1_pct']:.2f}%)",
        )
    console.print(tbl)

    scorecard = {
        "tier": args.tier,
        "elapsed_sec": round(time.time() - t0, 2),
        "single_model_load_meta": load_meta,
        "checkpoints": results,
        "deltas": {
            "sft_vs_base_nll_delta": round(sft_ppl["eval_loss_token_nll"] - base_ppl["eval_loss_token_nll"], 4),
            "dpo_vs_base_nll_delta": round(dpo_ppl["eval_loss_token_nll"] - base_ppl["eval_loss_token_nll"], 4),
            "sft_vs_base_ppl_delta": round(sft_ppl["eval_perplexity"] - base_ppl["eval_perplexity"], 4),
            "dpo_vs_base_ppl_delta": round(dpo_ppl["eval_perplexity"] - base_ppl["eval_perplexity"], 4),
            "dpo_reward_margin_gain": dpo_pref["dpo_reward_margin_vs_sft_ref"],
            "dpo_preference_accuracy_pct": dpo_pref["dpo_preference_acc_pct"],
            "pass_at_1_pct_progression": {k: results[k]["live_coding"]["compile_pass_at_1_pct"] for k in ("base", "sft", "dpo")},
        },
        "final_vram": query_dual_gpu_vram(),
    }
    out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved benchmark scorecard:[/bold green] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
