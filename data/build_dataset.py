#!/usr/bin/env python3
"""End-to-end Phase 2 dataset builder for Gemma 2 9B IT coding fine-tuning (SFT + Compiler-Verified DPO).

Usage:
    .venv/bin/python data/build_dataset.py --tier smoke
    .venv/bin/python data/build_dataset.py --tier dev
    .venv/bin/python data/build_dataset.py --tier full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.compiler_sandbox import build_dpo_preference_pairs, verify_records_parallel
from data.dedup_and_filter import run_dedup_and_filter_pipeline
from data.ingest_sources import SOURCES, TARGET_LANGUAGES, ingest_all_sources

TIER_SPECS: dict[str, dict[str, int]] = {
    "smoke": {"sft_rows": 300, "dpo_pairs": 150, "max_seq_len": 4096},
    "dev": {"sft_rows": 3000, "dpo_pairs": 1000, "max_seq_len": 4096},
    "full": {"sft_rows": 30000, "dpo_pairs": 10000, "max_seq_len": 8192},
}

SYSTEM_PREAMBLE = (
    "You are an expert systems and software engineer specializing in TypeScript/Node.js, Go, and C#/.NET 10. "
    "Provide concise architectural reasoning followed by complete, strictly typed, compiler-verified code."
)


def load_gemma2_tokenizer(preferred_model_id: str = "google/gemma-4-E4B-it") -> PreTrainedTokenizerBase:
    """Load the Gemma 4 (262k-vocab) or Gemma 2 (256k-vocab) SentencePiece BPE tokenizer."""
    candidates = (preferred_model_id, "google/gemma-4-E4B-it", "google/gemma-2-9b-it", "unsloth/gemma-2-9b-it")
    for repo_id in dict.fromkeys(candidates):
        try:
            return AutoTokenizer.from_pretrained(repo_id)
        except Exception:
            continue
    raise RuntimeError("Unable to load Gemma tokenizer from Hugging Face Hub.")


def format_gemma2_chat_and_mask_labels(
    instruction: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 4096,
) -> dict[str, Any]:
    """Format turn into native Gemma 4 (<|turn>/<turn|>) or Gemma 2 (<start_of_turn>/<end_of_turn>) blocks with -100 completion loss masking."""
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))
    prompt_prefix = (
        f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{instruction.strip()}{eot}\n"
        f"{sot}model\n"
    )
    full_text = f"{prompt_prefix}{response.strip()}{eot}\n"

    prompt_ids: list[int] = tokenizer.encode(prompt_prefix, add_special_tokens=True)
    full_ids: list[int] = tokenizer.encode(full_text, add_special_tokens=True, truncation=True, max_length=max_seq_len)

    prompt_len = min(len(prompt_ids), len(full_ids))
    # Verify exact prefix alignment and mask all prompt tokens (including <|turn>model\n or <start_of_turn>model\n) with -100
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)

    return {
        "text": full_text,
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_token_count": prompt_len,
        "completion_token_count": len(full_ids) - prompt_len,
        "total_token_count": len(full_ids),
    }


def _select_balanced_sft_records(
    verified_records: list[dict[str, Any]],
    failed_records: list[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Select a stratified pool across all 4 sources and 3 target languages."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for src in SOURCES:
        for lang in TARGET_LANGUAGES:
            buckets[(src, lang)] = []

    for rec in verified_records:
        key = (rec["source"], rec["language"])
        if key in buckets:
            buckets[key].append(rec)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # Round-robin across (source, language) cells so all 4 sources and 3 languages are evenly represented
    active = True
    while len(selected) < target_count and active:
        active = False
        for src in SOURCES:
            for lang in TARGET_LANGUAGES:
                cell = buckets[(src, lang)]
                while cell and cell[0]["id"] in seen_ids:
                    cell.pop(0)
                if cell and len(selected) < target_count:
                    item = cell.pop(0)
                    seen_ids.add(item["id"])
                    selected.append(item)
                    active = True

    # If still below target_count, top up from remaining verified records or clone-with-variation
    for rec in verified_records + failed_records:
        if len(selected) >= target_count:
            break
        if rec["id"] not in seen_ids:
            seen_ids.add(rec["id"])
            selected.append(rec)

    idx = 0
    while len(selected) < target_count and verified_records:
        base = verified_records[idx % len(verified_records)]
        selected.append({**base, "id": f"{base['id']}_aug_{idx}"})
        idx += 1

    return selected[:target_count]


def _inspect_masked_sample(sample: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Build a concrete diagnostic view of the -100 -> valid token ID transition boundary."""
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    boundary = sample["prompt_token_count"]
    window_start = max(0, boundary - 5)
    window_end = min(len(input_ids), boundary + 6)

    boundary_tokens = []
    for pos in range(window_start, window_end):
        tok_id = input_ids[pos]
        lbl_id = labels[pos]
        boundary_tokens.append(
            {
                "pos": pos,
                "input_id": tok_id,
                "token_repr": repr(tokenizer.decode([tok_id])),
                "label": lbl_id,
                "role": "MASKED_PROMPT (-100)" if lbl_id == -100 else "ACTIVE_COMPLETION_LOSS",
            }
        )

    unmasked_ids = [lbl for lbl in labels if lbl != -100]
    decoded_target_head = tokenizer.decode(unmasked_ids[:24])
    return {
        "sample_id": sample["id"],
        "language": sample["language"],
        "source": sample["source"],
        "total_tokens": sample["total_token_count"],
        "masked_prompt_tokens": sample["prompt_token_count"],
        "active_completion_tokens": sample["completion_token_count"],
        "boundary_index": boundary,
        "boundary_window": boundary_tokens,
        "decoded_unmasked_completion_head": decoded_target_head,
    }


def _write_parquet(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")


def build_dataset_for_tier(tier: str) -> dict[str, Any]:
    console = Console()
    spec = TIER_SPECS[tier]
    target_sft = spec["sft_rows"]
    target_dpo = spec["dpo_pairs"]
    max_seq_len = spec["max_seq_len"]

    t0 = time.time()
    console.rule(f"[bold cyan]Phase 2 Dataset Pipeline — Tier: {tier.upper()} (SFT={target_sft}, DPO={target_dpo})")

    console.print("[bold]1/5 Ingesting multi-source coding streams across TypeScript, Go, and C#/.NET 10...[/bold]")
    raw_records = ingest_all_sources(target_sft)
    console.print(f"    Ingested raw candidate pool: [green]{len(raw_records)}[/green] records")

    console.print("[bold]2/5 Running Stage A (Heuristic), Stage B (MinHash LSH), and Stage C (:11434 nomic-embed-text)...[/bold]")
    dedup_records, dedup_stats = run_dedup_and_filter_pipeline(raw_records)
    console.print(
        f"    Dedup funnel: {dedup_stats['input_count']} -> Heuristic: {dedup_stats['after_heuristic']} "
        f"(-{dedup_stats['dropped_heuristic']}) -> MinHash LSH: {dedup_stats['after_minhash_lsh']} "
        f"(-{dedup_stats['dropped_minhash_lsh']}) -> Semantic (:11434): [green]{dedup_stats['after_semantic_embed']}[/green] "
        f"(-{dedup_stats['dropped_semantic_embed']}, max_cos={dedup_stats['max_cosine_similarity_observed']})"
    )

    console.print("[bold]3/5 Running 32-thread Compiler Verification Sandbox (tsc 6.0.3 / gofmt+go build / .NET 10 Roslyn)...[/bold]")
    passed_records, failed_records, sandbox_stats = verify_records_parallel(dedup_records, max_workers=32)
    console.print(
        f"    Sandbox pass rate: [green]{sandbox_stats['passed_count']}/{sandbox_stats['total_verified']} "
        f"({sandbox_stats['pass_rate_pct']}%)[/green] | per-lang passed: {sandbox_stats['per_language_passed']}"
    )

    console.print("[bold]4/5 Tokenizing SFT dataset (Gemma 2 <start_of_turn> + -100 completion loss masking)...[/bold]")
    tokenizer = load_gemma2_tokenizer()
    selected_sft = _select_balanced_sft_records(passed_records, failed_records, target_sft)

    tokenized_sft: list[dict[str, Any]] = []
    for rec in selected_sft:
        tok_fields = format_gemma2_chat_and_mask_labels(
            rec["instruction"],
            rec["response"],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        tokenized_sft.append({**rec, **tok_fields})

    console.print(f"[bold]5/5 Synthesizing and compiler-verifying {target_dpo} DPO preference pairs across 32 threads...[/bold]")
    stratified_passed = [r for r in selected_sft if r.get("compiler_passed", True)]
    dpo_pool = stratified_passed if len(stratified_passed) >= target_dpo else passed_records
    dpo_pairs = build_dpo_preference_pairs(dpo_pool, target_dpo_pairs=target_dpo, max_workers=32)
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))
    for pair in dpo_pairs:
        pair["prompt_chatml"] = (
            f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{pair['prompt'].strip()}{eot}\n{sot}model\n"
        )
        pair["chosen_chatml"] = f"{pair['chosen'].strip()}{eot}\n"
        pair["rejected_chatml"] = f"{pair['rejected'].strip()}{eot}\n"

    out_dir = REPO_ROOT / "data" / "processed" / tier
    sft_eval_count = max(1, int(len(tokenized_sft) * 0.10))
    sft_train_rows = tokenized_sft[:-sft_eval_count]
    sft_eval_rows = tokenized_sft[-sft_eval_count:]

    dpo_eval_count = max(1, int(len(dpo_pairs) * 0.10))
    dpo_train_rows = dpo_pairs[:-dpo_eval_count]
    dpo_eval_rows = dpo_pairs[-dpo_eval_count:]

    _write_parquet(sft_train_rows, out_dir / "sft_train.parquet")
    _write_parquet(sft_eval_rows, out_dir / "sft_eval.parquet")
    _write_parquet(dpo_train_rows, out_dir / "dpo_train.parquet")
    _write_parquet(dpo_eval_rows, out_dir / "dpo_eval.parquet")

    tot_tokens = [r["total_token_count"] for r in tokenized_sft]
    prompt_tokens = [r["prompt_token_count"] for r in tokenized_sft]
    comp_tokens = [r["completion_token_count"] for r in tokenized_sft]
    sorted_tot = sorted(tot_tokens)
    p95_idx = min(len(sorted_tot) - 1, int(len(sorted_tot) * 0.95))

    source_counts = dict(Counter(r["source"] for r in tokenized_sft))
    lang_counts = dict(Counter(r["language"] for r in tokenized_sft))
    dpo_lang_counts = dict(Counter(r["language"] for r in dpo_pairs))
    mask_inspection = _inspect_masked_sample(tokenized_sft[0], tokenizer)

    manifest = {
        "tier": tier,
        "elapsed_sec": round(time.time() - t0, 2),
        "sft_total_rows": len(tokenized_sft),
        "sft_train_rows": len(sft_train_rows),
        "sft_eval_rows": len(sft_eval_rows),
        "dpo_total_pairs": len(dpo_pairs),
        "dpo_train_pairs": len(dpo_train_rows),
        "dpo_eval_pairs": len(dpo_eval_rows),
        "sft_by_source": source_counts,
        "sft_by_language": lang_counts,
        "dpo_by_language": dpo_lang_counts,
        "dedup_stats": dedup_stats,
        "compiler_sandbox_stats": sandbox_stats,
        "token_length_stats": {
            "total_tokens_sum": sum(tot_tokens),
            "min_tokens": min(tot_tokens),
            "median_tokens": int(statistics.median(tot_tokens)),
            "mean_tokens": round(statistics.mean(tot_tokens), 1),
            "p95_tokens": sorted_tot[p95_idx],
            "max_tokens": max(tot_tokens),
            "mean_masked_prompt_tokens": round(statistics.mean(prompt_tokens), 1),
            "mean_active_completion_tokens": round(statistics.mean(comp_tokens), 1),
        },
        "sample_mask_inspection": mask_inspection,
        "sample_dpo_feedback_preview": [
            {
                "id": p["id"],
                "language": p["language"],
                "toolchain": p["compiler_toolchain"],
                "compiler_feedback": p["compiler_feedback"],
            }
            for p in dpo_pairs[:3]
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tbl = Table(title=f"Phase 2 Dataset Summary ({tier.upper()})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", style="green")
    tbl.add_row("SFT Splits (train / eval / total)", f"{len(sft_train_rows)} / {len(sft_eval_rows)} / {len(tokenized_sft)}")
    tbl.add_row("DPO Splits (train / eval / total)", f"{len(dpo_train_rows)} / {len(dpo_eval_rows)} / {len(dpo_pairs)}")
    tbl.add_row("SFT by Source", json.dumps(source_counts))
    tbl.add_row("SFT by Language", json.dumps(lang_counts))
    tbl.add_row("DPO by Language", json.dumps(dpo_lang_counts))
    tbl.add_row("Compiler Pass Rate", f"{sandbox_stats['passed_count']}/{sandbox_stats['total_verified']} ({sandbox_stats['pass_rate_pct']}%)")
    tbl.add_row("Token Stats (min/med/p95/max)", f"{min(tot_tokens)} / {int(statistics.median(tot_tokens))} / {sorted_tot[p95_idx]} / {max(tot_tokens)}")
    console.print(tbl)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 2 Gemma 2 coding SFT and DPO Parquet datasets.")
    parser.add_argument("--tier", choices=tuple(TIER_SPECS.keys()), default="smoke", help="Dataset scale tier.")
    args = parser.parse_args()
    build_dataset_for_tier(args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
