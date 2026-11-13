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
from data.mine_oss_repos import DEFAULT_CORPUS_ROOT, load_oss_records

TIER_SPECS: dict[str, dict[str, int]] = {
    "smoke": {"sft_rows": 300, "dpo_pairs": 150, "max_seq_len": 4096},
    "dev": {"sft_rows": 3000, "dpo_pairs": 1000, "max_seq_len": 4096},
    "full": {"sft_rows": 30000, "dpo_pairs": 10000, "max_seq_len": 8192},
}

SYSTEM_PREAMBLE = (
    "You are an expert systems and software engineer specializing in .NET/C#, React/TypeScript, Go, and C/C++. "
    "Provide concise architectural reasoning followed by complete, strictly typed, compiler-verified code or unified diffs."
)


def resolve_processed_dir(
    tier: str, domain: str = "all", oss_only: bool = False, output_dir: Path | str | None = None
) -> Path:
    """Resolve deterministic dataset directory for a tier and domain filter."""
    if output_dir:
        return Path(output_dir)
    if domain == "all" and not oss_only:
        return REPO_ROOT / "data" / "processed" / tier
    suffix = "oss" if (oss_only and domain == "all") else domain
    return REPO_ROOT / "data" / "processed" / f"{tier}-{suffix}"


def load_gemma2_tokenizer(preferred_model_id: str = "google/gemma-4-E4B-it") -> PreTrainedTokenizerBase:
    """Load the Gemma 4 (262k-vocab) or Gemma 2 (256k-vocab) SentencePiece BPE tokenizer."""
    candidates = (preferred_model_id, "google/gemma-4-E4B-it", "google/gemma-2-9b-it", "unsloth/gemma-2-9b-it")
    for repo_id in dict.fromkeys(candidates):
        try:
            return AutoTokenizer.from_pretrained(repo_id)
        except Exception:
            continue
    raise RuntimeError("Unable to load Gemma tokenizer from Hugging Face Hub.")


def format_gemma_messages_and_mask_labels(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 4096,
) -> dict[str, Any]:
    """Format multi-turn messages into native Gemma 4/2 turns with assistant-only completion loss masking."""
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))

    sys_text = next((str(m.get("content", "")).strip() for m in messages if m.get("role") == "system"), "")
    preamble = sys_text or SYSTEM_PREAMBLE
    convo = [m for m in messages if m.get("role") != "system"]

    running_text = ""
    labels: list[int] = []
    injected_preamble = False
    for msg in convo:
        role = str(msg.get("role", "user")).strip().lower()
        body = str(msg.get("content", "")).strip()
        if role in ("assistant", "model"):
            running_text += f"{sot}model\n"
            header_ids = tokenizer.encode(running_text, add_special_tokens=True)
            labels.extend([-100] * max(0, len(header_ids) - len(labels)))
            running_text += f"{body}{eot}\n"
            turn_ids = tokenizer.encode(running_text, add_special_tokens=True)
            delta = len(turn_ids) - len(labels)
            if delta > 0:
                labels.extend(turn_ids[-delta:])
        else:
            if not injected_preamble:
                body = f"{preamble}\n\n{body}"
                injected_preamble = True
            running_text += f"{sot}user\n{body}{eot}\n"
            user_ids = tokenizer.encode(running_text, add_special_tokens=True)
            labels.extend([-100] * max(0, len(user_ids) - len(labels)))

    full_ids: list[int] = tokenizer.encode(
        running_text, add_special_tokens=True, truncation=True, max_length=max_seq_len
    )
    labels = (labels + [-100] * max(0, len(full_ids) - len(labels)))[: len(full_ids)]
    masked_count = sum(1 for lbl in labels if lbl == -100)
    return {
        "text": running_text,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_token_count": masked_count,
        "completion_token_count": len(full_ids) - masked_count,
        "total_token_count": len(full_ids),
    }


def format_gemma2_chat_and_mask_labels(
    instruction: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 4096,
) -> dict[str, Any]:
    """Format a single-turn pair into native Gemma 4/2 turns with -100 completion loss masking."""
    return format_gemma_messages_and_mask_labels(
        [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )


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
    """Summarize the -100 -> valid token ID transition boundary."""
    input_ids, labels, boundary = sample["input_ids"], sample["labels"], sample["prompt_token_count"]
    w_start, w_end = max(0, boundary - 4), min(len(input_ids), boundary + 4)
    window = [
        {"pos": p, "tok": repr(tokenizer.decode([input_ids[p]])), "role": "MASK(-100)" if labels[p] == -100 else "LOSS"}
        for p in range(w_start, w_end)
    ]
    unmasked = [lbl for lbl in labels if lbl != -100]
    return {
        "sample_id": sample["id"],
        "language": sample["language"],
        "source": sample["source"],
        "total_tokens": sample["total_token_count"],
        "masked_prompt_tokens": sample["prompt_token_count"],
        "active_completion_tokens": sample["completion_token_count"],
        "boundary_index": boundary,
        "boundary_window": window,
        "decoded_unmasked_completion_head": tokenizer.decode(unmasked[:24]),
    }


def _write_parquet(rows: list[dict[str, Any]], out_path: Path, empty_schema: pa.Schema | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=empty_schema)
    pq.write_table(table, out_path, compression="zstd")


def _build_sft_parquet_row(rec: dict[str, Any], tok_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(rec.get("id", "")),
        "source": str(rec.get("source", "")),
        "language": str(rec.get("language", "")),
        "record_type": str(rec.get("record_type", "snippet")),
        "domain": str(rec.get("domain", "general")),
        "instruction": str(rec.get("instruction", "")),
        "response": str(rec.get("response", "")),
        "compiler_passed": bool(rec.get("compiler_passed", True)),
        **tok_fields,
    }


def _build_dpo_parquet_row(pair: dict[str, Any], sot: str, eot: str) -> dict[str, Any]:
    return {
        "id": str(pair.get("id", "")),
        "source": str(pair.get("source", "")),
        "language": str(pair.get("language", "")),
        "record_type": str(pair.get("record_type", "snippet")),
        "domain": str(pair.get("domain", "general")),
        "prompt": str(pair["prompt"]),
        "chosen": str(pair["chosen"]),
        "rejected": str(pair["rejected"]),
        "compiler_toolchain": str(pair.get("compiler_toolchain", "")),
        "compiler_feedback": str(pair.get("compiler_feedback", "")),
        "prompt_chatml": f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{str(pair['prompt']).strip()}{eot}\n{sot}model\n",
        "chosen_chatml": f"{str(pair['chosen']).strip()}{eot}\n",
        "rejected_chatml": f"{str(pair['rejected']).strip()}{eot}\n",
    }


def build_dataset_for_tier(
    tier: str,
    domain: str = "all",
    oss_only: bool = False,
    oss_corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    output_dir: Path | str | None = None,
    oss_raw_dir: Path | str | None = None,
) -> dict[str, Any]:
    console = Console()
    spec = TIER_SPECS[tier]
    target_sft = spec["sft_rows"]
    target_dpo = spec["dpo_pairs"]
    max_seq_len = spec["max_seq_len"]
    out_dir = resolve_processed_dir(tier, domain=domain, oss_only=oss_only, output_dir=output_dir)
    raw_oss = Path(oss_raw_dir) if oss_raw_dir else (out_dir / "_raw_oss" if output_dir else REPO_ROOT / "data" / "raw")
    t0 = time.time()
    console.rule(f"[bold cyan]Dataset Pipeline — Tier: {tier.upper()} | Domain: {domain} (SFT={target_sft}, DPO={target_dpo})")
    raw_records = ingest_all_sources(
        target_sft, domain=domain, oss_only=oss_only, oss_raw_dir=raw_oss, oss_corpus_root=oss_corpus_root
    )
    if not raw_records:
        raise RuntimeError(f"Zero records ingested for domain={domain!r}, oss_only={oss_only}. Run scripts/clone_oss_corpus.zsh first.")
    dedup_records, dedup_stats = run_dedup_and_filter_pipeline(raw_records)
    passed_records, failed_records, sandbox_stats = verify_records_parallel(dedup_records, max_workers=32)
    tokenizer = load_gemma2_tokenizer()
    selected_sft = _select_balanced_sft_records(passed_records, failed_records, min(target_sft, len(passed_records) + len(failed_records)))

    tokenized_sft: list[dict[str, Any]] = []
    for rec in selected_sft:
        msgs = rec.get("messages")
        tok_fields = (
            format_gemma_messages_and_mask_labels(msgs, tokenizer=tokenizer, max_seq_len=max_seq_len)
            if isinstance(msgs, list) and msgs
            else format_gemma2_chat_and_mask_labels(rec["instruction"], rec["response"], tokenizer=tokenizer, max_seq_len=max_seq_len)
        )
        tokenized_sft.append(_build_sft_parquet_row(rec, tok_fields))

    console.print(f"[bold]5/5 Building DPO preference pairs & GRPO worktree tasks...[/bold]")
    oss_dpo = load_oss_records("dpo", raw_dir=raw_oss, corpus_root=oss_corpus_root, domain=domain)
    dpo_pool = [r for r in selected_sft if r.get("compiler_passed", True) and r.get("record_type") != "oss_diff"]
    synth_dpo = build_dpo_preference_pairs(dpo_pool, target_dpo_pairs=max(0, target_dpo - len(oss_dpo)), max_workers=32) if dpo_pool else []
    combined_dpo = (oss_dpo + synth_dpo)[:target_dpo]
    sot = str(tokenizer.special_tokens_map.get("sot_token", "<start_of_turn>"))
    eot = str(tokenizer.special_tokens_map.get("eot_token", "<end_of_turn>"))
    dpo_rows = [_build_dpo_parquet_row(pair, sot, eot) for pair in combined_dpo]

    grpo_raw = load_oss_records("grpo", raw_dir=raw_oss, corpus_root=oss_corpus_root, domain=domain)
    grpo_rows = [
        {**g, "prompt_chatml": f"{sot}user\n{SYSTEM_PREAMBLE}\n\n{str(g['prompt']).strip()}{eot}\n{sot}model\n"}
        for g in grpo_raw
    ]

    sft_eval_n = max(1, int(len(tokenized_sft) * 0.10)) if len(tokenized_sft) > 1 else 0
    sft_train_rows, sft_eval_rows = (tokenized_sft[:-sft_eval_n], tokenized_sft[-sft_eval_n:]) if sft_eval_n else (tokenized_sft, tokenized_sft)
    dpo_eval_n = max(1, int(len(dpo_rows) * 0.10)) if len(dpo_rows) > 1 else 0
    dpo_train_rows, dpo_eval_rows = (dpo_rows[:-dpo_eval_n], dpo_rows[-dpo_eval_n:]) if dpo_eval_n else (dpo_rows, dpo_rows)

    dpo_schema = pa.schema([(k, pa.string()) for k in ("id", "source", "language", "record_type", "domain", "prompt", "chosen", "rejected", "compiler_toolchain", "compiler_feedback", "prompt_chatml", "chosen_chatml", "rejected_chatml")])
    grpo_schema = pa.schema([("id", pa.string()), ("source", pa.string()), ("record_type", pa.string()), ("domain", pa.string()), ("ecosystem", pa.string()), ("language", pa.string()), ("repo_name", pa.string()), ("repo_path", pa.string()), ("parent_sha", pa.string()), ("commit_sha", pa.string()), ("test_files", pa.list_(pa.string())), ("prompt", pa.string()), ("reference_diff", pa.string()), ("prompt_chatml", pa.string())])

    _write_parquet(sft_train_rows, out_dir / "sft_train.parquet")
    _write_parquet(sft_eval_rows, out_dir / "sft_eval.parquet")
    _write_parquet(dpo_train_rows, out_dir / "dpo_train.parquet", empty_schema=dpo_schema)
    _write_parquet(dpo_eval_rows, out_dir / "dpo_eval.parquet", empty_schema=dpo_schema)
    _write_parquet(grpo_rows, out_dir / "grpo_tasks.parquet", empty_schema=grpo_schema)

    tot_tokens = [r["total_token_count"] for r in tokenized_sft]
    prompt_tokens = [r["prompt_token_count"] for r in tokenized_sft]
    comp_tokens = [r["completion_token_count"] for r in tokenized_sft]
    sorted_tot = sorted(tot_tokens)
    p95_idx = min(len(sorted_tot) - 1, int(len(sorted_tot) * 0.95))

    source_counts = dict(Counter(r["source"] for r in tokenized_sft))
    lang_counts = dict(Counter(r["language"] for r in tokenized_sft))
    dpo_lang_counts = dict(Counter(r["language"] for r in dpo_rows))
    mask_inspection = _inspect_masked_sample(tokenized_sft[0], tokenizer)

    manifest = {
        "tier": tier,
        "domain": domain,
        "oss_only": oss_only,
        "output_dir": str(out_dir),
        "elapsed_sec": round(time.time() - t0, 2),
        "sft_total_rows": len(tokenized_sft),
        "sft_train_rows": len(sft_train_rows),
        "sft_eval_rows": len(sft_eval_rows),
        "dpo_total_pairs": len(dpo_rows),
        "dpo_train_pairs": len(dpo_train_rows),
        "dpo_eval_pairs": len(dpo_eval_rows),
        "grpo_tasks": len(grpo_rows),
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
            {"id": p["id"], "language": p["language"], "toolchain": p["compiler_toolchain"], "compiler_feedback": p["compiler_feedback"]}
            for p in dpo_rows[:3]
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tbl = Table(title=f"Dataset Summary ({tier.upper()} / {domain})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", style="green")
    tbl.add_row("SFT Splits (train/eval/tot)", f"{len(sft_train_rows)} / {len(sft_eval_rows)} / {len(tokenized_sft)}")
    tbl.add_row("DPO Splits (train/eval/tot)", f"{len(dpo_train_rows)} / {len(dpo_eval_rows)} / {len(dpo_rows)}")
    tbl.add_row("GRPO Worktree Tasks", str(len(grpo_rows)))
    tbl.add_row("SFT by Source / Language", f"{json.dumps(source_counts)} | {json.dumps(lang_counts)}")
    tbl.add_row("Sandbox Pass Rate", f"{sandbox_stats['passed_count']}/{sandbox_stats['total_verified']} ({sandbox_stats['pass_rate_pct']}%)")
    console.print(tbl)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Gemma 4 coding SFT, DPO, and GRPO Parquet datasets.")
    parser.add_argument("--tier", choices=tuple(TIER_SPECS.keys()), default="smoke")
    parser.add_argument("--domain", choices=["all", "dotnet", "react", "go", "cpp", "general", "oss"], default="all")
    parser.add_argument("--oss-only", action="store_true")
    parser.add_argument("--oss-corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    build_dataset_for_tier(
        args.tier,
        domain=args.domain,
        oss_only=args.oss_only,
        oss_corpus_root=args.oss_corpus_root,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
