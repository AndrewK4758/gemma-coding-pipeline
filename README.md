# Gemma 4 (`E4B` & `26B-A4B MoE`) Dual-GPU Coding & Multi-Language OSS Fine-Tuning Pipeline

End-to-end local data curation, OSS commit mining, git-worktree & compiler verification, **Dual-GPU `bfloat16` / 4-bit NF4 QLoRA SFT**, **Compiler-Verified DPO**, **Online GRPO**, and multi-engine export for **`llama.cpp` (`Q8_0`, `imatrix Q4_K_M`, standalone `--lora` GGUF)**, **Ollama (`Modelfile`)**, and **vLLM (`serve_vllm.sh`)** across **C# / .NET 10**, **React / TypeScript**, **Go**, and **C / C++**.

---

## 1. Hardware Topology & Architecture Design

Designed for an asymmetric Dual-GPU CUDA Linux workstation (`16 GB` + `12 GB` VRAM):

| Component | Target Profile | Pipeline Role |
| :--- | :--- | :--- |
| **Host CPU** | **Multi-Core x86_64 CPU** (`16C/32T`) | Runs 32-thread compiler sandboxes (`tsc`, `go`, `.NET 10 Roslyn`, `gcc`/`g++`), `MinHash LSH`, and optional local `:11434` semantic deduplication (`0 MB` GPU VRAM). |
| **Primary GPU (`cuda:1`)** | **16 GB CUDA GPU** | Holds `model.language_model.layers.0..26` (`27` layers), `embed_tokens_per_layer` (`5.25 GiB` PLE table), and frozen multimodal towers (`~11.0 GiB` weights, `~12.6–13.0 GiB` peak training VRAM). |
| **Secondary GPU (`cuda:0`)** | **12 GB CUDA GPU** | Holds `embed_tokens`, `model.language_model.layers.27..41` (`15` layers), `norm`, and tied `lm_head` (`~3.9 GiB` weights, `~5.1–5.3 GiB` peak training VRAM). |
| **System RAM & Storage** | **>= 32 GiB RAM + Swap / Fast NVMe** | Hosts Arrow/Parquet datasets, `PagedAdamW8bit` overflow pages, streaming `bfloat16` LoRA shard merging, and GGUF quantization. |

### Key Systems Optimizations
1. **Static Asymmetric Layer Sharding Over Slow/Asymmetric PCIe (`p2p_to_cuda=False`)**:
   Instead of `DDP` or `FSDP`/`ZeRO-3` (which choke asymmetric PCIe links with multi-gigabyte weight collectives), layers are partitioned statically across `cuda:1` (`Layers 0..26`) and `cuda:0` (`Layers 27..41`). Only a single `~8 MB` hidden-state activation tensor crosses the PCIe boundary per micro-step.
2. **256-Token Chunked Vocabulary Cross-Entropy (`262,144` Vocab)**:
   Materializing full-sequence `float32` logits (`[1, 4096, 262144]`) for Gemma 4's `262K` vocabulary would consume `8.58 GiB` on `cuda:0` alone. [`train/model_loader.py`](train/model_loader.py) filters down to unmasked completion tokens (`labels != -100`) and evaluates `lm_head` + `30.0 * tanh(logits / 30.0)` + cross-entropy in **256-token rematerialized chunks**, cutting `cuda:0` activation VRAM by `>6 GiB` with zero mathematical approximation.
3. **Zero-VRAM Reference Policy Caching in DPO**:
   [`train/dpo_compiler.py`](train/dpo_compiler.py) pre-computes exact reference sequence log-probabilities ($\log \pi_{\text{ref}}(y_w \mid x)$ and $\log \pi_{\text{ref}}(y_l \mid x)$) in one `torch.no_grad()` pass using the Phase 3 SFT adapter before Step 1—anchoring DPO strictly to the SFT checkpoint while using `0 MB` of extra reference-model VRAM and halving step time.

---

## 2. Quick Start: Single-Command End-to-End Orchestrator

[`run_pipeline.py`](run_pipeline.py) sequences data ingestion/filtering (optionally keeping any `$LOCAL_OFFLOAD_SERVICE` endpoint on `:11434` active for CPU embeddings), switches GPUs to `train` mode (`26.2 GiB` usable VRAM), runs training, evaluation, and GGUF export, and guarantees optional `$LOCAL_OFFLOAD_SERVICE` restoration via a `try...finally` guard:

```bash
cd gemma-coding-pipeline

# 1. Smoke Tier (300 SFT rows / 150 DPO pairs — verifies full pipeline in ~4 mins)
.venv/bin/python run_pipeline.py --tier smoke --stages data,sft,dpo,eval,export

# 2. Dev Tier (3,000 SFT rows / 1,000 DPO pairs — fast iteration run in ~35-45 mins)
.venv/bin/python run_pipeline.py --tier dev --stages data,sft,dpo,eval,export

# 3. Full Production Tier (30,000 SFT rows / 10,000 DPO pairs + Online GRPO — overnight)
.venv/bin/python run_pipeline.py --tier full --stages data,sft,dpo,grpo,eval,export

# 4. Clone MIT/Apache-2.0/BSD-3 .NET, React UI, Go & C/C++ OSS corpus + train modular domain LoRAs
./scripts/clone_oss_corpus.zsh --only all
.venv/bin/python run_pipeline.py --tier dev --domain dotnet --adapter-name dotnet --oss-only --stages data,sft,dpo,grpo,eval,export
.venv/bin/python run_pipeline.py --tier dev --domain react  --adapter-name react  --oss-only --stages data,sft,dpo,grpo,eval,export
.venv/bin/python run_pipeline.py --tier dev --domain go     --adapter-name go     --oss-only --stages data,sft,dpo,grpo,eval,export
.venv/bin/python run_pipeline.py --tier dev --domain cpp    --adapter-name cpp    --oss-only --stages data,sft,dpo,grpo,eval,export

# 5. Optional 26B-A4B MoE QLoRA run (all 4 languages unified or per-domain; auto-enables 4-bit NF4 experts + streaming CPU merge)
./scripts/train_26b_all_languages.zsh --tier dev --mode unified
./scripts/train_26b_all_languages.zsh --tier dev --mode per-domain
```

Monitor training curves in real time with local TensorBoard:
```bash
.venv/bin/tensorboard --logdir runs/ --port 6006
```

---

## 3. Step-by-Step Manual Workflow

### Step 1 — Data Acquisition, Deduplication & Compiler Verification (Local Embeddings `:11434`)

If running a local OpenAI-compatible embedding service (`./scripts/gpu-mode.zsh status`), [`data/dedup_and_filter.py`](data/dedup_and_filter.py) queries `http://127.0.0.1:11434/v1/embeddings` (`nomic-embed-text` / `embedding`) for semantic deduplication.

```bash
.venv/bin/python data/build_dataset.py --tier dev
```

#### What `data/build_dataset.py` Does
1. **Streams 4 Data Sources Across TypeScript, Go & C#/.NET 10** ([`data/ingest_sources.py`](data/ingest_sources.py)):
   - **`codefeedback_magicoder`**: `m-a-p/CodeFeedback-Filtered-Instruction` + `ise-uiuc/Magicoder-Evol-Instruct-110K` (`streaming=True`).
   - **`commitpack_diffs`**: `bigcode/commitpackft` real Git commit diffs and instructions (`typescript`, `go`, `c#`).
   - **`arch_explanations`**: `HuggingFaceTB/cosmopedia-v2` + architectural Q&A.
   - **`local_projects`**: Harvests TypeScript, Go, C#, and C/C++ modules and tests from `$LOCAL_PROJECTS_ROOT` (default `~/projects`).
2. **3-Stage Deduplication Funnel** ([`data/dedup_and_filter.py`](data/dedup_and_filter.py)):
   - **Stage A (Heuristic Filter)**: Strips auto-generated headers, minified lines (`>500` chars), and empty/trivial snippets.
   - **Stage B (Lexical `MinHash LSH`)**: 5-gram shingling (`num_perm=128`, `threshold=0.85`) removes near-identical code in $O(N)$ time.
   - **Stage C (Semantic Embedding Filter)**: Batch-embeds candidates on `:11434` (`nomic-embed-text`) and removes semantic clones with cosine similarity `> 0.94`.
3. **32-Thread Compiler Sandbox & DPO Pair Generation** ([`data/compiler_sandbox.py`](data/compiler_sandbox.py)):
   - Compiles every extracted snippet across 32 worker threads:
     - **TypeScript**: TypeScript `6.0.3` compiler API (`ts.createProgram` + `ts.getPreEmitDiagnostics`) on Node `v26.12.0`
     - **Go**: `gofmt -e` + `go build` (`go1.27.1`)
     - **C# / .NET 10**: `.NET 10.0.400` Roslyn `csc.dll` against `net10.0` reference assemblies
   - Passing snippets populate `sft_train.parquet` / `sft_eval.parquet` (with `-100` prompt completion masking) and serve as `chosen` ($y_w$) in `dpo_train.parquet` / `dpo_eval.parquet`, paired against compiler-verified failing mutations (`rejected` $y_l$) with verbatim compiler diagnostics.

#### How to Add Custom Data Sources
- **Add another local repository**: Place the repository under `$LOCAL_PROJECTS_ROOT` (or `~/workspace`) so [`data/ingest_sources.py`](data/ingest_sources.py) discovers its `.ts`, `.go`, `.cs`, `.c`, and `.cpp` files automatically.
- **Add a new Hugging Face dataset or local JSONL file**: Add a generator function in [`data/ingest_sources.py`](data/ingest_sources.py) yielding:
  ```python
  {
      "id": "custom_001",
      "instruction": "<prompt or problem statement>",
      "response": "<reasoning and fenced ```typescript|go|csharp code block>",
      "language": "typescript",  # "typescript" | "go" | "csharp"
      "source": "my_custom_source",
  }
  ```

#### Where to Get Training Data Outside of Hugging Face
Beyond Hugging Face Hub, you can feed [`data/ingest_sources.py`](data/ingest_sources.py) from seven major non-HF sources:

| Source Category | Specific Origin & Access Method | What Signal It Provides |
| :--- | :--- | :--- |
| **1. Your Own Local & Private Git Repos** | `git log -p --no-merges` across `$LOCAL_PROJECTS_ROOT/*` | Real `(pre-commit file + commit message) -> post-commit code` surgical edit pairs, test-to-implementation pairs (`FooTests.cs <-> Foo.cs`), and Fill-in-the-Middle (`FIM`) spans. |
| **2. GitHub CLI (`gh`) / Gitea / GitLab PR & Issue APIs** | `gh pr list --state merged --json title,body,files` & `gh api repos/<owner>/<repo>/pulls/<id>/comments` | Merged pull request descriptions paired with exact code diffs and senior reviewer comments (`requested_changes -> fixed_commit`). |
| **3. Stack Exchange Official XML Dumps (Archive.org)** | `https://archive.org/download/stackexchange` (`stackoverflow.com-Posts.7z`, `codereview.stackexchange.com.7z`) | Millions of accepted answers (`AcceptedAnswerId`), high-upvote architectural explanations, and code-review refactors filtered by `<typescript>`, `<go>`, and `<c#>`. |
| **4. Google BigQuery Public Datasets** | `bq query` on `bigquery-public-data.github_repos` & `bigquery-public-data.stackoverflow` (1 TB/month free tier via `gcloud`) | SQL-queryable petabyte-scale GitHub files, commit messages, and Stack Overflow Q&A filtered by license (`Apache-2.0`, `MIT`) and file extension. |
| **5. Official Docs & Specification Repositories** | `git clone` of markdown doc repos: `dotnet/docs`, `dotnet/csharplang`, `golang/go/doc`, `microsoft/TypeScript-Website`, IETF RFCs | Authoritative language specs, `.NET 10` breaking-change guides, compiler error references, and stdlib idioms (avoids stale pre-2024 training cutoffs). |
| **6. Local Teacher Synthetic Distillation (`:11434`)** | Prompting your local `gemma-4-26B-A4B` (`http://127.0.0.1:11434/v1/chat/completions`) + 32-thread compiler sandbox | Zero-cost synthetic problems, multi-turn debugging traces, and "back-translated" prompts generated directly from your local source files and verified by `tsc` / `go build` / `Roslyn`. |
| **7. Kaggle, Zenodo & Academic Benchmarks** | `kaggle datasets download`, Software Heritage (`swh`), Defect4J / BugsInPy / Exercism tracks | Bug-fix corpora, security vulnerability patches (CVE before/after diffs), and unit-test-backed coding exercises. |

---

### Step 2 — Switch GPUs to Training Mode

Before launching GPU training, optionally stop `$LOCAL_OFFLOAD_SERVICE` to free full Dual-GPU VRAM (`~26+ GiB` usable):

```bash
./scripts/gpu-mode.zsh train
```

---

### Step 3 — Stage 3: Dual-GPU `bfloat16` LoRA Supervised Fine-Tuning (SFT)

Trains `bf16` LoRA adapters (`r=32, lora_alpha=64, lora_dropout=0.05`, `69.76M` trainable params = `0.87%` of `8.01B`) across all 7 linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) using `PagedAdamW8bit`:

```bash
# Primary 4,096-token SFT pass
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python train/sft_lora.py \
  --tier dev \
  --max-seq-length 4096 \
  --learning-rate 2e-4

# Optional 8,192-token long-context extension pass
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python train/sft_lora.py \
  --tier dev \
  --max-seq-length 8192 \
  --max-steps 50 \
  --learning-rate 5e-5
```
* **Artifacts**: `checkpoints/sft-<tier>/final_adapter/`, `checkpoints/sft-<tier>/training_metrics.json`, and TensorBoard logs in `runs/sft-<tier>/`.

---

### Step 4 — Stage 4: Compiler-Verified DPO & Optional Online GRPO

```bash
# Stage 4A: Offline Compiler-Verified DPO
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python train/dpo_compiler.py \
  --tier dev \
  --sft-adapter checkpoints/sft-dev/final_adapter \
  --beta 0.1 \
  --learning-rate 5e-5

# Stage 4B (Optional): Online GRPO with live compiler rollouts (G=4 completions/prompt)
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python train/grpo_online.py \
  --tier dev \
  --dpo-adapter checkpoints/dpo-dev/final_adapter \
  --num-generations 4
```
* **Artifacts**: `checkpoints/dpo-<tier>/final_adapter/`, `checkpoints/dpo-<tier>/dpo_metrics.json`, and `checkpoints/grpo-<tier>/final_adapter/`.

---

### Step 5 — Benchmark, Export to GGUF (`Q8_0` & `imatrix Q4_K_M`) & Restore `:11434`

```bash
# 1. Head-to-head evaluation (Base vs. SFT vs. SFT+DPO in a single GPU load)
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python eval/benchmark_coding.py --tier dev

# 2. Fold LoRA into bfloat16, calibrate imatrix, and quantize Q8_0 + Q4_K_M GGUFs
CUDA_VISIBLE_DEVICES=0,1 CUDA_MODULE_LOADING=LAZY \
  .venv/bin/python export/merge_and_quantize_gguf.py \
  --tier dev \
  --adapter-dir checkpoints/dpo-dev/final_adapter

# 3. Optionally restore inference service ($LOCAL_OFFLOAD_SERVICE)
./scripts/gpu-mode.zsh inference
```

---

### Step 6 — Using Your Trained Model Live Across `llama.cpp`, Ollama & vLLM

Your exported GGUF models can be placed into `$LLM_MODELS_DIR` and registered in `<inference-server-config>.yaml` under two dedicated model IDs:

| Server Model ID | Aliases | Backing GGUF File | Precision & Size | Context & GPU |
| :--- | :--- | :--- | :---: | :--- |
| **`gemma-coder`** | `gemma-4-e4b-coder`, `gemma-4-e4b-coder-q8` | `gemma-4-E4B-coder-Q8_0.gguf` | `Q8_0` (`7.48 GiB`) | `32K` ctx (`cuda:1` `16 GB` GPU, `75.6 tok/s`) |
| **`gemma-coder-q4`** | `gemma-4-e4b-coder-q4`, `gemma-4-e4b-coder-imatrix` | `gemma-4-E4B-coder-Q4_K_M-imatrix.gguf` | `Q4_K_M + imatrix` (`4.97 GiB`) | `64K` ctx (`cuda:1` `16 GB` GPU, `90.8 tok/s`) |

#### 1. Query It Directly via OpenAI-Compatible API (`curl` / Python / IDE)
```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-coder",
    "messages": [
      {"role": "user", "content": "Write a thread-safe generic worker pool in Go using context cancellation."}
    ],
    "temperature": 0.1,
    "max_tokens": 512
  }' | jq -r '.choices[0].message.content'
```

#### 2. Two Ways to Serve New LoRA Trainings
1. **Merged GGUF Replacement (Zero Runtime Overhead — Default)**:
   Whenever you run `export/merge_and_quantize_gguf.py` (or `run_pipeline.py --stages export`), it overwrites `outputs/gguf/gemma-4-E4B-coder-Q8_0.gguf` and `gemma-4-E4B-coder-Q4_K_M-imatrix.gguf`, and also emits a standalone Ollama `Modelfile` and `serve_vllm.sh`. If `$LLM_MODELS_DIR/gemma-4-E4B-coder-*.gguf` symlinks to those outputs, the next server reload uses your newly trained model automatically.
2. **Dynamic Unmerged `--lora` Adapter Hot-Loading (For Multiple Per-Project LoRAs)**:
   If you train multiple `134 MB` project-specific adapters (e.g., `checkpoints/dpo-custom/final_adapter`), you can convert the PEFT adapter directly to a standalone GGUF LoRA file without merging the base model:
   ```bash
   .venv/bin/python "${LLAMA_CPP_DIR:-$HOME/llama.cpp}/convert_lora_to_gguf.py" \
     --base ./outputs/merged-bf16 \
     --outtype f16 \
     --outfile "${LLM_MODELS_DIR:-./outputs/gguf}/lora-custom.gguf" \
     checkpoints/dpo-smoke/final_adapter
   ```
   Then pass `--lora ${models}/lora-custom.gguf` (or `--lora-scaled ${models}/lora-custom.gguf 1.0`) in your `llama-server` command or `<inference-server-config>.yaml`.

---

## 4. Verified Baseline & Smoke Pipeline Benchmark Results

### A. Held-Out Evaluation & Compiler `Pass@1` Progression
| Checkpoint | Held-Out Token NLL | Held-Out Perplexity (`PPL`) | Length-Norm LogP Margin | DPO Reward Margin vs. SFT | TypeScript `Pass@1` (`tsc 6.0.3`) | Go `Pass@1` (`go 1.27.1`) | C# `Pass@1` (`.NET 10 Roslyn`) | Total `Compile Pass@1` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Base (`google/gemma-4-E4B-it`)** | `0.9233` | `2.5176` | `+0.2331` | `+3.5579` | `1/5` (`20%`) | `2/5` (`40%`) | `0/5` (`0%`) | **`3/15` (`20.0%`)** |
| **2. Phase 3 SFT (`sft-smoke`)** | **`0.4394`** | **`1.5517`** | `+0.1940` | `0.0000` (ref) | `4/5` (`80%`) | `5/5` (`100%`) | `5/5` (`100%`) | **`14/15` (`93.3%`)** |
| **3. Phase 4 SFT+DPO (`dpo-smoke`)** | **`0.4523`** | **`1.5719`** | **`+0.3575`** (`+84.3%`) | **`+8.4220`** | **`5/5` (`100%`)** | **`5/5` (`100%`)** | **`5/5` (`100%`)** | **`15/15` (`100.0%`)** |

### B. Exported GGUF Artifacts & Dual-GPU `llama-bench` Throughput
| Artifact | Output Path | Size | Prompt Speed (`pp256`) | Generation Speed (`tg64`) |
| :--- | :--- | :---: | :---: | :---: |
| **Merged HF `bfloat16`** | `outputs/merged-bf16/model.safetensors` | `14.89 GiB` | — | — |
| **Coding Importance Matrix** | `outputs/gguf/gemma-4-E4B-coder.imatrix` | `4.22 MiB` | — | — |
| **GGUF `Q8_0` (8-bit)** | `outputs/gguf/gemma-4-E4B-coder-Q8_0.gguf` | **`7.48 GiB`** | **`5,668.6 tok/s`** | **`75.6 tok/s`** |
| **GGUF `Q4_K_M` + `imatrix`** | `outputs/gguf/gemma-4-E4B-coder-Q4_K_M-imatrix.gguf` | **`4.97 GiB`** | **`5,485.6 tok/s`** | **`90.8 tok/s`** |

---

## 5. Repository Layout

| Path | Description |
| :--- | :--- |
| [`.tool-versions`](.tool-versions) & [`pyproject.toml`](pyproject.toml) | Pins Python `3.13` and CUDA PyTorch (`torch==2.14.1+cu130`), `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `datasketch`, `tensorboard`. |
| [`scripts/gpu-mode.zsh`](scripts/gpu-mode.zsh) | Switches Dual GPUs between `train` mode (`$LOCAL_OFFLOAD_SERVICE` stopped) and `inference` / `serve` mode (`:11434` health-checked). |
| [`scripts/clone_oss_corpus.zsh`](scripts/clone_oss_corpus.zsh) | Shallow-clones curated MIT/Apache-2.0/BSD-3 `.NET`, `React UI`, `Go`, and `C/C++` repositories (`--depth=500`) and warms package caches offline. |
| [`scripts/verify_dual_gpu_sharding.py`](scripts/verify_dual_gpu_sharding.py) | Phase 1 hardware probe verifying Dual-GPU compute capability, asymmetric `device_map`, cross-PCIe autograd, and `PagedAdamW8bit`. |
| [`scripts/train_26b_all_languages.zsh`](scripts/train_26b_all_languages.zsh) | All-language (`dotnet`, `react`, `go`, `cpp`) runner for `26B-A4B-it` and `E4B-it` (`unified`, `oss-unified`, `per-domain`, `all`). |
| [`data/mine_oss_repos.py`](data/mine_oss_repos.py) | Mines atomic `impl + test` commits into Gemma 4 `<\|channel>thought` SFT, DPO, and GRPO JSONLs (combined and domain-split). |
| [`data/ingest_sources.py`](data/ingest_sources.py) | Streams Hugging Face coding datasets, harvests `$LOCAL_PROJECTS_ROOT`, and merges mined `.NET`/`React`/`Go`/`C++` OSS commits. |
| [`data/dedup_and_filter.py`](data/dedup_and_filter.py) | Heuristic quality filter + `datasketch` 5-gram `MinHash LSH` + `:11434` semantic deduplication. |
| [`data/compiler_sandbox.py`](data/compiler_sandbox.py), [`data/cpp_sandbox.py`](data/cpp_sandbox.py) & [`data/oss_sandbox.py`](data/oss_sandbox.py) | 32-thread compiler sandbox (`tsc`, `go build`, `.NET 10 Roslyn`, `gcc`/`g++`) plus isolated `/tmp` git-worktree test runner for GRPO rollouts. |
| [`data/build_dataset.py`](data/build_dataset.py) | Builds tokenized single/multi-turn SFT (`-100` masked), DPO, and `grpo_tasks.parquet` splits (`data/processed/<tier>/`). |
| [`train/model_loader.py`](train/model_loader.py) & [`train/moe_quant.py`](train/moe_quant.py) | Static asymmetric Dual-GPU `bfloat16`/4-bit QLoRA loader (including 4-bit NF4 MoE `QuantizedGemma4Experts4bit`) + chunked cross-entropy. |
| [`train/sft_lora.py`](train/sft_lora.py) | Phase 3 Supervised Fine-Tuning trainer with TensorBoard and JSON step telemetry. |
| [`train/dpo_compiler.py`](train/dpo_compiler.py) | Phase 4A Offline Compiler-Verified DPO trainer with zero-VRAM reference log-prob pre-computation. |
| [`train/grpo_online.py`](train/grpo_online.py) | Phase 4B Online GRPO trainer with snippet compiler and live OSS git-worktree test grading. |
| [`eval/benchmark_coding.py`](eval/benchmark_coding.py) | Phase 5 multi-adapter single-load benchmark (`Base` vs. `SFT` vs. `DPO`/`GRPO`). |
| [`export/merge_and_quantize_gguf.py`](export/merge_and_quantize_gguf.py) & [`export/multi_engine.py`](export/multi_engine.py) | Streaming shard-by-shard CPU LoRA merge, `Q8_0`/`Q4_K_M-imatrix` GGUFs, standalone `lora-*.gguf`, Ollama `Modelfile`, and `vLLM` launcher. |
| [`tests/`](tests/) | Automated unit and synthetic-model regression tests (`test_oss_pipeline.py`, `test_go_cpp_pipeline.py`, `test_dotnet_sdk_routing.py`, `test_model_loading_and_export.py`). |
| [`run_pipeline.py`](run_pipeline.py) | Top-level end-to-end CLI orchestrator (`--tier`, `--stages`, `--domain`, `--adapter-name`, `--oss-only`, `--load-in-4bit`). |
