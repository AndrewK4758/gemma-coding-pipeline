# Gemma 4 E4B (`8.0B`) Dual-GPU Coding Fine-Tuning & RLVR Pipeline

End-to-end local data curation, compiler verification, **16-bit `bfloat16` Dual-GPU LoRA Supervised Fine-Tuning (SFT)**, **Compiler-Verified Direct Preference Optimization (DPO)**, **Online Group Relative Policy Optimization (GRPO)**, and **`llama.cpp` GGUF (`Q8_0` + `imatrix Q4_K_M`) export** pipeline for **TypeScript/Node**, **Go**, and **C# / .NET 10**.

---

## 1. Hardware Topology & Architecture Design

Designed for an asymmetric Dual-GPU Blackwell Linux workstation:

| Component | Detected Spec | Pipeline Role |
| :--- | :--- | :--- |
| **CPU & L3 Cache** | **AMD Ryzen 9 9955HX3D** (`16C/32T`, `128 MB` 3D V-Cache) | Runs 32-thread compiler sandboxes (`tsc 6.0.3`, `go 1.27.1`, `.NET 10.0.400 Roslyn`), `MinHash LSH`, and `:11434` `nomic-embed-text` semantic deduplication (`0 MB` GPU VRAM). |
| **External eGPU (`cuda:1`)** | **NVIDIA GeForce RTX 5080** (`16 GB` GDDR7, `400W`, Blackwell `sm_120`, TB5 PCIe Gen3×4) | Holds `model.language_model.layers.0..26` (`27` layers), `embed_tokens_per_layer` (`5.25 GiB` PLE table), and frozen multimodal towers (`~11.0 GiB` weights, `~12.6–13.0 GiB` peak training VRAM). |
| **Internal dGPU (`cuda:0`)** | **NVIDIA GeForce RTX 5070 Ti Laptop** (`12 GB` GDDR7, `140W`, Blackwell `sm_120`, PCIe Gen5×8) | Holds `embed_tokens`, `model.language_model.layers.27..41` (`15` layers), `norm`, and tied `lm_head` (`~3.9 GiB` weights, `~5.1–5.3 GiB` peak training VRAM). |
| **System RAM & NVMe** | **30 GiB DDR5 + 95 GiB Swap; 1.4+ TB Free NVMe** | Hosts Arrow/Parquet datasets, `PagedAdamW8bit` overflow pages, `bfloat16` LoRA weight folding, and GGUF quantization. |

### Key Systems Optimizations
1. **Static Asymmetric Layer Sharding Over Thunderbolt (`p2p_to_cuda=False`)**:
   Instead of `DDP` or `FSDP`/`ZeRO-3` (which choke the PCIe Gen3×4 Thunderbolt link with multi-gigabyte weight collectives), layers are partitioned statically across `cuda:1` (`Layers 0..26`) and `cuda:0` (`Layers 27..41`). Only a single `~8 MB` hidden-state activation tensor crosses the PCIe boundary per micro-step.
2. **256-Token Chunked Vocabulary Cross-Entropy (`262,144` Vocab)**:
   Materializing full-sequence `float32` logits (`[1, 4096, 262144]`) for Gemma 4's `262K` vocabulary would consume `8.58 GiB` on `cuda:0` alone. [`train/model_loader.py`](train/model_loader.py) filters down to unmasked completion tokens (`labels != -100`) and evaluates `lm_head` + `30.0 * tanh(logits / 30.0)` + cross-entropy in **256-token rematerialized chunks**, cutting `cuda:0` activation VRAM by `>6 GiB` with zero mathematical approximation.
3. **Zero-VRAM Reference Policy Caching in DPO**:
   [`train/dpo_compiler.py`](train/dpo_compiler.py) pre-computes exact reference sequence log-probabilities ($\log \pi_{\text{ref}}(y_w \mid x)$ and $\log \pi_{\text{ref}}(y_l \mid x)$) in one `torch.no_grad()` pass using the Phase 3 SFT adapter before Step 1—anchoring DPO strictly to the SFT checkpoint while using `0 MB` of extra reference-model VRAM and halving step time.

---

## 2. Quick Start: Single-Command End-to-End Orchestrator

[`run_pipeline.py`](run_pipeline.py) sequences data ingestion/filtering (while keeping `local-ai-offload.service` on `:11434` active for CPU embeddings), switches GPUs to `train` mode (`26.2 GiB` usable VRAM), runs training, evaluation, and GGUF export, and guarantees `local-ai-offload.service` is restored via a `try...finally` guard:

```bash
cd gemma-coding-pipeline

# 1. Smoke Tier (300 SFT rows / 150 DPO pairs — verifies full pipeline in ~4 mins)
.venv/bin/python run_pipeline.py --tier smoke --stages data,sft,dpo,eval,export

# 2. Dev Tier (3,000 SFT rows / 1,000 DPO pairs — fast iteration run in ~35-45 mins)
.venv/bin/python run_pipeline.py --tier dev --stages data,sft,dpo,eval,export

# 3. Full Production Tier (30,000 SFT rows / 10,000 DPO pairs + Online GRPO — overnight)
.venv/bin/python run_pipeline.py --tier full --stages data,sft,dpo,grpo,eval,export
```

Monitor training curves in real time with local TensorBoard:
```bash
.venv/bin/tensorboard --logdir runs/ --port 6006
```

---

## 3. Step-by-Step Manual Workflow

### Step 1 — Data Acquisition, Deduplication & Compiler Verification (`llama-swap` ON)

Keep `local-ai-offload.service` running (`./scripts/gpu-mode.zsh status`) so [`data/dedup_and_filter.py`](data/dedup_and_filter.py) can query your local `nomic-embed-text` CPU embedding endpoint (`http://127.0.0.1:11434/v1/embeddings`).

```bash
.venv/bin/python data/build_dataset.py --tier dev
```

#### What `data/build_dataset.py` Does
1. **Streams 4 Data Sources Across TypeScript, Go & C#/.NET 10** ([`data/ingest_sources.py`](data/ingest_sources.py)):
   - **`codefeedback_magicoder`**: `m-a-p/CodeFeedback-Filtered-Instruction` + `ise-uiuc/Magicoder-Evol-Instruct-110K` (`streaming=True`).
   - **`commitpack_diffs`**: `bigcode/commitpackft` real Git commit diffs and instructions (`typescript`, `go`, `c#`).
   - **`arch_explanations`**: `HuggingFaceTB/cosmopedia-v2` + architectural Q&A.
   - **`local_projects`**: Harvests TypeScript, Go, and C#/.NET 10 modules and tests from `~/projects` (or `$LOCAL_PROJECTS_ROOT`).
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
- **Add another local repository**: Place the repository under `~/projects` (or set `LOCAL_PROJECTS_ROOT=/path/to/repos`) so [`data/ingest_sources.py`](data/ingest_sources.py) discovers its `.ts`, `.go`, and `.cs` files automatically.
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
| **1. Your Own Local & Private Git Repos** | `git log -p --no-merges` across `~/projects/*` (`$LOCAL_PROJECTS_ROOT`) | Real `(pre-commit file + commit message) -> post-commit code` surgical edit pairs, test-to-implementation pairs (`FooTests.cs <-> Foo.cs`), and Fill-in-the-Middle (`FIM`) spans. |
| **2. GitHub CLI (`gh`) / Gitea / GitLab PR & Issue APIs** | `gh pr list --state merged --json title,body,files` & `gh api repos/<owner>/<repo>/pulls/<id>/comments` | Merged pull request descriptions paired with exact code diffs and senior reviewer comments (`requested_changes -> fixed_commit`). |
| **3. Stack Exchange Official XML Dumps (Archive.org)** | `https://archive.org/download/stackexchange` (`stackoverflow.com-Posts.7z`, `codereview.stackexchange.com.7z`) | Millions of accepted answers (`AcceptedAnswerId`), high-upvote architectural explanations, and code-review refactors filtered by `<typescript>`, `<go>`, and `<c#>`. |
| **4. Google BigQuery Public Datasets** | `bq query` on `bigquery-public-data.github_repos` & `bigquery-public-data.stackoverflow` (1 TB/month free tier via `gcloud`) | SQL-queryable petabyte-scale GitHub files, commit messages, and Stack Overflow Q&A filtered by license (`Apache-2.0`, `MIT`) and file extension. |
| **5. Official Docs & Specification Repositories** | `git clone` of markdown doc repos: `dotnet/docs`, `dotnet/csharplang`, `golang/go/doc`, `microsoft/TypeScript-Website`, IETF RFCs | Authoritative language specs, `.NET 10` breaking-change guides, compiler error references, and stdlib idioms (avoids stale pre-2024 training cutoffs). |
| **6. Local Teacher Synthetic Distillation (`:11434`)** | Prompting your local `gemma-4-26B-A4B` (`http://127.0.0.1:11434/v1/chat/completions`) + 32-thread compiler sandbox | Zero-cost synthetic problems, multi-turn debugging traces, and "back-translated" prompts generated directly from your local source files and verified by `tsc` / `go build` / `Roslyn`. |
| **7. Kaggle, Zenodo & Academic Benchmarks** | `kaggle datasets download`, Software Heritage (`swh`), Defect4J / BugsInPy / Exercism tracks | Bug-fix corpora, security vulnerability patches (CVE before/after diffs), and unit-test-backed coding exercises. |

---

### Step 2 — Switch GPUs to Training Mode

Before launching GPU training, unload `local-ai-offload.service` to free all `26.2 GiB` across both Blackwell GPUs:

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

# 3. Restore local-ai-offload.service (:11434)
./scripts/gpu-mode.zsh inference
```

---

### Step 6 — Using Your Trained Model Live in `llamaswap-server` (`:11434`)

Your exported GGUF models can be symlinked into `~/.ollama/models/symlinks/` (`$LLM_MODELS_DIR`) and registered in `llama-swap.egpu.yaml` and `llama-swap.single.yaml` under two dedicated model IDs:

| `llama-swap` Model ID | Aliases | Backing GGUF File | Precision & Size | Context & GPU |
| :--- | :--- | :--- | :---: | :--- |
| **`gemma-coder`** | `gemma-4-e4b-coder`, `gemma-4-e4b-coder-q8`, `openai/gemma-coder` | `gemma-4-E4B-coder-Q8_0.gguf` | `Q8_0` (`7.48 GiB`) | `32K` ctx on RTX 5080 (`75.6 tok/s`) |
| **`gemma-coder-q4`** | `gemma-4-e4b-coder-q4`, `gemma-4-e4b-coder-imatrix`, `openai/gemma-coder-q4` | `gemma-4-E4B-coder-Q4_K_M-imatrix.gguf` | `Q4_K_M + imatrix` (`4.97 GiB`) | `64K` ctx on RTX 5080 (`90.8 tok/s`) |

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

#### 2. Two Ways to Serve New LoRA Trainings in `llama-swap`
1. **Merged GGUF Replacement (Zero Runtime Overhead — Default)**:
   Whenever you run `export/merge_and_quantize_gguf.py` (or `run_pipeline.py --stages export`), it overwrites `outputs/gguf/gemma-4-E4B-coder-Q8_0.gguf` and `gemma-4-E4B-coder-Q4_K_M-imatrix.gguf`. Because `~/.ollama/models/symlinks/gemma-4-E4B-coder-*.gguf` points directly to those files, the very next request to `"model": "gemma-coder"` on `:11434` loads your newly trained model automatically.
2. **Dynamic Unmerged `--lora` Adapter Hot-Loading (For Multiple Per-Project LoRAs)**:
   If you train multiple `134 MB` project-specific adapters (e.g., `checkpoints/dpo-custom/final_adapter`), you can convert the PEFT adapter directly to a standalone GGUF LoRA file without merging the 15 GB base model:
   ```bash
   .venv/bin/python ~/projects/llama.cpp/convert_lora_to_gguf.py \
     --base ./outputs/merged-bf16 \
     --outtype f16 \
     --outfile ~/.ollama/models/symlinks/lora-custom.gguf \
     checkpoints/dpo-smoke/final_adapter
   ```
   Then pass `--lora ${models}/lora-custom.gguf` (or `--lora-scaled ${models}/lora-custom.gguf 1.0`) in the `llama-server` command inside `llama-swap.egpu.yaml`.

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
| [`.tool-versions`](.tool-versions) & [`pyproject.toml`](pyproject.toml) | Pins `asdf` `python 3.13.17` and Blackwell `sm_120` PyTorch (`torch==2.14.1+cu130`), `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `datasketch`, `tensorboard`. |
| [`scripts/gpu-mode.zsh`](scripts/gpu-mode.zsh) | Switches Dual GPUs between `train` mode (`local-ai-offload` stopped) and `inference` mode (`:11434` active). |
| [`scripts/verify_dual_gpu_sharding.py`](scripts/verify_dual_gpu_sharding.py) | Phase 1 hardware probe verifying `sm_120`, asymmetric `device_map`, cross-PCIe autograd, and `PagedAdamW8bit`. |
| [`data/ingest_sources.py`](data/ingest_sources.py) | Streams Hugging Face coding datasets and harvests local `~/projects` TS, Go, and C#/.NET 10 repositories. |
| [`data/dedup_and_filter.py`](data/dedup_and_filter.py) | Heuristic quality filter + `datasketch` 5-gram `MinHash LSH` + `:11434` `nomic-embed-text` semantic deduplication. |
| [`data/compiler_sandbox.py`](data/compiler_sandbox.py) | 32-thread compiler verification sandbox (`tsc 6.0.3`, `go 1.27.1`, `.NET 10 Roslyn`) and DPO mutation generator. |
| [`data/build_dataset.py`](data/build_dataset.py) | Builds tokenized SFT (`-100` prompt masked) and DPO Parquet splits (`data/processed/<tier>/`). |
| [`train/model_loader.py`](train/model_loader.py) | Static 42-layer Dual-GPU `bfloat16` loader, PEFT LoRA injector, and 256-token chunked vocabulary cross-entropy loss. |
| [`train/sft_lora.py`](train/sft_lora.py) | Phase 3 Supervised Fine-Tuning trainer with TensorBoard and JSON step telemetry. |
| [`train/dpo_compiler.py`](train/dpo_compiler.py) | Phase 4A Offline Compiler-Verified DPO trainer with zero-VRAM reference log-prob pre-computation. |
| [`train/grpo_online.py`](train/grpo_online.py) | Phase 4B Online GRPO trainer with live multi-generation rollouts and compiler sandbox grading. |
| [`eval/benchmark_coding.py`](eval/benchmark_coding.py) | Phase 5 multi-adapter single-load benchmark (`Base` vs. `SFT` vs. `DPO` perplexity, preference margin, and `Pass@1`). |
| [`export/merge_and_quantize_gguf.py`](export/merge_and_quantize_gguf.py) | Folds LoRA into `bfloat16`, calibrates coding `imatrix`, exports `Q8_0` and `Q4_K_M` GGUFs, and runs `llama-bench`. |
| [`run_pipeline.py`](run_pipeline.py) | Top-level end-to-end CLI orchestrator (`--tier smoke|dev|full`). |
