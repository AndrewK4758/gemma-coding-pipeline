#!/usr/bin/env zsh
# Train Gemma 4 26B-A4B MoE (or E4B-it) across all 4 coding ecosystems:
# dotnet (C#/.NET 11), react (TS/React), go (Go 1.27), and cpp (C/C++20).
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

TIER="dev"
MODE="unified"
MODEL_ID="google/gemma-4-26B-A4B-it"
STAGES="data,sft,dpo,grpo,eval,export"
CORPUS_DIR="${OSS_CORPUS_DIR:-}"
OSS_ONLY=0
RESUME=0
DRY_RUN=0

while (( $# > 0 )); do
  case "$1" in
    --tier)
      TIER="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --model-id)
      MODEL_ID="${2:-}"
      shift 2
      ;;
    --stages|--stage)
      STAGES="${2:-}"
      shift 2
      ;;
    --oss-corpus-root)
      CORPUS_DIR="${2:-}"
      shift 2
      ;;
    --oss-only)
      OSS_ONLY=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: train_26b_all_languages.zsh [options]
  --tier {smoke|dev|full}                Scale tier (default: dev)
  --mode {unified|oss-unified|per-domain|all}
                                         Training layout (default: unified)
  --model-id <hf-id>                     HF model ID (default: google/gemma-4-26B-A4B-it)
  --stages <csv>                         Stages CSV (default: data,sft,dpo,grpo,eval,export)
  --oss-corpus-root <dir>                Cloned OSS corpus root
  --oss-only                             Restrict unified/per-domain runs to mined OSS repos
  --resume                               Resume SFT stage from latest checkpoint if available
  --dry-run                              Print planned commands without training
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  unified|oss-unified|per-domain|all) ;;
  *)
    echo "Unsupported --mode: ${MODE} (expected unified|oss-unified|per-domain|all)" >&2
    exit 1
    ;;
esac

if [[ -z "${CORPUS_DIR}" ]]; then
  if [[ -d "${HOME}/projects/ml/training_data" ]]; then
    CORPUS_DIR="${HOME}/projects/ml/training_data"
  else
    CORPUS_DIR="${HOME}/corpus/oss_repos"
  fi
fi

VENV_PY="${VENV_PYTHON:-}"
if [[ -z "${VENV_PY}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    VENV_PY="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${HOME}/projects/ml/gemma-coding-pipeline/.venv/bin/python" ]]; then
    VENV_PY="${HOME}/projects/ml/gemma-coding-pipeline/.venv/bin/python"
  else
    VENV_PY="$(command -v python3)"
  fi
fi

if [[ -z "${LOCAL_OFFLOAD_SERVICE:-}" ]] && command -v systemctl >/dev/null 2>&1; then
  if systemctl --user cat local-ai-offload.service >/dev/null 2>&1; then
    export LOCAL_OFFLOAD_SERVICE="local-ai-offload.service"
  fi
fi

export OSS_CORPUS_DIR="${CORPUS_DIR}"
ADAPTER_PREFIX="e4b"
if [[ "${MODEL_ID:l}" == *26b* ]]; then
  ADAPTER_PREFIX="26b"
fi

run_one() {
  local domain="$1"
  local adapter="$2"
  local use_oss="$3"
  local -a cmd=(
    "${VENV_PY}" "${REPO_ROOT}/run_pipeline.py"
    --tier "${TIER}"
    --model-id "${MODEL_ID}"
    --domain "${domain}"
    --adapter-name "${adapter}"
    --oss-corpus-root "${CORPUS_DIR}"
    --stages "${STAGES}"
  )
  if (( use_oss )); then
    cmd+=(--oss-only)
  fi
  if (( RESUME )); then
    cmd+=(--resume)
  fi
  if (( DRY_RUN )); then
    cmd+=(--dry-run)
  fi
  echo ">>> Running [${MODEL_ID}] domain=${domain} adapter=${adapter} tier=${TIER} oss_only=${use_oss}"
  (cd "${REPO_ROOT}" && "${cmd[@]}")
}

echo "Repo: ${REPO_ROOT} | Python: ${VENV_PY} | Corpus: ${CORPUS_DIR} | Mode: ${MODE}"
case "${MODE}" in
  unified)
    run_one "all" "${ADAPTER_PREFIX}-all" "${OSS_ONLY}"
    ;;
  oss-unified)
    run_one "all" "${ADAPTER_PREFIX}-oss" 1
    ;;
  per-domain)
    for eco in dotnet react go cpp; do
      run_one "${eco}" "${ADAPTER_PREFIX}-${eco}" "${OSS_ONLY}"
    done
    ;;
  all)
    run_one "all" "${ADAPTER_PREFIX}-all" "${OSS_ONLY}"
    for eco in dotnet react go cpp; do
      run_one "${eco}" "${ADAPTER_PREFIX}-${eco}" "${OSS_ONLY}"
    done
    ;;
esac
