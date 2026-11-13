#!/usr/bin/env zsh
# Clones curated MIT/Apache-2.0 .NET and React UI OSS repositories into ~/corpus/oss_repos
# and pre-warms local NuGet + pnpm/npm caches for offline GRPO worktree verification.
set -euo pipefail

CORPUS_DIR="${OSS_CORPUS_DIR:-${HOME}/corpus/oss_repos}"
CLONE_DEPTH="${OSS_CLONE_DEPTH:-500}"
ONLY_DOMAIN="all"
SKIP_RESTORE=0

while (( $# > 0 )); do
  case "$1" in
    --only)
      ONLY_DOMAIN="${2:-all}"
      shift 2
      ;;
    --skip-restore)
      SKIP_RESTORE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--only dotnet|react|go|cpp|all] [--skip-restore]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "${ONLY_DOMAIN}" in
  all|dotnet|react|go|cpp) ;;
  *)
    echo "Unsupported --only domain: ${ONLY_DOMAIN} (expected dotnet|react|go|cpp|all)" >&2
    exit 1
    ;;
esac

DOTNET_REPOS=(
  "https://github.com/dotnet/eShop.git"
  "https://github.com/kgrzybek/modular-monolith-with-ddd.git"
  "https://github.com/jasontaylordev/CleanArchitecture.git"
  "https://github.com/ardalis/CleanArchitecture.git"
  "https://github.com/jbogard/ContosoUniversityDotNetCore-Pages.git"
  "https://github.com/MassTransit/MassTransit.git"
  "https://github.com/fullstackhero/dotnet-starter-kit.git"
)

REACT_REPOS=(
  "https://github.com/alan2207/bulletproof-react.git"
  "https://github.com/epicweb-dev/epic-stack.git"
  "https://github.com/shadcn-ui/ui.git"
  "https://github.com/radix-ui/primitives.git"
  "https://github.com/TanStack/query.git"
  "https://github.com/TanStack/router.git"
  "https://github.com/excalidraw/excalidraw.git"
)

GO_REPOS=(
  "https://github.com/gin-gonic/gin.git"
  "https://github.com/go-chi/chi.git"
  "https://github.com/spf13/cobra.git"
  "https://github.com/cli/cli.git"
  "https://github.com/uber-go/zap.git"
  "https://github.com/prometheus/prometheus.git"
  "https://github.com/etcd-io/etcd.git"
)

CPP_REPOS=(
  "https://github.com/ggml-org/llama.cpp.git"
  "https://github.com/curl/curl.git"
  "https://github.com/valkey-io/valkey.git"
  "https://github.com/libuv/libuv.git"
  "https://github.com/fmtlib/fmt.git"
  "https://github.com/simdjson/simdjson.git"
  "https://github.com/nlohmann/json.git"
)

clone_or_update_group() {
  local eco="$1"
  shift
  local target_base="${CORPUS_DIR}/${eco}"
  mkdir -p "${target_base}"
  for url in "$@"; do
    local owner_repo="${url#https://github.com/}"
    owner_repo="${owner_repo%.git}"
    local safe_name="${owner_repo//\//__}"
    local target="${target_base}/${safe_name}"
    if [[ ! -d "${target}/.git" ]]; then
      echo "  [clone] ${owner_repo} (depth=${CLONE_DEPTH}) -> ${target}"
      git clone --depth="${CLONE_DEPTH}" --single-branch "${url}" "${target}"
    else
      echo "  [fetch] ${owner_repo} -> ${target}"
      git -C "${target}" fetch --depth="${CLONE_DEPTH}" --prune origin || true
    fi
  done
}

if [[ "${ONLY_DOMAIN}" == "all" || "${ONLY_DOMAIN}" == "dotnet" ]]; then
  echo "[1/5] Syncing .NET repositories into ${CORPUS_DIR}/dotnet..."
  clone_or_update_group "dotnet" "${DOTNET_REPOS[@]}"
fi

if [[ "${ONLY_DOMAIN}" == "all" || "${ONLY_DOMAIN}" == "react" ]]; then
  echo "[2/5] Syncing React UI & TypeScript repositories into ${CORPUS_DIR}/react..."
  clone_or_update_group "react" "${REACT_REPOS[@]}"
fi

if [[ "${ONLY_DOMAIN}" == "all" || "${ONLY_DOMAIN}" == "go" ]]; then
  echo "[3/5] Syncing Go OSS repositories into ${CORPUS_DIR}/go..."
  clone_or_update_group "go" "${GO_REPOS[@]}"
fi

if [[ "${ONLY_DOMAIN}" == "all" || "${ONLY_DOMAIN}" == "cpp" ]]; then
  echo "[4/5] Syncing C/C++ OSS repositories into ${CORPUS_DIR}/cpp..."
  clone_or_update_group "cpp" "${CPP_REPOS[@]}"
fi

if (( SKIP_RESTORE == 0 )); then
  echo "[5/5] Pre-warming local NuGet, pnpm/npm, and Go module caches for offline GRPO execution..."
  if [[ -d "${CORPUS_DIR}/dotnet" ]]; then
    for dir in "${CORPUS_DIR}/dotnet/"*(N/); do
      echo "  [nuget] ${dir:t}"
      (cd "$dir" && dotnet restore --verbosity quiet || true)
    done
  fi
  if [[ -d "${CORPUS_DIR}/react" ]]; then
    for dir in "${CORPUS_DIR}/react/"*(N/); do
      if [[ -f "${dir}/package.json" ]]; then
        echo "  [node]  ${dir:t}"
        (
          cd "$dir"
          if [[ -f pnpm-lock.yaml ]] && command -v pnpm >/dev/null 2>&1; then
            pnpm install --frozen-lockfile --ignore-scripts || true
          elif command -v npm >/dev/null 2>&1; then
            npm ci --ignore-scripts --prefer-offline || npm install --ignore-scripts --prefer-offline || true
          fi
        )
      fi
    done
  fi
  if [[ -d "${CORPUS_DIR}/go" ]] && command -v go >/dev/null 2>&1; then
    for dir in "${CORPUS_DIR}/go/"*(N/); do
      if [[ -f "${dir}/go.mod" ]]; then
        echo "  [gomod] ${dir:t}"
        (cd "$dir" && go mod download || true)
      fi
    done
  fi
fi

echo "OSS corpus ready at ${CORPUS_DIR}"
