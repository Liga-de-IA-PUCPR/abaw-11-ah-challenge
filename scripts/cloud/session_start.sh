#!/bin/bash
# =============================================================================
# Hook de SessionStart (ver .claude/settings.json) — só age em sessões na nuvem.
#
# - aponta o uv/PATH para a venv criada pelo setup script (/opt/abaw/venv);
# - `uv sync --frozen` incremental: no-op com o cache quente, instala o que mudou
#   se a branch tiver outro uv.lock (ou tudo, se o setup script não achou o repo).
#
# O stdout deste hook entra no contexto do Claude: mantenha-o curto.
# =============================================================================
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
set -uo pipefail

ABAW_HOME=/opt/abaw
export UV_PROJECT_ENVIRONMENT="${ABAW_VENV:-$ABAW_HOME/venv}"
export UV_CACHE_DIR="${ABAW_UV_CACHE:-$ABAW_HOME/uv-cache}"
export UV_PYTHON_DOWNLOADS=never
SYNC_GROUPS="${ABAW_UV_GROUPS:-neural gnn dev}"
SYNC_GROUPS="${SYNC_GROUPS//,/ }"
LOG=/tmp/abaw-session-start.log

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
mkdir -p "$ABAW_HOME" 2>/dev/null

# Persiste p/ todos os comandos que o Claude rodar nesta sessão.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat >> "$CLAUDE_ENV_FILE" <<EOF
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PYTHON_DOWNLOADS=never
export PATH="$UV_PROJECT_ENVIRONMENT/bin:\$PATH"
EOF
fi

args=()
for g in $SYNC_GROUPS; do args+=(--group "$g"); done
if uv sync --frozen "${args[@]}" >"$LOG" 2>&1; then
  status="venv ok ($SYNC_GROUPS)"
elif uv sync --frozen --group neural --group dev >>"$LOG" 2>&1; then
  status="venv ok (neural dev — '$SYNC_GROUPS' falhou, ver $LOG)"
else
  status="uv sync FALHOU (ver $LOG)"
fi

echo "[cloud] $status · sem dataset (só código/testes): valide com 'make check'"
exit 0
