#!/bin/bash
# =============================================================================
# Setup script do cloud environment (Claude Code na nuvem / claude.ai/code).
#
# COLE ESTE ARQUIVO INTEIRO no campo "Setup script" do ambiente (ver
# references/cloud_environment.md). Este arquivo no repo é só a cópia versionada:
# editá-lo aqui NÃO atualiza o ambiente nem reconstrói o cache.
#
# Escopo: escrever código, rodar lint/typecheck/testes e abrir PRs. Sem dataset,
# sem Hugging Face, sem W&B (os testes usam dados sintéticos).
#
# Roda como root no Ubuntu 24.04, depois do clone do repo e antes do Claude Code
# subir. Se terminar em < ~5 min, a Anthropic tira um snapshot do disco e as
# próximas sessões já começam com a venv pronta (cache de ~7 dias).
#
# O clone do repo é automático (GitHub App / `/web-setup`) — não clonamos aqui.
# A venv fica FORA do repo (/opt/abaw/venv) porque cada sessão parte de um clone
# novo; o hook scripts/cloud/session_start.sh aponta o uv para ela.
# =============================================================================
set -uo pipefail

ABAW_HOME=/opt/abaw
export UV_PROJECT_ENVIRONMENT="${ABAW_VENV:-$ABAW_HOME/venv}"
export UV_CACHE_DIR="${ABAW_UV_CACHE:-$ABAW_HOME/uv-cache}"
# Python do sistema (Ubuntu 24.04 = 3.12): os downloads de Python do uv vêm de
# GitHub releases, que o proxy do GitHub pode recusar (403) fora do repo da sessão.
export UV_PYTHON_DOWNLOADS=never
# gnn entra p/ os modelos GNN importarem e os testes que usam importorskip não serem pulados.
SYNC_GROUPS="${ABAW_UV_GROUPS:-neural gnn dev}"
SYNC_GROUPS="${SYNC_GROUPS//,/ }"

mkdir -p "$ABAW_HOME"
log() { echo "[abaw-setup] $*"; }

# Localiza o clone (o diretório exato não é documentado).
find_repo() {
  local d f
  for d in "${CLAUDE_PROJECT_DIR:-}" "$PWD" /home/*/* /root/* /workspace/* /workspaces/*; do
    [ -f "$d/pyproject.toml" ] && grep -q '^name = "abaw-11-ah-challenge"' "$d/pyproject.toml" \
      && { echo "$d"; return 0; }
  done
  while IFS= read -r f; do
    grep -q '^name = "abaw-11-ah-challenge"' "$f" && { dirname "$f"; return 0; }
  done < <(find / -maxdepth 5 -name pyproject.toml \
             -not -path '/proc/*' -not -path '/sys/*' -not -path '/usr/*' 2>/dev/null)
  return 1
}

# 1) Python 3.12 + uv (ambos já vêm na imagem; só instala se faltar).
command -v python3.12 >/dev/null 2>&1 \
  || { apt-get update -qq \
       && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12 python3.12-venv; } \
  || log "AVISO: python3.12 indisponível"
command -v uv >/dev/null 2>&1 || python3 -m pip install -q --break-system-packages uv \
  || log "AVISO: não consegui instalar o uv"

# 2) Dependências do projeto, a partir do uv.lock (torch Linux = wheels CUDA, ~3 GB).
if REPO=$(find_repo); then
  log "repo em $REPO — uv sync (grupos: $SYNC_GROUPS)"
  cd "$REPO"
  args=()
  for g in $SYNC_GROUPS; do args+=(--group "$g"); done
  # gnn-modalblocks vem do GitHub (fora do repo da sessão): se falhar, cai p/ neural+dev.
  uv sync --frozen "${args[@]}" \
    || { log "AVISO: sync com '$SYNC_GROUPS' falhou; tentando neural+dev"; \
         uv sync --frozen --group neural --group dev; } \
    || log "AVISO: uv sync falhou — o hook de SessionStart tenta de novo"
else
  log "AVISO: clone não encontrado — o hook de SessionStart faz o uv sync"
fi

# A sessão pode não rodar como root: deixa venv/cache graváveis p/ o hook.
chmod -R a+rwX "$ABAW_HOME" 2>/dev/null || true
log "ok"
exit 0
