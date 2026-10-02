# Ambiente na nuvem (Claude Code cloud sessions)

Sessões do Claude Code na nuvem (claude.ai/code, app Desktop, `claude --cloud`) para **escrever
código, rodar lint/typecheck/testes e subir commits/PRs**. Sem dataset, sem Hugging Face e sem
W&B: os testes usam dados sintéticos. Treino e avaliação de modelos continuam nas máquinas locais.
Docs oficiais: [cloud environments](https://code.claude.com/docs/en/cloud-environments).

| Onde | O quê |
|---|---|
| **Repo** (versionado) | `.claude/settings.json` (hook de SessionStart) · `scripts/cloud/session_start.sh` · `scripts/cloud/setup.sh` (cópia do setup script) |
| **Diálogo do ambiente** (claude.ai) | nome · acesso à rede · variáveis de ambiente · setup script |

Fluxo de uma sessão nova: **clone do repo → setup script** (só quando não há cache) **→ Claude
Code sobe → hook de SessionStart → sessão pronta**.

## 1. Acesso ao repo (o clone é automático)

Não há clone a fazer no setup script: a sessão já começa com um clone novo da branch escolhida.
O repo é privado (`Liga-de-IA-PUCPR/abaw-11-ah-challenge`), então uma das duas:

- **Claude GitHub App** instalado na organização `Liga-de-IA-PUCPR` (pede aprovação de um owner
  da org) — feito no onboarding de claude.ai/code;
- **`/web-setup`** num `claude` de terminal: envia o token do seu `gh` (precisa enxergar o repo).

Push e PR passam por um proxy do GitHub que guarda as credenciais fora da VM. Ele aceita push
de branches; recusa apagar branch e push de tags. Os arquivos acima só valem na nuvem depois de
estarem **no GitHub, na branch** em que a sessão abre — abra as sessões a partir da `develop`
(a `main` está congelada; ver [`.claude/CLAUDE.md`](../.claude/CLAUDE.md)).

## 2. Criar o ambiente

claude.ai/code (ou a caixa de prompt do Desktop) → ícone de nuvem / seletor de ambiente →
novo ambiente:

- **Name:** `abaw-11`
- **Network access:** **Trusted** (o default) — cobre PyPI e GitHub, que é tudo o que o
  `uv sync` usa.
- **Environment variables:** nenhuma obrigatória. Opcional: `ABAW_UV_GROUPS=neural,gnn,dev`
  (o default; acrescente `vision` p/ o Face Mesh).
- **Setup script:** cole o conteúdo inteiro de [`scripts/cloud/setup.sh`](../scripts/cloud/setup.sh).

O setup script faz `uv sync --frozen` numa venv **fora do repo** (`/opt/abaw/venv`, porque cada
sessão parte de um clone novo). Se terminar em < ~5 min, o disco vira snapshot e as próximas
sessões pulam essa etapa (cache de ~7 dias, refeito quando o script ou a rede mudam). Editar
`scripts/cloud/setup.sh` no repo **não** atualiza o ambiente: cole de novo no diálogo.

O hook (`session_start.sh`) roda em toda sessão, inclusive retomada: aponta `uv`/`PATH` p/ a
venv, faz um `uv sync --frozen` incremental (no-op com o cache quente; instala o que mudou se a
branch tiver outro `uv.lock`) e imprime uma linha de status. Fora da nuvem ele sai sem fazer nada
(`CLAUDE_CODE_REMOTE != true`), então não afeta as sessões locais da equipe.

## 3. Limites e diagnóstico

- VM Ubuntu 24.04 x86_64: 4 vCPU, 16 GB RAM, 30 GB de disco, **sem GPU**.
- `make check` (format-check + lint + compile + typecheck + testes) é o portão antes do PR.
- O grupo `gnn` (torch-geometric + `gnn-modalblocks`, via git do GitHub) é instalado para os
  modelos GNN importarem e os testes com `importorskip` não serem pulados. Se o proxy recusar o
  git de `gnn-modalblocks`, o sync cai para `neural,dev` e a linha de status avisa.
- O torch do `uv.lock` no Linux é o build CUDA (~3 GB de wheels nvidia): ocupa disco, mas roda
  em CPU normalmente.
- Log do hook: `/tmp/abaw-session-start.log`.
