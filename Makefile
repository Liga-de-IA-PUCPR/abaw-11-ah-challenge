# =============================================================================
# BAH AH-Challenge (ABAW11) — Makefile
# =============================================================================
# Centraliza TODOS os comandos do pipeline áudio+texto (sem digitar python na mão).
# Tudo roda dentro do ambiente gerenciado pelo `uv` (uv run ...).
#
# Início rápido:
#   make setup            # ambiente core (RF, CPU/MPS) — sem Lightning
#   make data             # extrai áudio -> índice/janelas -> features (Parquet)
#   make train            # baseline RandomForest (CPU)
#   make evaluate         # Macro-F1/AP no split de validação
#   make ci               # quality gate: format-check + lint + compile
#
# Modelo neural (opcional, Apple Metal / MPS):
#   make setup-neural     # adiciona Lightning + torchmetrics
#   make train-neural     # cross-attention sobre sequência de janelas (device=mps)
#
# Parâmetros (sobrescreva na linha de comando):
#   DEVICE=auto|cpu|mps|cuda   SPLIT=val|test   OUT=<arquivo>
#   EXPERIMENT=<preset>        ARGS="<overrides Hydra extras>"   SWEEP="<args multirun>"
# Exemplos:
#   make featurize DEVICE=mps
#   make train ARGS="model.n_estimators=800 data.window.size_s=4"
#   make evaluate SPLIT=test
#   make sweep SWEEP="model.lr=1e-3,5e-4 model.num_heads=4,8"
# =============================================================================

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
UV            := uv
RUN           := $(UV) run
PY            := $(RUN) python
MAIN          := main.py
# ruff standalone (uvx): lint/format sem precisar sincronizar o ambiente pesado.
RUFF          := uvx ruff

# Parâmetros (?= permite sobrescrever via CLI)
DEVICE        ?= auto
SPLIT         ?=
OUT           ?= outputs/submission.txt
EXPERIMENT    ?=
ARGS          ?=
SWEEP         ?=

# MPS (Apple Metal): habilita fallback p/ CPU em ops não suportadas pelo Metal.
MPS_FALLBACK  := PYTORCH_ENABLE_MPS_FALLBACK=1

# Preset Hydra opcional (ex.: EXPERIMENT=cross_attention -> "+experiment=cross_attention")
_EXP          := $(if $(EXPERIMENT),+experiment=$(EXPERIMENT),)
# Override de split opcional (vazio = usa o default do mode no main.py)
_SPLIT        := $(if $(SPLIT),split=$(SPLIT),)

.DEFAULT_GOAL := help

.PHONY: help setup setup-neural setup-all ffmpeg-check \
        extract-audio preprocess featurize data \
        train train-rf train-neural sweep evaluate submit pipeline \
        lint format format-check typecheck test compile ci check \
        clean clean-cache clean-outputs clean-all

# ----------------------------------------------------------------------------
# Ajuda (alvo default)
# ----------------------------------------------------------------------------
help:
	@echo "BAH AH-Challenge — comandos disponíveis (make <alvo>):"
	@echo ""
	@echo "  Ambiente:"
	@echo "    setup            uv sync (core: RF/CPU + dev) — SEM Lightning"
	@echo "    setup-neural     + grupo neural (Lightning, torchmetrics) p/ cross_attention"
	@echo "    setup-all        core + dev + neural"
	@echo "    ffmpeg-check     verifica se o ffmpeg (extração de áudio) está instalado"
	@echo ""
	@echo "  Dados (FASE 2/3):"
	@echo "    extract-audio    mp4 -> flac 16 kHz mono (src/scripts/extract_audio.py)"
	@echo "    preprocess       índice de vídeos + janela deslizante (mode=preprocess)"
	@echo "    featurize        janelas -> embeddings -> Parquet (mode=featurize, DEVICE=$(DEVICE))"
	@echo "    data             extract-audio + preprocess + featurize (prep completo)"
	@echo ""
	@echo "  Treino / avaliação (FASE 4/5):"
	@echo "    train            baseline RandomForest (CPU) [== train-rf]"
	@echo "    train-neural     cross-attention via Lightning (Apple Metal, DEVICE=mps)"
	@echo "    sweep            multirun Hydra (SWEEP=\"model.lr=1e-3,5e-4 ...\")"
	@echo "    evaluate         Macro-F1/AP num split (SPLIT=val por default)"
	@echo "    submit           gera arquivo de submissão (SPLIT=test, OUT=$(OUT))"
	@echo "    pipeline         data + train + evaluate (ponta a ponta)"
	@echo ""
	@echo "  Qualidade (CI/CD):"
	@echo "    ci               format-check + lint + compile (gate rápido)"
	@echo "    check            ci + typecheck + test (gate completo)"
	@echo "    lint / format    ruff check / ruff format (+ --fix)"
	@echo "    typecheck        mypy   |   test  pytest   |   compile  py_compile"
	@echo ""
	@echo "  Limpeza:"
	@echo "    clean            remove caches (__pycache__, .ruff_cache, .pytest_cache)"
	@echo "    clean-cache      remove artefatos derivados (data/interim, data/processed)"
	@echo "    clean-outputs    remove outputs/ multirun/ wandb/"
	@echo "    clean-all        clean + clean-cache + clean-outputs"
	@echo ""
	@echo "  Parâmetros: DEVICE=$(DEVICE)  SPLIT  OUT  EXPERIMENT  ARGS  SWEEP"

# ----------------------------------------------------------------------------
# Ambiente
# ----------------------------------------------------------------------------
setup:
	$(UV) sync

setup-neural:
	$(UV) sync --group neural

setup-all:
	$(UV) sync --group neural --group dev

ffmpeg-check:
	@command -v ffmpeg >/dev/null 2>&1 \
	  && echo "✓ ffmpeg encontrado: $$(ffmpeg -version | head -1)" \
	  || (echo "✗ ffmpeg não encontrado — instale com: brew install ffmpeg"; exit 1)

# ----------------------------------------------------------------------------
# Dados (FASE 2/3)
# ----------------------------------------------------------------------------
extract-audio: ffmpeg-check
	$(PY) -m src.scripts.extract_audio

preprocess:
	$(PY) $(MAIN) mode=preprocess $(ARGS)

featurize:
	$(PY) $(MAIN) mode=featurize device=$(DEVICE) $(ARGS)

data: extract-audio preprocess featurize
	@echo "✓ Dados prontos: áudio extraído, janelas indexadas e features em data/processed/"

# ----------------------------------------------------------------------------
# Treino / avaliação (FASE 4/5)
# ----------------------------------------------------------------------------
train: train-rf

# Baseline RandomForest — 100% CPU, NÃO importa Lightning.
train-rf:
	$(PY) $(MAIN) mode=train model=random_forest trainer=sklearn $(ARGS)

# Cross-attention (Lightning, opcional). Requer `make setup-neural`.
# device=mps por default neste alvo (Apple Metal) + fallback p/ CPU.
train-neural:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=cross_attention \
	  device=$(if $(filter auto,$(DEVICE)),mps,$(DEVICE)) $(ARGS)

# Varredura de hiperparâmetros (Hydra multirun + launcher joblib).
sweep:
	@test -n "$(SWEEP)" || (echo "Defina SWEEP, ex.: make sweep SWEEP=\"model.lr=1e-3,5e-4\""; exit 1)
	$(PY) $(MAIN) -m $(_EXP) $(SWEEP) $(ARGS)

evaluate:
	$(PY) $(MAIN) mode=evaluate $(_EXP) $(_SPLIT) device=$(DEVICE) $(ARGS)

submit:
	$(PY) $(MAIN) mode=submit $(_EXP) $(if $(SPLIT),split=$(SPLIT),split=test) \
	  out=$(OUT) device=$(DEVICE) $(ARGS)

# Ponta a ponta (dados -> treino -> avaliação).
pipeline: data train evaluate
	@echo "✓ Pipeline completo executado."

# ----------------------------------------------------------------------------
# Qualidade / CI-CD
# ----------------------------------------------------------------------------
lint:
	$(RUFF) check src $(MAIN)

format:
	$(RUFF) format src $(MAIN)
	$(RUFF) check --fix src $(MAIN)

format-check:
	$(RUFF) format --check src $(MAIN)

typecheck:
	$(RUN) mypy src $(MAIN)

test:
	$(RUN) pytest

# Compilação rápida (syntax) sem instalar deps — usa o python do sistema.
compile:
	@python3 -m py_compile $$(find src -name '*.py') $(MAIN) && echo "✓ py_compile OK"

# Gate rápido (estático): roda no que é determinístico/verde sem dataset.
ci: format-check lint compile
	@echo "✓ CI OK (format-check + lint + compile)"

# Gate completo (inclui type-check e testes).
check: ci typecheck test
	@echo "✓ Check completo OK"

# ----------------------------------------------------------------------------
# Limpeza
# ----------------------------------------------------------------------------
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	@echo "✓ Caches removidos"

clean-cache:
	rm -rf data/interim data/processed
	@echo "✓ Artefatos derivados removidos (data/interim, data/processed)"

clean-outputs:
	rm -rf outputs multirun wandb
	@echo "✓ Saídas removidas (outputs/, multirun/, wandb/)"

clean-all: clean clean-cache clean-outputs
	@echo "✓ Limpeza completa"
