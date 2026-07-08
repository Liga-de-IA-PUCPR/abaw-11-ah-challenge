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
MODEL_NAME    ?= random_forest     # modelo para sweep-windows (ex: make sweep-windows MODEL_NAME=xgboost)

# MPS (Apple Metal): habilita fallback p/ CPU em ops não suportadas pelo Metal.
MPS_FALLBACK  := PYTORCH_ENABLE_MPS_FALLBACK=1

# Preset Hydra opcional (ex.: EXPERIMENT=cross_attention -> "+experiment=cross_attention")
_EXP          := $(if $(EXPERIMENT),+experiment=$(EXPERIMENT),)
# Override de split opcional (vazio = usa o default do mode no main.py)
_SPLIT        := $(if $(SPLIT),split=$(SPLIT),)

.DEFAULT_GOAL := help

.PHONY: help setup setup-neural setup-all ffmpeg-check \
        extract-audio preprocess featurize data \
        train train-rf train-xgboost train-lightgbm train-neural \
        train-extra-trees train-logistic-regression train-catboost train-mlp train-stacking \
        train-stacking-catboost train-stacking-cat-rf \
        train-all-models train-all-sklearn \
        sweep sweep-rf sweep-xgboost sweep-lightgbm \
        sweep-rf-stage2 sweep-xgboost-stage2 sweep-lightgbm-stage2 \
        sweep-extra-trees sweep-logistic-regression sweep-catboost sweep-catboost-stage2 sweep-catboost-stage3 sweep-mlp \
        sweep-all best-model \
        preprocess-windows featurize-windows sweep-windows train-all-windows \
        evaluate submit pipeline cv-select compare \
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
	@echo "    data             preprocess (extrai áudio + janelas) + featurize (prep completo)"
	@echo ""
	@echo "  Treino / avaliação (FASE 4/5):"
	@echo "    train            baseline RandomForest (CPU) [== train-rf]"
	@echo "    train-xgboost           treina XGBoost (CPU)"
	@echo "    train-lightgbm          treina LightGBM (CPU)"
	@echo "    train-extra-trees       treina ExtraTrees (CPU)"
	@echo "    train-logistic-regression  treina Logistic Regression (CPU)"
	@echo "    train-catboost          treina CatBoost (CPU)"
	@echo "    train-mlp               treina MLP densa (CPU)"
	@echo "    train-stacking          treina Stacking RF+XGB+LGBM (CPU, ~5× mais lento)"
	@echo "    train-stacking-catboost treina Stacking + CatBoost como base estimator (RF+XGB+LGBM+Cat)"
	@echo "    train-stacking-cat-rf   treina Stacking reduzido CatBoost+RF (sem XGB/LGBM)"
	@echo "    train-all-models        RF + XGBoost + LightGBM em sequência (baseline)"
	@echo "    train-all-sklearn       todos os 7 modelos sklearn em sequência"
	@echo "    train-neural            cross-attention via Lightning (Apple Metal, DEVICE=mps)"
	@echo "    sweep                   multirun livre (SWEEP=\"model.lr=1e-3,5e-4 ...\")"
	@echo "    sweep-rf                sweep RF    (96 combinações)"
	@echo "    sweep-xgboost           sweep XGBoost (108 combinações)"
	@echo "    sweep-lightgbm          sweep LightGBM (108 combinações)"
	@echo "    sweep-extra-trees       sweep ExtraTrees (96 combinações)"
	@echo "    sweep-logistic-regression  sweep LogReg — apenas C (6 combinações)"
	@echo "    sweep-catboost          sweep CatBoost (81 combinações)"
	@echo "    sweep-mlp               sweep MLP (36 combinações)"
	@echo "    sweep-all               sweep de TODOS os modelos em sequência"
	@echo "    best-model              top-1 run do compare (melhor modelo)"
	@echo "    preprocess-windows      preprocess para small+medium+large"
	@echo "    featurize-windows       featurize para small+medium+large"
	@echo "    sweep-windows           3 janelas × 1 modelo (MODEL_NAME=$(MODEL_NAME)) — use após sweep-all"
	@echo "    train-all-windows       3 modelos × 3 janelas (9 runs, exploração inicial)"
	@echo "    cv-select        Fase 2: GroupKFold no treino p/ selecionar família de modelo"
	@echo "    evaluate         Macro-F1/AP num split (SPLIT=val por default)"
	@echo "    submit           gera arquivo de submissão (SPLIT=test, OUT=$(OUT))"
	@echo "    pipeline         data + train + evaluate (ponta a ponta)"
	@echo "    compare          tabela comparativa de todos os runs avaliados"
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

data: preprocess featurize
	@echo "✓ Dados prontos: áudio extraído, janelas indexadas e features em data/processed/"

# ----------------------------------------------------------------------------
# Treino / avaliação (FASE 4/5)
# ----------------------------------------------------------------------------
train: train-rf

# --- Modelos sklearn (CPU) ---------------------------------------------------
# Baseline RandomForest — 100% CPU, NÃO importa Lightning.
train-rf:
	$(PY) $(MAIN) mode=train model=random_forest trainer=sklearn $(ARGS)

train-xgboost:
	$(PY) $(MAIN) mode=train model=xgboost trainer=sklearn $(ARGS)

train-lightgbm:
	$(PY) $(MAIN) mode=train model=lightgbm trainer=sklearn $(ARGS)

train-extra-trees:
	$(PY) $(MAIN) mode=train model=extra_trees trainer=sklearn $(ARGS)

train-logistic-regression:
	$(PY) $(MAIN) mode=train model=logistic_regression trainer=sklearn $(ARGS)

train-catboost:
	$(PY) $(MAIN) mode=train model=catboost trainer=sklearn $(ARGS)

train-mlp:
	$(PY) $(MAIN) mode=train model=mlp trainer=sklearn $(ARGS)

train-stacking:
	$(PY) $(MAIN) mode=train model=stacking trainer=sklearn $(ARGS)

train-stacking-catboost:
	$(PY) $(MAIN) mode=train model=stacking_catboost trainer=sklearn $(ARGS)

train-stacking-cat-rf:
	$(PY) $(MAIN) mode=train model=stacking_cat_rf trainer=sklearn $(ARGS)

# Treina os 3 modelos sklearn originais em sequência.
train-all-models: train-rf train-xgboost train-lightgbm
	@echo "✓ RF + XGBoost + LightGBM treinados."

# Treina todos os modelos sklearn (incluindo os novos) em sequência.
train-all-sklearn: train-rf train-xgboost train-lightgbm train-extra-trees train-logistic-regression train-catboost train-mlp
	@echo "✓ Todos os modelos sklearn treinados (RF, XGB, LGBM, ET, LR, CatBoost, MLP)."

# Cross-attention (Lightning, opcional). Requer `make setup-neural`.
# device=mps por default neste alvo (Apple Metal) + fallback p/ CPU.
train-neural:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=cross_attention \
	  device=$(if $(filter auto,$(DEVICE)),mps,$(DEVICE)) $(ARGS)

# --- Sweep de hiperparâmetros ------------------------------------------------
# Varredura livre (overrides arbitrários via SWEEP="...").
sweep:
	@test -n "$(SWEEP)" || (echo "Defina SWEEP, ex.: make sweep SWEEP=\"model.lr=1e-3,5e-4\""; exit 1)
	$(PY) $(MAIN) -m $(_EXP) $(SWEEP) $(ARGS)

# Sweeps por modelo usando os configs pré-definidos em configs/sweep/.
sweep-rf:
	$(PY) $(MAIN) -m +sweep=rf $(ARGS)

sweep-xgboost:
	$(PY) $(MAIN) -m +sweep=xgboost $(ARGS)

sweep-lightgbm:
	$(PY) $(MAIN) -m +sweep=lightgbm $(ARGS)

# Stage-2: busca fina após identificar o melhor modelo no stage-1.
# Fixe os hiperparâmetros dominantes via ARGS="model.max_depth=10 model.n_estimators=400"
sweep-rf-stage2:
	$(PY) $(MAIN) -m +sweep=rf_stage2 $(ARGS)

sweep-xgboost-stage2:
	$(PY) $(MAIN) -m +sweep=xgboost_stage2 $(ARGS)

sweep-lightgbm-stage2:
	$(PY) $(MAIN) -m +sweep=lightgbm_stage2 $(ARGS)

sweep-extra-trees:
	$(PY) $(MAIN) -m +sweep=extra_trees $(ARGS)

sweep-logistic-regression:
	$(PY) $(MAIN) -m +sweep=logistic_regression $(ARGS)

sweep-catboost:
	$(PY) $(MAIN) -m +sweep=catboost $(ARGS)

sweep-catboost-stage2:
	$(PY) $(MAIN) -m +sweep=catboost_stage2 $(ARGS)

# Stage-3: refina além das bordas do stage-2 (n_estimators<50, depth>10);
# learning_rate e l2_leaf_reg ficam fixos (convergiram no stage-2).
sweep-catboost-stage3:
	$(PY) $(MAIN) -m +sweep=catboost_stage3 $(ARGS)

sweep-mlp:
	$(PY) $(MAIN) -m +sweep=mlp $(ARGS)

# Roda o sweep de TODOS os modelos em sequência → use 'make compare' para ver o ranking.
sweep-all: sweep-rf sweep-extra-trees sweep-logistic-regression sweep-catboost sweep-mlp sweep-xgboost sweep-lightgbm
	@echo ""
	@echo "✓ Sweep completo de todos os modelos concluído."
	@echo "  Use 'make compare' para ver o ranking por Macro-F1."

# Mostra o melhor modelo de todos os runs (top-1 do compare).
best-model:
	@$(PY) -m src.scripts.compare_runs --root outputs 2>/dev/null | head -4

# --- Sweep de janelamento ----------------------------------------------------
# ⚠️  Cada janelamento precisa de cache próprio (preprocess + featurize).
# preprocess-windows e featurize-windows constroem os 3 caches em sequência.
preprocess-windows:
	@for w in small medium large; do \
	  echo ">>> Preprocessando janela: $$w"; \
	  $(PY) $(MAIN) mode=preprocess window=$$w $(ARGS); \
	done

featurize-windows:
	@for w in small medium large; do \
	  echo ">>> Featurizando janela: $$w (device=$(DEVICE))"; \
	  $(PY) $(MAIN) mode=featurize window=$$w device=$(DEVICE) $(ARGS); \
	done

# Testa os 3 janelamentos para UM modelo (use após identificar o melhor modelo com sweep-all).
# Uso: make sweep-windows MODEL_NAME=xgboost
# ⚠️  Requer caches já construídos: make preprocess-windows featurize-windows
sweep-windows:
	@echo ">>> Testando janelamentos (small/medium/large) para modelo: $(MODEL_NAME)"
	@for w in small medium large; do \
	  echo "  >> window=$$w"; \
	  $(PY) $(MAIN) mode=train model=$(MODEL_NAME) trainer=sklearn window=$$w $(ARGS); \
	done
	@echo "✓ 3 janelamentos treinados para $(MODEL_NAME). Use 'make compare' para ver o ranking."

# 9 runs: 3 modelos × 3 janelas (baseline de janelamento — para exploração inicial).
# Para uso pós-seleção de modelo, prefira: make sweep-windows MODEL_NAME=<melhor>
# Para construir os caches: make preprocess-windows featurize-windows
train-all-windows:
	@for w in small medium large; do \
	  for m in random_forest xgboost lightgbm; do \
	    echo ">>> Treinando $$m (window=$$w)"; \
	    $(PY) $(MAIN) mode=train model=$$m trainer=sklearn window=$$w $(ARGS); \
	  done; \
	done
	@echo "✓ 9 runs (3 modelos × 3 janelas) concluídos."

evaluate:
	$(PY) $(MAIN) mode=evaluate $(_EXP) $(_SPLIT) device=$(DEVICE) $(ARGS)

submit:
	$(PY) $(MAIN) mode=submit $(_EXP) $(if $(SPLIT),split=$(SPLIT),split=test) \
	  out=$(OUT) device=$(DEVICE) $(ARGS)

# Ponta a ponta (dados -> treino -> avaliação).
pipeline: data train evaluate
	@echo "✓ Pipeline completo executado."

# Fase 2 — seleção de família via GroupKFold no treino (sem tocar o val).
# Avalia os modelos listados e imprime ranking por Macro-F1 médio ± desvio.
# Uso: make cv-select  |  make cv-select ARGS="--models xgboost lightgbm --n-splits 3"
cv-select:
	$(PY) -m src.scripts.cv_select $(ARGS)

# Tabela comparativa de todos os runs (eval_val/metrics.json OU train_result.json).
# eval_val tem precedência quando ambos existem (métricas mais completas).
compare:
	$(PY) -m src.scripts.compare_runs --root outputs

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
