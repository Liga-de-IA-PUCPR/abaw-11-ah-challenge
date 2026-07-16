# =============================================================================
# BAH AH-Challenge (ABAW11) — Makefile
# =============================================================================
# Centralizes ALL commands of the audio+text pipeline (no hand-typed python).
# Everything runs inside the uv-managed environment (uv run ...).
#
# Quick start:
#   make setup            # core environment (RF, CPU/MPS) — no Lightning
#   make data             # extract audio -> index/windows -> features (Parquet)
#   make train            # RandomForest baseline (CPU)
#   make evaluate         # Macro-F1/AP on the validation split
#   make ci               # quality gate: format-check + lint + compile
#
# Paper result (ensemble_5, traditional train/val/test protocol):
#   make setup-neural
#   make reproduce-best   # data + 5-seed ensemble + smooth calibration on val + test eval
#
# Neural model (optional, Apple Metal / CUDA):
#   make setup-neural     # adds Lightning + torchmetrics
#   make train-neural     # cross-attention over the window sequence
#
# Parameters (override on the command line):
#   DEVICE=auto|cpu|mps|cuda   SPLIT=val|test   OUT=<file>
#   EXPERIMENT=<preset>        ARGS="<extra Hydra overrides>"   SWEEP="<multirun args>"
# Examples:
#   make featurize DEVICE=mps
#   make train ARGS="model.n_estimators=800 data.window.size_s=4"
#   make evaluate SPLIT=test
#   make sweep SWEEP="model.lr=1e-3,5e-4 model.num_heads=4,8"
# =============================================================================

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
UV            := uv
RUN           := $(UV) run
PY            := $(RUN) python
MAIN          := main.py
# standalone ruff (uvx): lint/format without syncing the heavy environment.
RUFF          := uvx ruff

# Parameters (?= allows overriding via CLI)
DEVICE        ?= auto
SPLIT         ?=
OUT           ?= outputs/submission.txt
EXPERIMENT    ?=
ARGS          ?=
SWEEP         ?=
# Seed ensemble (cross-attention): seeds to train + manifest with the run dirs.
# Default = the paper's 5 seeds {42, 1, 2, 3, 4}.
SEEDS             ?= 42 1 2 3 4
ENSEMBLE_MANIFEST ?= outputs/cross_attention/ensemble_manifest.txt
# Threshold calibration used for the ensemble (probability averaging smooths the
# val curve -> smooth/argmax transfer better than base_rate; see paper §3.4).
ENS_CALIB         ?= smooth

# MPS (Apple Metal): fall back to CPU for ops not yet supported by Metal.
MPS_FALLBACK  := PYTORCH_ENABLE_MPS_FALLBACK=1

# Optional Hydra preset (e.g. EXPERIMENT=cross_attention -> "+experiment=cross_attention")
_EXP          := $(if $(EXPERIMENT),+experiment=$(EXPERIMENT),)
# Optional split override (empty = mode's default in main.py)
_SPLIT        := $(if $(SPLIT),split=$(SPLIT),)

.DEFAULT_GOAL := help

.PHONY: help setup setup-neural setup-all ffmpeg-check \
        extract-audio preprocess featurize data \
        train train-rf train-neural sweep evaluate submit pipeline \
        train-ensemble ensemble-evaluate ensemble-submit reproduce-best \
        lint format format-check typecheck test compile ci check \
        clean clean-cache clean-outputs clean-all

# ----------------------------------------------------------------------------
# Help (default target)
# ----------------------------------------------------------------------------
help:
	@echo "BAH AH-Challenge — available commands (make <target>):"
	@echo ""
	@echo "  Environment:"
	@echo "    setup            uv sync (core: RF/CPU + dev) — WITHOUT Lightning"
	@echo "    setup-neural     + neural group (Lightning, torchmetrics) for cross_attention"
	@echo "    setup-all        core + dev + neural"
	@echo "    ffmpeg-check     checks that ffmpeg (audio extraction) is installed"
	@echo ""
	@echo "  Data (PHASE 2/3):"
	@echo "    extract-audio    mp4 -> flac 16 kHz mono (src/scripts/extract_audio.py)"
	@echo "    preprocess       video index + sliding windows (mode=preprocess)"
	@echo "    featurize        windows -> embeddings -> Parquet (mode=featurize, DEVICE=$(DEVICE))"
	@echo "    data             preprocess (audio + windows) + featurize (full data prep)"
	@echo ""
	@echo "  Training / evaluation (PHASE 4/5):"
	@echo "    train            RandomForest baseline (CPU) [== train-rf]"
	@echo "    train-neural     cross-attention via Lightning (DEVICE=$(DEVICE))"
	@echo "    sweep            Hydra multirun (SWEEP=\"model.lr=1e-3,5e-4 ...\")"
	@echo "    evaluate         Macro-F1/AP on a split (SPLIT=val by default)"
	@echo "    submit           writes the submission file (SPLIT=test, OUT=$(OUT))"
	@echo "    train-ensemble   trains N seeds (SEEDS=\"$(SEEDS)\") for the ensemble"
	@echo "    ensemble-evaluate  Macro-F1/AP of the ensemble (probability averaging)"
	@echo "    ensemble-submit    submission from the ensemble"
	@echo "    reproduce-best   PAPER RESULT: data + 5-seed ensemble + eval on test"
	@echo "    pipeline         data + train + evaluate (end to end)"
	@echo ""
	@echo "  Quality (CI/CD):"
	@echo "    ci               format-check + lint + compile (fast gate)"
	@echo "    check            ci + typecheck + test (full gate)"
	@echo "    lint / format    ruff check / ruff format (+ --fix)"
	@echo "    typecheck        mypy   |   test  pytest   |   compile  py_compile"
	@echo ""
	@echo "  Cleaning:"
	@echo "    clean            removes caches (__pycache__, .ruff_cache, .pytest_cache)"
	@echo "    clean-cache      removes derived artifacts (data/interim, data/processed)"
	@echo "    clean-outputs    removes outputs/ multirun/ wandb/"
	@echo "    clean-all        clean + clean-cache + clean-outputs"
	@echo ""
	@echo "  Parameters: DEVICE=$(DEVICE)  SPLIT  OUT  EXPERIMENT  ARGS  SWEEP  SEEDS  ENS_CALIB"

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
setup:
	$(UV) sync

setup-neural:
	$(UV) sync --group neural

setup-all:
	$(UV) sync --group neural --group dev

ffmpeg-check:
	@command -v ffmpeg >/dev/null 2>&1 \
	  && echo "✓ ffmpeg found: $$(ffmpeg -version | head -1)" \
	  || (echo "✗ ffmpeg not found — install with: brew install ffmpeg (or apt install ffmpeg)"; exit 1)

# ----------------------------------------------------------------------------
# Data (PHASE 2/3)
# ----------------------------------------------------------------------------
extract-audio: ffmpeg-check
	$(PY) -m src.scripts.extract_audio

preprocess:
	$(PY) $(MAIN) mode=preprocess $(ARGS)

featurize:
	$(PY) $(MAIN) mode=featurize device=$(DEVICE) $(ARGS)

data: preprocess featurize
	@echo "✓ Data ready: audio extracted, windows indexed, features in data/processed/"

# ----------------------------------------------------------------------------
# Training / evaluation (PHASE 4/5)
# ----------------------------------------------------------------------------
train: train-rf

# RandomForest baseline — 100% CPU, never imports Lightning.
train-rf:
	$(PY) $(MAIN) mode=train model=random_forest trainer=sklearn $(ARGS)

# Cross-attention (Lightning, optional). Requires `make setup-neural`.
# device=auto resolves MPS ▸ CUDA ▸ CPU; MPS gets a CPU fallback for unsupported ops.
train-neural:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=cross_attention \
	  device=$(DEVICE) $(ARGS)

# Hyperparameter sweep (Hydra multirun + joblib launcher).
sweep:
	@test -n "$(SWEEP)" || (echo "Set SWEEP, e.g.: make sweep SWEEP=\"model.lr=1e-3,5e-4\""; exit 1)
	$(PY) $(MAIN) -m $(_EXP) $(SWEEP) $(ARGS)

evaluate:
	$(PY) $(MAIN) mode=evaluate $(_EXP) $(_SPLIT) device=$(DEVICE) $(ARGS)

submit:
	$(PY) $(MAIN) mode=submit $(_EXP) $(if $(SPLIT),split=$(SPLIT),split=test) \
	  out=$(OUT) device=$(DEVICE) $(ARGS)

# --- Seed ensemble (cross-attention) ----------------------------------------
# Trains N seeds (same architecture/hyperparameters, only the seed changes) and logs
# the run dirs into a manifest. The ensemble AVERAGES per-video probabilities →
# dissolves seed-to-seed variance (the model saturates within 1–3 epochs on the small
# val set) and beats the single-model ceiling.
# SEEDS="42 1 2" changes the seeds; ARGS="..." passes overrides (e.g. text_embedder, lr).
train-ensemble:
	@mkdir -p outputs/cross_attention
	@: > $(ENSEMBLE_MANIFEST)
	@for s in $(SEEDS); do \
	  echo ">>> training seed=$$s"; \
	  $(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=cross_attention \
	    device=$(DEVICE) seed=$$s $(ARGS) || exit 1; \
	  ls -dt outputs/cross_attention/2*/ | head -1 | sed 's#/$$##' >> $(ENSEMBLE_MANIFEST); \
	done
	@echo "✓ Ensemble trained ($(words $(SEEDS)) seeds). Manifest: $(ENSEMBLE_MANIFEST)"
	@cat $(ENSEMBLE_MANIFEST)

# Manifest run dirs in Hydra list format (dir1,dir2,...).
_ENS_LIST = $(shell paste -sd, $(ENSEMBLE_MANIFEST) 2>/dev/null)

# Evaluates the manifest ensemble (probability averaging + threshold recalibrated on
# the calibration split — data.calib_split, default val — with ENS_CALIB).
ensemble-evaluate:
	@test -s $(ENSEMBLE_MANIFEST) || (echo "Empty manifest: run 'make train-ensemble' first."; exit 1)
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=evaluate +experiment=cross_attention \
	  device=$(DEVICE) $(if $(SPLIT),split=$(SPLIT),split=test) \
	  aggregation.calibration=$(ENS_CALIB) "ensemble=[$(_ENS_LIST)]" $(ARGS)

# Writes the submission file from the manifest ensemble.
ensemble-submit:
	@test -s $(ENSEMBLE_MANIFEST) || (echo "Empty manifest: run 'make train-ensemble' first."; exit 1)
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=submit +experiment=cross_attention \
	  device=$(DEVICE) $(if $(SPLIT),split=$(SPLIT),split=test) out=$(OUT) \
	  aggregation.calibration=$(ENS_CALIB) "ensemble=[$(_ENS_LIST)]" $(ARGS)

# --- Paper reproduction ------------------------------------------------------
# Best result of the paper (Table 2, "MIL + 74 support features, 5 seeds"):
# Macro-F1 0.722 · AP 0.875 on the 525-video public test split, using the
# TRADITIONAL protocol — model weights fit on train (778 videos), threshold
# calibrated on val (124 videos, `smooth` plateau selection), metrics on test.
# Runs end to end: data prep → 5-seed training → ensemble evaluation.
# Prerequisites: `make setup-neural` + BAH dataset under data/raw/data/ (README §3).
reproduce-best:
	$(MAKE) data
	$(MAKE) train-ensemble SEEDS="42 1 2 3 4"
	$(MAKE) ensemble-evaluate SPLIT=test ENS_CALIB=smooth ARGS="+ensemble_name=ensemble_5 $(ARGS)"
	@echo "✓ Done. Reference: Macro-F1 0.722 · AP 0.875 (test, threshold calibrated on val)."
	@echo "  Reports: outputs/cross_attention/ensemble_5/eval_test/"

# End to end (data -> train -> evaluate).
pipeline: data train evaluate
	@echo "✓ Full pipeline executed."

# ----------------------------------------------------------------------------
# Quality / CI-CD
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

# Fast syntax check without installing deps — uses the system python.
compile:
	@python3 -m py_compile $$(find src -name '*.py') $(MAIN) && echo "✓ py_compile OK"

# Fast (static) gate: runs whatever is deterministic/green without the dataset.
ci: format-check lint compile
	@echo "✓ CI OK (format-check + lint + compile)"

# Full gate (includes type-check and tests).
check: ci typecheck test
	@echo "✓ Full check OK"

# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	@echo "✓ Caches removed"

clean-cache:
	rm -rf data/interim data/processed
	@echo "✓ Derived artifacts removed (data/interim, data/processed)"

clean-outputs:
	rm -rf outputs multirun wandb
	@echo "✓ Outputs removed (outputs/, multirun/, wandb/)"

clean-all: clean clean-cache clean-outputs
	@echo "✓ Full cleanup"
