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
# Multimodal (Rodrigo's GNNs + video; see references/integration_plan.md):
#   make setup-gnn        # + torch-geometric + gnn-modalblocks (setup-vision adds MediaPipe)
#   make featurize-w2v    # wav2vec2 cache used by the GNN presets
#   make train-gnn        # heterogeneous GNN (GNN_EXPERIMENT=hetero_gnn_v2_tune_wav2vec2)
#   make ensemble-multimodal   # cross-attention seeds + GNN (+ face) run dirs, prob. averaging
#
# MoE plan (text-anchored fusion + SoftMoE; see references/improvement_plan.md):
#   make featurize-moe    # extra Parquet columns (transcript, ASR timing, markers, audio, face)
#   make oof EXPERIMENT=moe_r1_text           # OOF protocol (5 participant folds, fixed τ)
#   make oof EXPERIMENT=moe_r2_text_tab TEXT_RUN=outputs/oof/moe-r1-text/<ts> BASELINE=<run>
#   make route MEMBERS="<oof run> <oof run> ..."   # MoERouter over the members
#
# Outputs archive (private Hugging Face bucket, versioned; push never deletes there):
#   make outputs-push     # local → bucket: what changed becomes a new version
#   make outputs-pull     # bucket → local (PREFIX=<dir under outputs/> or RUN_DIR=<run dir>)
#   make outputs-sync     # push + pull   |   outputs-status   what each direction would do
#   make outputs-free     # push + deletes local weights the bucket already has (same hash)
#   make outputs-log      # version history: who, when, git commit, what changed
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
# Multimodal (Rodrigo's GNNs + video). The cross-attention reads the default (librosa)
# cache; the GNN/face presets read the wav2vec2 cache (W2V_PARQUET). Each ensemble member
# loads its own cache for the same videos.
GNN_EXPERIMENT    ?= hetero_gnn_v2_tune_wav2vec2
FACE_EXPERIMENT   ?= face_gnn_ts_roi
W2V_PARQUET       ?= data/processed/text_audio_windows_w2v.parquet
GNN_RUN           ?= outputs/hetero_gnn_contrastive/20260713_162753
FACE_RUN          ?=
MM_ENS_NAME       ?= ensemble_multimodal
# MoE plan: extra columns + OOF protocol + MoERouter.
MOE_COLUMNS       ?= transcript asr_timing hesitation_markers
MOE_AUDIO         ?= wav2vec2_emotion_large
MOE_VISION        ?= vit_face_expression
BASELINE          ?=
TEXT_RUN          ?=
MEMBERS           ?=

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
        setup-gnn setup-vision featurize-w2v featurize-face train-gnn train-face \
        eval-run eval-ensemble-members ensemble-multimodal ensemble-multimodal-submit \
        meta-router featurize-moe featurize-moe-audio featurize-moe-face featurize-moe-scene \
        oof route \
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
	@echo "  Multimodal — Rodrigo's GNNs + video (references/integration_plan.md):"
	@echo "    setup-gnn        + gnn group (torch-geometric, gnn-modalblocks)"
	@echo "    setup-vision     + vision group (MediaPipe Face Mesh, OpenCV)"
	@echo "    featurize-w2v    wav2vec2 cache ($(W2V_PARQUET)) for the GNN presets"
	@echo "    featurize-face   Face Mesh landmarks -> face_landmarks column (FACE_EXPERIMENT)"
	@echo "    train-gnn        trains GNN_EXPERIMENT=$(GNN_EXPERIMENT)"
	@echo "    train-face       trains FACE_EXPERIMENT=$(FACE_EXPERIMENT)"
	@echo "    eval-run         RUN_DIR=<dir> EXPERIMENT=<preset>: eval_{train,val,test} + predictions.csv"
	@echo "    eval-ensemble-members  eval-run for every run in the cross-attention manifest"
	@echo "    ensemble-multimodal    manifest CA seeds + GNN_RUN (+ FACE_RUN), prob. averaging"
	@echo "    ensemble-multimodal-submit  submission from that ensemble (OUT=$(OUT))"
	@echo "    meta-router      CA⊕GNN router over the members' predictions.csv"
	@echo ""
	@echo "  MoE plan — text-anchored fusion + SoftMoE (references/improvement_plan.md):"
	@echo "    featurize-moe    extra columns: MOE_COLUMNS=\"$(MOE_COLUMNS)\""
	@echo "    featurize-moe-audio  audio column (MOE_AUDIO=$(MOE_AUDIO))"
	@echo "    featurize-moe-face   face/eyes/mouth column (MOE_VISION=$(MOE_VISION))"
	@echo "    featurize-moe-scene  optional scene column (VideoMAE)"
	@echo "    oof              OOF protocol of EXPERIMENT (+ gate vs BASELINE=<oof run>)"
	@echo "    route            MoERouter over MEMBERS=\"<oof run> ...\""
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
	@echo "  Outputs archive — private HF bucket $(HF_BUCKET) (scripts/outputs_bucket.py):"
	@echo "    outputs-status   both sides + what push/pull would do (transfers nothing)"
	@echo "    outputs-push     local → bucket: new/changed files become a VERSION; the content"
	@echo "                     it overwrites is archived in .versions/ (never deletes there)"
	@echo "    outputs-pull     bucket → local: missing/newer files, bucket mtime restored"
	@echo "                     (never deletes here nor replaces local content the bucket lacks)"
	@echo "    outputs-sync     push + pull (bidirectional: newest mtime wins, nothing is lost)"
	@echo "    outputs-free     push + deletes local *.ckpt/*.joblib with the bucket's Xet hash"
	@echo "    outputs-log      version history (who, when, git commit, what changed)"
	@echo "    target: PREFIX=<file|dir under outputs/> or RUN_DIR=outputs/<model>/<run>"
	@echo "    flags:  ARGS=\"--dry-run\" | \"--exclude '*.ckpt'\" | \"--rehydrate\" (pull/sync)"
	@echo ""
	@echo "  Parameters: DEVICE=$(DEVICE)  SPLIT  OUT  EXPERIMENT  ARGS  SWEEP  SEEDS  ENS_CALIB"
	@echo "              GNN_EXPERIMENT  FACE_EXPERIMENT  GNN_RUN  FACE_RUN  W2V_PARQUET  RUN_DIR"

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
setup:
	$(UV) sync

setup-neural:
	$(UV) sync --group neural

setup-all:
	$(UV) sync --group neural --group dev

# Rodrigo's heterogeneous GNNs (torch-geometric + gnn-modalblocks from git).
setup-gnn:
	$(UV) sync --group neural --group gnn --group dev

# + video: MediaPipe Face Mesh for mode=featurize_face.
setup-vision:
	$(UV) sync --group neural --group gnn --group vision --group dev

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

# --- Multimodal: Rodrigo's GNNs + video on top of this pipeline --------------
# The GNN/face presets (configs/experiment/hetero_gnn*, multimodal_*, face_gnn_ts*)
# consume the SAME Parquet contract; wav2vec2 presets point to W2V_PARQUET.
comma := ,

featurize-w2v:
	$(PY) $(MAIN) mode=featurize +experiment=featurize_deep device=$(DEVICE) \
	  data.paths.parquet_path=$(W2V_PARQUET) $(ARGS)

# Adds the face_landmarks column (478x3 per window) to the preset's Parquet.
# Needs the .mp4 files under data/raw/data/Videos and `make setup-vision`.
featurize-face:
	$(PY) $(MAIN) mode=featurize_face +experiment=$(FACE_EXPERIMENT) $(ARGS)

train-gnn:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=$(GNN_EXPERIMENT) \
	  device=$(DEVICE) $(ARGS)

train-face:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=train +experiment=$(FACE_EXPERIMENT) \
	  device=$(DEVICE) $(ARGS)

# eval_{train,val,test}/ (metrics + plots + predictions.csv) for ONE run dir.
# EXPERIMENT = the preset it was trained with (selects the model and its cache).
eval-run:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR=<run dir> (and EXPERIMENT=<training preset>)"; exit 1)
	@for s in train val test; do \
	  $(MPS_FALLBACK) $(PY) $(MAIN) mode=evaluate $(_EXP) checkpoint=$(RUN_DIR) split=$$s \
	    device=$(DEVICE) $(ARGS) || exit 1; \
	done

# predictions.csv for every cross-attention seed of the manifest (meta-router input).
eval-ensemble-members:
	@test -s $(ENSEMBLE_MANIFEST) || (echo "Empty manifest: run 'make train-ensemble' first."; exit 1)
	@for r in $$(cat $(ENSEMBLE_MANIFEST)); do \
	  $(MAKE) --no-print-directory eval-run RUN_DIR=$$r EXPERIMENT=cross_attention || exit 1; \
	done

# Heterogeneous ensemble: manifest CA seeds (librosa cache) + GNN_RUN (+ FACE_RUN), each
# member reading its own cache; threshold recalibrated on val (ENS_CALIB) as usual.
# (items built with $(comma): a literal comma inside $(if ...) would split its arguments)
_MM_GNN_ITEM  = {checkpoint:$(GNN_RUN)$(comma)parquet_path:$(W2V_PARQUET)}
_MM_FACE_ITEM = {checkpoint:$(FACE_RUN)$(comma)parquet_path:$(W2V_PARQUET)}
_MM_MEMBERS   = $(_ENS_LIST),$(_MM_GNN_ITEM)$(if $(FACE_RUN),$(comma)$(_MM_FACE_ITEM))

ensemble-multimodal:
	@test -s $(ENSEMBLE_MANIFEST) || (echo "Empty manifest: run 'make train-ensemble' first."; exit 1)
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=evaluate +experiment=cross_attention \
	  device=$(DEVICE) $(if $(SPLIT),split=$(SPLIT),split=test) \
	  aggregation.calibration=$(ENS_CALIB) "ensemble=[$(_MM_MEMBERS)]" \
	  +ensemble_name=$(MM_ENS_NAME) $(ARGS)

ensemble-multimodal-submit:
	@test -s $(ENSEMBLE_MANIFEST) || (echo "Empty manifest: run 'make train-ensemble' first."; exit 1)
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=submit +experiment=cross_attention \
	  device=$(DEVICE) $(if $(SPLIT),split=$(SPLIT),split=test) out=$(OUT) \
	  aggregation.calibration=$(ENS_CALIB) "ensemble=[$(_MM_MEMBERS)]" $(ARGS)

# Rodrigo's CA⊕GNN meta-router (logreg on disagreements) over the members'
# predictions.csv (run `make eval-ensemble-members` + `make eval-run RUN_DIR=$(GNN_RUN) ...`).
meta-router:
	$(PY) scripts/meta_router_ca_gnn.py \
	  --ca-runs $$(for r in $$(cat $(ENSEMBLE_MANIFEST)); do basename $$r; done) \
	  --gnn-run $(GNN_RUN) $(ARGS)

# --- MoE plan (text-anchored fusion + SoftMoE) -------------------------------
# Extra Parquet columns (mode=featurize_columns): video-level text/ASR signals, then the
# per-window audio/face embeddings of the chosen embedders. Needs `make data` first.
_COLS = $(subst $(eval) ,$(comma),$(strip $(1)))

featurize-moe:
	$(PY) $(MAIN) mode=featurize_columns "columns=[$(call _COLS,$(MOE_COLUMNS))]" $(ARGS)

featurize-moe-audio:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=featurize_columns "columns=[audio]" \
	  audio_embedder=$(MOE_AUDIO) device=$(DEVICE) $(ARGS)

featurize-moe-face:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=featurize_columns "columns=[face_crops]" \
	  vision_embedder=$(MOE_VISION) device=$(DEVICE) $(ARGS)

featurize-moe-scene:
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=featurize_columns "columns=[scene]" device=$(DEVICE) $(ARGS)

# OOF protocol of EXPERIMENT (any registry model). BASELINE=<oof run> adds the paired gate;
# TEXT_RUN=<oof run of moe_r1_text> makes each fold start from that fold's text model.
oof:
	@test -n "$(EXPERIMENT)" || (echo "Set EXPERIMENT=<preset>, e.g. EXPERIMENT=moe_r1_text"; exit 1)
	$(MPS_FALLBACK) $(PY) $(MAIN) mode=oof $(_EXP) device=$(DEVICE) \
	  $(if $(BASELINE),oof.baseline=$(BASELINE)) \
	  $(if $(TEXT_RUN),"model.branches.text.init_from='$(TEXT_RUN)/fold{fold}'") $(ARGS)

# MoERouter over OOF runs (same folds): evaluated on the members' folds + final prediction.
route:
	@test -n "$(MEMBERS)" || (echo "Set MEMBERS=\"<oof run> <oof run> ...\""; exit 1)
	$(PY) $(MAIN) mode=route "route.members=[$(call _COLS,$(MEMBERS))]" $(ARGS)

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
# Outputs archive — private Hugging Face Storage Bucket (scripts/outputs_bucket.py)
# ----------------------------------------------------------------------------
# outputs/ is versioned in a PRIVATE bucket (checkpoints and predictions derive from the
# BAH dataset, whose EULA forbids redistribution; the script refuses a public bucket).
# Nothing is ever lost: push never deletes in the bucket, and before overwriting a file
# it archives the previous content under .versions/<version>/archive/ (server-side copy,
# no re-upload); every push that changes something is a version with a manifest (who,
# when, git commit, what changed). Pull never deletes local files nor replaces local
# content the bucket does not have. Free only deletes what the bucket holds with the
# same Xet hash, and a full pull does not bring it back (PREFIX/RUN_DIR does).
# The script runs in its OWN environment (PEP 723 + scripts/outputs_bucket.py.lock): it
# needs huggingface_hub>=2, and transformers pins the project's to <2.
# Auth: `hf auth login` or HF_TOKEN (write token). Another bucket: HF_BUCKET=<ns>/<name>.
HF_BUCKET      ?= LF-BF/abaw-11-ah-challenge-outputs
OUTPUTS_BUCKET := $(UV) run scripts/outputs_bucket.py --bucket $(HF_BUCKET)
# Target (empty = all of outputs/): PREFIX=<file|dir under outputs/> or RUN_DIR=outputs/<...>
_OB_PREFIX     = $(or $(PREFIX),$(patsubst outputs/%,%,$(RUN_DIR)))
_OB_TARGET     = $(if $(_OB_PREFIX),--prefix "$(_OB_PREFIX)",)

.PHONY: outputs-status outputs-push outputs-pull outputs-sync outputs-free outputs-log

outputs-status:
	$(OUTPUTS_BUCKET) status $(_OB_TARGET) $(ARGS)

outputs-push:
	$(OUTPUTS_BUCKET) push $(_OB_TARGET) $(ARGS)

outputs-pull:
	$(OUTPUTS_BUCKET) pull $(_OB_TARGET) $(ARGS)

outputs-sync:
	$(OUTPUTS_BUCKET) sync $(_OB_TARGET) $(ARGS)

# The script pushes first: free only deletes what the bucket already holds.
outputs-free:
	$(OUTPUTS_BUCKET) free $(_OB_TARGET) $(ARGS)

outputs-log:
	$(OUTPUTS_BUCKET) log $(_OB_TARGET) $(ARGS)

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
