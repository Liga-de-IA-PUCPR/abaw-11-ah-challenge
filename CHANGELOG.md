# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
e este projeto adere a [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added

- Protocolo de avaliação OOF (`src/eval/protocol.py`): CV agrupada por participante
  (5 folds `StratifiedGroupKFold`, sem vazamento de participante entre folds),
  Macro-F1/AP com limiar fixo e IC por bootstrap pareado, mais `scripts/ensemble_ap.py`
  (ensemble ponderado por AP e re-score do meta-router CA⊕GNN com limiar fixo 0.5),
  para reduzir o sobreajuste ao val de 124 vídeos (FASE 0 de `docs/improvement_plan.md`).
- Fine-tune de texto GoEmotions (`src/models/text_finetune.py`, `model=text_finetune`,
  `text_embedder=roberta_goemotions`): `SamLowe/roberta-base-go_emotions` com embeddings
  + 4 primeiras camadas congeladas, CLS pooling, R-Drop, label smoothing 0.1, BCE
  class-balanced e cabeça auxiliar sobre o vetor de hesitação (74-d, peso 0.3) — FASE 1
  de `docs/improvement_plan.md` (`configs/experiment/text_goemotions_finetune.yaml`).
- ASR-erased time (`src/features/asr_timing.py`): 16 features determinísticas dos gaps
  entre chunks do Whisper (`VideoRecord.transcript_chunks`), tratando corretamente o
  reset de linha do tempo de 30s do decoder (gaps só contados onde monotônico) — FASE 2a
  de `docs/improvement_plan.md` (`tests/test_asr_timing.py`).
