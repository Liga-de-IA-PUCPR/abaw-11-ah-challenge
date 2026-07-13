# Audit Digest — BAH AH-Challenge (ABAW11)

> Resumo consolidado de auditorias e entregas verificadas.

---

## Audit Runs

### Run audit-001 — Ensemble e modelos GNN multimodais

> 2026-07-13 | Scope: pipeline GNN + ensemble + face | Artefatos: 5 claims RESOLVED

**Contexto:** Otimização de ensemble sem retreino, usando predições CSV exportadas dos três
melhores checkpoints Lightning.

**Entregas verificadas:**

| Artefato | Caminho |
|----------|---------|
| Varredura de pesos | `scripts/ensemble_sweep.py` |
| Resultado da varredura | `outputs/ensemble_eval/weight_sweep.json` |
| Config otimizada | `configs/ensemble/optimized.yaml` |
| Submissão otimizada | `outputs/submission_ensemble_optimized.txt` |
| Procedimento GNN/face | `references/gnn_training_procedure.md` |

**Resultados (Macro-F1):**

| Ensemble | Val F1 | Test F1 (público) | Limiar |
|----------|--------|---------------------|--------|
| antigo: full + gnn (média) | 0.7019 | 0.6692 | 0.46 val / 0.62 test |
| **novo: full + gnn + face (ponderado)** | **0.7169** | **0.6843** | 0.46 |

**Pesos otimizados (validação):**

| Modelo | Peso |
|--------|------|
| `multimodal_hetero_full` | 0.416667 |
| `hetero_gnn_contrastive` | 0.541667 |
| `multimodal_hetero_face` | 0.041667 |

**Métricas adicionais (ensemble otimizado, teste público):**
- AP classe positiva: 0.8392
- F1 classe 0: 0.6368 | F1 classe 1: 0.7318
- Matriz de confusão: [[142, 65], [97, 221]]

**Claims:** C001–C005 → RESOLVED. Ver `claims.md`.

**Observações:**
- `hetero_gnn_contrastive` domina o peso (~54%); face contribui pouco mas ajuda na combinação.
- Varredura é barata (CPU, sem GPU); repetível após novos treinos exportando CSVs.
- Submissão rodou com `data.num_workers=2` para não saturar a máquina.

---

### Run audit-002 — CatBoost + temperature scaling

> 2026-07-13 | Scope: sklearn tabular | Checkpoint: `outputs/catboost/20260713_145505`

**Pipeline:** CatBoost por janela (`auto_class_weights: Balanced`) → `mean_proba` →
temperature scaling (NLL, T=5.0 na val) → limiar smooth (τ=0.44).

**Resultados (test público, 525 vídeos):**

| Métrica | Valor |
|---------|-------|
| Macro-F1 | 0.6562 |
| Accuracy | 0.6590 |
| ROC-AUC | 0.7551 |
| AP | 0.8237 |
| Recall classe 0 / 1 | 0.720 / 0.619 |
| Matriz confusão | [[149, 58], [121, 197]] |

**Artefatos:** `outputs/catboost/20260713_145505/eval_test/` (metrics.json, plots/, predictions.csv).

**Comparação:** branch isa reportou F1 test **0.701** com CatBoost + smooth (sem temperature).
Gap provável: Parquet atual usa embeddings deep (768+768 wav2vec2/RoBERTa) vs features
librosa da isa; temperature saturou em T=5.0 (limite superior do otimizador).

**Melhor caminho competição:** ensemble GNN otimizado (F1 test 0.6843) > CatBoost tabular neste cache.

---

### Run audit-003 — Pipeline completa: features de suporte + `base_rate`

> 2026-07-13 | Scope: featurize + CatBoost + calibração Luiz | Checkpoint: `outputs/catboost/20260713_152213`

**Mudanças integradas (branch `origin/luiz`):**
- `hesitation.py` + `text_features.py` → tabular **d=74** (antes 17)
- Calibração `base_rate` + platô central em `smooth` (`aggregation.py`)
- `ensemble_evaluate` calibra limiar **sempre na val** (protocolo correto)

**Featurize:** `+experiment=featurize_deep` + `+data.force=true` →
`data/processed/text_audio_windows.parquet` (15622 janelas, d_text=768, d_audio=768, d_tab=74).

**Pipeline vencedor:** CatBoost → `mean_proba` → **`base_rate`** (sem temperature).

| Modelo | Calibração | Val F1 | **Test F1** | Limiar (val) |
|--------|------------|--------|-------------|--------------|
| CatBoost + suporte | `base_rate` | 0.6120 | **0.7007** ✅ | 0.195 |
| CatBoost + suporte | `smooth` | 0.6314 | 0.6988 | 0.200 |
| Ensemble GNN (3 membros) | `base_rate` | 0.7132 | 0.6854 | 0.423 |
| Ensemble GNN | `argmax` | 0.7169 | 0.6843 | 0.460 |

**Teste público (CatBoost vencedor, 525 vídeos):**
- Macro-F1: **0.7007** | Accuracy: 0.7105 | ROC-AUC: 0.7837 | AP: 0.8443
- Matriz: [[139, 68], [84, 234]]

**Comando reprodução:**
```bash
uv run python main.py +experiment=featurize_deep mode=featurize device=cuda +data.force=true wandb.mode=disabled
uv run python main.py +experiment=catboost_baseline mode=train wandb.mode=disabled
uv run python main.py +experiment=catboost_baseline mode=evaluate split=test \
  checkpoint=outputs/catboost/20260713_152213 wandb.mode=disabled
```

**Melhor caminho competição (atualizado):** CatBoost com features de suporte + `base_rate` (**F1 test 0.7007**).

---

_Last updated: 2026-07-13_
