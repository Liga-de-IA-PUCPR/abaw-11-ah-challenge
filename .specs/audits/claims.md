# Claims Ledger — BAH AH-Challenge (ABAW11)

> Registro persistente de claims de auditoria e entregas verificadas.
> Claims RESOLVED não são reabertos sem nova evidência.

## Headers

| ID | Category | Status | Title | Evidence |
|----|----------|--------|-------|----------|
| C001 | Performance | RESOLVED | Ensemble ponderado supera média simples (full + gnn) | `outputs/ensemble_eval/weight_sweep.json`, `configs/ensemble/optimized.yaml` |
| C002 | Tooling | RESOLVED | Varredura de pesos sem retreino (`scripts/ensemble_sweep.py`) | `scripts/ensemble_sweep.py` |
| C003 | Delivery | RESOLVED | Submissão otimizada gerada com sucesso | `outputs/submission_ensemble_optimized.txt` |
| C004 | Architecture | RESOLVED | Pipeline ensemble (evaluate + submit) integrado ao `main.py` | `main.py` (`ensemble_evaluate`, `ensemble_submit`) |
| C005 | Model | RESOLVED | `multimodal_hetero_face` integrado ao ensemble com peso baixo | `configs/ensemble/optimized.yaml` |

## Claims

### C001 — Ensemble ponderado supera média simples (full + gnn)

- **Category:** Performance
- **Status:** RESOLVED (2026-07-12)
- **Confidence:** CONFIRMED (métricas reproduzíveis a partir de CSVs exportados)

**Baseline (antigo):** `+ensemble=default` — `combine: mean`, membros `multimodal_hetero_full` + `hetero_gnn_contrastive`.

| Ensemble | Val Macro-F1 | Test Macro-F1 (público) |
|----------|--------------|-------------------------|
| antigo full + gnn (média) | 0.7019 | 0.6692 |
| novo ponderado full + gnn + face | **0.7169** | **0.6843** |

Ganho: **+1.5 pp** val, **+1.5 pp** test (estimado no split público).

**Melhor config (validação):**

```yaml
combine: weighted
members:
  - model: multimodal_hetero_full
    weight: 0.416667
  - model: hetero_gnn_contrastive
    weight: 0.541667
  - model: multimodal_hetero_face
    weight: 0.041667
threshold: 0.46
```

**Evidência:**
- Baseline: `outputs/ensemble_eval/report_val.json`, `outputs/ensemble_eval/report_test.json`
- Otimizado: `outputs/ensemble_eval/weight_sweep.json` (`best_by_val`)
- Config persistida: `configs/ensemble/optimized.yaml`

---

### C002 — Varredura de pesos sem retreino

- **Category:** Tooling
- **Status:** RESOLVED (2026-07-12)
- **File:** `scripts/ensemble_sweep.py`

Script lê `predictions.csv` já exportados (val + test), varre grade de pesos normalizados
(`--step 0.05` default), escolhe limiar e pesos **somente na validação**, reporta teste uma vez.
Não carrega modelos nem usa GPU.

Saída: `outputs/ensemble_eval/weight_sweep.json` (top-25 por val F1).

---

### C003 — Submissão otimizada gerada

- **Category:** Delivery
- **Status:** RESOLVED (2026-07-12)
- **File:** `outputs/submission_ensemble_optimized.txt`

Submissão gerada via `mode=ensemble_submit` com `+ensemble=optimized`, `data.num_workers=2`.
525 vídeos + header (526 linhas). Comando loga `Submissão ensemble: …`.

Comparável em tamanho com `outputs/submission_ensemble.txt` (baseline).

---

### C004 — Pipeline ensemble no `main.py`

- **Category:** Architecture
- **Status:** RESOLVED (2026-07-12)
- **Files:** `main.py`, `configs/ensemble/default.yaml`, `configs/ensemble/optimized.yaml`

Modos Hydra:
- `mode=ensemble_evaluate` — combina probas, calibra limiar, grava `outputs/ensemble_eval/report_{split}.json`
- `mode=ensemble_submit` — calibra na val, infere no test, grava submissão

`combine` suportado: `mean` | `weighted` (peso por membro em `ensemble.members[].weight`).

---

### C005 — Face mesh no ensemble

- **Category:** Model
- **Status:** RESOLVED (2026-07-12)
- **Files:** `src/models/multimodal_hetero_face.py`, `configs/experiment/multimodal_hetero_face.yaml`

Checkpoint usado: `outputs/multimodal_hetero_face/20260712_141548`.

Peso otimizado **0.041667** (~4%) — contribuição marginal mas melhora val F1 quando combinado
com full + gnn. Ver procedimento em `references/gnn_training_procedure.md` § Face Mesh.

---

_Last updated: 2026-07-13 (C001–C005 resolved — ensemble otimizado e submissão)_
