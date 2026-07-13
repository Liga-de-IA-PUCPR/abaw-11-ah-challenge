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

_Last updated: 2026-07-13_
