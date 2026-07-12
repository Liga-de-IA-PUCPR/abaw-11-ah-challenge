# Calibração automática do limiar (`src/training`)

A métrica oficial é **Macro-F1 a nível de vídeo**, mas os modelos produzem um
**score contínuo** por vídeo. Transformar score → decisão 0/1 exige um **limiar**,
e a escolha desse limiar é a *calibração*. Tudo vive em
[`aggregation.py`](aggregation.py) e é controlado pelo grupo Hydra
[`configs/aggregation/`](../../configs/aggregation/default.yaml).

## Como funciona (2 passos)

1. **Agrega janela → vídeo** (`aggregate_to_video` / `video_scores`): o RF gera 1
   proba por *janela*; agregamos num score por *vídeo* (`method`, default
   `mean_proba`). O `cross_attention` já emite 1 score por vídeo → `method="identity"`.
2. **Calibra o limiar na validação** (`calibrate_threshold`): varre limiares em
   `[0,1]`, mede o Macro-F1 **na val** em cada um e escolhe o melhor. O limiar é
   gravado no checkpoint e **reusado** em `evaluate`/`submit`.

> O limiar é um **parâmetro do modelo**: aprende-se na **val**, nunca no test.
> Otimizar no test infla a métrica e não generaliza para a *hidden test* oficial.

## Por que o default é `smooth` (e não o pico cru)

A curva Macro-F1 × limiar é uma **função degrau**. Com val pequena (~124 vídeos)
ela fica serrilhada, e o `argmax` cru pode fisgar um **pico de sorte** que não
transfere. Caso real deste repo (calibrado **só na val**):

| `calibration` | limiar (da val) | F1 val | **F1 no test** |
|---|---|---|---|
| `argmax` | 0.30 (spike) | 0.719 | 0.614 ❌ |
| `smooth` | 0.63 (platô) | 0.707 | **0.701** ✅ |

`smooth` suaviza a curva (média móvel de largura `smooth_window`) antes do
`argmax`, escolhendo o **centro do platô estável** — que generaliza. Ver
`_select_threshold_index` em [`aggregation.py`](aggregation.py).

## Configuração (`configs/aggregation/default.yaml`)

| campo | valores | o que faz |
|---|---|---|
| `method` | `mean_proba`·`max_proba`·`frac_positive`·`any` | agregação janela→vídeo (RF) |
| `threshold` | `auto` · `<float>` | `auto` calibra na val; um float **fixa** e pula a calibração |
| `calibration` | `smooth` · `argmax` | estratégia de escolha do limiar (default `smooth`) |
| `smooth_window` | float (ex. `0.10`) | largura da média móvel, em unidades de limiar |
| `metric` | `macro_f1` | métrica maximizada (lida de `metrics.primary`) |

## Como sobrescrever (sem editar YAML)

Overrides Hydra via `ARGS="..."` no Makefile:

```bash
# fixar o limiar já no treino (pula a calibração)
make train-neural ARGS="aggregation.threshold=0.6"

# voltar ao pico cru (para comparação)
make train-rf ARGS="aggregation.calibration=argmax"

# janela de suavização maior (platô mais largo)
make train-rf ARGS="aggregation.smooth_window=0.15"

# trocar o limiar no test SEM re-treinar (reaproveita o checkpoint)
make evaluate EXPERIMENT=cross_attention SPLIT=test ARGS="aggregation.threshold=0.6"
```

`aggregation.threshold=<float>` em `evaluate`/`submit` sobrepõe o limiar salvo no
checkpoint (`main._apply_threshold_override`) — útil para varrer o ponto de
operação sem treinar de novo.

## Onde está no código

| arquivo | papel |
|---|---|
| [`aggregation.py`](aggregation.py) | `calibrate_threshold`, `_select_threshold_index`, `aggregate_to_video`, `threshold_curve` |
| [`sklearn_trainer.py`](sklearn_trainer.py) · [`lightning_trainer.py`](lightning_trainer.py) | leem `aggregation.*` e chamam `calibrate_threshold` no `fit` |
| [`../../main.py`](../../main.py) | `_apply_threshold_override` (override em evaluate/submit) |
