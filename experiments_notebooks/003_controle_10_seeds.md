# EXP-003 — Controle: 10 seeds × mistura de arquiteturas

- **Data**: 2026-10-14 (abertura: plano e comandos, antes dos runs) · — (conclusão)
- **Status**: em execução — os runs foram feitos no servidor, mas os resultados não chegaram ao bucket: a última versão é `20261014T220619.896Z-e594` (a do [EXP-002](002_protocolo_oficial_calibracao.md)). O `make outputs-push` da sessão `ctl-fim` provavelmente foi recusado porque o bucket ficou público.
- **Modelo / preset(s)**: `cross_attention` · `moe_r1_text` → `moe_r5_face` · ensembles de 10 seeds · protocolo tradicional
- **Runs** (esperados, `outputs/` do servidor `alceu01`):
  - seeds novas 5, 6, 7, 8, 9: `oficial/exp0-extra/s<seed>/<ts>`, `oficial/r1-extra/s<seed>/<ts>`, `oficial/exp1-extra/s<seed>/<ts>`
  - ensembles de 10: `ensemble/oficial-exp0x10` (manifesto `oficial/exp0x10_manifest.txt`) e `ensemble/oficial-exp1x10` (`oficial/exp1x10_manifest.txt`)
  - comparação: `ensemble/oficial-exp4` (Exp 0 + Exp 1, 5 + 5 seeds, do EXP-002)
- **Commit**: desconhecido até a versão do bucket existir (o código no servidor era o `b00df98`, segundo o push do EXP-002)
- **Relacionados**: [EXP-002](002_protocolo_oficial_calibracao.md) (os ensembles de 5 seeds e o Exp 4)

> **XYZ** — Obter **o Exp 4 (5 seeds do artigo + 5 do MoE) acima do artigo
> com 10 seeds e do MoE com 10 seeds, no AP e no Macro-F1 do test, com o IC 95%
> do Δ pareado inteiro acima de 0**, medido por **Macro-F1 a nível de vídeo no
> test oficial (525 vídeos) com o limiar escolhido só no val (124 vídeos) por
> cada estratégia, em ensembles por média de probabilidades
> (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
> `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
> `eval_test/predictions.csv`**, fazendo **o número de seeds de cada
> arquitetura de 5 para 10 (`SEEDS="5 6 7 8 9"` somadas às 5 do EXP-002), com o
> ensemble sempre de 10 membros**.

## 1. Explicação

No [EXP-002](002_protocolo_oficial_calibracao.md), o Exp 4 (o artigo + o MoE do
Rodrigo) foi o melhor ensemble em todas as colunas (AP 0,8844 contra 0,8759 do
artigo), mas sem significância e com o dobro de membros: 10 seeds contra 5.
Este experimento separa as duas explicações. Se o artigo com 10 seeds empatar
com o Exp 4, o ganho era só o número de seeds; se o Exp 4 superar o artigo com
10 seeds e o MoE com 10 seeds, misturar as duas arquiteturas ajuda. A regra de
decisão foi fixada antes dos runs:

- artigo com 10 seeds ≈ Exp 4 → o ganho era do número de seeds → fica o artigo
  sozinho, com 10 seeds;
- Exp 4 > artigo com 10 seeds e > MoE com 10 seeds → misturar as arquiteturas
  ajuda → fica a mistura.

## 2. Objetivo — XYZ planejado

- **X — resultado esperado**: o Exp 4 (5 seeds do artigo + 5 do MoE) acima do
  artigo com 10 seeds e do MoE com 10 seeds, no AP e no Macro-F1 do test, com o
  IC 95% do Δ pareado inteiro acima de 0.
- **Y — evidência**: Macro-F1 a nível de vídeo no test oficial (525 vídeos)
  com o limiar escolhido só no val (124 vídeos) por cada estratégia, em
  ensembles por média de probabilidades
  (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
  `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
  `eval_test/predictions.csv`.
- **Z — intervenção**: o número de seeds de cada arquitetura de 5 para 10
  (`SEEDS="5 6 7 8 9"` somadas às 5 do EXP-002), com o ensemble sempre de 10
  membros.

## 3. Métodos

### 3.1 Protocolo de avaliação (Y)

| Item | Valor |
|---|---|
| Métrica | `eval_f1` e `eval_ap` de `thresholds.json` (as 5 estratégias do EXP-002) |
| Split / protocolo | tradicional, igual ao EXP-002: train 778 → val 124 → test 525 |
| Seleção do checkpoint | por seed, early stopping e checkpoint por `val_ap` |
| Limiar | as 5 estratégias escolhidas no val; a comparação principal usa o fixo 0,5 e o `smooth` (a do artigo), como no EXP-002 |
| Agregação | ensemble de 10 membros por média de probabilidades: artigo × 10, MoE × 10 e Exp 4 (5 + 5) |
| Baseline | `ensemble/oficial-exp4` (EXP-002) contra `ensemble/oficial-exp0x10` e `ensemble/oficial-exp1x10` |
| Seeds / repetições | seeds 42, 1, 2, 3, 4 (EXP-002) + 5, 6, 7, 8, 9 (novas): um treino por seed |

### 3.2 Configuração e execução (Z)

- **Variável manipulada**: o número de seeds por arquitetura (5 → 10).
- **Variáveis controladas**: receitas, protocolo, Parquet de janelas, as 5
  seeds do EXP-002 (reaproveitadas, não retreinadas), o tamanho do ensemble
  (10 membros nas três condições).
- **Comando** (fila do tmux; o código no servidor é o do EXP-002):

  GPU 1, o artigo com 10 seeds:

  ```bash
  make train-seeds EXPERIMENT=cross_attention NAME=oficial/exp0-extra SEEDS="5 6 7 8 9" DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  cat outputs/oficial/exp0/manifest.txt outputs/oficial/exp0-extra/manifest.txt > outputs/oficial/exp0x10_manifest.txt
  make ensemble-evaluate ENSEMBLE_MANIFEST=outputs/oficial/exp0x10_manifest.txt SPLIT=test DEVICE=cuda ARGS="+ensemble_name=oficial-exp0x10"
  ```

  GPU 0, o MoE com 10 seeds (cada seed da Rodada 5 parte do texto da mesma
  seed da Rodada 1):

  ```bash
  make train-seeds EXPERIMENT=moe_r1_text NAME=oficial/r1-extra SEEDS="5 6 7 8 9" DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make train-seeds EXPERIMENT=moe_r5_face NAME=oficial/exp1-extra TEXT_NAME=oficial/r1-extra SEEDS="5 6 7 8 9" DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  cat outputs/oficial/exp1/manifest.txt outputs/oficial/exp1-extra/manifest.txt > outputs/oficial/exp1x10_manifest.txt
  make ensemble-evaluate ENSEMBLE_MANIFEST=outputs/oficial/exp1x10_manifest.txt SPLIT=test DEVICE=cuda ARGS="+ensemble_name=oficial-exp1x10"
  ```

  Depois das duas filas:

  ```bash
  make thresholds RUNS="outputs/ensemble/oficial-exp0 outputs/ensemble/oficial-exp0x10 outputs/ensemble/oficial-exp1 outputs/ensemble/oficial-exp1x10 outputs/ensemble/oficial-exp4"
  make outputs-push
  ```

- **Diferença de config**: só a seed e a pasta (`run_dirs.train`) entre as
  seeds de uma receita.
- **Recursos**: *(a preencher quando houver runs)*

## 4. Discussão

*(a preencher quando houver runs)*

## 5. Conclusões e resultados

### 5.1 Resultados

*(a preencher quando houver runs)*

### 5.2 XYZ realizado

*(a preencher quando houver runs)*

### 5.3 Próximos passos

*(a preencher quando houver runs)*
