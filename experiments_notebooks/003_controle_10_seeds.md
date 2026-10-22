# EXP-003 — Controle: 10 seeds × mistura de arquiteturas

- **Data**: 2026-10-14 (abertura: plano e comandos, antes dos runs) · 2026-10-22 (verificação dos artefatos) · — (conclusão)
- **Status**: em execução — resultados do controle ainda indisponíveis no bucket em 2026-10-22. O bucket está **privado e acessível**; a listagem recursiva e o histórico só confirmam os artefatos do EXP-001 e do [EXP-002](002_protocolo_oficial_calibracao.md), com última versão `20261014T220619.896Z-e594`. Não foi possível confirmar os runs extras nem concluir o controle. A hipótese anterior de recusa do push por bucket público não explica o estado atual e não foi confirmada por log da sessão `ctl-fim`.
- **Modelo / preset(s)**: `cross_attention` · `moe_r1_text` → `moe_r5_face` · ensembles de 10 seeds · protocolo tradicional
- **Runs** (caminhos relativos a `outputs/` do servidor `alceu01`; disponibilidade verificada no bucket em 2026-10-22):
  - **ausentes no bucket**, seeds novas 5, 6, 7, 8, 9: `oficial/exp0-extra/s<seed>/<ts>`, `oficial/r1-extra/s<seed>/<ts>`, `oficial/exp1-extra/s<seed>/<ts>`
  - **ausentes no bucket**, ensembles de 10: `ensemble/oficial-exp0x10` (manifesto `oficial/exp0x10_manifest.txt`) e `ensemble/oficial-exp1x10` (`oficial/exp1x10_manifest.txt`)
  - **disponível**, comparação: `ensemble/oficial-exp4` (Exp 0 + Exp 1, 5 + 5 seeds, do EXP-002), com `eval_test/{thresholds.json,predictions.csv,metrics.json}` e `oficial/exp4_manifest.txt`, na versão `20261014T220619.896Z-e594`
- **Commit**: desconhecido para os runs extras, ainda sem manifesto no bucket. Para a referência Exp 4, `b00df988` é o **commit do push** (`.versions/20261014T220619.896Z-e594/manifest.json → git.commit`), não uma confirmação do commit de execução dos runs extras.
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
- **Recursos**: GPUs e filas conforme os comandos planejados acima; tempo de
  execução e conclusão dos runs extras não confirmados por artefatos.
- **Verificação do arquivo durável (2026-10-22)**: `HfApi.bucket_info`
  confirmou `private=True` e `total_files=561`; `list_bucket_tree` com
  `recursive=True` não encontrou `oficial/{exp0-extra,r1-extra,exp1-extra}`,
  `oficial/{exp0x10_manifest,exp1x10_manifest}.txt` nem
  `ensemble/{oficial-exp0x10,oficial-exp1x10}`. O histórico contém somente
  `20261006T233147.300Z-f769` e `20261014T220619.896Z-e594`.
  Foram baixados apenas os artefatos sem pesos de `oficial/` e `ensemble/`
  para uma pasta temporária, preservando o `outputs/` local. Para repetir:

  ```bash
  uv run scripts/outputs_bucket.py --bucket LF-BF/abaw-11-ah-challenge-outputs log
  # Use uma pasta temporária nova para cada verificação.
  uv run scripts/outputs_bucket.py --bucket LF-BF/abaw-11-ah-challenge-outputs pull --local <pasta-temporaria> --prefix ensemble --exclude '*.ckpt' --exclude '*.joblib'
  uv run scripts/outputs_bucket.py --bucket LF-BF/abaw-11-ah-challenge-outputs pull --local <pasta-temporaria> --prefix oficial --exclude '*.ckpt' --exclude '*.joblib' --exclude '*wandb*'
  ```

## 4. Discussão

- **X** — a mistura tem uma referência medida, mas o efeito de aumentar as
  seeds de cada arquitetura ainda não é observável. **Y** — o Exp 4 tem AP
  0,8844 e Macro-F1 0,7328 com τ = 0,5, no test; os dois ensembles homogêneos
  de 10 seeds estão ausentes (§5.1). **Z** — os 10 membros do Exp 4 são as
  5 seeds de cada arquitetura já usadas no EXP-002; sua presença no bucket
  não comprova a execução do controle com seeds extras.
- **X** — o impedimento atual é a ausência dos artefatos, com acesso ao bucket
  funcionando. **Y** — a consulta de 2026-10-22 confirmou bucket privado,
  histórico sem nova versão e ausência dos caminhos esperados (§3.2).
  **Z** — é necessário localizar o destino do upload novo ou completar o
  envio no servidor antes da comparação.

**Ameaças à validade**: a referência disponível tem 5 + 5 membros; compará-la
somente aos ensembles homogêneos de 5 seeds do EXP-002 mantém a confusão
entre tamanho do ensemble e mistura de arquiteturas. Não rejeitar a diferença
no bootstrap também não prova equivalência nem que o ganho se deve apenas ao
número de seeds. Seleção do checkpoint e calibração usam o mesmo val; o test
já foi consultado no EXP-002, portanto este controle é uma análise adicional
nesse mesmo conjunto, sem novo test independente.

## 5. Conclusões e resultados

### 5.1 Resultados

Referência já arquivada, **reutilizada do EXP-002**, sem resultados novos do
controle:

| Condição | Run | Estratégia | Macro-F1 | AP | Positivos previstos | Limiar |
|---|---|---|---|---|---|---|
| Exp 4 (artigo 5 + MoE 5) | `ensemble/oficial-exp4` | `fixo_0.5` | 0,7328 | 0,8844 | 0,5219 | 0,5000 |
| Exp 4 (artigo 5 + MoE 5) | `ensemble/oficial-exp4` | `argmax` | 0,7324 | 0,8844 | 0,6400 | 0,4000 |
| Exp 4 (artigo 5 + MoE 5) | `ensemble/oficial-exp4` | `smooth` | 0,7314 | 0,8844 | 0,6171 | 0,4200 |
| Exp 4 (artigo 5 + MoE 5) | `ensemble/oficial-exp4` | `base_rate` | 0,7379 | 0,8844 | 0,5562 | 0,4622 |
| Exp 4 (artigo 5 + MoE 5) | `ensemble/oficial-exp4` | `bootstrap` | 0,7340 | 0,8844 | 0,6286 | 0,4100 |
| Artigo × 10 | `ensemble/oficial-exp0x10` | pendente | — | — | — | — |
| MoE × 10 | `ensemble/oficial-exp1x10` | pendente | — | — | — | — |

Fonte de todas as linhas medidas: bucket
`LF-BF/abaw-11-ah-challenge-outputs`, versão
`20261014T220619.896Z-e594`,
`ensemble/oficial-exp4/eval_test/thresholds.json`:
`strategies[].strategy`, `strategies[].eval_f1`, `eval_ap`,
`strategies[].eval_pos_rate_pred`, `strategies[].threshold`.
O arquivo registra `n_calib=124` e `n_eval=525`;
`oficial/exp4_manifest.txt` confirma os membros (seeds 42, 1, 2, 3, 4 de
`oficial/exp0` e de `oficial/exp1`).

ΔAP, ΔMacro-F1 e IC 95% do controle permanecem pendentes: faltam as
predições dos ensembles `oficial-exp0x10` e `oficial-exp1x10`.

### 5.2 XYZ realizado

- **X — obtido**: ainda não medido para o controle. A referência Exp 4
  disponível tem AP 0,8844 e Macro-F1 0,7328 (`fixo_0.5`) / 0,7314 (`smooth`)
  no test (§5.1). Esperado na §2: o Exp 4 (5 seeds do artigo + 5 do MoE)
  acima do artigo com 10 seeds e do MoE com 10 seeds, no AP e no Macro-F1 do
  test, com o IC 95% do Δ pareado inteiro acima de 0.
- **Y — evidência**: Macro-F1 a nível de vídeo no test oficial (525 vídeos)
  com o limiar escolhido só no val (124 vídeos) por cada estratégia, em
  ensembles por média de probabilidades
  (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
  `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
  `eval_test/predictions.csv`.
- **Z — intervenção**: permanece planejada como na §2; não é possível
  confirmar as seeds adicionais e a composição dos ensembles homogêneos
  sem seus manifestos e resultados.
- **Veredito**: pendente — a hipótese ainda não pode ser avaliada. O status
  continua `em execução` e o XYZ do topo permanece como objetivo.

### 5.3 Próximos passos

- Confirmar em qual bucket/caminho foram enviados os resultados do controle;
  se continuarem apenas no servidor, completar `make outputs-push` e
  conferir a nova versão e os caminhos da §3.2.
- Conferir as seeds extras, os manifestos com 10 membros únicos e os
  overrides; baixar os artefatos sem pesos para uma pasta temporária.
- Recalcular as diferenças pareadas Exp 4 − artigo × 10 e Exp 4 − MoE × 10
  em AP e nas 5 estratégias de Macro-F1, com 2000 reamostras, alinhando
  `video_id` e validando os mesmos rótulos. Preencher os ICs antes de decidir.
- Atualizar o XYZ do topo, o veredito, a data de conclusão e o índice só
  quando houver evidência do controle; não escolher uma receita com base
  apenas na referência do EXP-002.
