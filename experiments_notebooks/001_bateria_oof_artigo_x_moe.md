# EXP-001 — Bateria OOF: cross-attention do artigo × MoE do Rodrigo

- **Data**: 2026-10-06 (runs das 20:16 às 20:23, horário do servidor; push às 23:31 UTC) · 2026-10-13 (conclusão: análise dos artefatos). Registro retroativo, escrito em 2026-10-15.
- **Status**: concluído
- **Modelo / preset(s)**: `cross_attention` (artigo) · `moe_r1_text` · `moe_r5_face` · `cross_attention_video` · `moe_luiz_features` · MoERouter (`mode=route`) · protocolo OOF
- **Runs** (`outputs/` do servidor `alceu01`; bucket `LF-BF/abaw-11-ah-challenge-outputs`, versão `20261006T233147.300Z-f769`):
  - Exp 0 (baseline): `oof/exp0-cross-attention/20261006_201715`
  - R1 (texto da Rodada 1, ponto de partida do Exp 1): `oof/moe-r1-text/20261006_201635`
  - Exp 1: `oof/moe-r5-face/20261006_201918`
  - Exp 2: `oof/cross-attention-video/20261006_201827`
  - Exp 3: `oof/moe-luiz-features/20261006_202016`
  - Exp 4: `route/exp4-baselines/20261006_202255`; variante com os 4 membros: `route/final-todos/20261006_202305`
- **Commit**: `89534c3` (commit do push, branch `feat/moe-fusion`; `.versions/20261006T233147.300Z-f769/manifest.json → git.commit`)
- **Relacionados**: PR #7 (`feat/moe-fusion`, plano MoE em `references/improvement_plan.md`); [EXP-002](002_protocolo_oficial_calibracao.md) (a mesma comparação no protocolo do artigo)

> **XYZ** — Obtive **nenhuma variante acima do cross-attention do artigo:
> todos os gates de Macro-F1 deram `empate` (Exp 1 −0,0121 [−0,0405, +0,0165];
> Exp 2 −0,0195 [−0,0491, +0,0074]; Exp 3 −0,0210 [−0,0488, +0,0067]; roteador
> Exp 0 + Exp 1 contra o melhor membro +0,0067 [−0,0036, +0,0170])**, medido por
> **Macro-F1 a nível de vídeo com τ = 0,5 sobre as predições OOF dos 902 vídeos
> de train+val (5 dobras StratifiedGroupKFold por participante, `oof.seed` 42),
> com o gate pareado por bootstrap (1000 reamostras) contra o Exp 0
> (`oof_metrics.json → macro_f1`, `gate`; `oof/comparacoes.txt`)**, fazendo
> **a troca da receita (`EXPERIMENT=<preset>`) com o resto do protocolo OOF
> fixo**.

*Registro retroativo: os runs vieram antes do caderno. O X da §2 foi
reconstituído do pedido do Luiz em 2026-10-05 (comparar a arquitetura nova do
Rodrigo com a do artigo em quatro experimentos) e do critério que o próprio
protocolo OOF aplica: veredito `melhora` quando o IC 95% do ΔMacro-F1 fica
inteiro acima de 0.*

## 1. Explicação

O PR #7 implementou o plano MoE do Rodrigo (`references/improvement_plan.md`):
fusão ancorada em texto com SoftMoE, colunas novas no Parquet de janelas
(texto GoEmotions fine-tunado, ASR timing, marcadores de hesitação, áudio
wav2vec2-emotion, recortes de rosto) e o MoERouter sobre membros, tudo medido
pelo protocolo OOF. Esta bateria compara essa arquitetura com o cross-attention
do artigo em quatro experimentos: o MoE completo (Exp 1, `moe_r5_face`, com o
texto iniciado pela Rodada 1 da mesma dobra), o modelo do artigo com o vídeo do
Rodrigo (Exp 2), o MoE com o áudio e o texto do artigo (Exp 3) e o ensemble dos
dois baselines pelo MoERouter (Exp 4). O test oficial é medido uma única vez,
no fim, pelo roteador.

## 2. Objetivo — XYZ planejado

- **X — resultado esperado**: cada variante (Exp 1, 2 e 3) e o roteador
  (Exp 4) superam o cross-attention do artigo (Exp 0): veredito `melhora` no
  gate pareado, com o IC 95% do ΔMacro-F1@0,5 OOF inteiro acima de 0.
- **Y — evidência**: Macro-F1 a nível de vídeo com τ = 0,5 sobre as predições
  OOF dos 902 vídeos de train+val (5 dobras StratifiedGroupKFold por
  participante, `oof.seed` 42), com o gate pareado por bootstrap (1000
  reamostras) contra o Exp 0 (`oof_metrics.json → macro_f1`, `gate`;
  `oof/comparacoes.txt`).
- **Z — intervenção**: a troca da receita (`EXPERIMENT=<preset>`) com o resto
  do protocolo OOF fixo — `cross_attention` (Exp 0) × `moe_r5_face` (Exp 1) ×
  `cross_attention_video` (Exp 2) × `moe_luiz_features` (Exp 3); o Exp 4 é o
  `make route` sobre Exp 0 e Exp 1.

## 3. Métodos

### 3.1 Protocolo de avaliação (Y)

| Item | Valor |
|---|---|
| Métrica | `macro_f1` (Macro-F1 a nível de vídeo, τ = 0,5) e `ap` (AP da classe positiva), de `oof_metrics.json` |
| Split / protocolo | OOF: 902 vídeos de train+val (778 + 124, 225 participantes), 5 dobras StratifiedGroupKFold por participante, `oof.seed` 42; prevalência 0,5100 (`pos_rate_true`) |
| Seleção do checkpoint | por dobra, early stopping e checkpoint por `val_ap` num holdout interno de 15% dos participantes de treino (`oof.inner_val_frac`) |
| Limiar | τ fixo 0,5 (`oof.threshold`) |
| Agregação | predições das 5 dobras juntas; test = média das 5 dobras (`pred_test.csv`), medido uma vez com `route.measure_splits=[test]` |
| Baseline | Exp 0, `oof/exp0-cross-attention/20261006_201715`: o cross-attention do artigo no mesmo protocolo e nas mesmas dobras |
| Seeds / repetições | um treino por dobra e condição |

### 3.2 Configuração e execução (Z)

- **Variável manipulada**: o preset (receita inteira: arquitetura e features).
- **Variáveis controladas**: dobras (`oof.seed` 42), τ, `oof.inner_val_frac`,
  Parquet de janelas, `wandb.mode=offline`, `data.num_workers=4`.
- **Comando** de cada condição (fila do tmux: GPU 0 com R1 → Exp 1; GPU 1 com
  Exp 0 → Exp 2 → Exp 3; o Exp 4 depois das duas). Os overrides abaixo são os
  do `.hydra/overrides.yaml` de cada run:

  ```bash
  make oof EXPERIMENT=cross_attention DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4 experiment_name=exp0-cross-attention"
  make oof EXPERIMENT=moe_r1_text DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make oof EXPERIMENT=moe_r5_face TEXT_RUN=outputs/oof/moe-r1-text/20261006_201635 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make oof EXPERIMENT=cross_attention_video BASELINE=outputs/oof/exp0-cross-attention/20261006_201715 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make oof EXPERIMENT=moe_luiz_features BASELINE=outputs/oof/exp0-cross-attention/20261006_201715 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make route MEMBERS="outputs/oof/exp0-cross-attention/20261006_201715 outputs/oof/moe-r5-face/20261006_201918" ARGS="route.name=exp4-baselines route.measure_splits=[test]"
  make route MEMBERS="<os 4 runs: exp0, cross-attention-video, moe-r5-face, moe-luiz-features>" ARGS="route.name=final-todos route.measure_splits=[test]"
  make oof-compare A=<run> B=<run>   # saída em oof/comparacoes.txt
  ```

  O `TEXT_RUN` vira `model.branches.text.init_from='outputs/oof/moe-r1-text/20261006_201635/fold{fold}'`:
  cada dobra do Exp 1 parte do texto da mesma dobra da R1.
- **Diferença de config**: pelo `.hydra/overrides.yaml`, só o preset, o
  `oof.baseline` (Exp 2 e 3), o `init_from` (Exp 1) e o `experiment_name`.
- **Recursos**: 2× RTX 5090 no servidor `alceu01`. Cada OOF de 5 dobras levou
  de 1,2 a 3,3 min (`main.log`): Exp 0 20:17:15 → 20:18:26, R1 20:16:35 →
  20:19:16, Exp 1 20:19:18 → 20:22:38, Exp 2 20:18:27 → 20:20:15, Exp 3
  20:20:16 → 20:22:37; os roteadores, ~10 s.

## 4. Discussão

- **X** — Nenhuma variante supera o Exp 0 no Macro-F1 OOF. **Y** — gates de
  `oof/comparacoes.txt`: Exp 1 −0,0121 [−0,0405, +0,0165] (p = 0,444), Exp 2
  −0,0195 [−0,0491, +0,0074] (p = 0,160), Exp 3 −0,0210 [−0,0488, +0,0067]
  (p = 0,156); Exp 3 × Exp 1 −0,0089 [−0,0400, +0,0222] (p = 0,568). Todos
  `empate`. **Z** — Com 902 vídeos, o IC do Δ tem cerca de ±0,03: diferenças
  de 1 a 2 pontos entre receitas não se separam do ruído.
- **X** — O texto sozinho empata com o modelo multimodal do artigo. **Y** —
  R1 0,7202 contra Exp 0 0,7194 (`oof_metrics.json → macro_f1`); ΔF1 +0,0008
  [−0,0305, +0,0301], ΔAP +0,0020 [−0,0237, +0,0297]
  ([`analises/001_oof_reanalise.py`](analises/001_oof_reanalise.py)). **Z** —
  O RoBERTa-GoEmotions fine-tunado carrega quase todo o sinal que o artigo
  extrai de texto, áudio e tabular.
- **X** — O AP das 5 dobras juntas pune o MoE: o Exp 1 tem o maior AP por
  dobra e um dos menores AP juntos. **Y** — Exp 1: AP médio por dobra 0,8248
  e AP juntas 0,7415; Exp 0: 0,8171 e 0,8056
  ([`analises/001`](analises/001_oof_reanalise.py); `oof_metrics.json → ap`).
  Os limiares que o holdout interno calibrou nas dobras do Exp 1 vão de 0,3197
  a 0,9086; os do Exp 0, de 0,3495 a 0,5454 (`fold<k>/trainer_state.json →
  threshold`). **Z** — Cada dobra do MoE sai numa escala de probabilidade
  diferente; juntar as dobras mistura escalas e derruba o ranking global. O
  ΔAP "significativo" do Exp 1 contra o Exp 0 (−0,0641 [−0,1023, −0,0328],
  `oof/comparacoes.txt`) vem da régua, não de um ranking pior.
- **X** — O τ = 0,5 fixo pune o Exp 2. **Y** — O Exp 2 prevê 0,3736 de
  positivos (`pos_rate_pred`; real 0,5100) com AP por dobra 0,8181, igual ao do
  Exp 0 (0,8171); os limiares do holdout interno das dobras dele vão de 0,0095
  a 0,4324. **Z** — As colunas de rosto deslocam a escala para baixo; a perda
  de F1 vem do corte, não do ranking.
- **X** — O roteador não ganha do melhor membro e usa quase só o artigo.
  **Y** — Roteador Exp 0 + Exp 1: 0,7261 no OOF, `gate_vs_best` +0,0067
  [−0,0036, +0,0170] (`empate`), uso 0,8822 do Exp 0 e 0,1178 do Exp 1
  (`route_metrics.json → router`, `gate_vs_best`, `usage`); com os 4 membros,
  0,7183 e `gate_vs_best` −0,0011 [−0,0216, +0,0170]. **Z** — As predições dos
  membros andam juntas: no OOF não há complementaridade para o roteador
  aproveitar.
- **X** — No test (medição única), nada supera o Exp 0 com significância.
  **Y** — `route_metrics.json → measure_test`: Exp 0 0,7272 / AP 0,8767; Exp 1
  0,7113 / 0,8727; Exp 2 0,6933 / 0,8651; Exp 3 0,6955 / 0,8608; roteador de
  2 membros 0,7304 [0,6907, 0,7700] / 0,8820; roteador de 4 membros 0,7431
  [0,7026, 0,7806] / 0,8733. Pareado contra o Exp 0
  ([`analises/001`](analises/001_oof_reanalise.py)): roteador de 4 membros
  +0,0159 [−0,0032, +0,0355] (p = 0,117); roteador de 2 membros +0,0033
  [−0,0082, +0,0152], com ΔAP +0,0053 [+0,0003, +0,0115]. **Z** — No OOF o
  mesmo roteador de 4 membros não superou o melhor membro, então o +0,0159 do
  test é compatível com ruído.
- **X** — No test, com τ = 0,5, todos os modelos preveem menos positivos do
  que os 60,6% reais, e quem mais erra o corte mais perde F1. **Y** —
  Positivos previstos no test: Exp 0 0,5448, Exp 1 0,4952, Exp 2 0,3905,
  R1 0,4267 (real 0,6057); o R1 fica −0,0361 [−0,0715, −0,0008] (p = 0,045)
  abaixo do Exp 0 no F1 com AP 0,8739 contra 0,8767
  ([`analises/001`](analises/001_oof_reanalise.py)). **Z** — O test tem
  prevalência maior que o train+val (0,6057 × 0,5100); um corte fixo calibrado
  para ~51% fica baixo lá.
- **X** — Os modelos saturam cedo, e a bateria inteira levou ~7 min. **Y** —
  Melhor época por dobra (`fold<k>/trainer_state.json → ckpt_path`): Exp 0
  [2, 2, 0, 1, 3], Exp 1 [0, 11, 1, 0, 2], R1 [2, 2, 4, 6, 5]; os horários do
  `main.log` na §3.2. **Z** — Cada dobra treina com ~720 vídeos (menos os 15% de participantes
  do holdout interno), e o `val_ap` do holdout atinge o máximo nas primeiras
  épocas.

**Ameaças à validade**: um treino por dobra, sem repetição de seed; τ fixo
(pune modelos com escala deslocada); o AP das dobras juntas mistura escalas; os
presets comparam receitas inteiras (o Exp 1 muda arquitetura e features ao
mesmo tempo); o test vem da média de modelos treinados em train+val, então não
se compara diretamente com o número do artigo (treinado só no train); as
features foram recalculadas na GPU do servidor, não no Mac do artigo.

## 5. Conclusões e resultados

### 5.1 Resultados

OOF (902 vídeos, τ = 0,5):

| Condição | Run | Macro-F1 [IC 95%] | AP juntas | AP por dobra | Positivos previstos | ΔF1 vs Exp 0 [IC 95%] |
|---|---|---|---|---|---|---|
| Exp 0 — artigo (baseline) | `oof/exp0-cross-attention/20261006_201715` | 0,7194 [0,6907, 0,7482] | 0,8056 | 0,8171 | 0,5067 | — |
| R1 — texto GoEmotions | `oof/moe-r1-text/20261006_201635` | 0,7202 [0,6906, 0,7498] | 0,8076 | 0,8229 | 0,4157 | +0,0008 [−0,0305, +0,0301] |
| Exp 1 — MoE completo | `oof/moe-r5-face/20261006_201918` | 0,7073 [0,6777, 0,7383] | 0,7415 | 0,8248 | 0,4922 | −0,0121 [−0,0405, +0,0165] |
| Exp 2 — artigo + vídeo | `oof/cross-attention-video/20261006_201827` | 0,6999 [0,6676, 0,7288] | 0,7975 | 0,8181 | 0,3736 | −0,0195 [−0,0491, +0,0074] |
| Exp 3 — MoE + áudio/texto do artigo | `oof/moe-luiz-features/20261006_202016` | 0,6984 [0,6683, 0,7283] | 0,7448 | 0,8003 | 0,4967 | −0,0210 [−0,0488, +0,0067] |
| Exp 4 — roteador Exp 0 + Exp 1 | `route/exp4-baselines/20261006_202255` | 0,7261 [0,6969, 0,7538] | 0,8075 | — | 0,4823 | +0,0067 [−0,0036, +0,0170] (vs o melhor membro) |
| roteador com os 4 membros | `route/final-todos/20261006_202305` | 0,7183 [0,6885, 0,7450] | 0,7956 | — | 0,4723 | −0,0011 [−0,0216, +0,0170] (vs o melhor membro) |

Fonte: `oof_metrics.json → macro_f1`, `macro_f1_ci`, `ap`, `pos_rate_pred`;
`route_metrics.json → router`, `gate_vs_best`; ΔF1 do Exp 1, 2 e 3 de
`oof/comparacoes.txt`; AP por dobra e ΔF1 do R1 de
[`analises/001_oof_reanalise.py`](analises/001_oof_reanalise.py).

Test oficial (525 vídeos, 0,6057 positivos; média das 5 dobras, τ = 0,5,
medição única):

| Condição | Macro-F1 | AP | Positivos previstos | ΔF1 vs Exp 0 [IC 95%] | ΔAP vs Exp 0 [IC 95%] |
|---|---|---|---|---|---|
| Exp 0 — artigo | 0,7272 | 0,8767 | 0,5448 | — | — |
| R1 — texto | 0,6911 | 0,8739 | 0,4267 | −0,0361 [−0,0715, −0,0008] | −0,0028 [−0,0229, +0,0164] |
| Exp 1 — MoE completo | 0,7113 | 0,8727 | 0,4952 | −0,0158 [−0,0535, +0,0201] | −0,0039 [−0,0241, +0,0154] |
| Exp 2 — artigo + vídeo | 0,6933 | 0,8651 | 0,3905 | −0,0338 [−0,0689, +0,0004] | −0,0116 [−0,0245, +0,0014] |
| Exp 3 — MoE + áudio/texto do artigo | 0,6955 | 0,8608 | 0,5276 | −0,0316 [−0,0645, +0,0008] | −0,0159 [−0,0321, −0,0001] |
| Exp 4 — roteador Exp 0 + Exp 1 | 0,7304 | 0,8820 | 0,5276 | +0,0033 [−0,0082, +0,0152] | +0,0053 [+0,0003, +0,0115] |
| média simples Exp 0 + Exp 1 | 0,7196 | 0,8822 | 0,5162 | −0,0075 [−0,0337, +0,0174] | +0,0056 [−0,0046, +0,0156] |
| roteador com os 4 membros | 0,7431 | 0,8733 | 0,5124 | +0,0159 [−0,0032, +0,0355] | −0,0033 [−0,0208, +0,0100] |

Fonte: `route_metrics.json → measure_test` (`members`, `router`,
`mean_of_members`) dos dois roteadores; a linha do R1 e os Δ pareados
(bootstrap, 2000 reamostras) de
[`analises/001_oof_reanalise.py`](analises/001_oof_reanalise.py).

### 5.2 XYZ realizado

- **X — obtido**: nenhuma variante acima do cross-attention do artigo: todos
  os gates de Macro-F1 deram `empate` (Exp 1 −0,0121 [−0,0405, +0,0165];
  Exp 2 −0,0195 [−0,0491, +0,0074]; Exp 3 −0,0210 [−0,0488, +0,0067];
  roteador Exp 0 + Exp 1 contra o melhor membro +0,0067 [−0,0036, +0,0170])
  (esperado na §2: veredito `melhora`).
- **Y — evidência**: Macro-F1 a nível de vídeo com τ = 0,5 sobre as predições
  OOF dos 902 vídeos de train+val (5 dobras StratifiedGroupKFold por
  participante, `oof.seed` 42), com o gate pareado por bootstrap (1000
  reamostras) contra o Exp 0 (`oof_metrics.json → macro_f1`, `gate`;
  `oof/comparacoes.txt`).
- **Z — intervenção**: a planejada. Entraram também o R1 (ponto de partida do
  Exp 1, medido como receita) e o roteador com os 4 membros (`final-todos`).
- **Veredito**: refutada — a arquitetura MoE e as variantes não superam o
  cross-attention do artigo; no máximo empatam.

### 5.3 Próximos passos

- Repetir a comparação no protocolo do artigo (train → val → test, ensemble
  de 5 seeds) e testar o limiar calibrado no val em vez do τ fixo:
  [EXP-002](002_protocolo_oficial_calibracao.md).
- Reportar o AP médio por dobra ao lado do AP das dobras juntas no
  `oof_metrics.json` (mudança de código a registrar no PR do protocolo OOF):
  hoje a régua de AP do OOF pune modelos com escala instável entre dobras.
