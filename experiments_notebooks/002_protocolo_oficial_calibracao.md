# EXP-002 — Protocolo oficial (train → val → test) com 5 estratégias de limiar

- **Data**: 2026-10-14 (runs das 18:55 às 19:04, horário do servidor; push às 22:06 UTC) · 2026-10-14 (conclusão). Registro retroativo, escrito em 2026-10-15.
- **Status**: concluído
- **Modelo / preset(s)**: `cross_attention` · `moe_r1_text` → `moe_r5_face` · `cross_attention_video` · `moe_luiz_features` · ensembles de 5 seeds · protocolo tradicional
- **Runs** (`outputs/` do servidor `alceu01`; bucket `LF-BF/abaw-11-ah-challenge-outputs`, versão `20261014T220619.896Z-e594`):
  - treinos: `oficial/<exp>/s<seed>/<ts>` com `<exp>` ∈ {`r1`, `exp0`, `exp1`, `exp2`, `exp3`} e seeds 42, 1, 2, 3, 4; manifestos `oficial/<exp>/manifest.txt` e `oficial/exp4_manifest.txt`
  - ensembles: `ensemble/oficial-exp0` (baseline) · `ensemble/oficial-exp1` · `ensemble/oficial-exp2` · `ensemble/oficial-exp3` · `ensemble/oficial-exp4`, cada um com `eval_test/`
- **Commit**: `b00df98` (commit do push, branch `feat/moe-fusion`; `.versions/20261014T220619.896Z-e594/manifest.json → git.commit`). Inclui a seleção `bootstrap` (`933f039`), o `make train-seeds` (`3705d7b`) e o `thresholds.json` no `mode=evaluate` (`3723bd9`).
- **Relacionados**: [EXP-001](001_bateria_oof_artigo_x_moe.md) (as mesmas receitas no protocolo OOF); referência do artigo em `references/integration_plan.md` (F2); [EXP-003](003_controle_10_seeds.md) (controle de 10 seeds)

> **XYZ** — Obtive **nenhuma estratégia de limiar acima do τ = 0,5 fixo
> (média nos 5 ensembles: `base_rate` 0,7077, fixo 0,7074, `smooth` 0,7047,
> `bootstrap` 0,7001, `argmax` 0,6995; nenhum Δ pareado contra o fixo com o IC
> inteiro acima de 0) e nenhuma receita acima do artigo (a melhor, Exp 0 +
> Exp 1: ΔF1 +0,0063 [−0,0237, +0,0350] com `smooth`, ΔAP +0,0085 [−0,0023,
> +0,0190])**, medido por **Macro-F1 a nível de vídeo no test oficial (525
> vídeos) com o limiar escolhido só no val (124 vídeos) por cada estratégia,
> em ensembles de 5 seeds por média de probabilidades
> (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
> `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
> `eval_test/predictions.csv`**, fazendo **a estratégia de limiar
> (`aggregation.calibration`: fixo 0,5 · `argmax` · `smooth` · `base_rate` ·
> `bootstrap`) sobre as 5 receitas treinadas no protocolo do artigo**.

*Registro retroativo. O X da §2 foi reconstituído dos dois princípios que o
Luiz pediu em 2026-10-14 — calibrar o limiar em vez de fixar, com um método
novo para val pequeno, e restaurar as separações oficiais train/val/test — e
da régua dos outros experimentos (IC 95% do Δ pareado acima de 0). As
estratégias foram fixadas no código antes de os runs existirem.*

## 1. Explicação

O [EXP-001](001_bateria_oof_artigo_x_moe.md) mediu as receitas no protocolo OOF,
com τ fixo e as dobras de train+val juntas, e mostrou duas distorções da régua:
o τ fixo pune modelos com escala deslocada e o AP das dobras juntas pune
escalas instáveis. Este experimento volta ao protocolo do artigo (`make
reproduce-best`): pesos só no train, early stopping e limiar no val, test medido
uma vez, cada receita como ensemble de 5 seeds. Sobre os mesmos ensembles,
compara 5 estratégias de limiar escolhidas só no val. Uma delas é nova, a
`bootstrap`: a mediana, sobre 1000 reamostras estratificadas do val, do limiar
que maximiza o Macro-F1 em cada uma. Ela mira o mesmo ótimo do `argmax` com
menos variância, sem a janela fixa do `smooth` nem a hipótese do `base_rate`
(ótimo = prevalência).

## 2. Objetivo — XYZ planejado

- **X — resultado esperado**: (1) alguma estratégia escolhida só no val — em
  especial a `bootstrap` — supera o τ = 0,5 fixo no Macro-F1 do test, com o IC
  95% do Δ pareado inteiro acima de 0; (2) no protocolo oficial, alguma receita
  supera o Exp 0 (o artigo) pelo mesmo critério.
- **Y — evidência**: Macro-F1 a nível de vídeo no test oficial (525 vídeos)
  com o limiar escolhido só no val (124 vídeos) por cada estratégia, em
  ensembles de 5 seeds por média de probabilidades
  (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
  `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
  `eval_test/predictions.csv`.
- **Z — intervenção**: a estratégia de limiar (`aggregation.calibration`:
  fixo 0,5 · `argmax` · `smooth` · `base_rate` · `bootstrap`) sobre as 5
  receitas treinadas no protocolo do artigo.

## 3. Métodos

### 3.1 Protocolo de avaliação (Y)

| Item | Valor |
|---|---|
| Métrica | `eval_f1` (Macro-F1 a nível de vídeo no test, com o limiar da estratégia) e `eval_ap` (AP), de `thresholds.json`; `metrics.json → metrics.macro_f1` é a estratégia `smooth` (a do artigo) |
| Split / protocolo | tradicional: pesos no train (778 vídeos), early stopping e limiar no val (124 vídeos, 30 participantes), test oficial (525 vídeos, 75 participantes, prevalência 0,6057) medido uma vez |
| Seleção do checkpoint | por seed, early stopping e checkpoint por `val_ap` |
| Limiar | 5 estratégias escolhidas no val do ensemble (`compare_threshold_strategies`, `src/training/aggregation.py`) |
| Agregação | ensemble de 5 seeds por média de probabilidades; Exp 4 = as 10 seeds do Exp 0 e do Exp 1 |
| Baseline | Exp 0, `ensemble/oficial-exp0` — o protocolo do `make reproduce-best`; referência do artigo 0,7226 / AP 0,8750 (`references/integration_plan.md`, F2) |
| Seeds / repetições | seeds 42, 1, 2, 3, 4: um treino por seed e receita |

### 3.2 Configuração e execução (Z)

- **Variável manipulada**: `aggregation.calibration`, com o fixo 0,5 como
  referência. As receitas (presets) são o segundo eixo, como no EXP-001.
- **Variáveis controladas**: o Parquet de janelas e as colunas do PR #7, as
  seeds, o early stopping por `val_ap`, as mesmas predições do ensemble para
  todas as estratégias (só o corte muda).
- **Comando** de cada condição (fila do tmux: GPU 0 com R1 → Exp 1; GPU 1 com
  Exp 0 → Exp 2 → Exp 3; Exp 4 e a tabela no fim):

  ```bash
  make train-seeds EXPERIMENT=cross_attention NAME=oficial/exp0 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make train-seeds EXPERIMENT=moe_r1_text NAME=oficial/r1 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make train-seeds EXPERIMENT=moe_r5_face NAME=oficial/exp1 TEXT_NAME=oficial/r1 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make train-seeds EXPERIMENT=cross_attention_video NAME=oficial/exp2 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make train-seeds EXPERIMENT=moe_luiz_features NAME=oficial/exp3 DEVICE=cuda ARGS="wandb.mode=offline data.num_workers=4"
  make ensemble-evaluate ENSEMBLE_MANIFEST=outputs/oficial/<exp>/manifest.txt SPLIT=test DEVICE=cuda ARGS="+ensemble_name=oficial-<exp>"
  cat outputs/oficial/exp0/manifest.txt outputs/oficial/exp1/manifest.txt > outputs/oficial/exp4_manifest.txt
  make thresholds RUNS="outputs/ensemble/oficial-exp0 … outputs/ensemble/oficial-exp4"
  ```

  Cada seed vira `mode=train +experiment=<preset> device=cuda seed=<s>
  run_dirs.train=oficial/<exp>/s<s>` (`.hydra/overrides.yaml`); no Exp 1, mais
  `model.branches.text.init_from=outputs/oficial/r1/s<s>/<ts>` (o texto da
  mesma seed da R1). Cada ensemble vira `mode=evaluate +experiment=cross_attention
  split=test aggregation.calibration=smooth ensemble=[...]
  +ensemble_name=oficial-<exp>` (`logs/evaluate/<ts>/.hydra/overrides.yaml`).
- **Diferença de config**: entre as estratégias, nenhuma — as 5 leem as mesmas
  probabilidades. Entre as receitas, o preset (e o `init_from` do Exp 1).
- **Recursos**: 2× RTX 5090 no servidor `alceu01`. Cada treino levou de 16 a
  55 s e cada avaliação de ensemble de 26 s a 1 min 22 s (`main.log`); a
  bateria inteira, das 18:55 às 19:04.

## 4. Discussão

- **X** — O protocolo reproduz o artigo. **Y** — Exp 0 com `smooth`: Macro-F1
  0,7251 e AP 0,8759, limiar 0,4700 (`ensemble/oficial-exp0/eval_test/metrics.json
  → metrics.macro_f1`, `metrics.average_precision`, `metrics.threshold`),
  contra 0,7226 / 0,8750 do artigo (`references/integration_plan.md`, F2).
  **Z** — Mesma receita e mesmo protocolo; a diferença de +0,0025 vem das
  features recalculadas na GPU do servidor e do não determinismo das seeds.
- **X** — Nenhuma estratégia de limiar supera o fixo 0,5. **Y** — Média nos 5
  ensembles: `base_rate` 0,7077, fixo 0,7074, `smooth` 0,7047, `bootstrap`
  0,7001, `argmax` 0,6995; nenhum Δ contra o fixo tem o IC inteiro acima de 0
  ([`analises/002`](analises/002_oficial_reanalise.py)). A calibração piora o
  Exp 0 (`base_rate` −0,0310 [−0,0645, +0,0003]; `bootstrap` −0,0306 [−0,0638,
  +0,0018]) e o Exp 1 (`smooth` e `bootstrap` −0,0178 [−0,0388, +0,0005]) e só
  ajuda o Exp 3 (`base_rate` +0,0278 [−0,0006, +0,0571]). **Z** — Explicado no
  próximo item.
- **X** — O ótimo do val não fica onde está o do test, e erra para os dois
  lados. **Y** — `argmax` do val contra o limiar ótimo do próprio test
  (diagnóstico, com os rótulos do test;
  [`analises/002`](analises/002_oficial_reanalise.py)): Exp 0 0,40 × 0,46;
  Exp 1 0,57 × 0,40; Exp 2 0,25 × 0,44; Exp 3 0,42 × 0,31; Exp 4 0,40 × 0,44.
  **Z** — O val tem 124 vídeos de 30 participantes; o formato da curva dele
  difere do test (75 participantes) como um todo. É viés da amostra, não a
  variância de um ou dois vídeos.
- **X** — A `bootstrap` (nova) acompanha o `argmax` do val e não corrige o
  erro. **Y** — Limiar `bootstrap` × `argmax` (`thresholds.json →
  strategies[].threshold`): 0,41 × 0,40 (Exp 0), 0,57 × 0,57 (Exp 1), 0,34 ×
  0,25 (Exp 2), 0,43 × 0,42 (Exp 3), 0,41 × 0,40 (Exp 4). **Z** — A reamostragem
  reduz a variância do pico dentro do val, mas não corrige um val enviesado
  inteiro; e reamostra vídeos, não participantes (são só 30), então subestima a
  incerteza real.
- **X** — O limiar vale de 0,2 a 1,6 ponto nos ensembles bons; a receita vale
  até 7,8. **Y** — Ótimo do test − fixo: Exp 0 0,7310 − 0,7292 = +0,0018;
  Exp 1 0,7439 − 0,7284 = +0,0155; Exp 2 0,7033 − 0,6917 = +0,0116; Exp 3
  0,6895 − 0,6552 = +0,0343; Exp 4 0,7429 − 0,7328 = +0,0101
  ([`analises/002`](analises/002_oficial_reanalise.py); `thresholds.json`). No
  fixo, as receitas vão de 0,6552 a 0,7328. **Z** — Mesmo o corte perfeito, que
  não é implantável, rende pouco perto da escolha do modelo.
- **X** — O MoE empata com o artigo; o vídeo no artigo e o MoE com as features
  do artigo pioram; Exp 0 + Exp 1 é o melhor em todas as colunas, sem
  significância. **Y** — Pareado contra o Exp 0
  ([`analises/002`](analises/002_oficial_reanalise.py)): Exp 1 ΔF1 −0,0145
  [−0,0494, +0,0200] com `smooth` e −0,0008 com o fixo, ΔAP −0,0027; Exp 2
  −0,0375 [−0,0698, −0,0024] com o fixo (p = 0,042); Exp 3 −0,0628 [−0,1015,
  −0,0242] com `smooth`, ΔAP −0,0271 [−0,0452, −0,0103]; Exp 4 +0,0063 [−0,0237,
  +0,0350] com `smooth`, ΔAP +0,0085 [−0,0023, +0,0190]. **Z** — A ordem bate
  com o [EXP-001](001_bateria_oof_artigo_x_moe.md). O AP 0,8844 do Exp 4 é o
  maior no test entre os runs deste caderno, mas ele tem o dobro de seeds do Exp 0: o ganho pode
  ser só de seeds ([EXP-003](003_controle_10_seeds.md)).
- **X** — Uma seed sozinha oscila e satura cedo. **Y** — Macro-F1 do val por
  seed (`main.log`, linha "Treino concluído"): Exp 0 0,6693 ± 0,0348, Exp 1
  0,6896 ± 0,0229, Exp 3 0,6660 ± 0,0327; melhor época do Exp 0 [0, 1, 0, 1, 2];
  limiar salvo pelas seeds do Exp 3 entre 0,0143 e 0,4421
  ([`analises/002`](analises/002_oficial_reanalise.py)). **Z** — A escala de
  probabilidade de cada seed do MoE varia muito; a média de 5 seeds a
  estabiliza.

**Ameaças à validade**: o val decide o early stopping e o limiar (seleção e
calibração no mesmo val; o único número honesto é o do test); 5 seeds por
receita; o Exp 4 tem o dobro de membros dos outros; o limiar ótimo do test é
diagnóstico (usa os rótulos do test); as 5 estratégias são comparadas no mesmo
test sem correção para múltiplas comparações; os presets comparam receitas
inteiras; um único test (525 vídeos, 75 participantes).

## 5. Conclusões e resultados

### 5.1 Resultados

Macro-F1 no test (525 vídeos, 0,6057 positivos), limiar escolhido no val:

| Ensemble (5 seeds) | Pasta | AP | fixo 0,5 | `argmax` | `smooth` | `base_rate` | `bootstrap` |
|---|---|---|---|---|---|---|---|
| Exp 0 — artigo (baseline) | `ensemble/oficial-exp0` | 0,8759 | 0,7292 | 0,7076 | 0,7251 | 0,6981 | 0,6986 |
| Exp 1 — MoE completo | `ensemble/oficial-exp1` | 0,8732 | 0,7284 | 0,7105 | 0,7105 | 0,7260 | 0,7105 |
| Exp 2 — artigo + vídeo | `ensemble/oficial-exp2` | 0,8639 | 0,6917 | 0,6717 | 0,6944 | 0,6937 | 0,6869 |
| Exp 3 — MoE + áudio/texto do artigo | `ensemble/oficial-exp3` | 0,8488 | 0,6552 | 0,6753 | 0,6623 | 0,6830 | 0,6703 |
| Exp 4 — Exp 0 + Exp 1 (10 seeds) | `ensemble/oficial-exp4` | 0,8844 | 0,7328 | 0,7324 | 0,7314 | 0,7379 | 0,7340 |
| média das 5 linhas | — | — | 0,7074 | 0,6995 | 0,7047 | 0,7077 | 0,7001 |

Fonte: `eval_test/thresholds.json → eval_ap`, `strategies[].eval_f1`; média de
[`analises/002`](analises/002_oficial_reanalise.py).

Limiar escolhido no val (positivos previstos no test entre parênteses) e o
ótimo do test, só como diagnóstico:

| Ensemble | fixo | `argmax` | `smooth` | `base_rate` | `bootstrap` | ótimo do test |
|---|---|---|---|---|---|---|
| Exp 0 | 0,50 (0,5181) | 0,40 (0,6800) | 0,47 (0,5676) | 0,4154 (0,6514) | 0,41 (0,6610) | 0,46 → 0,7310 |
| Exp 1 | 0,50 (0,5295) | 0,57 (0,4743) | 0,57 (0,4743) | 0,4884 (0,5352) | 0,57 (0,4743) | 0,40 → 0,7439 |
| Exp 2 | 0,50 (0,5276) | 0,25 (0,7295) | 0,31 (0,6838) | 0,3673 (0,6286) | 0,34 (0,6571) | 0,44 → 0,7033 |
| Exp 3 | 0,50 (0,4686) | 0,42 (0,5143) | 0,48 (0,4800) | 0,3207 (0,5905) | 0,43 (0,5048) | 0,31 → 0,6895 |
| Exp 4 | 0,50 (0,5219) | 0,40 (0,6400) | 0,42 (0,6171) | 0,4622 (0,5562) | 0,41 (0,6286) | 0,44 → 0,7429 |

Fonte: `thresholds.json → strategies[].threshold`, `eval_pos_rate_pred`; ótimo
do test de [`analises/002`](analises/002_oficial_reanalise.py).

Comparações pareadas (bootstrap, 2000 reamostras;
[`analises/002`](analises/002_oficial_reanalise.py)):

| Contra o Exp 0 | ΔF1 `smooth` | ΔF1 fixo 0,5 | ΔAP |
|---|---|---|---|
| Exp 1 | −0,0145 [−0,0494, +0,0200] | −0,0008 [−0,0350, +0,0351] | −0,0027 [−0,0237, +0,0163] |
| Exp 2 | −0,0307 [−0,0689, +0,0070] | −0,0375 [−0,0698, −0,0024] | −0,0120 [−0,0263, +0,0023] |
| Exp 3 | −0,0628 [−0,1015, −0,0242] | −0,0739 [−0,1094, −0,0375] | −0,0271 [−0,0452, −0,0103] |
| Exp 4 | +0,0063 [−0,0237, +0,0350] | +0,0036 [−0,0231, +0,0322] | +0,0085 [−0,0023, +0,0190] |

| Contra o fixo 0,5, no mesmo ensemble | `smooth` | `base_rate` | `bootstrap` |
|---|---|---|---|
| Exp 0 | −0,0041 [−0,0243, +0,0140] | −0,0310 [−0,0645, +0,0003] | −0,0306 [−0,0638, +0,0018] |
| Exp 1 | −0,0178 [−0,0388, +0,0005] | −0,0024 [−0,0103, +0,0036] | −0,0178 [−0,0388, +0,0005] |
| Exp 2 | +0,0027 [−0,0325, +0,0363] | +0,0020 [−0,0266, +0,0288] | −0,0047 [−0,0379, +0,0255] |
| Exp 3 | +0,0070 [−0,0007, +0,0165] | +0,0278 [−0,0006, +0,0571] | +0,0150 [−0,0010, +0,0309] |
| Exp 4 | −0,0014 [−0,0281, +0,0264] | +0,0051 [−0,0110, +0,0210] | +0,0012 [−0,0269, +0,0296] |

Seeds (val, fim do treino; [`analises/002`](analises/002_oficial_reanalise.py)
sobre o `main.log` e o `trainer_state.json` de cada seed):

| Receita | Macro-F1 val | AP val | Melhor época por seed | Limiar salvo (`base_rate`) |
|---|---|---|---|---|
| R1 | 0,6829 ± 0,0270 | 0,8585 ± 0,0054 | 2, 2, 1, 3, 2 | 0,3992–0,4674 |
| Exp 0 | 0,6693 ± 0,0348 | 0,8475 ± 0,0065 | 0, 1, 0, 1, 2 | 0,3084–0,5340 |
| Exp 1 | 0,6896 ± 0,0229 | 0,8637 ± 0,0074 | 8, 0, 1, 1, 7 | 0,2284–0,7365 |
| Exp 2 | 0,6626 ± 0,0239 | 0,8519 ± 0,0069 | 4, 5, 0, 9, 3 | 0,1872–0,5666 |
| Exp 3 | 0,6660 ± 0,0327 | 0,8506 ± 0,0099 | 8, 1, 5, 11, 4 | 0,0143–0,4421 |

### 5.2 XYZ realizado

- **X — obtido**: nenhuma estratégia de limiar acima do τ = 0,5 fixo (média
  nos 5 ensembles: `base_rate` 0,7077, fixo 0,7074, `smooth` 0,7047,
  `bootstrap` 0,7001, `argmax` 0,6995; nenhum Δ pareado contra o fixo com o IC
  inteiro acima de 0) e nenhuma receita acima do artigo (a melhor, Exp 0 +
  Exp 1: ΔF1 +0,0063 [−0,0237, +0,0350] com `smooth`, ΔAP +0,0085 [−0,0023,
  +0,0190]) (esperado na §2: uma estratégia, em especial a `bootstrap`, e uma
  receita acima das referências com o IC do Δ acima de 0).
- **Y — evidência**: Macro-F1 a nível de vídeo no test oficial (525 vídeos)
  com o limiar escolhido só no val (124 vídeos) por cada estratégia, em
  ensembles de 5 seeds por média de probabilidades
  (`ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`,
  `eval_ap`), com a comparação pareada por bootstrap (2000 reamostras) sobre
  `eval_test/predictions.csv`.
- **Z — intervenção**: a planejada.
- **Veredito**: refutada — com este val, calibrar o limiar não supera o τ = 0,5
  fixo, a `bootstrap` não corrige o viés do val, e nenhuma receita supera o
  artigo com significância.

### 5.3 Próximos passos

- Separar o efeito do número de seeds do efeito de misturar arquiteturas no
  Exp 4: [EXP-003](003_controle_10_seeds.md).
- Default de limiar: o τ = 0,5 fixo (ou o `base_rate`) para comparar e para o
  número final; a `bootstrap` fica disponível, mas não como default. Decisão a
  registrar em `references/` junto do merge do PR #7.
- Descartar as receitas do Exp 2 (vídeo no artigo) e do Exp 3 (MoE com as
  features do artigo).
