# EXP-NNN — Título curto do experimento

- **Data**: AAAA-MM-DD (abertura) · AAAA-MM-DD (conclusão)
- **Status**: planejado | em execução | concluído | abandonado
- **Modelo / preset(s)**: `cross_attention` | `moe_r5_face` | … · protocolo tradicional | OOF
- **Runs**: pastas em `outputs/` (uma por condição, baseline incluído) + a versão do bucket onde estão arquivadas
- **Commit**: hash do código (o run não grava o commit; o manifesto da versão do bucket grava o commit do push)
- **Relacionados**: experimentos anteriores, PRs, seções de `references/`

> **XYZ** — Obtive **X** (resultado observável, com número e baseline), medido
> por **Y** (métrica + split/protocolo + limiar + seeds + arquivo), fazendo
> **Z** (a intervenção, como override ou preset).

Enquanto o experimento não termina, a linha XYZ é a hipótese ("Obter X, medido
por Y, fazendo Z"); na conclusão ela é reescrita com o X obtido e o MESMO Y.

## 1. Explicação

O que é o experimento e por que ele existe, em um parágrafo: a pergunta, o que
a motivou (um resultado anterior, uma seção do plano) e o que muda no pipeline.

## 2. Objetivo — XYZ planejado

- **X — resultado esperado**: o critério de sucesso, observável e com número
  e comparação (ex.: Macro-F1 no test ≥ baseline + 0,01, com o IC 95% do Δ
  pareado acima de 0). "Melhorar o modelo" não é X.
- **Y — evidência**: a métrica e o protocolo que decidem X. O Y fica congelado
  aqui: a §5.2 usa exatamente este texto.
- **Z — intervenção**: o que é feito para obter X — UMA variável manipulada,
  escrita como override (`model.extra_columns=[...]`) ou preset
  (`EXPERIMENT=<preset>`).

## 3. Métodos

### 3.1 Protocolo de avaliação (Y)

| Item | Valor |
|---|---|
| Métrica | chave exata (`macro_f1`, `ap`, `eval_f1`, …) e o que ela mede |
| Split / protocolo | tradicional (train 778 → val 124 → test 525) ou OOF (5 dobras por participante sobre 902) |
| Seleção do checkpoint | early stopping e checkpoint por `val_ap` (no val ou no holdout interno da dobra) |
| Limiar | τ fixo 0,5 ou a estratégia escolhida no val (`thresholds.json`) |
| Agregação | média de probabilidades entre dobras ou seeds; AP juntas × AP por dobra |
| Baseline | run de comparação e por que ele é comparável |
| Seeds / repetições | `seed` e quantos treinos por condição |

### 3.2 Configuração e execução (Z)

- **Variável manipulada**: a chave (ou o preset) e os valores comparados.
- **Variáveis controladas**: o que fica fixo entre as condições (Parquet de
  janelas, dobras, limiar, seeds, épocas, early stopping).
- **Comando** de cada condição:

  ```bash
  make train-seeds EXPERIMENT=<preset> NAME=expNNN/<condição> DEVICE=cuda ARGS="<chave>=<valor>"
  ```

- **Diferença de config**: as linhas do `.hydra/overrides.yaml` que diferem
  entre a condição e o baseline — idealmente só a variável manipulada e o nome
  da pasta.
- **Recursos**: GPU, tempo por treino (do `main.log`), sessões do tmux.

## 4. Discussão

Um achado por item, sempre na ordem X → Y → Z:

- **X** — o que foi observado. **Y** — a evidência (número, run, arquivo,
  chave). **Z** — o que explica X (a intervenção, um mecanismo, uma
  comparação com o artigo ou com outro experimento).

**Ameaças à validade**: o que pode invalidar a leitura acima (poucas seeds,
seleção e calibração no mesmo val, receitas inteiras em vez de uma variável,
AP comparado entre conjuntos de prevalências diferentes, um único test).

## 5. Conclusões e resultados

### 5.1 Resultados

| Condição | Run | Macro-F1 | AP | Positivos previstos | Limiar |
|---|---|---|---|---|---|
| baseline | | | | | |
| | | | | | |

Fonte: o arquivo e a chave de cada coluna (`oof_metrics.json`,
`eval_test/thresholds.json`, `route_metrics.json`, …).

### 5.2 XYZ realizado

- **X — obtido**: o resultado, com o número (esperado na §2: …).
- **Y — evidência**: o texto da §2, copiado. Se o Y mudou, registre o desvio:
  Y planejado, Y usado e o motivo.
- **Z — intervenção**: o que foi efetivamente feito, com as diferenças em
  relação ao plano.
- **Veredito**: confirmada | refutada | inconclusiva — em uma frase.

### 5.3 Próximos passos

- O que este resultado abre ou fecha (novo experimento, mudança de default a
  registrar em `references/`).
