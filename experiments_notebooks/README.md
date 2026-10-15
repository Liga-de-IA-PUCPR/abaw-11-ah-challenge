# experiments_notebooks/ — caderno de experimentos

Registro dos experimentos do projeto: **um arquivo Markdown = um
experimento**, com a pergunta, o objetivo, o protocolo, a discussão e as
conclusões. É o lado interpretativo dos runs — o que se queria medir, o que se
mediu e o que se concluiu. Os números continuam vindo dos artefatos dos runs
(`outputs/` no servidor de GPU, arquivado no bucket do Hugging Face
`LF-BF/abaw-11-ah-challenge-outputs`): o caderno os cita com a fonte e nunca
os substitui.

## O método XYZ

Todo documento monta o raciocínio pela fórmula X-Y-Z (Laszlo Bock, Google):
*"obtive **X**, medido por **Y**, fazendo **Z**"* — primeiro o resultado
observável, depois a evidência verificável, por último o método.

| Letra | O que é | Exemplo neste projeto |
|---|---|---|
| **X** | o resultado observável, com número e comparação | Macro-F1 no test ≥ cross-attention do artigo + 0,01, com o IC 95% do Δ pareado acima de 0 |
| **Y** | a evidência que decide X, verificável por outra pessoa | Macro-F1 a nível de vídeo no test oficial (525), limiar escolhido só no val, ensemble de 5 seeds, `ensemble/<nome>/eval_test/thresholds.json → strategies[].eval_f1`, bootstrap pareado |
| **Z** | o método: a intervenção que produz X | `model.extra_columns=[face_crops_vit_face_expression]` (ou o preset `EXPERIMENT=cross_attention_video`), o resto igual ao baseline |

A ordem vale em três escalas:

1. **Documento** — a linha XYZ no topo dá o resultado antes de tudo; o corpo
   segue Objetivo (X) → Métodos (Y, depois Z) → Discussão → Conclusões e
   resultados.
2. **Objetivo × conclusão** — o XYZ planejado (§2) e o XYZ realizado (§5.2)
   usam o MESMO Y. Trocar a métrica ou o protocolo depois de ver o resultado é
   desvio, registrado como tal (Y planejado, Y usado, motivo).
3. **Discussão** — cada achado é X (o que se observou) → Y (a evidência) → Z
   (o que o explica).

## Regras

- **Número sem fonte não entra.** Todo valor cita o run (pasta em
  `outputs/`), o arquivo e a chave (`oof_metrics.json → macro_f1`), e a versão
  do bucket onde ele está arquivado. Número calculado depois dos runs cita o
  script de [`analises/`](analises/) que o reproduz.
- **Uma variável manipulada por experimento**, como override Hydra ou preset
  trocado; o resto é igual ao baseline, e o `.hydra/overrides.yaml` de cada run
  confirma isso. Trocar o preset compara receitas inteiras (arquitetura e
  features juntas): registre isso nas ameaças à validade.
- **Protocolo explícito**: tradicional (train → val → test) ou OOF (5 dobras
  por participante); a estratégia de limiar; as seeds e o ensemble; o conjunto
  em que o AP foi medido (o piso dele é a prevalência).
- **Nada se apaga**: experimento que não andou fica com status `abandonado` e
  o motivo. Um resultado negativo também é resultado.
- **Não é o diário de decisões**: se a conclusão motivar mudança de código ou
  de default, ela vai para `references/` (o plano em vigor) e para o PR, e o
  experimento a cita.
- **O `outputs/` local não recebe downloads**: ele guarda ablações antigas que
  não estão no bucket. Para ler resultados, baixe para uma pasta temporária
  (`outputs_bucket.py pull --local <pasta> --exclude '*.ckpt'`).
- Datas absolutas (AAAA-MM-DD), em pt-BR, como o resto da documentação.

## Como registrar

No Claude Code: `/experimento <descrição>` abre um experimento novo (status
`planejado`); `/experimento EXP-NNN <runs, resultados ou notas>` atualiza um
existente — preenche resultados, discussão e conclusões a partir dos
artefatos dos runs.

À mão:

1. Copie [`_template.md`](_template.md) para `NNN_slug.md` (NNN = próximo
   número, 3 dígitos).
2. Preencha §1–§3 antes de rodar; §4 e §5 quando houver runs.
3. Adicione (ou atualize) a linha no índice abaixo.
4. Commit: `docs(experimentos): registra EXP-NNN — <resumo>` na abertura e
   `docs(experimentos): conclui EXP-NNN — <resumo>` na conclusão.

Sugestão: dê a cada experimento uma pasta própria no `outputs/`
(`experiment_name=expNNN-<condição>` no OOF, `NAME=expNNN/<condição>` no
`make train-seeds`) para os runs não se misturarem.

Análises: os scripts de [`analises/`](analises/) leem uma pasta com os outputs
baixados (sem os pesos) e recalculam o que não sai pronto dos runs — as
comparações pareadas no test e o AP por dobra. Rode-os num checkout com o
código que gerou os runs.

## Índice

| # | Experimento | Modelo | Status | Resultado (X) | Data |
|---|---|---|---|---|---|
| [001](001_bateria_oof_artigo_x_moe.md) | Bateria OOF: cross-attention do artigo × MoE do Rodrigo (4 variantes + roteador) | `cross_attention`, `moe_*` | concluído | todos os gates de Macro-F1 `empate` (Exp 1 −0,0121; roteador +0,0067 vs o melhor membro); no test, nada acima do artigo com significância: refutada | 2026-10-06 |
| [002](002_protocolo_oficial_calibracao.md) | Protocolo oficial (train → val → test, 5 seeds) com 5 estratégias de limiar | `cross_attention`, `moe_*` | concluído | média das estratégias: `base_rate` 0,7077 ≈ fixo 0,7074 > `bootstrap` 0,7001; artigo reproduzido (0,7251 / 0,8759); Exp 0 + Exp 1 ΔAP +0,0085 sem significância: refutada | 2026-10-14 |
