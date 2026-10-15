---
description: Registra um experimento em experiments_notebooks/ (abre ou atualiza um EXP-NNN) com o raciocínio montado pelo método XYZ
argument-hint: <descrição do experimento> | EXP-NNN <runs, resultados ou notas>
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(ls:*), Bash(git rev-parse:*), Bash(git log:*), Bash(uv run scripts/outputs_bucket.py:*), Bash(uv run python experiments_notebooks/analises/:*)
---

# /experimento — caderno de experimentos

PEDIDO: $ARGUMENTS

Você registra UM experimento em `experiments_notebooks/`, e só isso: não
treina, não roda bateria na GPU, não muda código nem config e não faz commit.
No fim, mostra o arquivo e sugere a mensagem de commit.

## Antes de escrever

1. Leia `experiments_notebooks/README.md` (método XYZ, regras, índice) e
   `experiments_notebooks/_template.md`. O template é a única fonte da
   estrutura: não invente seções nem remova as que existem.
2. Decida o modo pelo PEDIDO:
   - começa com `EXP-NNN` ou cita um arquivo do caderno → **atualizar**;
   - qualquer outra coisa → **abrir** um experimento novo;
   - vazio → pergunte qual experimento abrir ou atualizar.

## O método XYZ

*"Obtive **X**, medido por **Y**, fazendo **Z**"* (Laszlo Bock): resultado
observável → evidência verificável → método, sempre nessa ordem.

- **X** tem número e comparação (Macro-F1 no test ≥ referência + 0,01, com o
  IC 95% do Δ pareado acima de 0). "Melhorar", "testar" ou "ver se funciona"
  não são X.
- **Y** é reproduzível por outra pessoa: a chave exata da métrica, o split e o
  protocolo (tradicional ou OOF), a estratégia de limiar, as seeds e o arquivo
  de onde o número sai.
- **Z** é UMA variável manipulada, escrita como override Hydra
  (`model.extra_columns=[...]`) ou como o preset trocado (`EXPERIMENT=<preset>`);
  o resto é igual ao baseline. Trocar o preset compara receitas inteiras: diga
  isso nas ameaças à validade.
- X e Y viajam **verbatim** do Objetivo (§2) para o XYZ realizado (§5.2): copie
  o texto, não o reescreva. Se o Y mudou, a §5.2 registra Y planejado, Y usado
  e o motivo — nunca troque a métrica em silêncio depois de ver o resultado.
- Na Discussão (§4), cada achado é um item X → Y → Z: o que se observou, a
  evidência, o que o explica.

## Abrir

1. Número: o maior `NNN` em `experiments_notebooks/[0-9][0-9][0-9]_*.md` + 1
   (`001` se não houver nenhum). Slug curto em snake_case, sem acento.
2. Copie o template para `experiments_notebooks/NNN_slug.md`, status
   `planejado`, data de abertura = hoje (AAAA-MM-DD).
3. Preencha §1 (explicação) e §2 (XYZ planejado) a partir do PEDIDO e do
   repositório. Se faltar o que torna X, Y ou Z concretos — a métrica, o
   limiar, o baseline, a variável —, faça UMA pergunta com todas as lacunas.
   Não escolha no lugar do dono do projeto.
4. Preencha §3 com o que já se sabe (protocolo, comando, variáveis
   controladas). §4 e §5 ficam com *(a preencher quando houver runs)*.
5. Escreva a linha XYZ do topo como hipótese: "Obter X, medido por Y, fazendo Z."
6. Adicione a linha no índice do `README.md` (Resultado (X) = "—").

## Atualizar

1. Leia o arquivo inteiro e mude o status conforme o PEDIDO (`em execução`,
   `concluído` ou `abandonado` — este com o motivo).
2. Colete os números (seção abaixo) e preencha, nesta ordem: §5.1
   (resultados) → §4 (discussão) → §5.2 (XYZ realizado + veredito) → §5.3
   (próximos passos).
3. Reescreva a linha XYZ do topo no passado com o X obtido e o MESMO Y; data
   de conclusão quando o status for `concluído`.
4. Atualize a linha do índice (status e X obtido, em poucas palavras).

## Fontes dos números

Todo número cita a fonte (`<run>` + arquivo + chave), com 4 casas decimais
para Macro-F1 e AP. Nunca estime, arredonde de memória nem invente um valor.

- Os runs rodam no servidor de GPU e ficam no `outputs/` de lá (não
  versionado); o arquivo durável é o bucket do Hugging Face
  `LF-BF/abaw-11-ah-challenge-outputs` (`make outputs-push`). Para ler aqui,
  baixe SEM os pesos e NUNCA para o `outputs/` local, que guarda ablações que
  não estão no bucket:
  `uv run scripts/outputs_bucket.py --bucket LF-BF/abaw-11-ah-challenge-outputs pull --local <pasta temporária> --exclude '*.ckpt' --exclude '*.joblib'`.
  Se a ferramenta recusar ou o run não estiver no bucket, peça ao usuário os
  arquivos (ou a saída da sessão do tmux) — não reconstrua números.
- Layout do `outputs/` (pastas relativas a ele):
  - `oof/<experiment_name>/<ts>/` (`mode=oof`) — `oof_metrics.json`
    (`macro_f1`, `macro_f1_ci`, `ap`, `per_fold`, `pos_rate_pred`, `gate`),
    `oof_predictions.csv`, `pred_test.csv`, `fold<k>/trainer_state.json`
    (`ckpt_path` com a melhor época, `threshold`), `.hydra/overrides.yaml`,
    `main.log`;
  - `route/<route.name>/<ts>/route_metrics.json` (`mode=route`) — `router`,
    `members_metrics`, `mean_of_members`, `gate_vs_best`, `gate_vs_mean`,
    `usage`, `measure_test`;
  - `<pasta>/s<seed>/<ts>/` (`make train-seeds`) — `trainer_state.json`,
    `main.log` (linha "Treino concluído": Macro-F1 e AP do val e o limiar),
    `.hydra/overrides.yaml`; o manifesto do ensemble é `<pasta>/manifest.txt`;
  - `ensemble/<ensemble_name>/eval_test/` (`make ensemble-evaluate`) —
    `metrics.json` (`metrics.macro_f1`, `metrics.average_precision`,
    `metrics.threshold`), `results.txt`, `predictions.csv` e `thresholds.json`
    (`strategies[].eval_f1`, `threshold`, `eval_pos_rate_pred`; `eval_ap`).
- Número calculado depois dos runs (comparação pareada, AP por dobra) cita o
  script de `experiments_notebooks/analises/` que o reproduz.
- Commit: o run não grava o commit. O manifesto da versão do bucket
  (`.versions/<versão>/manifest.json → git.commit`) grava o commit da máquina
  no PUSH: cite-o como "commit do push". Sem manifesto, "desconhecido" — não
  infira pela data.

## Protocolo do projeto (para a §3.1 e as ameaças à validade)

- Splits oficiais do BAH, participantes disjuntos: train 778 vídeos (195
  participantes), val 124 (30), test 525 (75; rotulado, 60,6% positivos). O
  private test do desafio (152 vídeos) não é mais usado: o desafio encerrou.
- Métricas a nível de vídeo: Macro-F1 (a oficial, depende do limiar) e AP da
  classe positiva (sem limiar). O AP só se compara dentro do mesmo conjunto:
  o piso é a prevalência (51,0% no OOF de train+val, 60,6% no test).
- Protocolo tradicional (o do artigo, `make reproduce-best`): pesos no train,
  early stopping (`val_ap`) e limiar no val, test medido uma vez. Referência:
  ensemble de 5 seeds, Macro-F1 0,7226 / AP 0,8750
  (`references/integration_plan.md`, F2).
- Protocolo OOF (`mode=oof`): 5 dobras StratifiedGroupKFold por participante
  sobre train+val (902 vídeos), `oof.seed` 42, holdout interno de 15% dos
  participantes para o early stopping, τ fixo 0,5, gate pareado por bootstrap
  contra `oof.baseline`. O test é predito pela média das 5 dobras e medido uma
  única vez (`route.measure_splits=[test]`). O AP das 5 dobras juntas pune
  modelos cuja escala muda entre dobras: reporte também o AP médio por dobra.
- Limiar: no protocolo tradicional, o `mode=evaluate` no test grava
  `eval_test/thresholds.json` com 5 estratégias escolhidas só no val
  (`fixo_0.5`, `argmax`, `smooth`, `base_rate`, `bootstrap`). O val tem 30
  participantes e o ótimo dele erra o do test nas duas direções
  ([EXP-002](../../experiments_notebooks/002_protocolo_oficial_calibracao.md)): no
  protocolo tradicional, seleção e calibração usam o mesmo val, e só o test é
  honesto — diga isso quando ele decidir o veredito.
- Seeds: uma seed não decide (Macro-F1 do val varia ±0,02–0,035 entre seeds;
  a melhor época quase sempre é 0–2). Compare ensembles de seeds e use o
  bootstrap pareado no mesmo conjunto (`src/eval/protocol.py`); sem
  repetições, registre isso nas ameaças à validade.

## Estilo

pt-BR, direto, sem floreio; sem adjetivos no lugar de números. Links
relativos ao arquivo. Se a conclusão motivar uma mudança de código ou de
default, sugira registrá-la em `references/` (o plano em vigor) e no PR — não
a faça aqui.

## Ao terminar

Mostre o caminho do arquivo e a linha XYZ do topo, liste o que ficou em aberto
(lacunas, números pendentes) e sugira o commit, sem trailer de co-autoria:
`docs(experimentos): registra EXP-NNN — <resumo>` (abertura) ou
`docs(experimentos): conclui EXP-NNN — <resumo>` (conclusão).
