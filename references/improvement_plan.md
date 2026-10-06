# Plano v2: workflow iterativo de agregação de sinais via MoE

## Figuras

![Arquitetura proposta](figures/improvement_plan/architecture.png)

![Fluxo de execução](figures/improvement_plan/workflow.png)

Geradas por `scripts/plot_improvement_plan.py` (Graphviz/`dot`, PNG + PDF em `docs/figures/`).

## Status da implementação (branch `feat/moe-fusion`)

Toda a arquitetura da figura está implementada sobre o pipeline da `develop`, opt-in (nenhum
default muda; o caminho do artigo segue idêntico). **Nenhuma rodada foi medida ainda** — os
experimentos rodam na máquina com GPU; os números entram nesta seção.

### Mapa figura → código

| Bloco da figura | Onde está | Config |
|---|---|---|
| Sinais novos como colunas do Parquet de janelas (sem refazer o `featurize`) | `src/features/columns.py`, `src/pipeline/featurize_columns.py` (`mode=featurize_columns`) | `columns=[...]` |
| transcript → **RoBERTa-GoEmotions** (4 camadas congeladas, CLS) | ramo `hf_text` em `src/models/encoders.py`; transcrição tokenizada no `VideoSequenceDataset` (o `TextSequenceDataset` pendente da Rodada 1) | `configs/branch/text_goemotions.yaml` |
| transcript → **11 marcadores de hesitação** + cabeça auxiliar (0.3) | `src/features/hesitation_markers.py` (léxicos reusados de `text_features.py`) | `configs/branch/hesitation_markers.yaml` (`role: head`, `aux_weight: 0.3`) |
| Whisper → **ASR-erased time** (16) | `src/features/asr_timing.py` (já existia), coluna `asr_timing` | `configs/branch/asr_timing.yaml` |
| waveform → **wav2vec2-emotion** (1024-d) + cabeça temporal + supervisão por janela | coluna `audio_emb_<audio_embedder>`; GRU + logit por janela (rótulos de `time_detailed_ah`) | `configs/audio_embedder/wav2vec2_emotion_large.yaml`, `configs/branch/audio_emotion.yaml` |
| **recortes face/olhos/boca** → backbone → Transformer → `[μ,σ,μΔ,σΔ]` | `src/features/vision_embedder.py` (`FaceCropEmbedder`); no modelo, SoftMoE dos recortes + Transformer + `StatsPool` | `configs/vision_embedder/vit_face_expression.yaml`, `configs/branch/face_crops.yaml` |
| **Scene** (opcional, tracejado) — VideoMAE-v2 congelado, 16 frames | `SceneEmbedder` (mesmo módulo; aceita código remoto, entrada `(B,C,T,H,W)` e saída já vetorial) | `configs/scene_embedder/videomae_v2.yaml` (ou `videomae.yaml`, v1), `configs/branch/scene.yaml` |
| **projection block** `d_m`, **reliability gate** `g_m = σ(MLP[b; h_m])`, `z = LN(b + Σ g_m·d_m(h_m))` | `ProjectionBlock`, `ReliabilityGate` (`src/models/blocks.py`), `TextAnchoredMoE` (`src/models/moe_fusion.py`) | `model.fusion.*` (`configs/model/moe_fusion.yaml`) |
| **SoftMoE (MoEFusionHead)**, K=3–4 ExpertMLP | `SoftMoE` (`src/models/blocks.py`) | `model.fusion.head: moe`, `num_experts` |
| **load-balancing loss** (só se 1 expert > 70%) | `load_balance_loss` — KL(uso‖uniforme) aplicado só quando o uso máximo no lote passa do limiar; `train_moe_max_usage` logado | `model.loss.balance_weight`, `balance_threshold: 0.7` |
| **todo** membro expõe o **vetor pré-logit** | `LightningTrainer.predict_outputs`: o `moe_fusion` devolve `embedding` no `predict_step`; nos demais (cross-attention, GNNs) ele é capturado na entrada da `Linear(·,1)` que gera o logit (a camada cujo `sigmoid` reproduz a `proba`). Vai para `eval_<split>/embeddings.npy` (linhas = `predictions.csv`) e `oof_embeddings.npy` | — |
| **MoERouter** (sucessor do router CA/GNN), τ = 0.5 fixo | `src/models/moe_router.py`, `src/pipeline/route.py` (`mode=route`): avaliação nas dobras dos membros, medição única no public test e submissão oficial no private test | bloco `route` (`measure_splits`, `submit_split`) |
| Gate OOF de cada rodada (Rodada 0+) | `src/pipeline/oof.py` (`mode=oof`) sobre `src/eval/protocol.py` (+ `paired_bootstrap_ap`); os modelos das dobras também predizem o test e Parquets extras (private test) | bloco `oof` (`predict_splits`, `predict_parquets`) |
| Re-treino final com holdout de ~8% (Rodada 6) | split especial `holdout` em `src/data/datasets.py` | `data.calib_split=holdout`, `data.holdout_frac` |

Tudo é trocável pela config: um embedder é uma **coluna** (`audio_embedder=hubert` →
`audio_emb_hubert`, depois `model.branches.audio.column=audio_emb_hubert`); a arquitetura de
um ramo é `encoder` (`sequence|vector|hf_text`) / `temporal` (`identity|gru|transformer`) /
`pool` (`mean|max|attention|stats`) / `mixer` (`linear|moe`); os presets compõem ramos de
`configs/branch/` e a CLI acrescenta/remove (`+branch@model.branches.scene=scene`,
`~model.branches.asr`). Um ramo só é o **membro unimodal** da mesma classe.

### Como rodar cada rodada (máquina com GPU)

Pré-requisitos: dataset extraído em `data/raw/data/` (inclui `cropped-aligned-faces/`) e o
ambiente `make setup-vision` (traz `torchvision`, exigido pelo `AutoImageProcessor` do
transformers 5).

```bash
make data                          # preprocess + featurize (Parquet canônico)
make featurize-moe                 # transcript + asr_timing + hesitation_markers
make featurize-moe-audio           # audio_emb_wav2vec2_emotion_large
make featurize-moe-face            # face_crops_vit_face_expression
make featurize-moe-scene MOE_SCENE=videomae_v2   # (opcional) scene_emb_videomae_v2 (código remoto)

# Rodada 0 — régua: cross-attention do artigo (e, se quiser, o GNN) no protocolo OOF
make oof EXPERIMENT=cross_attention ARGS="experiment_name=r0-cross-attention"
make oof EXPERIMENT=hetero_gnn_v2_tune_wav2vec2 ARGS="experiment_name=r0-hetero-gnn"  # make featurize-w2v antes
make route MEMBERS="outputs/oof/r0-cross-attention/<ts> outputs/oof/r0-hetero-gnn/<ts>" \
     ARGS="route.name=r0-router route.use_embeddings=false"    # "router atual" reavaliado

# Rodada 1 — texto como âncora (gate: OOF AP ≥ 0.80); variante barata congelada como régua
make oof EXPERIMENT=moe_r1_text BASELINE=outputs/oof/r0-cross-attention/<ts>
make oof EXPERIMENT=moe_r1_text_frozen BASELINE=outputs/oof/r0-cross-attention/<ts>

# Rodadas 2–5 — cada uma contra a anterior; o texto parte do modelo da Rodada 1 da MESMA dobra
R1=outputs/oof/moe-r1-text/<ts>
make oof EXPERIMENT=moe_r2_text_tab TEXT_RUN=$R1 BASELINE=$R1
make oof EXPERIMENT=moe_r3_asr      TEXT_RUN=$R1 BASELINE=outputs/oof/moe-r2-text-tab/<ts>
make oof EXPERIMENT=moe_r4_audio    TEXT_RUN=$R1 BASELINE=outputs/oof/moe-r3-asr/<ts>
make oof EXPERIMENT=moe_r5_face     TEXT_RUN=$R1 BASELINE=outputs/oof/moe-r4-audio/<ts>
# sinal reprovado no gate → "testado, não incorporado": remova o ramo nas rodadas seguintes,
# ex.: make oof EXPERIMENT=moe_r4_audio TEXT_RUN=$R1 ARGS="~model.branches.asr"

# membros unimodais do MoERouter
make oof EXPERIMENT=moe_unimodal_audio
make oof EXPERIMENT=moe_unimodal_asr

# Rodada 6 — MoERouter final sobre os membros que passaram (+ CA/GNN se ajudarem no OOF).
# 1) colunas no Parquet do private test (caminhos isolados p/ não sobrescrever os do raw):
EXT="data.paths.data_root=data/external/data data.paths.interim_dir=data/interim/external \
     data.paths.parquet_path=data/processed/external_windows.parquet"
make preprocess ARGS="$EXT"
make featurize-moe ARGS="$EXT" && make featurize-moe-audio ARGS="$EXT" && make featurize-moe-face ARGS="$EXT"
# 2) cada membro roda o OOF predizendo também o private test (os modelos das 5 dobras):
make oof EXPERIMENT=<membro> ... \
     ARGS="\"oof.predict_parquets={external:'data/processed/external_windows.parquet'}\""
# 3) roteador: medição ÚNICA no public test + submissão oficial no private test (τ = 0.5)
make route MEMBERS="outputs/oof/moe-r5-face/<ts> $R1 outputs/oof/moe-unimodal-audio/<ts> ..." \
     ARGS="route.predict_splits=[test,external] route.measure_splits=[test] \
           route.submit_split=external submission_reference=data/external/reference_trial-0.txt \
           submission_probabilities=true"
```

Cada `make oof` grava em `outputs/oof/<experiment_name>/<ts>/`: `oof_predictions.csv`,
`oof_embeddings.npy`, `oof_metrics.json` (Macro-F1@0.5, AP, IC bootstrap, por dobra e o gate
pareado vs `BASELINE`: ΔF1/ΔAP com IC → `melhora`/`empate`/`piora`), `pred_test.csv` (média
das 5 dobras, **sem métricas** — o public test é medido uma vez só) e `fold<k>/` (checkpoints).
O `make route` grava `route_metrics.json` (gate vs melhor membro e vs média, uso de cada
membro e, se pedido, `measure_test` — roteador × cada membro × média no public test),
`pred_<nome>.csv` e a submissão (`submission_external.txt`, ou `out=`). Alternativa de
modelo único: re-treino com holdout, com o ramo de texto re-treinado nos mesmos splits (ou
`model.branches.text.trainable=true`) —
`uv run python main.py mode=train +experiment=<melhor> "data.train_splits=[train,val,test]"
data.calib_split=holdout` e depois `mode=submit split=test` com
`data.paths.parquet_path=data/processed/external_windows.parquet`.

### Comparação com o cross-attention do artigo (Luiz × Rodrigo)

Todos os runs no mesmo protocolo OOF (mesmas dobras) → qualquer par se compara depois com
`make oof-compare A=<run> B=<run>` (ΔF1@τ e ΔAP com IC por bootstrap pareado).

| Experimento | Preset | O que muda |
|---|---|---|
| referência: cross-attention do artigo | `cross_attention` | — |
| baseline do Rodrigo (arquitetura MoE completa) | `moe_r5_face` (texto da `moe_r1_text`, `TEXT_RUN`) | arquitetura + features dele |
| artigo + vídeo do Rodrigo | `cross_attention_video` | `model.extra_columns=[face_crops_…]`: recortes faciais fundidos por token antes do pooling (opt-in; sem a coluna o modelo do artigo fica idêntico) |
| MoE com áudio/texto do artigo + vídeo dele | `moe_luiz_features` | `moe_r5_face` com librosa 320 (`audio_emb`) e RoBERTa-emotion congelado (`text_emb`) no lugar de wav2vec2-emotion e GoEmotions |
| ensemble dos baselines | `make route MEMBERS="<artigo> <moe_r5_face>"` | roteador + média simples, com gate vs o melhor membro |

`featurize-moe-audio` e `featurize-moe-face` podem rodar ao mesmo tempo (uma GPU cada): o merge
de colunas no Parquet é feito sob trava de arquivo e relê o arquivo antes de gravar.

### Decisões de implementação (desvios do texto do plano)

- **SoftMoE local, não importado da `vision-toolbelt-liga`.** Mesma formulação (router
  `Linear → softmax(K)`, experts MLP de 2 camadas com GELU, soma ponderada), mas o MoE é o
  agregador de TODAS as rodadas (inclusive texto + tabular): depender da toolbelt puxaria
  `mlflow`, `timm`, `pytorch-metric-learning` e o `__init__` de `architectures` inteiro só por
  esse bloco, e o `forward` dela não devolve os pesos do roteador (necessários p/ o
  balanceamento e a telemetria).
- **Recortes faciais sobre os rostos já alinhados do BAH** (`cropped-aligned-faces`, 256×256):
  olhos e boca ficam em posições canônicas, então os recortes são caixas fracionárias
  configuráveis (`vision_embedder.crops`) — sem rodar detector/Face Mesh por frame. Os
  `EyesMeshCrop`/`MouthMeshCrop` da toolbelt usam a API legada `mp.solutions`, ausente no
  MediaPipe 1.x. O backbone é qualquer `AutoModel` de visão do HuggingFace (default
  `trpakov/vit-face-expression`); a única dependência nova é `torchvision` no grupo `vision`.
- **`text_finetune.py` não foi reportado:** o modelo de texto da Rodada 1 é o próprio
  `moe_fusion` com um ramo `hf_text` (mesma receita: GoEmotions, 4 camadas congeladas, CLS,
  R-Drop α=4, label smoothing 0.1, `pos_weight` auto). Os membros unimodais (texto, áudio,
  ASR) são presets da mesma classe — um código só.
- **Cabeça auxiliar dos marcadores ligada ao caminho principal.** No `text_finetune` original
  a cabeça auxiliar era um `Linear` isolado sobre o vetor tabular (perda própria, sem
  parâmetros compartilhados — não influenciava a predição). Aqui o ramo dos marcadores entra
  na cabeça MoE (`role: head`) e tem o classificador auxiliar (estilo AMF), então a perda
  auxiliar molda uma representação que a cabeça usa.
- **Empilhamento sem vazamento entre rodadas:** da Rodada 2 em diante o ramo de texto parte
  do modelo da Rodada 1 da MESMA dobra (`init_from=<R1>/fold{fold}`, formatado pelo runner) e
  fica fixo (`trainable: false`).
- **MoERouter:** os "experts" são os membros treinados; o roteador lê
  `[proj(embedding_m) ‖ logit_m]` e combina os logits com `softmax` sobre os membros. É
  avaliado nas dobras dos próprios membros (stacking sem vazamento); no test, cada dobra é
  roteada com as saídas dos modelos daquela dobra e as 5 são promediadas. Ressalva: embeddings
  de modelos de dobras diferentes vivem em espaços diferentes (o `LayerNorm` por membro só
  alinha escala) — `route.use_embeddings=false` dá o roteador só de logits p/ comparar.
- **Mamba** não foi implementado (decisão do plano: só se a cabeça temporal virar gargalo).
- **VideoMAE-v2 é opt-in** (`MOE_SCENE=videomae_v2`): o repositório roda código remoto
  (`trust_remote_code`) e o checkpoint é CC BY-NC 4.0; o default do `make` segue o VideoMAE v1,
  que roda com o transformers puro.

### Observações sobre os dados (conferir antes das Rodadas 3 e 5)

- Os chunks do Whisper são **contíguos** dentro de cada segmento de ~30 s (o fim de um chunk é
  o início do próximo): os gaps do `asr_timing` saem quase sempre zerados (≈0,6 gap por vídeo
  em train+val). O "tempo apagado" do IISERB pede timestamps por palavra — a Rodada 3 tende a
  ser fraca com os chunks atuais.
- A duração inferida dos chunks pode **passar da duração real** do vídeo (chunks sobrepostos
  na timeline acumulada): num vídeo de 82,5 s as janelas iam até 115 s. As janelas finais
  ficam sem áudio/rosto (zeros). É comportamento pré-existente do janelamento — não foi
  alterado aqui para não mudar os caches nem a reprodução do artigo.

## Por que reestruturar (contexto desta revisão)

O plano original (`references/improvement_plan.md`, já parcialmente implementado —
F0/F1/F2a; contexto e arquivos herdados resumidos mais abaixo) era uma sequência
linear F0→F4: cada fase
adicionava um sinal nas suas próprias configs/modelos, e só na Fase 4 tudo convergia
num único "Text Residual Fusion" com um gate escalar por modalidade.

Essa estrutura tem um problema: só saberíamos se a aposta estava certa no fim, depois
de já ter implementado tudo. O pedido desta revisão é trocar isso por um **workflow
iterativo**: a cada sinal novo adicionado, rodar o protocolo OOF (já existe, Fase 0) e
decidir *na hora* se aquele sinal entra ou não — nunca acumular trabalho não validado.

A peça central dessa mudança é o **Mixture-of-Experts**: em vez de um MLP fixo no fim
da fusão e um router separado (regressão logística) decidindo entre CA/GNN, o MoE vira
o **único mecanismo de agregação**, usado em dois níveis:
1. Como classificador final de qualquer sub-fusão (troca o `MLP classifier` do Text
   Residual Fusion).
2. Como sucessor do router atual (`scripts/meta_router_ca_gnn.py`), decidindo por
   amostra qual membro (ou combinação) confiar — generalização natural do reliability
   gate `g_m = σ(MLP[b;h_m])` já desenhado, só que com K experts em vez de 1 peso por
   modalidade.

Cada "rodada" do workflow (ver abaixo) testa se um MoE com mais um sinal bate o MoE
anterior no protocolo OOF (`src/eval/protocol.py`, já implementado). Se não bater, o
sinal fica registrado como "testado, não incorporado" — nunca é descartado
silenciosamente, mas também nunca infla a arquitetura sem provar valor.

**Mamba**: pesquisado (vision-toolbelt-liga tem um backbone Mamba solto, não integrado
ao MoE por padrão; `mamba-ssm`/`causal-conv1d` não estão instalados no projeto e
exigem build com kernels CUDA compilados). RAS só usa Mamba numa sequência curta (16
frames de cena) — no nosso caso as sequências temporais também são curtas (janelas de
5s). Decisão: Mamba **não entra como default** em nenhuma cabeça temporal; fica marcado
como experimento opcional na Rodada 4 (áudio/cena), a testar só se a cabeça temporal
simples (GRU/Transformer pequeno) virar gargalo de custo comprovado — nunca bloqueia
uma rodada.

---

## Mecanismo central: `SoftMoE` como agregador único

**Fonte:** `SoftMoE` em
`/home/rwp/code/project_lib/vision-toolbelt-liga/src/vision_toolbelt/architectures/moe.py`.
Já é backbone-agnóstico — aceita `(B, C)` (vetores de embedding) diretamente, sem
precisar da parte de CNN do `MoEClassificationModel`/`build_moe_classifier` (essa parte
é só para imagem). Roteamento denso/soft: `router = Linear(in_features, num_experts)`,
softmax sobre todos os experts, cada expert é um `ExpertMLP` (2 camadas, GELU,
dropout), saída combinada por soma ponderada. Sem loss de balanceamento de carga no
código original — **a adicionar** se um expert monopolizar o roteamento (ver Rodada 2).

**Novo módulo** `src/models/moe_fusion.py`, dois usos concretos:

1. **`MoEFusionHead`** — substitui o `MLP classifier` do Text Residual Fusion. Entrada:
   o vetor fundido `z = LN(b_text + Σ_m g_m·d_m(h_m))` (mesma equação de antes, texto
   como âncora). Saída: `SoftMoE(z) → Linear(K→2)`. K experts pequenos (3–4 pra
   começar) se especializam em diferentes "modos" de conflito A/H (hesitação verbal vs.
   microexpressão vs. pausa longa), em vez de um único MLP compartilhado ter que cobrir
   todos.
2. **`MoERouter`** — sucessor do router atual (`scripts/meta_router_ca_gnn.py`).
   Diferença importante descoberta na exploração: o router hoje só enxerga
   **probabilidades escalares** por membro (`predictions.csv` com `video_id, y_true,
   y_proba`), nunca embeddings — então um MoE de verdade (que ganha ao ver features,
   não só a probabilidade final) exige plumbing novo: estender o dump de predições
   para incluir o vetor pré-logit de cada membro (`outputs/<run>/eval_<split>/
   embeddings.npy` ou coluna serializada no mesmo CSV). Isso vai na Rodada 2 abaixo,
   não antes — é o primeiro ponto onde "MoE de verdade" diverge do router antigo.

**Regra de balanceamento:** se, ao inspecionar os pesos do router (`softmax` médio por
expert no conjunto OOF), um expert concentrar >70% do tráfego, adicionar uma
load-balancing loss leve (ex.: penalizar desvio da distribuição uniforme de uso dos
experts, receita padrão de Switch Transformer / Shazeer et al.) — só se o desbalanceio
for medido, não preventivamente.

---

## Workflow: rodadas iterativas, cada uma com gate de decisão

Cada rodada segue o mesmo ciclo: **adicionar 1 sinal → treinar o membro unimodal →
testar como expert candidato no MoE → medir no OOF (`src/eval/protocol.py`) → decidir**.
Nenhuma rodada decide "no escuro"; cada uma responde explicitamente "estamos indo na
direção certa?" antes de abrir a próxima.

### Rodada 0 — Fundação (já implementada, branch `improve-macro-f1-beyond-router`)
- Protocolo OOF (`src/eval/protocol.py`): 5-fold agrupado por participante, Macro-F1/AP
  com τ=0.5 fixo, bootstrap pareado.
- Baseline: router atual (CA⊕GNN) reavaliado nesse protocolo, para saber quanto do
  0.7454 é sorte do val de 124 vídeos vs. sinal real.
- **Gate de saída:** baseline OOF documentado — é a régua contra a qual toda rodada
  seguinte se mede. *(Concluído.)*

### Rodada 1 — Texto como âncora (já implementada, precisa de 1 peça: dataset tokenizado)
- `src/models/text_finetune.py`: RoBERTa-GoEmotions fine-tune + cabeça auxiliar de
  marcadores de hesitação (peso 0.3), já implementado e testado.
- **Pendência bloqueante:** falta `TextSequenceDataset` (tokeniza transcript bruto por
  vídeo) — hoje só existem consumidores de embeddings pré-computados. Sem isso o
  treino real não roda. *(Resolvida na `feat/moe-fusion`: coluna `transcript` +
  tokenização no `VideoSequenceDataset`; o modelo é o preset `moe_r1_text`.)*
- **Gate de saída:** treinar de fato e medir OOF AP do texto sozinho. Critério ≥ 0.80
  (referência dos dois papers). Só passa para a Rodada 2 se isso for atingido — texto é
  a âncora de todo o resto, um texto fraco compromete a arquitetura inteira.

### Rodada 2 — Primeiro MoE: 2 experts (texto + tabular/hesitação)
- Implementar `src/models/moe_fusion.py` (`MoEFusionHead`) com **apenas 2 sinais**:
  texto fine-tunado (Rodada 1) e as 74 features tabulares existentes.
- Objetivo desta rodada não é ganhar performance ainda — é **provar a plumbing**: MoE
  treina, roteia de forma sensata (não colapsa num único expert), e bate (ou empata) o
  texto sozinho no OOF.
- Estender o dump de `predictions.csv` para incluir embeddings pré-logit por membro
  (`outputs/<run>/eval_<split>/embeddings.npy`), habilitando o `MoERouter` para as
  rodadas seguintes.
- **Gate de saída:** MoE(texto, tabular) ≥ texto sozinho no OOF, com IC do bootstrap
  não incluindo 0 como piora. Se o MoE só empatar ou piorar aqui, é sinal de bug na
  plumbing, não de falta de sinal — não passar para a Rodada 3 sem resolver.

### Rodada 3 — ASR-erased time entra como terceiro expert
- `src/features/asr_timing.py` já implementado e testado (16 features de gaps).
- Resolver a pendência de wiring (vetor por vídeo replicado nas janelas, conforme já
  recomendado no plano original) e adicionar como 3º expert candidato no MoE.
- **Gate de saída:** MoE(texto, tabular, asr_timing) > MoE da Rodada 2 no OOF. Se não
  bater, ASR-timing fica registrado como "testado, não incorporado" — não é descartado
  do código, só não entra no ensemble final.

### Rodada 4 — Áudio de emoção (bloqueada até download do áudio)
- Trocar librosa por `wav2vec2` de emoção + cabeça temporal (GRU/Transformer pequeno
  por padrão; Mamba como experimento tracejado, só se a cabeça padrão for
  comprovadamente o gargalo).
- Supervisão auxiliar por janela via `time_detailed_ah` (já existe em
  `src/data/windowing.py:185-194`).
- **Gate de saída:** MoE com áudio > MoE da Rodada 3 no OOF. Mesma regra de
  "testado, não incorporado" se não bater.

### Rodada 5 — Canal visual (bloqueada até download dos vídeos)
- Extração via `vision-toolbelt-liga`: `FaceDetectionCrop`/`EyesMeshCrop`/
  `MouthMeshCrop` → backbone (`BACKBONES.build`) → estatísticas `[μ,σ,μΔ,σΔ]`.
- Aqui o `SoftMoE` da toolbelt pode ser reaproveitado **duas vezes**: (a) combinando os
  3 crops (face/olhos/boca) antes de entrar como expert único no MoE principal, e (b)
  no MoE principal em si. Documentar as duas instâncias sem confundir escopos.
- Canal de cena (VideoMAE-v2, 16 frames) só entra se sobrar tempo — é o canal mais
  fraco nos dois papers de referência.
- **Gate de saída:** MoE com face > MoE da Rodada 4 no OOF.

### Rodada 6 — Consolidação e submissão
- MoE final com todos os experts que passaram no gate das rodadas anteriores.
- Re-treino em train+val+test com holdout estratificado de ~8% (receita já definida no
  plano original).
- Medição **única** no public test — nunca antes disso.
- Checar taxa de positivos prevista no private test (~50–55%, referência do baseline
  atual).

**Regra transversal de todas as rodadas:** o gate de decisão usa sempre
`paired_bootstrap_macro_f1` (`src/eval/protocol.py`) comparando MoE-com-sinal-novo vs.
MoE-sem-sinal-novo no mesmo conjunto OOF — nunca comparação informal de números soltos.

## Contexto herdado do plano original (ainda válido)

- **Produção hoje:** meta-router CA⊕GNN (áudio librosa + texto `twitter-roberta-emotion`
  + 74 features tabulares). Faz **0.724 no val** e **0.7454 no public test**. Pesos, C,
  τ e limiar do router escolhidos juntos no val de 124 vídeos (`scripts/meta_router_ca_gnn.py`)
  — é exatamente o padrão de overfitting que o protocolo OOF (Rodada 0) existe para medir.
- **Oráculo CA/GNN:** ~0.80 no test se sempre se escolhesse o melhor dos dois — ainda há
  ganho de roteamento/complementaridade não capturado, reforçando a aposta em MoE.
- **Referências:** RAS (2º lugar) fez 0.782 no private test com um único modelo
  text-residual. IISERB (1º lugar) mostrou que calibrar no val pequeno custa até 4 pontos.
- **Onde estamos fracos** (ver `references/visual_signal_lessons_from_top_teams.md`):
  texto com encoder de emoção genérico (não GoEmotions) e congelado; áudio só com
  prosódia (librosa), sem encoder de emoção; face com só 2 frames/janela e landmarks
  crus; sem canal de cena; sem uso do timing apagado pelo ASR.
- **Restrições:** vídeos brutos fora da máquina até novo download; GPU RTX 3060 12GB;
  usar a `vision-toolbelt-liga` do usuário para tudo relacionado a extração visual e MoE.

## Arquivos (consolidado)

- **Herdados** (portados na integração): `src/eval/protocol.py`, `scripts/ensemble_ap.py`,
  `src/features/asr_timing.py`. O `src/models/text_finetune.py` ficou na branch
  `improve-macro-f1-beyond-router`: a receita dele virou o preset `moe_r1_text`.
- **Implementados na `feat/moe-fusion`:**
  - dados/features: `src/features/{columns,hesitation_markers,vision_embedder}.py`,
    `src/pipeline/featurize_columns.py`; `VideoSequenceDataset` com colunas extras,
    transcrição tokenizada, rótulos por janela, `subset` e o split `holdout`;
  - modelo: `src/models/{blocks,encoders,moe_fusion,moe_router}.py`; dump de
    `embeddings.npy` no `mode=evaluate`;
  - avaliação: `src/pipeline/{oof,route}.py` (`mode=oof`, `mode=route`);
  - configs: `configs/model/moe_fusion.yaml`, `configs/branch/*.yaml`,
    `configs/experiment/moe_*.yaml`, `configs/{vision,scene}_embedder/`,
    `configs/audio_embedder/wav2vec2_emotion_large.yaml`, blocos `oof`/`route`;
  - testes: `tests/{test_moe_fusion,test_featurize_columns,test_oof_route}.py` (+ `conftest.py`).
- **Reuso:** janelas/rótulos `src/data/windowing.py`; Whisper `src/data/schema.py`;
  tabular `configs/data/default.yaml`; folds/bootstrap `src/eval/protocol.py`; trainer,
  registry e ensemble da `develop`.
- **Docs:** cada rodada medida ganha seus números OOF na seção de status acima.

## Próximos passos

1. Rodar as Rodadas 0–5 na máquina com GPU (comandos na seção de status) e registrar, por
   rodada, Macro-F1@0.5, AP e o veredito do gate pareado.
2. Conferir `train_moe_max_usage` nos logs: se um expert passar de 70% do tráfego de forma
   persistente, subir `model.loss.balance_weight`.
3. Rodada 6: MoERouter final sobre os membros aprovados, medição única no public test e
   re-treino com holdout de 8% para a submissão (taxa de positivos prevista ~50–55%).

## Verificação (aplicável a toda rodada)

- `uv run pytest` deve passar antes de qualquer gate de decisão ser avaliado.
- `make ci` (ruff) limpo.
- Toda rodada produz `outputs/oof/<experiment_name>/<ts>/oof_predictions.csv` e o
  `oof_metrics.json` (F1, AP, IC do bootstrap e o gate pareado vs `oof.baseline`) — é o
  artefato que sustenta o gate de saída.
- O public test só é avaliado **uma vez**, na Rodada 6 (consolidação) — nunca antes.
- Submissão final: re-treino com todos os dados + holdout de 8%, checando que a taxa de
  positivos prevista no private fica em torno de 50–55% (mesma faixa do baseline atual).
