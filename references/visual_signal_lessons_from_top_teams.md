# O que os times 1º e 2º lugar fizeram no canal visual — e o que aproveitar na nossa GNN

Contexto: no ABAW 2026 BAH Challenge (ambivalência/hesitação), ficamos em 3º lugar —
não porque a abordagem com GNN sobre Face Mesh fosse pior, mas porque não deu tempo
de submeter os resultados com GNN antes do prazo. Os dois times à nossa frente
(`data/external/Simple Features and Honest Calibration...pdf`, doravante **IISERB**, e
`data/external/Team RAS in 11th ABAW Competition...pdf`, doravante **RAS**) concordam
num ponto que também é a nossa maior fraqueza: **o canal visual é o elo mais fraco do
problema**. Ambos tratam o rosto/cena como modalidade auxiliar fraca e investem pouco
nela relativo a texto/áudio. Isso é uma pista, não uma sentença — os dois usam
representações visuais genéricas (ViT/VideoMAE/EmoAffectNet 2D, sem grafo, sem
geometria), então o "teto" que eles relatam pode ser um teto da *representação*, não do
canal. É exatamente aqui que nossa GNN sobre landmarks tem uma aposta diferente e ainda
não comparada de frente com essas baselines.

## 1. Resumo do que cada time fez no canal de imagem

### IISERB (1º lugar, `AP 0.731`)
- Testaram várias representações de vídeo: **VideoMAE-Base** (ação genérica, AP 0.650),
  **FER-ViT** (expressão facial, embedding 768 + 14 estatísticas FER, AP 0.636–0.669),
  e um conjunto de **36 features de gaze/AU via MediaPipe** (AP 0.665).
- Resultado: toda representação visual ficou numa faixa estreita de **AP 0.64–0.67**,
  para eles isso "lê mais como teto de canal do que falha de modelagem" — ou seja,
  concluem que não é a arquitetura que está errada, é que o rosto carrega menos sinal
  de A/H que a fala.
- Achado negativo explícito: features de **gaze/brow dinâmicas pioraram** o modelo
  fundido (AP 0.860 → 0.841 no ensemble). Ou seja, adicionar mais geometria facial
  *sem controle de ruído* piorou a fusão.
- O sinal não-verbal mais forte e mais **independente** dos outros (correlação
  0.11–0.36) não veio do rosto, veio do **timing do ASR** (gaps de fala erasados pelo
  Whisper) — 16 features determinísticas, AP 0.718, acima de qualquer feature visual.

### RAS (2º lugar, `MF1 0.7514` médio, `0.7824` no private test)
- Canal facial: **EmoAffectNet** (encoder 2D frame-a-frame, 512-dim) + **Transformer**
  temporal (1 camada, 8 heads) sobre até 500 frames, com **flow matching** como
  regularização extra (nos espaços de feature e de logit).
- Canal de cena (scene, não-face): **VideoMAE-v2** sobre 16 frames do vídeo completo
  (não recortado no rosto) — captura fundo, postura corporal, contexto — processado por
  um encoder **Mamba** (SSM) bidirecional.
- Resultado unimodal (Tabela 1 do paper): Text 72.33 > Audio 69.74 > **Face 62.67** >
  Scene 61.12 (MF1 médio Dev/Public). O rosto sozinho é o segundo pior canal.
- A fusão (**Text Residual Fusion**) trata texto como âncora e injeta ajustes residuais
  com *gates* aprendidos por modalidade (áudio, face, cena), ao invés de concatenar
  tudo de forma simétrica. O gate aprende "quanto confiar" em cada modalidade
  condicionado ao próprio texto — isso é o análogo, em espírito, ao gate de
  confiabilidade do AMF do IISERB.
- O ganho de fundir face+audio+scene sobre o texto puro foi de **+2.93 MF1** no
  desenvolvimento médio e **+4.03** no private test — nada desprezível, mas herda o
  mesmo padrão: cada canal visual isolado é fraco, o ganho vem de deixá-los corrigir o
  texto só quando ajudam (via gate), não de arquitetura sofisticada por canal.

## 2. Onde os dois convergem (e por que isso importa pra gente)

1. **Texto domina, visual é auxiliar fraco.** Nos dois times, texto sozinho já bate
   toda representação visual isolada por uma margem grande (IISERB: AP 0.811 vs 0.65–0.67;
   RAS: MF1 72.3 vs 62.7). Se nosso pipeline atual pondera a GNN facial de forma
   simétrica com texto/áudio, isso provavelmente está sub-ótimo — vale revisar se o
   `meta_router_ca_gnn` já trata isso ou se estamos deixando o canal visual "empatar"
   peso demais.
2. **Mais geometria facial ≠ mais sinal, se não for filtrada.** O achado negativo do
   IISERB com gaze/brow é um alerta direto pro nosso ROI `ah` (boca+olhos+sobrancelha+
   íris em `src/data/face_roi.py`): a aposta de restringir landmarks a essas regiões é
   coerente com a ideia deles, mas o resultado deles mostra que *mesmo* restringindo a
   regiões plausíveis (AU/gaze), o ganho pode não aparecer sem um gate que deixe o
   modelo ignorar o canal quando ele não tem sinal para aquele vídeo.
3. **O ganho real do canal visual, quando existe, é pequeno e frágil.** RAS mediu
   isso com clareza: canal facial sozinho é quase o pior canal, mas dentro da fusão com
   gate ele ainda contribui ganho mensurável no private test. A lição não é "abandonar
   o visual", é "não deixar ele competir em pé de igualdade sem calibração/gate".

## 3. O que é especificamente aplicável à nossa GNN (`face_gcn_ts.py`, `face_roi.py`)

Nem tudo que eles fizeram é transferível — eles usam CNN/ViT 2D ou Transformer/Mamba
sobre embeddings de frame, não grafos sobre geometria de landmark como nós. Mas várias
ideias são ortogonais à arquitetura e diretamente portáveis:

- **Estatísticas de dinâmica temporal, não só pooling.** RAS usa, para cada modalidade,
  `s_m = [μ, σ, μ_Δ, σ_Δ]` — média/desvio do sinal e da sua *primeira diferença* no
  tempo — como resumo compacto antes da fusão. Nossa GNN já modela o tempo via cadeia
  `t→t+1` (`FaceTemporalChainGCN`/`FaceLandmarkTemporalGCN`), mas o vetor final que sai
  para a fusão é só o pooling médio (`h.mean(dim=0)`). Vale testar concatenar também o
  desvio-padrão e as estatísticas da diferença temporal (velocidade média/variância do
  embedding, não só das coordenadas — já temos `use_velocity` para coordenadas cruas,
  mas não para o embedding pós-GCN).
- **Head auxiliar supervisionado no tempo, para localizar o evento.** O modelo de áudio
  do RAS usa uma soft-label por *token* de tempo (derivada das anotações frame-level do
  BAH) mais uma "event loss" que empurra o pico máximo do sinal temporal a ser
  discriminativo, já que A/H pode ocorrer só num trecho curto do vídeo. O dataset BAH
  tem anotações frame-level — se ainda não estamos usando isso na GNN, dá pra adicionar
  uma head auxiliar por janela temporal (supervisão fraca via frame-level labels)
  análoga, ao invés de só supervisionar no nível do vídeo inteiro.
- **Gate de confiabilidade condicionado, não peso fixo por canal.** Tanto o AMF do
  IISERB (`g = σ(W[h_t; h_v; h_a; h_m])`) quanto o Text Residual Fusion do RAS
  (`g_m = σ(MLP[b; h_m])`) usam um gate *aprendido e dependente da amostra* para decidir
  quanto confiar em cada modalidade, ao invés de um peso fixo ou média simples. Se o
  `meta_router_ca_gnn` atual pondera CA vs GNN com um roteador aprendido, isso já vai
  nessa direção — mas vale conferir se o roteador é condicionado no conteúdo (como os
  dois papers fazem) ou é uma mistura estática, porque a lição empírica dos dois
  primeiros colocados é que gate condicionado > peso fixo.
- **Canal de "cena" (corpo/contexto), separado do rosto recortado.** O RAS extrai um
  canal adicional de VideoMAE sobre o *frame inteiro* (não só o rosto), capturando
  postura e ambiente, e ele mede MF1 61.1 sozinho, comparável ao facial. É um canal que
  não estamos replicando com grafo (nosso Face Mesh é só rosto). Não é prioridade —
  ainda é o canal mais fraco dos dois — mas se sobrar tempo, um sinal de postura via
  MediaPipe Pose alimentando um grafo corporal seria a extensão natural da nossa
  abordagem de grafo geométrico para esse "scene" channel, ao invés de VideoMAE bruto.
- **Regularização por flow matching no espaço de features/logits (RAS, face model).**
  É uma técnica específica e não trivial de portar (steps de integração, duas losses
  extras); citamos como possibilidade de trabalho futuro, não como próximo passo —
  o ganho relativo dela isolado não é reportado (só o pipeline completo com FM), então
  não há evidência direta de que valha o custo de implementação antes de esgotar as
  ideias mais simples acima (estatísticas de diferença temporal, gate condicionado,
  supervisão frame-level).

## 4. O que os dois relatam que NÃO ajudou (evitar repetir)

- Prosódia (pausas, jitter, shimmer) sozinha teve AP baixo (0.666) e **piorou** a fusão
  do IISERB — sinal de que adicionar canais fracos sem gate de confiabilidade pode
  arrastar o ensemble pra baixo.
- Supervisionar uma head extra nos *tipos de cue* anotados no BAH **piorou** levemente o
  AP de teste do IISERB — supervisão auxiliar mal desenhada pode atrapalhar mais que
  ajudar; se formos adicionar a head frame-level sugerida acima, vale testar o peso da
  loss auxiliar com cuidado (o RAS usa pesos pequenos: 0.5/0.02/0.1 para os termos
  auxiliares de áudio).
- Trocar só o operador de "conflito" cross-modal (diferença absoluta vs. split
  ortogonal vs. nenhum) não moveu o resultado de forma consistente entre validação e
  teste no IISERB — architeture tweaks no fusion step têm efeito marginal comparado a
  calibração e ao canal em si.
- **Calibração de threshold/pesos ajustada no split de validação pequeno (124 vídeos)
  sobreajusta fortemente**: IISERB perdeu 4 pontos de macro-F1 (0.741 val → 0.690 test)
  fazendo isso, e resolveu fixando o threshold em 0.5 e pesando membros por AP
  (threshold-free). Isso não é sobre o canal visual especificamente, mas é relevante
  para qualquer ensemble/roteador que estejamos calibrando no nosso split pequeno de
  validação — vale checar se o `meta_router_ca_gnn` tem o mesmo risco.

## 5. Próximos passos sugeridos (exploratórios, não comprometidos)

1. Adicionar σ e estatísticas de Δ-temporal ao vetor de saída da GNN facial (baixo
   custo, direto em `face_gcn_ts.py`), e comparar contra o pooling médio atual.
2. Auditar se o roteador do `meta_router_ca_gnn` é condicionado por amostra (gate
   aprendido) ou usa pesos fixos/globais — se for fixo, testar uma versão gated no
   estilo AMF/Text-Residual-Fusion.
3. Conferir se a calibração de threshold do nosso pipeline é feita no split de
   validação pequeno; se sim, testar a alternativa "AP-weighted, threshold fixo" do
   IISERB antes de investir mais em arquitetura.
4. Avaliar anotações frame-level do BAH como supervisão auxiliar fraca por janela
   temporal na GNN (loss extra de baixo peso), inspirado no token-level head do áudio
   do RAS.
5. Como experimento de baixa prioridade: canal de "cena"/postura via grafo (MediaPipe
   Pose) para cobrir o que hoje só o VideoMAE de cena do RAS captura.

## Fontes
- Kumar, Mishra, Lone. *Simple Features and Honest Calibration for Ambivalence and
  Hesitancy Recognition in Video*. arXiv:2607.11120 (1º lugar, AP 0.731 public test).
- Ryumina et al. *Team RAS in 11th ABAW Competition: Multimodal Ambivalence
  Recognition Approach*. arXiv:2607.14702 (2º lugar, MF1 78.24% private test).
