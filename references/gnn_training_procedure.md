# Procedimento de treinamento — GNN heterogêneo (gnn-modalblocks)

Este documento descreve a abordagem **GNN heterogêneo** para o **ABAW 11th AH Video
Recognition Challenge**, implementada com a biblioteca
[`gnn-modalblocks`](/home/rwp/code/project_lib/gnn-modalblocks) e o pipeline de
dados da branch `origin/luiz`.

## 1. Objetivo

Classificação **binária em nível de vídeo**: prever se um vídeo contém
Ambivalência/Hesitação (A/H) — label `1` — ou não — label `0`.

Métrica oficial do challenge: **Macro F1** no test set privado (+ AP da classe positiva).

## 2. Relação com as branches de atenção

| Mecanismo (branches anteriores) | Equivalente no grafo GNN |
|--------------------------------|--------------------------|
| Local self-attention intra-modal (`matheus-local-attn`) | Arestas `temporal` entre janelas consecutivas (`audio→audio`, `text→text`) |
| Cross-attention áudio→texto (`luiz`, `matheus`) | Arestas `aligns` entre nós da mesma janela (`audio_i → text_i`) |
| Masked mean pool → logit de vídeo (`luiz`) | Nó `video` conectado a todas as janelas via `reports` + classificador no nó `video` |
| Features tabulares (usadas no RF, ignoradas na cross-attention) | Features iniciais do nó `video` (média tabular por janela) |

O **HeteroGAT** (`gnn_modalblocks.architectures.hetero_gat`) substitui camadas fixas
de atenção por **message passing aprendido** sobre essa topologia.

## Modelos GNN

| Modelo | Comando Hydra | Descrição |
|--------|---------------|-----------|
| **`gnn_baseline`** (default do preset) | `model=gnn_baseline` | MultimodalBlock + LatentCorrelationGCN — espelha cross-attention |
| **`hetero_gnn`** (avançado) | `model=hetero_gnn` | HeteroGAT com grafo áudio/texto/vídeo |
| **`hetero_gnn_contrastive`** | `model=hetero_gnn_contrastive` | HeteroGAT + BCE + SupCon (+ triplet opcional) |
| **`multimodal_hetero_full`** (competição) | `+experiment=multimodal_hetero_full` | Stack completo: MultimodalBlock → HetGAT (grafo fused) → LatentGCN → BiLSTM + contrastive multi-task |
| **`multimodal_hetero_face`** (competição + face) | `+experiment=multimodal_hetero_face` | Stack acima + **Face Mesh 468 pts** com GCN espacial (arestas por distância) + GCN temporal (GNN4TS-style) |

## 3. Arquiteturas de rede neural

Diagramas refletem o código em `src/models/` e `src/data/graph_builder.py`.
Dimensões entre parênteses vêm de `configs/model/*.yaml`.

> **Como ler os diagramas — código de cores por modalidade:**
>
> | Cor | Significado |
> |-----|-------------|
> | 🔵 azul | áudio |
> | 🟢 verde | texto |
> | 🟣 roxo | fusão áudio+texto (`fused`) |
> | 🟠 laranja | readout / nó `video` |
> | 🌸 rosa | face (visual) |
> | ⚪ cinza | saída (logit → probabilidade) |
> | 🟦 índigo | bloco interno (detalhe em 3.6) |
>
> Para reduzir ruído, os diagramas de modelo (3.2–3.4) mostram **só o forward-pass**.
> Os blocos internos estão em **3.6**; as perdas de treino em **3.7**.

### 3.1 Visão geral (uma linha)

Independente do modelo, o fluxo é o mesmo: janelas → embeddings → encoder → 1 logit por vídeo.

```mermaid
flowchart LR
    WIN["Janelas 5s<br/>(hop 2.5s)"] --> EMB["Embeddings por janela"]
    EMB --> ENC["Encoder<br/>(varia por modelo)"]
    ENC --> OUT["logit → sigmoid → P(A/H)"]

    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
    class WIN,EMB,ENC,OUT gray;
```

### 3.2 `hetero_gnn` — o modelo mais simples (comece por aqui)

Embeddings brutos viram um **grafo por vídeo**, o HeteroGAT propaga informação e o nó
`video` produz o logit.

```mermaid
flowchart LR
    A["áudio (768)"]:::audio --> G
    T["texto (768)"]:::text --> G
    TAB["tabular (32)"]:::video --> G
    G["grafo do vídeo<br/>(nós = janelas)"]:::video --> GAT["HeteroGAT<br/>3.6.2"]:::video
    GAT --> ZV["nó video (64)"]:::video
    ZV --> MLP["MLP"]:::gray --> OUT["P(A/H)"]:::gray

    classDef audio fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef text fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
```

O que é o "grafo do vídeo" (3.5 detalha): cada janela é um nó; arestas ligam janelas
vizinhas no tempo, alinham áudio↔texto e reportam ao nó `video`.

Arquivos: `src/models/hetero_gnn.py`, `src/models/hetero_gnn_contrastive.py`.

### 3.3 `multimodal_hetero_full` — 3 ramos em paralelo

A ideia central: fundir áudio+texto uma vez (`fused`) e olhar esse sinal por **3 lentes
complementares**, depois concatenar.

```mermaid
flowchart LR
    A["áudio (768)"]:::audio --> FUSED
    T["texto (768)"]:::text --> FUSED
    FUSED["fused (512)<br/>3.6.1"]:::fused

    FUSED --> R1["Ramo 1<br/>HeteroGAT 3.6.2 → 64"]:::video
    FUSED --> R2["Ramo 2<br/>LatentGCN 3.6.3 → 128"]:::video
    FUSED --> R3["Ramo 3<br/>BiLSTM 3.6.4 → 64"]:::video

    R1 --> CAT["concat<br/>(256)"]:::gray
    R2 --> CAT
    R3 --> CAT
    CAT --> MLP["MLP"]:::gray --> OUT["P(A/H)"]:::gray

    classDef audio fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef text fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef fused fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
```

**Para que serve cada ramo:**

| Ramo | Captura | Saída | Zoom interno |
|------|---------|-------|--------------|
| HeteroGAT | relações no grafo (temporal + cross-modal) | 64 | 3.6.2 |
| LatentGCN | correlações latentes entre janelas | 128 | 3.6.3 |
| BiLSTM | ordem temporal densa | 64 | 3.6.4 |
| **concat** | | **256** | |

O `fused` vem do **MultimodalBlock** (3.6.1).

Arquivo: `src/models/multimodal_hetero_full.py`.

### 3.4 `multimodal_hetero_face` — o `full` + 1 ramo visual

Reusa **todo** o `full` (256-d) e soma um 4º ramo de face (128-d) antes do MLP.

```mermaid
flowchart LR
    FULL["stack full<br/>(3.3) → 256"]:::video --> CAT
    FACE["face_seq<br/>(B,T,468,3)"]:::face --> SG["GCN espacial 3.6.5"]:::face
    SG --> TG["GCN temporal 3.6.6"]:::face
    TG --> ZF["face (128)"]:::face
    ZF --> CAT["concat<br/>(384)"]:::gray
    CAT --> MLP["MLP"]:::gray --> OUT["P(A/H)"]:::gray

    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef face fill:#fce7f3,stroke:#ec4899,color:#831843;
    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
```

Arquivo: `src/models/multimodal_hetero_face.py`, `src/models/face_gcn_ts.py`.

O ramo face abre em dois blocos: **GCN espacial** (3.6.5) e **GCN temporal** (3.6.6).

### 3.5 Zoom: o grafo heterogêneo de um vídeo

Só para entender o "grafo do vídeo" da 3.2/3.3. Cada janela `t` é um nó; três tipos de aresta:

```mermaid
flowchart LR
    A0["áudio t"]:::audio -->|temporal| A1["áudio t+1"]:::audio
    T0["texto t"]:::text -->|temporal| T1["texto t+1"]:::text
    A0 -.aligns.-> T0
    A1 -.aligns.-> T1
    A0 -->|reports| V["nó video"]:::video
    T0 -->|reports| V
    A1 -->|reports| V
    T1 -->|reports| V

    classDef audio fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef text fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
```

| Aresta | Papel | Equivalente clássico |
|--------|-------|----------------------|
| `temporal` | liga janelas vizinhas | self-attention local |
| `aligns` | liga áudio↔texto da mesma janela | cross-attention |
| `reports` | toda janela → nó `video` | mean pool + classificador |

No `multimodal_hetero_full` há ainda o nó `fused` com as mesmas arestas
(`src/data/graph_builder.py`, `BAH_FULL_GRAPH_METADATA`).

### 3.6 Zoom: blocos internos

Cada caixa dos diagramas 3.2–3.4 corresponde a um módulo concreto. Abaixo, o
**forward interno** de cada bloco (sem losses).

#### 3.6.1 MultimodalBlock — fusão áudio + texto

Gera o tensor `fused (512)` usado pelos 3 ramos do `full`. Config: `fusion: attention`.

```mermaid
flowchart LR
    A["áudio (512)<br/>já projetado"]:::audio --> ATT
    T["texto (512)<br/>já projetado"]:::text --> ATT
    ATT["softmax por modalidade<br/>peso aprendido"]:::block
    ATT --> SUM["soma ponderada"]:::block
    SUM --> OUT["fused (512)"]:::fused

    classDef audio fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef text fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef fused fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

| Etapa | O que faz |
|-------|-----------|
| Projeção | `Linear(768→512)` em áudio e texto (antes do block, em `multimodal_hetero_full`) |
| AttentionFusion | empilha modalidades, calcula peso `softmax(Linear(x))` por janela |
| Saída | embedding fundido `fused` por janela temporal |

Arquivo: `gnn-modalblocks/fusion/multimodal_block.py`, `fusion/attention.py`.

#### 3.6.2 HeteroGAT — message passing no grafo

2 camadas de `HeteroConv` + `GATConv`, uma por tipo de aresta (`temporal`, `aligns`, `reports` + reversas).

```mermaid
flowchart LR
    IN["HeteroData<br/>nós: audio, text, fused?, video"]:::video --> PROJ

    subgraph L1 ["Camada 1"]
        PROJ["Linear por tipo de nó<br/>ex: 768→128"]:::block
        GAT1["GATConv × cada aresta<br/>heads=4, aggr=sum"]:::block
        ELU["ELU"]:::block
        PROJ --> GAT1 --> ELU
    end

    subgraph L2 ["Camada 2"]
        GAT2["GATConv × cada aresta<br/>heads=1 → 64-d"]:::block
        ELU --> GAT2
    end

    GAT2 --> ZV["z_video (64)<br/>nó readout"]:::video

    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

| Etapa | Dimensões (default) | Papel |
|-------|---------------------|-------|
| Projeção | por tipo de nó → 128 | alinha dimensões antes do GAT |
| GAT camada 1 | 128 → 128 (4 heads) | propaga entre janelas e modalidades |
| GAT camada 2 | 128 → 64 | embedding final do nó `video` |

No `hetero_gnn`, `z_video` passa por um MLP externo (`refine`). No `full`, entra no `concat`.

Arquivo: `gnn-modalblocks/architectures/hetero_gat.py`.

#### 3.6.3 LatentGCN — adjacência aprendida entre janelas

Cada janela `fused` é um nó; o grafo **não é fixo** — a adjacência vem de atenção Q·K com sparsificação top-k.

```mermaid
flowchart LR
    IN["fused por janela<br/>(T, 512)"]:::fused --> QK

    subgraph ADJ ["Adjacência aprendida"]
        QK["Q, K por janela<br/>4 heads"]:::block
        TOPK["top-k=8 vizinhos<br/>por janela"]:::block
        SOFT["softmax → A (T×T)"]:::block
        QK --> TOPK --> SOFT
    end

    subgraph MSG ["Message passing"]
        V["V por janela"]:::block
        MP["h = A @ V"]:::block
        LIN["Linear → ReLU"]:::block
        SOFT --> MP
        V --> MP --> LIN
    end

    LIN --> POOL["mean pool<br/>sobre T janelas"]:::block
    POOL --> OUT["(128)"]:::video

    classDef fused fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

| Etapa | O que faz |
|-------|-----------|
| Atenção Q·K | score entre pares de janelas (correlação latente) |
| top-k | mantém só os 8 vizinhos mais relevantes (grafo esparso) |
| A @ V | GCN clássico sobre adjacência aprendida |
| mean pool | 1 vetor por vídeo |

Arquivo: `gnn-modalblocks/architectures/latent_gcn.py` (`LatentCorrelationGCN`).

#### 3.6.4 TemporalEncoder — BiLSTM sobre `fused`

Captura ordem temporal **densa** (complementar ao grafo esparso do LatentGCN).

```mermaid
flowchart LR
    IN["fused (B, T, 512)"]:::fused --> PACK["pack por length<br/>(ignora padding)"]:::block
    PACK --> LSTM["BiLSTM 2 camadas<br/>hidden=64"]:::block
    LSTM --> CAT["concat último fwd+bwd"]:::block
    CAT --> LIN["Linear → 64"]:::block
    LIN --> OUT["(64) por vídeo"]:::video

    classDef fused fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

Arquivo: `gnn-modalblocks/architectures/temporal_encoder.py`.

#### 3.6.5 Face Spatial GCN — grafo dos 468 landmarks (por janela)

Uma janela de vídeo → 468 nós (keypoints MediaPipe). Arestas k-NN por distância euclidiana.

```mermaid
flowchart LR
    IN["468 × (x,y,z)"]:::face --> ADJ

    subgraph SPATIAL ["Por janela temporal"]
        ADJ["k-NN top_k=8<br/>peso exp(−d²/σ²)"]:::block
        G1["GCN₁: adj @ x → Linear → ReLU<br/>3 → 64"]:::block
        G2["GCN₂: adj @ h → Linear → ReLU<br/>64 → 128"]:::block
        POOL["mean pool 468 nós"]:::block
        ADJ --> G1 --> G2 --> POOL
    end

    POOL --> OUT["embedding janela (128)"]:::face

    classDef face fill:#fce7f3,stroke:#ec4899,color:#831843;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

Arquivos: `src/models/face_gcn_ts.py` (`DistanceGCNLayer`), `src/data/face_graph.py`.

#### 3.6.6 Face Temporal GCN — cadeia entre janelas

Modo default `chain`: embeddings espaciais de cada janela ligados em cadeia `t ↔ t+1`.

```mermaid
flowchart LR
    IN["seq de embeddings<br/>(T, 128)"]:::face --> ADJ

    subgraph TEMPORAL ["Entre janelas"]
        ADJ["adjacência chain<br/>t↔t+1 + self-loop"]:::block
        L1["Linear → ReLU<br/>128 → 64"]:::block
        MP1["adj @ h"]:::block
        L2["Linear → ReLU<br/>64 → 128"]:::block
        MP2["adj @ h"]:::block
        POOL["mean pool temporal"]:::block
        ADJ --> L1 --> MP1 --> L2 --> MP2 --> POOL
    end

    POOL --> OUT["face readout (128)"]:::face

    classDef face fill:#fce7f3,stroke:#ec4899,color:#831843;
    classDef block fill:#e0e7ff,stroke:#6366f1,color:#312e81;
```

Modo alternativo `gnn4ts`: o GCN temporal roda **por landmark** (`468 × T`) em vez de
pool espacial antes — ver `FaceLandmarkTemporalGCN` em `face_gcn_ts.py`.

#### 3.6.7 Mapa: onde cada bloco aparece

```mermaid
flowchart TB
    subgraph blocks ["Blocos (gnn-modalblocks + face)"]
        MB["MultimodalBlock"]:::fused
        HG["HeteroGAT"]:::video
        LG["LatentGCN"]:::video
        BI["BiLSTM"]:::video
        FS["Face Spatial GCN"]:::face
        FT["Face Temporal GCN"]:::face
    end

    subgraph models ["Modelos"]
        GNN["hetero_gnn"]:::gray
        FULL["multimodal_hetero_full"]:::gray
        FACE["multimodal_hetero_face"]:::gray
    end

    MB --> FULL
    HG --> GNN
    HG --> FULL
    LG --> FULL
    BI --> FULL
    MB --> FULL
    FULL --> FACE
    FS --> FACE
    FT --> FACE

    classDef fused fill:#ede9fe,stroke:#8b5cf6,color:#4c1d95;
    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef face fill:#fce7f3,stroke:#ec4899,color:#831843;
    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
```

### 3.7 Perdas de treino (fora do forward-pass)

Todos treinam com BCE; os modelos "contrastive/full" somam termos auxiliares sobre o embedding.

| Perda | Onde se aplica | Modelos |
|-------|----------------|---------|
| BCE / Focal | logit final | todos |
| SupCon (λ≈0.1) | `video_emb` | contrastive, full, face |
| NT-Xent (λ≈0.05) | pares áudio↔texto | full, face |
| Triplet (λ≈0.05) | `video_emb` (miner batch-hard) | opcional |

### 3.8 Ensemble de inferência (produção)

Três checkpoints já treinados; combinação linear das probabilidades e 1 limiar calibrado.

```mermaid
flowchart LR
    M1["full<br/>w=0.417"]:::video --> C
    M2["gnn_contrastive<br/>w=0.542"]:::video --> C
    M3["face<br/>w=0.042"]:::face --> C
    C["score = soma(wi · Pi)"]:::gray --> THR["limiar 0.46"]:::gray --> PRED["pred binária"]:::gray

    classDef video fill:#ffedd5,stroke:#f97316,color:#7c2d12;
    classDef face fill:#fce7f3,stroke:#ec4899,color:#831843;
    classDef gray fill:#f3f4f6,stroke:#9ca3af,color:#111827;
```

Otimização offline dos pesos: `scripts/ensemble_sweep.py` → `configs/ensemble/optimized.yaml`.

## Face Mesh + GCN temporal (GNN4TS-style)

Pipeline visual opcional para A/H (expressão facial):

1. **`mode=featurize_face`** — MediaPipe Face Mesh extrai **468 landmarks** `(x,y,z)` por janela
   temporal, alinhados a `[t0, t1]` do mp4. Grava coluna `face_landmarks` no Parquet.
2. **Grafo espacial (por janela)** — 468 nós; arestas k-NN ponderadas por
   `exp(-d²/σ²)` entre distâncias euclidianas entre keypoints (`src/data/face_graph.py`).
3. **Grafo temporal (por vídeo)** — sequência de embeddings de janela conectada em cadeia
   `t→t+1` (`src/models/face_gcn_ts.py`), análogo a grafos que variam no tempo (GNN4TS).
4. **Modelo** — `multimodal_hetero_face` concatena o readout facial ao stack HetGAT+LatentGCN+BiLSTM.

```bash
# deps visuais (MediaPipe + OpenCV)
uv sync --group neural --group vision

# 1) featurize áudio+texto (já existente)
uv run python main.py mode=featurize device=cuda

# 2) adicionar face landmarks ao Parquet
uv run python main.py mode=featurize_face +face_embedder=mediapipe

# 3) treinar
uv run python main.py +experiment=multimodal_hetero_face mode=train device=cuda wandb.mode=disabled
```

Requer `data/raw/Videos/**/*.mp4` (mesmos paths do índice).

## Logging de experimentos

O treino neural usa **Weights & Biases** (`wandb`, config `wandb.mode`) — **não** MLflow local.
Checkpoints e métricas ficam em `outputs/<experiment_name>/...` (`.ckpt` + `trainer_state.json`).

Para desligar W&B: `wandb.mode=disabled`.

A lib `gnn-modalblocks` tem callback MLflow; integração futura se necessário.

**Construção do grafo:** `src/data/graph_builder.py` → `build_video_hetero_graph()`.

- Entrada: sequências `(T, d_audio)`, `(T, d_text)`, `(T, d_tab)` por vídeo.
- Saída: `torch_geometric.data.HeteroData`.
- Batching: `collate_graph_batch()` → `Batch.from_data_list`.
- Topologia do grafo: ver **3.5**; blocos internos: **3.6**.

## 4. Modelo

Arquivo: `src/models/hetero_gnn.py`

1. **Encoder:** `ENCODERS.build("hetero_gat", ...)` da lib `gnn-modalblocks`.
   - 2 camadas `HeteroConv` + `GATConv`.
   - Arestas reversas adicionadas automaticamente (`hetero_utils`).
   - `head_node_type="video"` — predição no nó de leitura global.

2. **Refino:** MLP sobre logits derivados do nó `video` (BCEWithLogits).

3. **Lightning:** `LitHeteroGnn` — mesmas métricas (Macro F1, AP) do caminho
   cross-attention.

Registro: `hetero_gnn` em `src/models/registry.py` (import lazy, family=`lightning`).

## 5. Pipeline de dados (reutilizado de `origin/luiz`)

O GNN consome o **mesmo cache Parquet** que a cross-attention:

| Fase | Comando | Artefato |
|------|---------|----------|
| Pré-processamento | `uv run python main.py mode=preprocess` | `data/interim/windows.parquet`, áudio `.flac` |
| Featurização | `uv run python main.py mode=featurize device=cuda` | `data/processed/text_audio_windows.parquet` |
| Treino GNN | ver 6 | checkpoint em `outputs/` |

Cada linha do Parquet = **uma janela** com:
- `audio_emb`, `text_emb`, `tabular`
- `video_label` (alvo final)
- `split` (participant-wise)

Janelamento: 5 s de tamanho, hop 2,5 s (50% overlap) — ver `configs/data/default.yaml`.

## 6. Comandos de treinamento

### Instalação

```bash
# ffmpeg necessário para extração de áudio
uv sync --group neural
```

O grupo `neural` inclui `torch-geometric` e `gnn-modalblocks` (path editável em
`../project_lib/gnn-modalblocks`).

### Memória (RTX 3060 12 GB ou RAM limitada)

O featurize processa **256 janelas por vez** (`data.featurize_chunk_size`) para não
carregar ~15k waveforms na RAM de uma vez. Se ainda travar:

```bash
uv run python main.py +experiment=hetero_gnn mode=featurize device=cuda \
  data.featurize_chunk_size=128 \
  audio_embedder.batch_size=2 \
  text_embedder.batch_size=4
```

Evite rodar `mode=featurize` **sem** `+experiment=hetero_gnn` — o default usa
**librosa na CPU** (lento e pesado em todas as threads).

### Sequência completa

```bash
# 1) Índice + janelas + extração de áudio
uv run python main.py mode=preprocess

# 2) Embeddings (wav2vec2 + RoBERTa-emotion + tabular)
uv run python main.py mode=featurize device=cuda

# 3) Treino do GNN heterogêneo
uv run python main.py \
  +experiment=hetero_gnn \
  mode=train \
  device=cuda \
  wandb.mode=disabled

# 4) Avaliação no split de validação
uv run python main.py \
  +experiment=hetero_gnn \
  mode=evaluate \
  split=val \
  device=cuda \
  wandb.mode=disabled

# 5) Submissão no test set público
uv run python main.py \
  +experiment=hetero_gnn \
  mode=submit \
  split=test \
  device=cuda \
  wandb.mode=disabled
```

### Hiperparâmetros (`configs/model/hetero_gnn.yaml`)

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `hidden_channels` | 128 | Dimensão oculta do GAT |
| `heads` | 4 | Cabeças de atenção |
| `out_channels` | 64 | Embedding por tipo de nó após conv |
| `dropout` | 0.1 | Dropout no refino |
| `lr` | 1e-3 | AdamW |
| `weight_decay` | 1e-2 | AdamW |

Overrides via CLI Hydra, por exemplo:

```bash
uv run python main.py +experiment=hetero_gnn model.heads=8 model.hidden_channels=256
```

## 7. Comparação com cross-attention

| Aspecto | Cross-attention (`luiz`) | GNN heterogêneo |
|---------|--------------------------|-----------------|
| Fusão cross-modal | `MultiheadAttention(q=audio, kv=text)` | Arestas `aligns` + GAT |
| Contexto temporal | Atenção densa sobre T (com mask) | Arestas `temporal` locais + multi-hop GAT |
| Readout | Mean pool mascarado | Nó `video` + MLP |
| Tabular | Ignorado | Nó `video` (média por janela) |
| Lib | PyTorch nativo | `gnn-modalblocks` + PyG |

## 8. Arquivos relevantes

```
src/data/graph_builder.py      # topologia HeteroData + collate
src/models/hetero_gnn.py       # HeteroGAT + Lightning
src/models/registry.py         # registro hetero_gnn
configs/model/hetero_gnn.yaml
configs/experiment/hetero_gnn.yaml
configs/ensemble/optimized.yaml
scripts/ensemble_sweep.py
main.py                        # collate graph + ensemble_evaluate/submit
```

## 9. Ensemble (produção)

Três checkpoints Lightning são combinados para a submissão final. A otimização de pesos
usa **predições CSV já exportadas** — sem retreino e sem GPU.

### Modelos no ensemble otimizado

| Modelo | Checkpoint | Peso |
|--------|------------|------|
| `multimodal_hetero_full` | `outputs/multimodal_hetero_full/20260706_213403` | 0.416667 |
| `hetero_gnn_contrastive` | `outputs/hetero_gnn_contrastive/20260706_211918` | 0.541667 |
| `multimodal_hetero_face` | `outputs/multimodal_hetero_face/20260712_141548` | 0.041667 |

Config: `configs/ensemble/optimized.yaml` (`combine: weighted`, limiar 0.46).

### Resultados (Macro-F1)

| Ensemble | Val F1 | Test F1 (público) |
|----------|--------|-------------------|
| antigo: full + gnn (média) | 0.7019 | 0.6692 |
| **novo: full + gnn + face (ponderado)** | **0.7169** | **0.6843** |

Evidência: `outputs/ensemble_eval/report_*.json` (baseline), `outputs/ensemble_eval/weight_sweep.json` (otimizado).

### Comandos

```bash
# 1) Exportar predições de cada membro (val + test)
uv run python main.py +experiment=multimodal_hetero_full mode=evaluate split=val device=cuda
uv run python main.py +experiment=multimodal_hetero_full mode=evaluate split=test device=cuda
# (repetir para hetero_gnn_contrastive e multimodal_hetero_face)

# 2) Varredura de pesos offline (CPU, sem carregar modelos)
uv run python scripts/ensemble_sweep.py \
  --member full:outputs/.../eval_val/predictions.csv:outputs/.../eval_test/predictions.csv \
  --member gnn:outputs/.../eval_val/predictions.csv:outputs/.../eval_test/predictions.csv \
  --member face:outputs/.../eval_val/predictions.csv:outputs/.../eval_test/predictions.csv

# 3) Avaliar ensemble com config otimizada
uv run python main.py mode=ensemble_evaluate +ensemble=optimized split=val device=cuda

# 4) Gerar submissão (calibra limiar na val, infere no test)
uv run python main.py mode=ensemble_submit +ensemble=optimized device=cuda \
  data.num_workers=2 out=outputs/submission_ensemble_optimized.txt
```

Submissão gerada: `outputs/submission_ensemble_optimized.txt` (525 vídeos).

Baseline (média simples, 2 membros): `configs/ensemble/default.yaml` → `outputs/submission_ensemble.txt`.

## 10. Referências

- BAH dataset: `data/raw/readme.md`
- Challenge: https://affective-behavior-analysis-in-the-wild.github.io/11th/
- gnn-modalblocks: `/home/rwp/code/project_lib/gnn-modalblocks/README.md`
- Paper BAH: https://arxiv.org/pdf/2505.19328
- Auditoria / claims: `.specs/audits/claims.md`, `.specs/audits/DIGEST.md`
