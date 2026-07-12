<a id="readme-top"></a>

# BAH AH-Challenge (ABAW11) — Áudio + Texto

Pipeline **multimodal áudio + texto (sem vídeo)** para prever **ambivalência/hesitação (A/H)**
em respostas faladas — *AH Video Recognition Challenge, 3ª edição (ABAW11 @ ECCV 2026)*.
Tarefa: **classificação binária a nível de vídeo** (`1` = há A/H, `0` = não há). Métrica
oficial: **Macro-F1**.

A abordagem usa uma **janela deslizante (5 s, hop 2,5 s)** alinhando áudio ⟷ texto por
timestamps do Whisper, extrai **embeddings de texto** (RoBERTa-emotion EN, configurável) e
**features de áudio** (LIBROSA ou wav2vec2/HuBERT), e classifica cada janela com um
**RandomForest** (baseline, CPU) ou uma **cross-attention temporal** (Lightning, opcional);
a predição final por vídeo vem da **agregação das janelas** com limiar calibrado.

> Projeto, decisões de arquitetura e detalhes de cada módulo:
> **[docs/implementation/](docs/implementation/README.md)** (plano em 7 fases).

<details open="open">
  <summary><b>Índice</b></summary>
  <ol>
    <li><a href="#1-requisitos">Requisitos</a></li>
    <li><a href="#2-instalação--inicialização">Instalação &amp; inicialização</a></li>
    <li><a href="#3-dataset">Dataset</a></li>
    <li><a href="#4-início-rápido">Início rápido</a></li>
    <li><a href="#5-comandos-make">Comandos (make)</a></li>
    <li><a href="#6-rodando-experimentos-hydra">Rodando experimentos (Hydra)</a></li>
    <li><a href="#7-configuração-dos-yaml">Configuração dos YAML</a></li>
    <li><a href="#8-estrutura-do-projeto">Estrutura do projeto</a></li>
    <li><a href="#9-saídas--submissão">Saídas &amp; submissão</a></li>
    <li><a href="#autores--licença">Autores &amp; licença</a></li>
  </ol>
</details>

---

## 1. Requisitos

| Requisito | Versão / nota |
|-----------|---------------|
| **Python** | ≥ 3.12 |
| **[uv](https://docs.astral.sh/uv/)** | gerenciador de ambiente/dependências |
| **ffmpeg** | dependência de **sistema** (extração de áudio mp4 → flac) |
| **GPU** | **opcional** — CPU funciona; **Apple Metal (MPS)** e CUDA suportados via `device` |

O caminho **RandomForest roda 100% em CPU**. O modelo neural (cross-attention) usa
**PyTorch Lightning**, que é uma dependência **opcional** (grupo `neural`).

---

## 2. Instalação & inicialização

```bash
# 1) uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh      # ou: brew install uv

# 2) ffmpeg (sistema)
brew install ffmpeg                                  # macOS  (Linux: apt install ffmpeg)

# 3) ambiente do projeto (core: RF/CPU + ferramentas de dev) — SEM Lightning
make setup            # == uv sync

# 4) (opcional) modelo neural cross-attention — adiciona Lightning + torchmetrics
make setup-neural     # == uv sync --group neural
```

Tudo roda **dentro do ambiente gerenciado pelo `uv`** (`uv run …`); não é preciso ativar
venv na mão. O pacote é instalado em modo editável (src-layout) e expõe o entrypoint
`bah` (= `main:main`).

**Tracking (Weights & Biases):** o W&B é opcional. Faça `wandb login` uma vez, ou rode
offline/desligado pelo config (`wandb.mode=offline|disabled`) ou pela env `WANDB_MODE`.

```bash
make ffmpeg-check     # confirma o ffmpeg
make ci               # gate rápido (format-check + lint + compile) — não precisa de dados
```

---

## 3. Dataset

O dataset **BAH** (300 participantes, ≤ 7 vídeos cada) deve ser baixado do desafio (requer
aceite da EULA) e colocado sob **`data/raw/data/`**:

```
data/raw/data/
├── Videos/<pid>/Visite_1/<pid>_Question_<q>_..._Video.mp4   # vídeos brutos (extraímos só o áudio)
├── transcription/                                           # transcrição Whisper (chunks + timestamps)
├── split/{train,val,test}.txt                               # splits participant-wise (id, classe, transcrição)
├── video_annotation_transcript.yaml                         # global_ah (rótulo de vídeo) + time_detailed_ah
├── meta_data.yml                                            # metadados demográficos por participante
└── bah-video.csv
```

Os caminhos são configuráveis em [`configs/data/default.yaml`](configs/data/default.yaml)
(`data.paths.*`). Artefatos derivados ficam em `data/interim/` (áudio `.flac`, índice de
janelas) e `data/processed/` (features em Parquet) — ambos ignorados pelo git.

---

## 4. Início rápido

```bash
# preparação dos dados: extrai áudio → indexa/janela → embeddings → Parquet
make data                       # featurize usa DEVICE=auto (MPS no Apple Silicon)

# baseline RandomForest (CPU) — treina, calibra agregação e avalia
make train
make evaluate                   # Macro-F1 / AP no split de validação

# modelo neural (opcional, Apple Metal)
make setup-neural
make train-neural               # cross-attention, device=mps + fallback p/ CPU

# arquivo de submissão (private test)
make submit                     # SPLIT=test, OUT=outputs/submission.txt
```

Equivalentes sem `make` (Hydra direto): ver [§6](#6-rodando-experimentos-hydra).

---

## 5. Comandos (make)

`make help` lista tudo. Principais alvos:

| Grupo | Alvo | O que faz |
|-------|------|-----------|
| **Ambiente** | `setup` · `setup-neural` · `setup-all` | `uv sync` (core) · + grupo `neural` · tudo |
| | `ffmpeg-check` | confirma o ffmpeg instalado |
| **Dados** | `extract-audio` · `preprocess` · `featurize` · `data` | mp4→flac · índice+janelas · embeddings→Parquet · os 3 |
| **Treino/aval.** | `train` (=`train-rf`) · `train-neural` · `sweep` | RF (CPU) · cross-attention (MPS) · multirun |
| | `evaluate` · `submit` · `pipeline` | métricas num split · submissão · ponta a ponta |
| **Qualidade (CI)** | `ci` · `check` | `format-check + lint + compile` · + `typecheck + test` |
| | `lint` · `format` · `typecheck` · `test` · `compile` | ruff · ruff --fix · mypy · pytest · py_compile |
| **Limpeza** | `clean` · `clean-cache` · `clean-outputs` · `clean-all` | caches · `data/interim,processed` · `outputs/…` · tudo |

**Parâmetros** (sobrescreva na linha de comando):

```bash
make featurize DEVICE=mps
make evaluate SPLIT=test
make train ARGS="model.n_estimators=800 data.window.size_s=4"
make sweep SWEEP="model.lr=1e-3,5e-4 model.num_heads=4,8"
```

`DEVICE` = `auto`(default) `|cpu|mps|cuda` · `SPLIT` = `val|test` · `OUT` = arquivo de saída ·
`EXPERIMENT` = preset · `ARGS` = overrides Hydra extras · `SWEEP` = grade do multirun.

---

## 6. Rodando experimentos (Hydra)

A CLI é um entrypoint **[Hydra](https://hydra.cc/)** (`main.py`). O **modo** é escolhido por
`mode=` e qualquer parâmetro pode ser sobrescrito na linha de comando:

```bash
uv run python main.py mode=preprocess                       # índice + janelas
uv run python main.py mode=featurize device=mps             # embeddings → Parquet
uv run python main.py                                       # mode=train (default) — RF baseline
uv run python main.py mode=evaluate split=val
uv run python main.py mode=submit split=test out=outputs/submission.txt
```

**Trocar de modelo/embedder** = selecionar outro arquivo do grupo, ou usar um **preset**:

```bash
# trocar peças individuais
uv run python main.py text_embedder=minilm audio_embedder=wav2vec2

# presets (configs/experiment/*) trocam vários grupos de uma vez
uv run python main.py +experiment=rf_baseline               # RF + librosa + sklearn (CPU)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python main.py +experiment=cross_attention device=mps   # cross-attention + wav2vec2 + Lightning
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python main.py +experiment=hetero_gnn mode=train device=cuda  # GNN heterogêneo (gnn-modalblocks)
```

Procedimento detalhado do GNN: [`references/gnn_training_procedure.md`](references/gnn_training_procedure.md).

**Multirun** (varredura paralela via launcher joblib):

```bash
uv run python main.py -m model.n_estimators=400,800 data.window.size_s=4,5,6
```

> **Device:** `device=auto` resolve para **MPS ▸ CUDA ▸ CPU**. No Apple Silicon, o caminho
> neural exporta `PYTORCH_ENABLE_MPS_FALLBACK=1` (o alvo `make train-neural` já faz isso) para
> cair em CPU nas operações ainda não suportadas pelo Metal.

---

## 7. Configuração dos YAML

A configuração é composta por **grupos Hydra** em [`configs/`](configs/) — um eixo por
diretório. O `config.yaml` raiz lista os defaults; trocar um eixo é só apontar outro arquivo
do grupo (`grupo=arquivo`) ou editar o YAML.

```
configs/
├── config.yaml                 # raiz: defaults list + seed, device, mode, wandb, launcher
├── data/default.yaml           # data.paths.* · data.audio.* · data.window.* · data.tabular
├── text_embedder/              # roberta_emotion (default) · minilm · bertimbau
├── audio_embedder/             # librosa (default, CPU) · wav2vec2 · hubert
├── model/                      # random_forest (family=sklearn) · cross_attention (family=lightning)
├── trainer/                    # sklearn (CPU) · lightning (accelerator derivado do device)
├── aggregation/default.yaml    # método (mean_proba) + limiar (auto, calibrado na val)
└── experiment/                 # presets: rf_baseline · cross_attention
```

Blocos mais úteis para calibrar:

| Onde | Chaves | Efeito |
|------|--------|--------|
| `config.yaml` | `seed`, `device`, `mode`, `wandb.mode` | semente, dispositivo, etapa, tracking |
| `data/default.yaml` | `data.window.{size_s,hop_s,min_overlap_for_positive}` | janela deslizante + rótulo da janela |
| `data/default.yaml` | `data.paths.*` | caminhos do dataset e dos artefatos |
| `text_embedder/*` | `model_name`, `pooling`, `max_length` | encoder de texto (HuggingFace) |
| `audio_embedder/*` | `backend` (`librosa\|wav2vec2\|hubert`), `n_mfcc`, `agg_stats` | features de áudio |
| `model/random_forest` | `n_estimators`, `max_depth`, `class_weight` | hiperparâmetros do RF |
| `model/cross_attention` | `common_dim`, `num_heads`, `dropout`, `lr` | rede de fusão temporal |
| `aggregation/default` | `method`, `threshold` | como janelas viram a predição do vídeo |

Três formas de ajustar, em ordem de praticidade:

```bash
# 1) override pontual na CLI (não altera arquivos)
uv run python main.py model.n_estimators=800 text_embedder=minilm

# 2) editar o arquivo do grupo (ex.: configs/model/random_forest.yaml)
# 3) criar um preset em configs/experiment/<nome>.yaml e usar +experiment=<nome>
```

> **Idioma do texto:** as transcrições do BAH são em **inglês**; o default é
> `cardiffnlp/twitter-roberta-base-emotion` (RoBERTa EN). Não use o `bertimbau` (PT) sobre o
> texto original — só com tradução. Detalhes em [docs §8](docs/implementation/README.md).

---

## 8. Estrutura do projeto

```
.
├── main.py                     # entrypoint Hydra (@hydra.main) — dispatch por cfg.mode
├── Makefile                    # comandos centralizados (pipeline + CI/CD)
├── pyproject.toml              # deps (uv) — core + grupos `neural`/`dev`
├── configs/                    # grupos Hydra (ver §7)
├── docs/implementation/        # plano de implementação (7 fases)
├── data/                       # raw/ (dataset) · interim/ (áudio, janelas) · processed/ (features)
└── src/
    ├── conf/                   # schemas tipados + resolve_device + seed_everything
    ├── logger.py
    ├── base/                   # ABCs: BaseEmbedder, BaseModel, BaseTrainer
    ├── data/                   # indexing, audio_io, windowing, datasets, schema
    ├── features/               # text_embedder, audio_embedder, tabular, builder
    ├── models/                 # registry, random_forest, cross_attention
    ├── training/               # factory, sklearn_trainer, lightning_trainer, aggregation, metrics, splits
    ├── outputs/                # wandb_logger, checkpoint, reporter, submission
    ├── pipeline/               # preprocess, featurize (orquestração)
    └── scripts/                # extract_audio (mp4 → flac 16 kHz)
```

---

## 9. Saídas & submissão

- Cada run grava em `outputs/<experiment_name>/<timestamp>/` (checkpoint, métricas, plots) e,
  se habilitado, no **W&B**. O Reporter local sempre roda como fallback offline.
- `make submit` (ou `mode=submit`) gera o arquivo de predições a nível de vídeo
  (`video_id, pred`) em `OUT` (default `outputs/submission.txt`).
- Métrica oficial: **Macro-F1 a nível de vídeo**; *private test* é enviado por e-mail
  (≤ 5 trials/semana, melhor conta).

---

## Autores & licença

- **Liga IA** — [Matheus Girardi](mailto:matheusmgirardi@gmail.com) ·
  [Luiz Fernando](mailto:lf.fonseca.0808@gmail.com)
- Licença: ver [LICENSE](LICENSE).

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>
