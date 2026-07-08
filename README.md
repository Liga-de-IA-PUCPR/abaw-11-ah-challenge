<a id="readme-top"></a>

# BAH AH-Challenge (ABAW11) — Áudio + Texto

Pipeline **multimodal áudio + texto (sem vídeo)** para prever **ambivalência/hesitação (A/H)**
em respostas faladas — *AH Video Recognition Challenge, 3ª edição (ABAW11 @ ECCV 2026)*.
Tarefa: **classificação binária a nível de vídeo** (`1` = há A/H, `0` = não há). Métrica
oficial: **Macro-F1**.

A abordagem usa uma **janela deslizante** (3 s / 5 s / 7 s, configurável) alinhando áudio ⟷
texto por timestamps do Whisper, extrai **embeddings de texto** (RoBERTa-emotion EN,
configurável) e **features de áudio** (LibROSA ou wav2vec2/HuBERT), classifica cada janela
com um dos **8 modelos sklearn** (CPU) ou uma **cross-attention temporal** (Lightning, opcional);
a predição final por vídeo vem da **agregação das janelas** com limiar calibrado na validação.

> 📚 Projeto, decisões de arquitetura e detalhes de cada módulo:
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

Os **8 modelos sklearn** (RF, XGBoost, LightGBM, ExtraTrees, LogReg, CatBoost, MLP, Stacking)
rodam 100% em CPU e estão incluídos nas dependências core (`uv sync`).
O modelo neural (cross-attention) usa **PyTorch Lightning**, dependência **opcional** (grupo `neural`).

---

## 2. Instalação & inicialização

```bash
# 1) uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) ffmpeg (dependência de sistema)
sudo apt install -y ffmpeg          # Linux / WSL
brew install ffmpeg                 # macOS

# 3) ambiente do projeto (core: todos os modelos sklearn + ferramentas dev, incluindo CatBoost)
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
janelas `windows_index_<size_s>s.parquet`) e `data/processed/` (features em
`text_audio_windows_<embedder>_<size_s>s.parquet`) — ambos ignorados pelo git. **Os dois
nomes incluem o tamanho da janela** (`data.window.size_s`): cada janelamento (small/medium/
large) tem seu próprio cache, então `preprocess`/`featurize` para janelas diferentes nunca
sobrescrevem o cache umas das outras (ver §7).

---

## 4. Início rápido

```bash
# preparação dos dados: extrai áudio → indexa/janela → embeddings → Parquet
make data                        # featurize usa DEVICE=auto (MPS no Apple Silicon)

# baseline rápido — treina todos os 7 modelos sklearn com hiperparâmetros default
make train-all-sklearn
make compare                     # ranking por Macro-F1 de todos os runs
make best-model                  # top-1 rápido

# sweep de hiperparâmetros por modelo (grade pré-definida em configs/sweep/)
make sweep-rf                    # 96 combinações
make sweep-catboost              # 81 combinações
make sweep-all                   # todos os 7 modelos em sequência (531 combinações)

# refinamento stage-2 do melhor modelo (fixe os melhores hiperparâmetros do stage-1)
make sweep-rf-stage2 ARGS="model.n_estimators=400 model.max_depth=10"

# CatBoost tem um stage-3 pronto (refina além das bordas do stage-2 — ver §6)
make sweep-catboost-stage2
make sweep-catboost-stage3

# sweep de janelamento — apenas com o melhor modelo (após sweep-all + compare)
make preprocess-windows featurize-windows    # constrói os 3 caches (small/medium/large)
make sweep-windows MODEL_NAME=xgboost       # 3 janelas × 1 modelo

# modelo neural (opcional, Apple Metal / CUDA)
make setup-neural
make train-neural                # cross-attention, device=mps + fallback p/ CPU

# avaliação detalhada e submissão
make evaluate SPLIT=val
make submit                      # SPLIT=test, OUT=outputs/submission.txt
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
| | `preprocess-windows` · `featurize-windows` | caches para os 3 janelamentos (small/medium/large) |
| | `sweep-windows` | 3 janelas × 1 modelo (`MODEL_NAME=<modelo>`) — use após `sweep-all` |
| **Treino** | `train` (=`train-rf`) | RF baseline (CPU) |
| | `train-xgboost` · `train-lightgbm` | XGBoost · LightGBM (CPU) |
| | `train-extra-trees` | ExtraTreesClassifier (CPU) |
| | `train-logistic-regression` | Regressão Logística + StandardScaler (CPU) |
| | `train-catboost` | CatBoost (CPU) |
| | `train-mlp` | MLP densa + StandardScaler (CPU) |
| | `train-stacking` | Stacking RF+XGB+LGBM → LogReg meta (CPU, ~5× mais lento) |
| | `train-all-models` | RF + XGBoost + LightGBM em sequência |
| | `train-all-sklearn` | todos os 7 modelos sklearn em sequência |
| | `train-all-windows` | 9 runs: 3 modelos × 3 janelas (assume caches prontos) |
| | `train-neural` | cross-attention via Lightning (MPS/CUDA) |
| **Sweep** | `sweep-rf` · `sweep-xgboost` · `sweep-lightgbm` | grade pré-definida (stage-1) |
| | `sweep-extra-trees` · `sweep-logistic-regression` · `sweep-catboost` · `sweep-mlp` | grade stage-1 por modelo |
| | `sweep-rf-stage2` · `sweep-xgboost-stage2` · `sweep-lightgbm-stage2` | refinamento após stage-1 |
| | `sweep-catboost-stage2` · `sweep-catboost-stage3` | refinamento em 2 estágios do CatBoost (stage-3 refina além das bordas do stage-2) |
| | `sweep-all` | todos os 7 modelos em sequência (531 combinações) |
| | `sweep SWEEP="..."` | multirun livre com overrides Hydra |
| **Resultados** | `compare` | ranking Macro-F1 de todos os runs (treino + avaliação); inclui janela, embedder de texto/áudio e modelo |
| | `best-model` | top-1 do ranking |
| **Aval./subm.** | `evaluate` · `submit` · `pipeline` | métricas num split · submissão · ponta a ponta |
| **Qualidade (CI)** | `ci` · `check` | `format-check + lint + compile` · + `typecheck + test` |
| | `lint` · `format` · `typecheck` · `test` · `compile` | ruff · ruff --fix · mypy · pytest · py_compile |
| **Limpeza** | `clean` · `clean-cache` · `clean-outputs` · `clean-all` | caches · `data/interim,processed` · `outputs/…` · tudo |

**Parâmetros** (sobrescreva na linha de comando):

```bash
make featurize DEVICE=mps
make evaluate SPLIT=test
make train-rf ARGS="model.n_estimators=800"
make sweep-rf ARGS="hydra.launcher.n_jobs=4"   # paralelizar runs dentro do sweep
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

**Selecionar modelo** = apontar outro arquivo do grupo `model`:

```bash
uv run python main.py model=random_forest           # RF baseline (default)
uv run python main.py model=xgboost                 # XGBoost
uv run python main.py model=lightgbm                # LightGBM
uv run python main.py model=extra_trees             # ExtraTrees
uv run python main.py model=logistic_regression     # LogReg + StandardScaler
uv run python main.py model=catboost                # CatBoost
uv run python main.py model=mlp                     # MLP + StandardScaler
uv run python main.py model=stacking                # Stacking RF+XGB+LGBM
```

**Trocar embedder** ou usar um **preset**:

```bash
# trocar peças individuais
uv run python main.py text_embedder=minilm audio_embedder=wav2vec2

# preset cross-attention (troca model + audio_embedder + trainer de uma vez)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python main.py +experiment=cross_attention device=mps
```

**Janelamento**: trocar o grupo `window` aponta para um cache Parquet diferente (nome inclui
`size_s` — não sobrescreve os outros tamanhos, ver §7) — se for a primeira vez com esse
tamanho, rode `preprocess` e `featurize` antes de `train`:

```bash
uv run python main.py mode=preprocess window=small
uv run python main.py mode=featurize window=small
uv run python main.py model=xgboost window=small
```

Ou use `make preprocess-windows featurize-windows train-all-windows` para rodar tudo em sequência.

**Sweep de hiperparâmetros** (multirun paralelo via launcher joblib):

```bash
# sweep livre
uv run python main.py -m model=xgboost model.n_estimators=300,500 model.max_depth=4,6,8

# sweep via config pré-definida (configs/sweep/<nome>.yaml)
uv run python main.py -m +sweep=rf                          # stage-1: grade ampla
uv run python main.py -m +sweep=rf_stage2 \
  model.n_estimators=400 model.max_depth=10                 # stage-2: refinamento

# paralelizar runs dentro do sweep
uv run python main.py -m +sweep=catboost hydra.launcher.n_jobs=4
```

Os configs de sweep disponíveis são:

| Config | Combinações | Hiperparâmetros varridos |
|--------|-------------|--------------------------|
| `+sweep=rf` | 96 | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| `+sweep=xgboost` | 108 | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| `+sweep=lightgbm` | 108 | `n_estimators`, `num_leaves`, `learning_rate`, `min_child_samples` |
| `+sweep=extra_trees` | 96 | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| `+sweep=logistic_regression` | 6 | `C` (regularização L2) |
| `+sweep=catboost` | 81 | `n_estimators`, `depth`, `learning_rate`, `l2_leaf_reg` |
| `+sweep=mlp` | 36 | `hidden_dim_1`, `hidden_dim_2`, `alpha` |
| `+sweep=rf_stage2` | 27 | refino de RF (fixe `n_estimators` e `max_depth` do stage-1) |
| `+sweep=xgboost_stage2` | 54 | refino de XGBoost (fixe `max_depth` e `learning_rate`) |
| `+sweep=lightgbm_stage2` | 54 | refino de LightGBM (fixe `num_leaves` e `learning_rate`) |
| `+sweep=catboost_stage2` | 81 | refino de CatBoost após o stage-1 (todos os 4 hiperparâmetros, grade deslocada) |
| `+sweep=catboost_stage3` | 16 | refino além das bordas do stage-2 (`n_estimators<50`, `depth>10`); `learning_rate`/`l2_leaf_reg` fixos (já convergidos) |

> 🍎 **Device:** `device=auto` resolve para **MPS ▸ CUDA ▸ CPU**. No Apple Silicon, o caminho
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
├── window/                     # small (3s/1.5s) · medium (5s/2.5s, default) · large (7s/3.5s)
├── text_embedder/              # roberta_emotion (default) · minilm · bertimbau
├── audio_embedder/             # librosa (default, CPU) · wav2vec2 · hubert
├── model/                      # random_forest · xgboost · lightgbm · extra_trees ·
│                               # logistic_regression · catboost · mlp · stacking ·
│                               # cross_attention (family=lightning)
├── trainer/                    # sklearn (CPU) · lightning (accelerator derivado do device)
├── aggregation/default.yaml    # método (mean_proba) + limiar (auto, calibrado na val)
├── sweep/                      # grades de hiperparâmetros para multirun (-m +sweep=<nome>)
│   ├── rf.yaml · rf_stage2.yaml
│   ├── xgboost.yaml · xgboost_stage2.yaml
│   ├── lightgbm.yaml · lightgbm_stage2.yaml
│   ├── extra_trees.yaml · logistic_regression.yaml
│   ├── catboost.yaml · catboost_stage2.yaml · catboost_stage3.yaml
│   ├── mlp.yaml
└── experiment/                 # preset: cross_attention (model + audio_embedder + trainer)
```

Blocos mais úteis para calibrar:

| Onde | Chaves | Efeito |
|------|--------|--------|
| `config.yaml` | `seed`, `device`, `mode`, `wandb.mode` | semente, dispositivo, etapa, tracking |
| `window/*` | `size_s`, `hop_s`, `min_overlap_for_positive` | janela deslizante + rótulo — cada tamanho tem cache próprio (ver abaixo) |
| `data/default.yaml` | `data.paths.*` | caminhos do dataset e dos artefatos; `window_index`/`parquet_path` incluem `${data.window.size_s}` |
| `text_embedder/*` | `model_name`, `pooling`, `max_length` | encoder de texto (HuggingFace) |
| `audio_embedder/*` | `backend` (`librosa\|wav2vec2\|hubert`), `n_mfcc`, `agg_stats` | features de áudio |
| `model/random_forest` · `model/extra_trees` | `n_estimators`, `max_depth`, `class_weight` | florestas aleatórias |
| `model/xgboost` | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` | XGBoost |
| `model/lightgbm` | `n_estimators`, `num_leaves`, `learning_rate`, `class_weight` | LightGBM |
| `model/logistic_regression` | `C`, `penalty`, `solver` | regularização L2 (saga) |
| `model/catboost` | `n_estimators`, `depth`, `learning_rate`, `l2_leaf_reg` | CatBoost |
| `model/mlp` | `hidden_dim_1`, `hidden_dim_2`, `alpha`, `learning_rate_init` | MLP densa |
| `model/stacking` | `cv`, `meta_C`, `base_*_n_estimators` | meta-ensemble |
| `model/cross_attention` | `common_dim`, `num_heads`, `dropout`, `lr` | rede de fusão temporal |
| `aggregation/default` | `method`, `threshold`, `calibration` | como janelas viram a predição do vídeo |

> ⚠️ **Cache por janela:** `data.paths.window_index` e `data.paths.parquet_path` incluem
> `${data.window.size_s}` no nome do arquivo — cada tamanho de janela (small/medium/large)
> grava seu próprio cache em `data/interim/` e `data/processed/`. Isso evita que
> `preprocess-windows`/`featurize-windows`/`sweep-windows` (que rodam as 3 janelas em
> sequência) sobrescrevam o cache umas das outras — bug real já corrigido neste repositório.

Três formas de ajustar, em ordem de praticidade:

```bash
# 1) override pontual na CLI (não altera arquivos)
uv run python main.py model.n_estimators=800 text_embedder=minilm

# 2) editar o arquivo do grupo (ex.: configs/model/random_forest.yaml)
# 3) criar um preset em configs/experiment/<nome>.yaml e usar +experiment=<nome>
```

> ⚠️ **Idioma do texto:** as transcrições do BAH são em **inglês**; o default é
> `cardiffnlp/twitter-roberta-base-emotion` (RoBERTa EN). Não use o `bertimbau` (PT) sobre o
> texto original — só com tradução.

---

## 8. Estrutura do projeto

```
.
├── main.py                     # entrypoint Hydra (@hydra.main) — dispatch por cfg.mode
├── Makefile                    # comandos centralizados (pipeline + CI/CD)
├── pyproject.toml              # deps (uv) — core + grupos `neural`/`dev`
├── configs/                    # grupos Hydra (ver §7)
├── data/                       # raw/ (dataset) · interim/ (áudio, janelas) · processed/ (features)
└── src/
    ├── conf/                   # schemas tipados + resolve_device + seed_everything
    ├── logger.py
    ├── base/                   # ABCs: BaseEmbedder, BaseModel, BaseTrainer
    ├── data/                   # indexing, audio_io, windowing, datasets, schema
    ├── features/               # text_embedder, audio_embedder, tabular, builder
    ├── models/                 # registry · random_forest · xgboost_model · lightgbm_model
    │                           # extra_trees · logistic_regression · catboost_model
    │                           # mlp_model · stacking_model · cross_attention
    ├── training/               # factory · sklearn_trainer · lightning_trainer
    │                           # aggregation · metrics · splits
    ├── outputs/                # wandb_logger · checkpoint · reporter · submission
    ├── pipeline/               # preprocess · featurize (orquestração)
    └── scripts/                # extract_audio · compare_runs
```

---

## 9. Saídas & submissão

- Cada run grava em `outputs/<model_name>/<timestamp>/` (checkpoint, métricas, plots) e,
  se habilitado, no **W&B**. O Reporter local sempre roda como fallback offline.
- **Timestamp do run**: `YYYYMMDD_HHMMSS_ffffff_<pid>` (microssegundos + PID). Necessário
  porque `hydra.launcher.n_jobs>1` (sweeps paralelos) pode terminar 2+ jobs no mesmo segundo;
  com granularidade de segundo apenas, dois jobs escreviam no mesmo diretório e um
  sobrescrevia o outro **silenciosamente** — bug real já corrigido (`resolve_output_dir`).
- Checkpoints são organizados por modelo: `outputs/random_forest/`, `outputs/xgboost/` etc.
  — `evaluate` e `submit` resolvem automaticamente o checkpoint mais recente **do modelo
  selecionado** (não do modelo mais recente de qualquer tipo).
- `make compare` lê `train_result.json` (gerado automaticamente no treino) e
  `eval_val/metrics.json` (gerado pelo `make evaluate`), preferindo o segundo; exibe ranking
  por Macro-F1 de todos os runs, incluindo **janela**, **embedder de texto** e **embedder de
  áudio** usados em cada run (runs treinados antes dessa metadata existir aparecem como `?`).
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
