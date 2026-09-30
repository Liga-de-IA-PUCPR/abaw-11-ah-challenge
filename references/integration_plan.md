# Plano de integração — pipeline do Luiz (main) + modelos do Rodrigo

Branch: `integrate-rodrigo-multimodal` (criada a partir de `main@c0c2755`).
Origem dos módulos do Rodrigo: `origin/improve-macro-f1-beyond-router@bfdf856` (28/09),
que contém toda a `origin/GNN+CA-meta-router@39df581` (13/07) + 7 commits.

## 1. Objetivo

Uma única base (a `main`) em que os modelos convivem e se escolhem por config:

| Frente | Toggle | O que roda |
|--------|--------|------------|
| **A — Cross-attention (Luiz)** | `+experiment=cross_attention` | modelo do artigo, inalterado (`make reproduce-best` continua dando o mesmo número) |
| **B — Modelos do Rodrigo com o pré-processamento do Luiz** | `+experiment=hetero_gnn_v2_tune_wav2vec2`, `face_gnn_ts_roi`, `multimodal_hetero_face_v3`, `catboost_baseline`, … | GNNs heterogêneos, GNN facial (Face Mesh), CatBoost — todos lendo o **mesmo contrato de Parquet** gerado por `mode=featurize` da main (+ coluna `face_landmarks` via `mode=featurize_face`) |
| **C — Ensemble áudio + texto + vídeo** | `ensemble=[<runs CA>, <run GNN/face>]` (ou `+experiment=ensemble_*`) | média (ou média ponderada) das probabilidades por vídeo de membros **heterogêneos**; cada membro recarrega o próprio modelo e pode ler o próprio Parquet. Meta-router CA⊕GNN continua disponível como script (`scripts/meta_router_ca_gnn*.py`) |

## 2. Ponto de partida

- As branches do Rodrigo saíram de `5c02cf7` (17/06), **antes** do pipeline da main existir; ele
  copiou o pipeline do Luiz de ~29/06 (`bbae582`) e evoluiu por cima. Um `git merge` direto gera
  **31 conflitos**, a maioria *add/add* (os dois lados criaram `main.py`, `src/conf/schema.py`,
  `configs/*` de forma independente). Por isso a integração é um **port** sobre a main, não um merge.
- 38 arquivos são idênticos nos dois lados (`src/base/*`, `src/data/{indexing,windowing,schema,audio_io}.py`,
  `src/features/{tabular,hesitation,text_features}.py`, `src/models/{cross_attention,random_forest}.py`, …).
- Nos arquivos centrais que os dois mexeram, a versão do Rodrigo **perdeu** evoluções posteriores da main:
  `train_splits`/`calib_split`, `calib_parquet_path`, `aggregation.recalibrate`, `EnsembleTrainer`
  (`ensemble=[...]`), monitor `val_ap`, formato oficial de submissão (`submission_reference`,
  `submission_probabilities`), hesitação embutida no librosa, `trust_remote_code`, `fuse_text_features`.
  E acrescentou generalizações que valem trazer (lista na §4).
- Dependência externa `gnn-modalblocks`: na branch dele é um editable local
  (`../project_lib/gnn-modalblocks`), inexistente fora da máquina dele. O pacote é público em
  `github.com/watasabi/gnn-modalblocks` (MIT, commit `f42fd44`) e contém todos os módulos usados
  (`ENCODERS`, `MultimodalBlock`, `contrastive.registry`, `architectures.hetero_utils`) →
  vira dependência git **fixada no commit**, num grupo opcional.

## 3. Princípios

1. **A main é a base.** Nenhum default muda: sem override, tudo roda exatamente como hoje
   (RF por padrão; `+experiment=cross_attention`; `ensemble=[dirs]`; calibração `base_rate`/`smooth`,
   monitor `val_ap`, formato oficial de submissão).
2. **Tudo o que é do Rodrigo entra opt-in**: modelos por registro *lazy* no `registry`
   (família `lightning`, ou `sklearn` p/ CatBoost), presets em `configs/experiment/`, dependências
   pesadas em grupos opcionais (`gnn`, `vision`) importadas *lazy*. O caminho RF continua sem torch.
3. **Um Parquet, várias visões.** Os modelos dele consomem o mesmo contrato de dados da main
   (`audio_emb`, `text_emb`, `tabular` por janela → `VideoSequenceDataset`); vídeo entra como coluna
   opcional `face_landmarks` (478×3) → `face_seq` no batch.
4. **Arquivo central com mudança dos dois lados = versão da main + generalizações dele como opção.**
5. **Checkpoint autodescritivo.** `trainer_state.json` passa a gravar `model_name`, `model_cfg`
   (arquitetura) e `parquet_path` → qualquer run dir pode virar membro de ensemble sem config extra
   (runs antigos caem no fallback: nome = pasta-pai do run, arquitetura inferida do `state_dict`).

## 4. Mapa de decisões por arquivo

### 4.1 Núcleo (fusão manual — base main)

| Arquivo | Decisão |
|---------|---------|
| `main.py` | main + modos `featurize_face`, `pretrain_gae`, `hard_mining`; `WeightedRandomSampler` (`data.hard_examples`); ensemble heterogêneo no `_resolve_trainer`. Os modos `ensemble_evaluate`/`ensemble_submit` dele **não** entram: o `ensemble=[...]` da main passa a aceitar membros de modelos diferentes (mesma função, uma interface só) |
| `src/conf/schema.py` | main + `data.hard_examples`, `data.featurize_chunk_size`, `data.force_face`, `trainer.accumulate_grad_batches`/`swa*`, `aggregation.score_calibration`, grupo `face_embedder`, membros de ensemble (str \| dict) |
| `src/data/datasets.py` | main + `face_seq` opcional (dataset e `collate_sequences`) |
| `src/training/lightning_trainer.py` | main (val_ap, `recalibrate_on_val`, `predict_scores`) + dele: `accumulate_grad_batches`, SWA, `build_lightning_module(trainer_cfg)`, `pos_weight`, init GAE, fine-tune de pesos (`checkpoint=` no `mode=train`), snapshot/restauração de arquitetura (`checkpoint_compat`) |
| `src/training/ensemble.py` | generalizado: membro = run dir **ou** `{checkpoint, model, weight, parquet_path}`; famílias misturadas (lightning + sklearn); `combine=mean\|weighted` |
| `src/training/sklearn_trainer.py` | main (guarda contra rótulo -1) + temperature scaling (`aggregation.score_calibration=temperature`) e `target_pos_rate` |
| `src/training/aggregation.py` | main + `calibrate_threshold_from_video_scores` (scores pós-calibração) |
| `src/training/metrics.py` | main + accuracy, recall por classe, ROC-AUC, classification report |
| `src/outputs/checkpoint.py` | main + filtro `model_name` no `resolve_latest_checkpoint` |
| `src/outputs/reporter.py` | main + curva ROC, **`predictions.csv`** por split (insumo do meta-router e do protocolo OOF), análise de erro por `question_type` |
| `src/features/builder.py`, `src/pipeline/featurize.py` | main + escrita do Parquet em lotes (`data.featurize_chunk_size`) — mesmo resultado, menos RAM/VRAM |
| `src/models/registry.py`, `__init__.py`, `src/training/factory.py` | main + registro lazy de `hetero_gnn`, `gnn_baseline`, `hetero_gnn_contrastive`, `multimodal_hetero_full`, `multimodal_hetero_face`, `face_gnn_ts`, `text_finetune` + `catboost` (sklearn) |
| `pyproject.toml` / `uv.lock` | main + grupos `gnn` (`torch-geometric`, `gnn-modalblocks` git@f42fd44) e `vision` (`mediapipe`, `opencv-python-headless`); `catboost` no core (import lazy); `optuna` no `dev` |
| `Makefile`, `README.md` | main + seção/targets novos (`setup-gnn`, `setup-vision`, `featurize-face`, `train-gnn`, `ensemble-multimodal`, …) |

### 4.2 Módulos do Rodrigo portados (sem conflito — entram como estão, só ajustes de caminho/import)

- Modelos: `src/models/{hetero_gnn,hetero_gnn_contrastive,gnn_baseline,hetero_gat_edge,hetero_gae_pretrain,multimodal_hetero_full,multimodal_hetero_face,face_gcn_ts,face_gnn_ts_model,text_finetune,tab_fusion,lightning_seq,lightning_utils,checkpoint_compat,catboost_model}.py`
- Dados/vídeo: `src/data/{graph_builder,face_graph,face_roi}.py`, `src/features/{face_mesh,asr_timing}.py`, `src/pipeline/featurize_face.py` (mesmo índice de janelas da main, `interim/windows_index.parquet`; escrita do Parquet passou a ser atômica)
- Treino/avaliação: `src/training/{gae_pretrain_trainer,score_calibration}.py`, `src/eval/protocol.py` (OOF agrupado por participante)
- Configs: `configs/model/*` dele, `configs/face_embedder/mediapipe.yaml`, presets de `configs/experiment/*` de GNN/face/texto/CatBoost (com `parquet_path` apontando para os caches da main, ex. `text_audio_windows_w2v.parquet` p/ os presets wav2vec2)
- Scripts (`scripts/`): meta-routers (`meta_router_ca_gnn.py` na versão parametrizável `--ca-runs/--gnn-run`, `_face`, `_face_gate`, `_svm`), `ensemble_ap.py`, `gated_ensemble.py`, `threshold_sweep.py`, `search_face_fusion.py`, `diagnose_face_vs_router.py`, `mine_hard_from_preds.py`, `optuna_gnn_tune.py`, `gnn_seed_ensemble.py` (adaptado p/ `mode=evaluate ensemble=[...]`)
- Testes: `tests/{test_asr_timing,test_eval_protocol,test_face_graph,test_text_finetune}.py`
- Documentação: `references/{gnn_training_procedure,meta_router_ca_gnn,improvement_plan,visual_signal_lessons_from_top_teams}.md` + figuras

### 4.3 Não entram (e por quê)

| Item | Motivo |
|------|--------|
| `configs/ensemble/*` (grupo Hydra `ensemble`) | colide com a chave `ensemble=[...]` da main (Hydra trataria `ensemble=` como seleção de grupo). Os presets viram `configs/experiment/ensemble_*.yaml` |
| modos `ensemble_evaluate`/`ensemble_submit` | substituídos pelo `ensemble=[...]` heterogêneo (a main já tem calibração, recalibração e formato oficial nesse caminho) |
| presets `cross_attention_luiz*`, `featurize_luiz*`, `luiz_*`, `rf_luiz`, `scripts/luiz_cross_attention_repro.py` | duplicam o que a main já é por padrão (`make reproduce-best`) |
| `scripts/{ensemble_sweep,ensemble_weight_sweep_live,export_and_sweep}.py` | dependem dos helpers `_ensemble_*` do `main.py` dele |
| `src/{config,modeling,services}/`, `src/main.py` | stubs vazios/legados |
| registro do `text_finetune` + preset `text_goemotions_finetune` | o modelo espera `input_ids`/`attention_mask` e nenhum dataset os produz (nem na branch dele). Módulo + teste portados; registro fica para quando existir a visão de dados de transcrição tokenizada |
| `asr_timing` no tabular | as 16 features existem (+ teste), mas não estão ligadas ao `TabularFeaturizer` (nem na branch dele) |
| `.specs/`, `CHANGELOG.md` | artefatos de processo da branch dele (o conteúdo relevante vai para este plano e para os `references/`) |

## 5. Fases e critérios de aceite (status em 30/09)

- [x] **F0** Branch + este plano.
- [x] **F1** Dependências: grupos `gnn`/`vision`, `uv lock` (único efeito colateral: protobuf 7.35 → 6.33, exigência do mediapipe/mlflow), ambiente sobe com `uv sync --group neural --group gnn --group vision --group dev`.
- [x] **F2** Núcleo fundido (§4.1). **Gate ✅:** `ensemble_5` do artigo (manifest `20260713_211*`) **bit a bit idêntico** à main — test F1 0.722590 / AP 0.874997 / mesma matriz de confusão / limiar 0.500; RF idêntico (F1 0.6753 / AP 0.8374, mesmo checkpoint avaliado pelo código da main e da branch).
- [x] **F3** Módulos portados (§4.2) + registry. **Gate ✅:** 42 testes verdes (4 dele + 2 novos: caminho facial e ensemble); caminho RF não importa torch/lightning.
- [x] **F4** Frente B. **Gate ✅:** checkpoint GNN `20260713_162753` avaliado pelo pipeline da main sobre `text_audio_windows_w2v.parquet`: test F1 **0.7109** (= reportado por ele), AP 0.8529 (ele: 0.8537); mediana |Δp| ≈ 0, 94% dos vídeos com |Δp| < 1e-3, Spearman 0.999 — o resíduo é diferença do cache wav2vec2 (featurizado em 10/07 nesta máquina). Smoke-train de 1 época OK: `hetero_gnn_contrastive`, `gnn_baseline`, `multimodal_hetero_full`, `hetero_gnn_v2_tune` (CA-edges), `hetero_gnn_v2_tuned` (3 camadas), `face_gnn_ts`, `multimodal_hetero_face`, `pretrain_gae`, GNN com `gae_init`, `hard_mining` → treino com `WeightedRandomSampler`, CatBoost train/evaluate/submit.
- [x] **F5** Frente C. **Gate ✅:** `ensemble=[5×CA, GNN]` (cada membro no seu cache) roda em evaluate/submit e gera relatório + `predictions.csv`; meta-router roda sobre os `predictions.csv` gerados pela main. Números na §7.
- [~] **F6** Vídeo ponta a ponta. Coberto por teste com extrator falso (join, zeros, escrita atômica, cache) + smoke com coluna facial sintética. **Falta:** rodar o `featurize_face` real — exige extrair `data/raw/data.zip` (mp4) e baixar o `face_landmarker.task` do MediaPipe (1ª execução).
- [x] **F7** Makefile + README + este plano.

### Achados da integração (corrigidos na branch)

1. **`gnn-modalblocks` público está atrás da cópia local do Rodrigo.** A versão publicada (`f42fd44`) tem `HeteroGAT` de 2 camadas fixas (`conv1`/`conv2`) sem `num_layers`; os checkpoints dele usam `convs.<i>` (versão local nunca publicada). `build_hetero_gat` (em `hetero_gat_edge.py`) usa a do pacote se aceitar `num_layers`, senão `HeteroGATEdgeAttr` sem `edge_attr` — mesma rede, mesmas chaves. **Pedir ao Rodrigo para publicar a versão local** e então subir o `rev` no `pyproject`.
2. **Membro de ensemble de preset perdia a arquitetura.** O `model_cfg` dele não guarda sub-blocos (`face`, `loss`…): um `face_gnn_ts_roi` recarregado pelo YAML base dava *size mismatch*. Agora o `trainer_state.json` grava o bloco `model` inteiro (`model_config`) e o `load_trainer` recria o modelo por ele; runs antigos: membro `{checkpoint: ..., experiment: <preset>}`.
3. **`num_workers>0` quebrava no macOS** (o dataset guardava o módulo `torch` num atributo; `spawn` não serializa). Corrigido no `VideoSequenceDataset`.
4. **Lotes do `featurize` dele podiam partir um vídeo** entre lotes e mudar as features de nível de resposta (texto A1/A3). Aqui os lotes só quebram na fronteira entre vídeos e são opt-in (`data.featurize_chunk_size`).
5. **Presets que incluíam outro experiment** (`face_gnn_ts_{chain,focal}` com `- /experiment: face_gnn_ts`) não compunham no Hydra; base expandida.

## 6. Ressalvas que valem para a decisão de modelagem

1. **O membro que levou o meta-router a 0.7454 não usa vídeo.** É o HeteroGAT
   `hetero_gnn_contrastive` (`20260713_162753`) sobre **áudio wav2vec2 + texto + tabular**. Os modelos
   com vídeo (`face_gnn_ts`, `multimodal_hetero_face`, ROI facial) existem, mas no doc dele a fusão/gate
   facial no router trouxe "ganho irrisório" e não entrou na produção. A frente C com vídeo é, portanto,
   um experimento a fazer — a integração deixa ele executável, não garante o ganho.
2. **Regime de limiar.** O +1.4 do router no test-525 coincide com prever ~58% de positivos
   (test-525 tem 60.6%; o externo tem ~47%). Toda comparação da frente C deve ser feita com limiar
   calibrado fora do split avaliado e, de preferência, com o protocolo OOF (`src/eval/protocol.py`).
3. **Calibração val-only.** Os modelos dele calibram na val (124 vídeos). Com a main, os mesmos
   modelos passam a aceitar `data.train_splits`/`data.calib_split` e `aggregation.recalibrate`.
4. **Vídeo exige os mp4.** `mode=featurize_face` lê `data/raw/data/Videos/*.mp4` e baixa o modelo
   `face_landmarker.task` do MediaPipe na 1ª execução.

## 7. Resultados medidos na Frente C (test-525, limiar calibrado na val)

| Sistema | Macro-F1 | AP | Positivos preditos |
|---------|---------:|---:|-------------------:|
| CA × 5 seeds (artigo, `ensemble_5`) | 0.7226 | 0.8750 | 55.2% |
| GNN `20260713_162753` sozinho (cache w2v) | 0.7109 | 0.8529 | — |
| **CA × 5 + GNN** (média simples, `make ensemble-multimodal`) | **0.7337** | 0.8741 | 59.8% |
| CA × 5 + GNN com o **mesmo nº de positivos** do CA × 5 (k=290) | 0.7226 | — | 55.2% |
| Meta-router CA⊕GNN (`make meta-router`, seeds CA do artigo) | 0.7041 (pesos otimizados) / 0.7213 (`--equal-weights`) | — | 0 trocas na val |

Leitura: o +1.1 do ensemble com o GNN vem **inteiro do limiar** (prevê 60% de positivos; o test-525
tem 60.6%). No mesmo número de positivos o F1 é idêntico ao do CA × 5, e o AP (ranking) não melhora.
O meta-router, com os seeds do artigo, não encontra discordâncias que valha trocar. Isso confirma a
ressalva 6.2: no corte da submissão externa (~47% positivos) esse ganho não deve aparecer. O membro
facial ainda não foi treinado com landmarks reais (F6).

## 8. Como usar

```bash
make setup-vision                      # neural + gnn + vision + dev

# A — cross-attention (artigo), inalterado
make reproduce-best

# B — modelos do Rodrigo sobre o pré-processamento da main
make featurize-w2v                     # cache wav2vec2 (se ainda não existir)
make train-gnn                         # GNN_EXPERIMENT=hetero_gnn_v2_tune_wav2vec2
make featurize-face                    # landmarks no cache (precisa dos .mp4)
make train-face                        # FACE_EXPERIMENT=face_gnn_ts_roi

# C — ensemble áudio + texto (+ vídeo)
make ensemble-multimodal GNN_RUN=outputs/hetero_gnn_contrastive/<run> FACE_RUN=outputs/face_gnn_ts/<run>
make eval-ensemble-members && make meta-router     # roteador CA⊕GNN sobre os predictions.csv
```

Membros de ensemble aceitam `{checkpoint, model, experiment, weight, parquet_path, calib_parquet_path}`
(`ensemble_weights=[...]` para pesos alinhados à lista).
