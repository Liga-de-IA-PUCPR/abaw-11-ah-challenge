# Plano de integração — pipeline do Luiz (main) + modelos do Rodrigo

Branch: `integrate-rodrigo-multimodal` (criada a partir de `main@c0c2755`).
Origem dos módulos do Rodrigo: `origin/improve-macro-f1-beyond-router@bfdf856` (28/09),
que contém toda a `origin/GNN+CA-meta-router@39df581` (13/07) + 7 commits.

## 1. Objetivo

Uma única base (a `main`) em que os modelos convivem e se escolhem por config:

| Frente | Toggle | O que roda |
|--------|--------|------------|
| **A — Cross-attention (Luiz)** | `+experiment=cross_attention` | modelo do artigo, inalterado (`make reproduce-best` continua dando o mesmo número) |
| **B — Modelos do Rodrigo com o pré-processamento do Luiz** | `+experiment=hetero_gnn_v2_tune_wav2vec2`, `face_gnn_ts_roi`, `multimodal_hetero_face_v3`, `text_goemotions_finetune`, `catboost_baseline`, … | GNNs heterogêneos, GNN facial (Face Mesh), fine-tune de texto, CatBoost — todos lendo o **mesmo Parquet** gerado por `mode=featurize` da main (+ coluna `face_landmarks` via `mode=featurize_face`) |
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
- Dados/vídeo: `src/data/{graph_builder,face_graph,face_roi}.py`, `src/features/{face_mesh,asr_timing}.py`, `src/pipeline/featurize_face.py` (ajuste: usa `data.paths.window_index` da main em vez de `interim/windows_index.parquet`)
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
| `.specs/`, `CHANGELOG.md` | artefatos de processo da branch dele (o conteúdo relevante vai para este plano e para os `references/`) |

## 5. Fases e critérios de aceite

- [ ] **F0** Branch + este plano.
- [ ] **F1** Dependências: grupos `gnn`/`vision`, `uv lock`, ambiente sobe (`uv sync --group neural --group gnn --group dev`).
- [ ] **F2** Núcleo fundido (§4.1). **Gate:** `ensemble_5` do artigo reproduz **exatamente** o mesmo F1/AP no test-525 (regressão).
- [ ] **F3** Módulos portados (§4.2) + registry. **Gate:** `pytest` verde; import lazy preservado (caminho RF não importa torch).
- [ ] **F4** Frente B com o pré-processamento da main. **Gate:** o checkpoint GNN `outputs/hetero_gnn_contrastive/20260713_162753` avaliado no Parquet wav2vec2 da main reproduz o reportado por ele (test-525: F1 0.7109 / AP 0.8537 @ thr 0.34); smoke-train de 1 época de cada modelo novo.
- [ ] **F5** Frente C. **Gate:** `ensemble=[5×CA, GNN]` roda em evaluate/submit (membros com Parquets diferentes), gera relatório + `predictions.csv`; meta-router roda sobre os `predictions.csv` gerados pela main.
- [ ] **F6** Vídeo ponta a ponta: `mode=featurize_face` → `face_landmarks` → `face_gnn_ts`/`multimodal_hetero_face`. **Gate (sem os mp4 extraídos):** teste com coluna facial sintética; com `data/raw/data.zip` extraído, rodar o featurize real.
- [ ] **F7** Makefile + README + atualização deste plano com os números medidos.

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
