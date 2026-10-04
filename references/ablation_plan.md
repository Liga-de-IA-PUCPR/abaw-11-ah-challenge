# Plano de ablações — modelos com vídeo do Rodrigo × pré-processamento do Luiz

Branch: `ablations-rodrigo-luiz` (a partir da `develop`, depois do merge do PR #3).
Pergunta: **os modelos novos do Rodrigo (com vídeo) melhoram quando recebem o pré-processamento
e as features de suporte do Luiz — e, em ensemble com a CA do artigo, isso transfere?**

## 1. O que a auditoria mostrou (antes de rodar qualquer coisa)

| Fato | Evidência | Consequência |
|------|-----------|--------------|
| Os modelos do Rodrigo **recebem** as 74 features de suporte (7 qtype + 18 hesitação + 49 texto) | `trainer_state.json` do GNN `20260713_162753` tem `dim_tab=74`; o cache wav2vec2 e o librosa têm as **mesmas 15 622 janelas e o mesmo `tabular`, bit a bit** | não falta feature; falta o **jeito** de usá-las |
| …mas só como **média crua** das janelas: nó `video` do grafo (`multimodal_hetero_face`) ou `Linear` (`face_gnn_ts`) | `graph_builder.build_video_hetero_graph`, `face_gnn_ts_model._tab_mean_pool` | sem BatchNorm (o tab tem `max\|x\|=532`, desvio por feature de 4e-8 a 22.6) e sem localização temporal — exatamente o que a CA corrigiu com BN + `tab_fusion=token` + MIL |
| O áudio dele é **wav2vec2-base**, não a prosódia librosa (que no artigo venceu o w2v: 0.7245 vs 0.7115) | presets `*_v2`, `face_gnn_ts_roi` | o "pré-processamento do Luiz" no lado do áudio = cache librosa |
| O librosa está em **escala crua** (centroide/rolloff espectral até ~8 000, desvio ~1 000) | inspeção do `text_audio_windows.parquet` | a CA absorve com `Linear + LayerNorm`; no GNN o áudio cru vira feature de nó do GAT — foi isso que derrubou o HeteroGAT librosa do Rodrigo a 0.495 sem projeção. Daí `model.audio_norm=true` nas células librosa |
| O modelo facial **nunca foi treinado com landmarks reais nesta máquina** | F6 do `integration_plan.md`; nenhum cache tem `face_landmarks` | a fase 0 extrai os landmarks |
| Ganho de ensemble/meta-router até aqui foi **efeito de limiar** | CA5+GNN 0.7337 → 0.7226 no mesmo nº de positivos; meta-router 0.7454 com PP≈58% | toda célula é comparada no **mesmo k** da CA × 5 (k=290) e no AP |

"Modelo novo dele" = `face_gnn_ts_roi` (28/09: ROI facial + velocity + GNN4TS, só face + suporte) e
`multimodal_hetero_face` (áudio + texto + suporte + face). O HeteroGAT sem vídeo já foi ablado por
ele no librosa (0.6697) e fica como referência (`gnn_w2v`).

## 2. O que foi implementado para as ablações (tudo opt-in; defaults = comportamento antigo)

| Onde | Opção | Valores |
|------|-------|---------|
| `multimodal_hetero_face` / `_full` | `model.tab_fusion` | `graph` (original: média crua no nó video) · `late` (BN → proj → atenção-MIL → concat no readout) · `token` (BN → proj por janela, fundido a cada token antes de GAT/LatentGCN/BiLSTM — a mesma fusão da CA) · `none` (controle) |
| idem | `model.audio_norm` | `true` = BatchNorm por feature no áudio (só janelas válidas) |
| idem | `model.face.enabled=false` | agora funciona (antes quebrava: o forward recebia `face_seq`) |
| `face_gnn_ts` | `model.tab_fusion` | `mean` (original) · `late` (BN + atenção-MIL) |
| idem | `model.face.enabled=false` | controle "só features de suporte" |
| `mode=featurize_face` | `data.face_from=<parquet>` | copia a coluna de landmarks de outro cache (MediaPipe roda **uma** vez) |
| `scripts/ablation_report.py` | — | tabela + ensembles offline + comandos `main.py` |

Checkpoints antigos carregam idênticos (`graph`/`mean` não criam parâmetros novos; testes em
`tests/test_tab_fusion_ablation.py`).

## 3. Protocolo de medida

- **Treino** no `train` (778), early-stop/limiar na `val` (124), **test-525 medido uma vez** por
  célula — o mesmo protocolo do `ensemble_5` do artigo. Hiperparâmetros = presets do Rodrigo
  (só muda o fator testado em cada célula).
- **3 seeds por célula** (`ABL_SEEDS="42 1 2"`): o GNN do Rodrigo variou de 0.7109 (1 seed) a 0.687
  (ensemble de 7 seeds novos). O sistema da célula = média das probas dos seeds.
- **Colunas do relatório** (`make ablation-report`):
  `AP` (sem limiar) · `AP/seed μ±σ` · `F1@thr_val` (smooth na val) · `%pos` ·
  **`F1@k=290`** (mesmo nº de positivos da CA × 5) · `Δ vs ref [IC95]` (bootstrap pareado no
  mesmo k) · `teto F1` (oráculo no test, não deployável).
- **Regra de leitura:** uma melhora só conta se aparecer em **AP** e em **F1@k**. F1@thr_val que
  sobe junto com `%pos` é deslocamento de limiar (o test-525 tem 60.6% positivos; o externo ~47%).

Validação do relatório com runs existentes (bate com o gate do artigo):

| sistema | AP | F1@thr_val | %pos | F1@k=290 |
|---------|---:|-----------:|-----:|---------:|
| ca5 (artigo) | 0.8750 | 0.7226 | 55.2 | 0.7226 |
| gnn_w2v | 0.8537 | 0.7109 | 66.1 | 0.7070 |
| ca5+gnn_w2v (peso igual por sistema) | 0.8701 | 0.7309 | 62.1 | 0.7187 (Δ −0.004 [−0.027, +0.021]) |

## 4. Fase 0 — dados (uma vez)

```bash
# → data/raw/data/{Videos,transcription,split,...}; pula os 918k recortes faciais (não usados aqui)
unzip -q data/raw/data.zip -d data/raw/ -x "__MACOSX/*" "data/cropped-aligned-faces/*"
make setup-vision                              # neural + gnn + vision (MediaPipe) + dev
make preprocess                                # índice de janelas (data/interim/windows_index.parquet)
make face-caches                               # MediaPipe 1× → *_face.parquet (librosa) e *_w2v_face.parquet
```

`face-caches` copia os caches antes de escrever: `text_audio_windows.parquet` (artigo) e
`text_audio_windows_w2v.parquet` nunca são modificados.

## 5. Fase 1 — modelo novo do Rodrigo isolado (`make ablate CELL=<id>`)

### 1A — `face_gnn_ts_roi` (face + features de suporte)

| Célula | Face | Suporte | Responde |
|--------|:----:|---------|----------|
| A1 | ✓ | — | quanto sinal a face carrega sozinha |
| A2 | ✓ | Rodrigo (média crua → Linear) | o modelo como ele usou |
| **A3** | ✓ | **Luiz (BN + atenção-MIL)** | **A3 − A2: o seu tratamento das features ajuda o modelo dele?** |
| A0 | — | Luiz (BN + atenção-MIL) | A3 − A0: a face soma algo às features de suporte? |

### 1B — `multimodal_hetero_face` (áudio + texto + suporte + face)

| Célula | Áudio | Suporte | Face | Responde |
|--------|-------|---------|:----:|----------|
| B1 | wav2vec2 | graph (média crua) | ✓ | baseline = o modelo como ele usaria |
| B2 | **librosa** + `audio_norm` | graph | ✓ | B2 − B1: o seu áudio |
| **B3** | librosa + `audio_norm` | **token** | ✓ | **B3 − B2: a sua fusão das features. B3 = seu pré-processamento + vídeo dele (o teste pedido)** |
| B4 | librosa + `audio_norm` | token | — | B3 − B4: o que o vídeo acrescenta nesse modelo |
| B5 *(extra)* | wav2vec2 | token | ✓ | fecha o 2×2 áudio × fusão |
| B6 *(extra)* | librosa + `audio_norm` | late | ✓ | late vs token |
| B7 *(extra)* | librosa + `audio_norm` | none | ✓ | as features de suporte importam nesse modelo? |

```bash
make ablate-list                               # células e overrides exatos
make ablate CELL=B3 DEVICE=cuda                # uma célula (3 seeds; retoma de onde parou)
make ablate-core DEVICE=cuda                   # A1 A2 A3 A0 B1 B2 B3 B4  (24 treinos)
make ablate-extra DEVICE=cuda                  # B5 B6 B7
make ablation-report                           # → outputs/ablations/report.md
```

Parâmetros: `ABL_SEEDS="42 1 2"`, `ABL_ROOT=outputs/ablations`, `ABL_WANDB=offline`,
`ARGS="..."` (overrides extras, ex.: `ARGS="data.num_workers=0"` no macOS). Cada seed grava
`outputs/ablations/<célula>/<modelo>/<timestamp>/eval_{train,val,test}/predictions.csv` e entra
em `manifest.txt` só depois das 3 avaliações — rodar de novo pula o que já terminou.

## 6. Fase 2 — ensembles com pré-processamentos diferentes

Cada membro lê o **seu** cache: CA × 5 no librosa do artigo, `gnn_w2v` no wav2vec2, células A/B
nos caches `_face`.

```bash
make ablation-ensembles                        # ca5+gnn_w2v, ca5+<célula>, ca5+gnn_w2v+<célula>
make ablation-report REPORT_ARGS="--combo ca5+A3+B3 --combo ca5:2+B3 --emit"
make ablation-report REPORT_ARGS="--combos --rank"   # média de ranks (escalas diferentes)
```

`--emit` imprime o comando `main.py mode=evaluate ... ensemble=[...] ensemble_weights=[...]` que
reproduz o ensemble pelo caminho canônico (troque por `mode=submit out=... submission_reference=...
submission_probabilities=true` para gerar a submissão). Peso igual **por sistema** (os 5 seeds da
CA dividem o peso dela), não por membro.

Roteadores do Rodrigo sobre os mesmos `predictions.csv`:

```bash
make meta-router GNN_RUN=<run de B3>                    # CA ⊕ multimodal no lugar do GNN
make meta-router-face FACE_RUN=<run de A3>              # CA ⊕ GNN ⊕ face (rescue quando CA == GNN)
```

Os roteadores escolhem limiar na val e não têm versão "mesmo k" — leia o `%pos` deles antes de
comparar com a tabela.

## 7. Critério de decisão

1. **Fase 1:** a célula vira candidata a membro se `AP ≥ 0.85` **ou** se for descorrelacionada da CA
   (o ganho de ensemble vem de errar em vídeos diferentes, não de ser forte sozinha).
2. **Fase 2:** um ensemble só substitui o CA × 5 se `Δ F1@k` > 0 com IC que não cruza zero **e** o AP
   subir. Melhora só em `F1@thr_val` = limiar; não submeter.
3. O test-525 é usado para escolher entre poucas configurações; a confirmação final é a submissão
   externa (≤ 5 trials/semana), cortada em k≈60–64 dos 152 vídeos (o ajuste de limiar do ensemble
   antigo já se esgotou em 0.7456; ganho novo tem de vir do ranking).

## 8. Ressalvas

- `face_gnn_ts_roi` calibra o limiar do run com `base_rate` e o multimodal com `smooth` (presets do
  Rodrigo); o relatório ignora o limiar salvo e recalibra todos com `smooth` na val.
- Monitor de early-stop = `val_f1_macro` (preset dele) em todas as células — constante dentro da
  grade. O artigo usa `val_ap`; testar depois só na melhor célula (`ARGS="trainer.monitor=val_ap"`).
- Face = 2 frames por janela (`face_embedder.max_frames=2`) e landmarks crus; se A1 ficar perto
  do acaso, o próximo passo é a extração do plano dele (recortes + backbone), não tuning da GNN.
  O `data.zip` já traz `data/cropped-aligned-faces/` (918k recortes alinhados) — insumo pronto
  para esse encoder, sem MediaPipe.
- ASR-timing (16 features do Rodrigo) continua fora do tabular — rodada posterior.
