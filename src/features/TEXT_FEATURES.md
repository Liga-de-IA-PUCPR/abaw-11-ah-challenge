# Features de texto (`text_features.py`) — referência técnica

Documenta como cada feature do `TextFeaturizer` é calculada, sua **granularidade** (resposta
vs janela), os hiperparâmetros e a base na literatura. É a feature de SUPORTE ao embedding de
texto para o construto do BAH: **ambivalência (A)** + **hesitância (H)**. Dim fixa (49 com os
defaults). Ablação isolada (RF só nessas features): **macro-F1 de val ≈ 0.64** (baseline
majoritário 0.38; áudio isolado ≈ 0.57).

## Princípio de granularidade (verificado nos dados)
1 vídeo = 1 pergunta = 1 resposta; ~24 palavras/janela vs ~117–151/vídeo; ~9 janelas/vídeo. Logo:
- **Ambivalência é propriedade da RESPOSTA** — por janela (~24 palavras) o léxico é esparso e os
  polos opostos ("amo fumar" / "odeio o que faz") caem em janelas diferentes. Por isso **A1 e a
  flutuação de A3 são computadas sobre o transcript do vídeo inteiro e broadcast** para as janelas.
- **Hesitância e contraste são pistas LOCAIS** — A4, H1 e a entropia de A3 ficam por janela.

`extract(texts, video_ids)` agrupa as janelas por vídeo, reconstrói o transcript (chunks únicos),
computa o bloco de resposta (broadcast) e o de janela, e monta cada vetor por lookup numa lista
canônica (à prova de desalinhamento). H2 (fillers) foi descartada: o Whisper descarta ~96% deles.

## Fontes de P e N (intensidade positiva/negativa) — nunca polaridade líquida
- **Léxico (VADER):** `P=pos`, `N=neg` (proporções unipolares separadas).
- **Emoção (RoBERTa-emotion, opcional `use_emotion`):** softmax da head de classificação →
  `P=Σ prob(positive_labels)`, `N=Σ prob(negative_labels)` (default joy+optimism / anger+sadness).
Separar P e N é o certo: sentimento líquido (P−N) confunde neutro (P,N baixos) com ambivalente
(P,N altos) — ambos dão ~0 (Kaplan 1972).

## Hiperparâmetros (`TextFeaturesConfig`)
| campo | default | efeito |
|---|---|---|
| `use_a1/use_a3/use_a4/use_h1` | true | liga cada grupo |
| `use_emotion` | true | carrega o classificador de emoção (P/N-emoção em A1 + A3) |
| `use_hedge_classifier` | false | H1 contextual (logreg de supervisão distante) |
| `emotion_model` | cardiffnlp/twitter-roberta-base-emotion | classificador de emoção |
| `positive_labels`/`negative_labels` | joy,optimism / anger,sadness | mapeamento de polos |
| `norm` | [per_word, per_100, raw] | normalizações emitidas p/ contagens léxicas |
| `max_length`/`batch_size` | 128 / 32 | tokenização/inferência da emoção |

Léxicos (hedges por subcategoria, contraste, desejo, resistência) são constantes editáveis no
topo de `text_features.py`.

---

## A1 — Ambivalência (RESPOSTA, broadcast)
Índices psicométricos, para cada fonte P/N (VADER `text_a1_vader_*` e emoção `text_a1_emo_*`):
```
P, N, intensity=(P+N)/2, inconsist=|P−N|, kaplan=min(P,N), griffin=(P+N)/2−|P−N|
```
Griffin/Kaplan só ficam altos quando P **e** N são fortes (definição de ambivalência). Mais o
eixo de domínio do professor sobre a resposta: `text_a1_desire`/`text_a1_resistance` (nas 3
normalizações), `text_a1_desire_x_resistance` (produto das taxas), `text_a1_desire_min_resistance`.
- Direção: `griffin`/`kaplan`/`inconsist` altos ⇒ mais ambivalência.
- Fontes: Thompson/Zanna/Griffin (1995); Kaplan (1972); Conner et al. (2021).

## A3 — Emoção: entropia (JANELA) + flutuação/conflito (RESPOSTA)
Requer `use_emotion`. Por janela: `text_a3_emotion_entropy` (mistura da distribuição),
`text_a3_emotion_top2gap` (proximidade das 2 top emoções). Por resposta (broadcast):
`text_a3_emotion_fluct_std`/`_range` (std/amplitude da valência `P−N` entre as janelas do vídeo —
"oscila entre estados"), `text_a3_pole_conflict=min(P_emo,N_emo)`, `text_a3_valence_resp=P_emo−N_emo`.
- Fontes: MEDA (IEEE T-AFFC 2020); HSEmotion (ABAW-8).

## A4 — Contraste + shift de polaridade (JANELA)
`text_a4_contrast` (nas 3 normalizações) conta marcadores (but/however/although/though/yet/on the
other hand/at the same time/...). `text_a4_polarity_shift=|VADER(1ª metade)−VADER(2ª metade)|` e
`text_a4_sign_flip` (1 se o sinal do sentimento vira entre as metades — vira-volta).
- Foi o sinal isolado mais forte (AUC de janela ≈ 0.60). Fonte: stance≠sentimento (Springer 2025).

## H1 — Hedges (JANELA)
Contagem + taxa/palavra + taxa/100 por subcategoria: `text_h1_fp` (1ª pessoa epistêmica: "i think"),
`text_h1_adv` (advérbios: "maybe/probably"), `text_h1_approx` (aproximadores: "sort of/kind of"),
`text_h1_unc` (incerteza explícita: "not sure/i don't know"), `text_h1_agentless` (atribuição
sem-agente: "some people say"), e `text_h1_total`. Opcional `text_h1_hedge_score`: logreg treinada
por **supervisão distante** (janelas com pista de alta precisão → rótulo fraco; features BoW ao
redor das pistas) — contextual, não dispara cego numa palavra-gatilho.
- Fontes: CoNLL-2010 (detecção de hedge/incerteza); Ganter & Strube (2009, weasel words).

---

## Wiring — early e late fusion (espelha o áudio)
- **Late fusion** (`data.tabular.use_text_features=true` + `data.tabular.text_features`): o
  `TabularFeaturizer` possui o `TextFeaturizer`; as colunas entram no `tabular`. O RF as recebe
  pelo concat de `WindowMatrixView`; o cross-attention pelo ramo `tab_seq` (`model.use_tabular`).
  **É o caminho do RandomForest** (sem wrapper extra) e não muda o modelo.
- **Early fusion** (`text_embedder.fuse_text_features=true`): o `FeatureBuilder` concatena o
  `TextFeaturizer` ao `text_emb` por janela → passa pelo `proj_b` → **atenção** do cross-attention.
  `dim_b` é inferido do cache no fit.
- ⚠️ Ligar os dois juntos duplica as features (o builder avisa).

O classificador contextual de H1 (se ligado) é treinado no split de TREINO via `fit()` (TabularFeaturizer
ou builder chamam `fit` nas janelas de treino — sem vazamento).

## Calibração / validação
`notebooks/calibrate_text_features.ipynb`: AUC por feature vs rótulo (janela e vídeo) + ablação RF
(macro-F1 de val ligando/desligando A1/A3/A4/H1, normalizações e fontes P/N). Decide léxico, índices
A1 e normalização a grão fino.

## Referências
1. Thompson, Zanna & Griffin (1995). "Let's not be indifferent about (attitudinal) ambivalence." In *Attitude Strength*.
2. Kaplan (1972). "On the ambivalence-indifference problem in attitude theory and measurement." Psychological Bulletin.
3. Conner et al. (2021). "Cognitive-Affective Inconsistency and Ambivalence." Pers. Soc. Psychol. Bulletin.
4. MEDA (2020). "Multi-Label Emotion Detection via Emotion-Specified Feature Extraction and Emotion Correlation Learning." IEEE T-AFFC.
5. Farkas et al. (2010). "The CoNLL-2010 Shared Task: Learning to Detect Hedges and their Scope." CoNLL.
6. Ganter & Strube (2009). "Finding Hedges by Chasing Weasels." ACL-IJCNLP.
7. Hutto & Gilbert (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." ICWSM. (P/N léxico)
8. Ryumina et al. — HSEmotion Team at ABAW-8 (agregação estatística/flutuação de emoção).
