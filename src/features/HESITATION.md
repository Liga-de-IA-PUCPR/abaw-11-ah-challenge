# Features de incerteza/hesitância (`hesitation.py`) — referência técnica

Documenta como cada feature do `HesitationExtractor` é calculada, quais hiperparâmetros
a moldam e a base na literatura. São escalares por janela; o vetor tem **dimensão fixa 18**.

## Escopo e princípio

O alvo do BAH é um construto **psicológico/comportamental** — **hesitância/incerteza**
(e ambivalência) — **não disfluência clínica**. Por isso este bloco NÃO detecta o
*evento* filled pause ("uh/um"); ele mede os **marcadores prosódicos do ESTADO de
(in)certeza** que a literatura de percepção de confiança identifica:

- fala **duvidosa/insegura**: mais **latência** antes de responder, mais **pausas**,
  **entonação terminal ascendente** (soa como pergunta), **volume mais baixo e instável**,
  **F0 mais variável** e faixa de pitch maior;
- fala **confiante**: contorno **descendente**, **mais alta e estável**, F0 mais baixo e
  menos variável, ritmo mais rápido.

Consequência importante: incerteza tem F0 **mais variável** — o **oposto** do "pitch
plano" de um filled pause. Por isso removemos as features específicas de disfluência
(estabilidade de formantes, platô de pitch, energia de alta frequência, fluxo espectral,
vogal sustentada): elas detectam o evento, não o estado. Ambivalência propriamente dita
é majoritariamente **semântica** (texto) + afeto; este bloco endereça a **hesitância**,
como sinal de suporte ao embedding.

## Entrada, engines e framing

- Entrada: waveform mono `float32` de uma janela, a `sample_rate` Hz (16000 no pipeline).
- Dois motores: **librosa/numpy** (timing/pausas e volume) e **Praat via parselmouth**
  (`use_praat=true`: F0, jitter, shimmer). Se o Praat falhar ou `use_praat=false`, o bloco
  de entonação/voz vira 0.0 (o resto continua).
- Análise por frame: `frame_length` é a janela de análise (amostras) do RMS; `hop_length`
  o passo. No Praat, `time_step = hop_length / sample_rate`.
- Robustez: janela menor que `frame_length`, ou falha de bloco, produz 0.0 (dimensão
  mantida). A ordem das colunas é canônica e determinística.

## Hiperparâmetros

| hiperparâmetro | default | o que controla | features afetadas |
|---|---|---|---|
| `frame_length` | 2048 | janela de análise (amostras) de RMS | pausas, loudness |
| `hop_length` | 512 | passo entre frames; `time_step` do Praat | resolução temporal |
| `silence_rms_threshold` | 0.1 | fração do RMS máx. abaixo da qual o frame é "silêncio" | `hes_pause_ratio` |
| `top_db` | 30.0 | dB abaixo do pico p/ o VAD (`librosa.effects.split`) | latência + todas as pausas via VAD |
| `min_pause_s` | 0.15 | duração mínima (s) p/ um vão contar como pausa | `hes_pause_rate`, `hes_mean_pause_dur` |
| `long_pause_s` | 0.5 | limiar (s) de pausa "longa" | `hes_long_pause_ratio` |
| `f0_min` / `f0_max` | 65 / 500 | piso/teto (Hz) da busca de F0 do Praat | F0, jitter, shimmer |
| `final_frac` | 0.30 | fração final dos frames vozeados p/ a subida terminal | `hes_f0_final_slope` |
| `nuclei_silence_db` | 25.0 | piso de silêncio (dB abaixo do pico) p/ picos de núcleo | taxa de fala/articulação |
| `nuclei_min_dip_db` | 2.0 | vale mínimo (dB) entre núcleos (de Jong & Wempe) | taxa de fala/articulação |
| `use_praat` | true | liga o bloco Praat (F0/jitter/shimmer) | entonação + qualidade de voz |

Constantes fixas do Praat p/ jitter/shimmer (defaults canônicos, não expostas):
`period_floor=0.0001 s`, `period_ceil=0.02 s`, `max_period_factor=1.3`, `max_amp_factor=1.6`.

---

## Bloco 1 — Timing / pausas (librosa/numpy)

Base: `intervals = librosa.effects.split(wav, top_db, frame_length, hop_length)` devolve
os trechos **não-silenciosos** (amostras); os **vãos internos** (`gaps`) são os silêncios
entre trechos, em segundos. `total_dur = n/sr`. `top_db` maior ⇒ VAD mais permissivo ⇒
menos pausas; menor ⇒ mais sensível.

### `hes_onset_latency`
Silêncio inicial antes do 1º trecho de fala (s): `intervals[0,0]/sr` (ou `total_dur` se
tudo silêncio). **Latência de resposta** é um dos marcadores mais fortes de incerteza.
- Hiperparâmetros: `top_db`.
- Direção: **maior ⇒ mais incerteza** (atraso antes de responder).
- Fonte: Brennan & Williams (1995); Smith & Clark (1993).
- Ressalva: como a janela é deslizante, é o silêncio inicial **da janela** — proxy do
  atraso de resposta real (medido do fim da pergunta), não idêntico a ele.

### `hes_pause_ratio`
Fração de frames com RMS abaixo do limiar de silêncio.
```
rms = librosa.feature.rms(wav, frame_length, hop_length)
hes_pause_ratio = mean(rms < silence_rms_threshold * max(rms))
```
- Hiperparâmetros: `silence_rms_threshold`. Não usa `top_db` (calcula direto do RMS).
- Direção: maior ⇒ mais silêncio ⇒ mais hesitação.
- Fonte: Krahmer & Swerts (pausas como marcador de incerteza).

### `hes_pause_rate`
Pausas por segundo: `len([g for g in gaps if g >= min_pause_s]) / total_dur`.
- Hiperparâmetros: `top_db`, `min_pause_s`.
- Direção: maior ⇒ fala mais fragmentada ⇒ incerteza.
- Fonte: Krahmer & Swerts.

### `hes_mean_pause_dur`
Duração média das pausas contadas (s); 0 se nenhuma.
- Hiperparâmetros: `top_db`, `min_pause_s`.
- Direção: maior ⇒ pausas mais longas.
- Fonte: Krahmer & Swerts.

### `hes_long_pause_ratio`
Fração da duração em pausas longas: `sum(g for g in gaps if g > long_pause_s) / total_dur`.
- Hiperparâmetros: `top_db`, `long_pause_s`.
- Direção: maior ⇒ mais tempo em silêncios longos ⇒ incerteza.
- Fonte: Krahmer & Swerts.

### `hes_voiced_fraction`
Fração de fala (não-silêncio): `1 - hes_pause_ratio`. Proxy de continuidade/ritmo (fala
lenta/entrecortada ⇒ menos confiança).
- Fonte: Jiang & Pell (ritmo mais lento na dúvida).

---

## Bloco 2 — Taxa de fala / articulação (librosa/scipy)

Detecção de **núcleos silábicos** pelo método de Jong & Wempe (2009): núcleos são picos
de intensidade (dB) acima de um piso de silêncio E separados por um vale (`dip`), com gate
de fala (dentro de trecho não-silencioso do VAD). Eixo de fluência **ortogonal às pausas**
— mede a densidade da fala, não o silêncio. Fala mais lenta ⇒ menos confiança (Jiang &
Pell 2017). Substitui, aqui, o `tempo` musical do librosa (que é BPM, não taxa de fala).

Base: `db = 20*log10(rms)`; `peaks = find_peaks(db, height=max(db)-nuclei_silence_db,
prominence=nuclei_min_dip_db)`; mantém só picos dentro dos intervalos não-silenciosos.
`n = nº de núcleos`; `phon_time` = soma dos trechos não-silenciosos (s).

### `hes_speech_rate`
`n / total_dur` (sílabas/s, inclui pausas).
- Hiperparâmetros: `nuclei_silence_db`, `nuclei_min_dip_db`, `top_db`.
- Direção: **menor ⇒ mais incerteza** (fala mais lenta).
- Fonte: de Jong & Wempe (2009); Jiang & Pell (2017).

### `hes_articulation_rate`
`n / phon_time` (sílabas/s do tempo de fala, exclui pausas).
- Direção: menor ⇒ mais incerteza. Ortogonal às pausas (mede densidade da fala em si).
- Fonte: de Jong & Wempe (2009); Cucchiarini et al. (2000).

### `hes_phonation_ratio`
`phon_time / total_dur` (fração do tempo efetivamente falando).
- Direção: menor ⇒ mais pausa/hesitação.
- Fonte: de Jong & Wempe (2009).

---

## Bloco 3 — Entonação (Praat via parselmouth)

Base: `pitch = snd.to_pitch(time_step, f0_min, f0_max)`; `f0v` = F0 dos frames vozeados;
contorno em semitons relativo à mediana. `to_pitch` do Praat usa autocorrelação.

### `hes_f0_cv`
Coeficiente de variação de F0: `std(f0v) / mean(f0v)`.
- Direção: **maior ⇒ mais incerteza** (F0 mais variável na dúvida; confiança é mais
  estável). Nota: é o **oposto** do platô de pitch de um filled pause.
- Fonte: Jiang & Pell (2017).

### `hes_f0_range_st`
Faixa de pitch em semitons: `12 * log2(max(f0v) / min(f0v))`.
- Direção: maior ⇒ dúvida.
- Fonte: Jiang & Pell (2017).

### `hes_f0_slope`
Inclinação global do contorno (semitons/s), regressão de `st` vs tempo sobre os frames
vozeados.
- Direção: **ascendente (positivo) ⇒ incerteza**; descendente ⇒ confiança.
- Fonte: Jiang & Pell (2017).

### `hes_f0_final_slope`
Inclinação **terminal**: mesma regressão, mas só na **fração final** (`final_frac`) dos
frames vozeados. Captura a subida terminal (soa como pergunta).
```
n_tail = round(final_frac * n_frames_vozeados)
hes_f0_final_slope = slope( st[-n_tail:] vs t[-n_tail:] )
```
- Hiperparâmetros: `final_frac`.
- Direção: **subida terminal positiva ⇒ incerteza**.
- Fonte: Brennan & Williams (1995); Krahmer & Swerts; Jiang & Pell (2017).

---

## Bloco 4 — Volume / energia (librosa/numpy)

Base: `rms = librosa.feature.rms(wav, frame_length, hop_length)`.

### `hes_loudness_mean`
Nível médio de energia: `mean(rms)`.
- Direção: **menor ⇒ menos confiante** (dúvida é mais silenciosa).
- Fonte: Jiang & Pell (2017).
- Ressalva: depende do ganho de gravação; é um valor **relativo** entre janelas.

### `hes_loudness_cv`
Instabilidade de volume: `std(rms) / mean(rms)` (invariante a ganho).
- Direção: **maior ⇒ mais hesitação** (volume que sobe e desce de forma instável).
- Fonte: prosódia de incerteza (volume instável como pista de hesitação/autodúvida).

---

## Bloco 5 — Qualidade de voz (Praat via parselmouth)

Pistas **afetivas** (tensão/nervosismo), do núcleo eGeMAPS. Complementam a hesitância
com o componente emocional; não são o marcador central de incerteza.

### `hes_jitter_local`
Jitter local: perturbação média relativa do **período** glotal, ciclo a ciclo, sobre um
PointProcess periódico.
```
pp = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
hes_jitter_local = call(pp, "Get jitter (local)", 0,0, 1e-4, 0.02, 1.3)   # NaN -> 0.0
```
- Hiperparâmetros: `f0_min`, `f0_max`.
- Fonte: eGeMAPS (Eyben et al. 2016); Praat.

### `hes_shimmer_local`
Shimmer local: perturbação média relativa da **amplitude**, ciclo a ciclo.
- Hiperparâmetros: `f0_min`, `f0_max`.
- Fonte: eGeMAPS; Praat.

---

## Bloco 6 — Score composto

### `hes_uncertainty_score`
Score heurístico em `[0, 1]` combinando os marcadores mais fortes de incerteza (cada
termo em `[0, 1]`, "maior ⇒ mais incerteza"). Termos de F0 são *gated* pelo vozeamento.
```
gate      = 1 se hes_voiced_fraction > 0.1, senão 0
s_latency = clip(hes_onset_latency, 0, 1)                 # segundos, satura em 1 s
s_pause   = clip(hes_long_pause_ratio, 0, 1)
s_rise    = gate * clip(hes_f0_final_slope / 4.0, 0, 1)   # subida terminal (st/s)
s_var     = gate * clip(hes_f0_cv / 0.3, 0, 1)            # variabilidade de pitch
s_vol     = clip(hes_loudness_cv, 0, 1)                    # volume instável
s_slow    = clip(1 - hes_articulation_rate / 6.0, 0, 1)   # articulação lenta

hes_uncertainty_score = clip( 0.20*s_latency + 0.20*s_pause + 0.15*s_rise
                              + 0.15*s_var + 0.15*s_vol + 0.15*s_slow , 0, 1)
```
- Aviso: heurística **não supervisionada**, apenas para inspeção. Pesos fixos/arbitrários.
  Não use como rótulo; deixe o modelo aprender a combinação a partir das features cruas.

---

## Dimensão e ordem

`dim = 18` (fixo). A ordem é definida por `_build_feature_names` e não depende de dados,
só da config. O cálculo devolve `{nome: valor}` e o vetor é montado por lookup nessa lista,
então computar em qualquer ordem nunca desalinha nome/valor.

Removidas nesta revisão (eram de disfluência, não de incerteza): `hes_formant{i}_*`,
`hes_hf_energy_ratio`, `hes_spectral_flux_mean`, `hes_f0_plateau_ratio`,
`hes_voiced_run_max`, `hes_f0_mean` (confundido pelo falante), `hes_intensity_slope`.

## Integração e calibração

- Onde entra: `data.tabular.use_hesitation=true` anexa o vetor à coluna `tabular`
  (acompanha qualquer `audio_embedder`). Alternativamente, o token `"hesitation"` no
  `feature_set` do backend `librosa` embute no `audio_emb` (early fusion; passa pela
  atenção). Ligar os dois duplica as features.
- Calibração dos limiares: `notebooks/calibrate_hesitation.ipynb` varre cada
  hiperparâmetro contra o rótulo de janela (AUC). Calibre no split de validação e confirme
  no macro-F1 por vídeo.

## Referências

1. Jiang, X. & Pell, M. D. (2017). "The sound of confidence and doubt." Speech
   Communication, 88, 106–126. Confiança: contorno descendente, mais alto/estável, F0
   menor e menos variável, ritmo mais rápido; dúvida: o oposto.
2. Brennan, S. E. & Williams, M. (1995). "The feeling of another's knowing: Prosody and
   filled pauses as cues to listeners about the metacognitive states of speakers."
   Journal of Memory and Language, 34, 383–398. Latência e entonação ascendente como
   pistas de (in)certeza.
3. Smith, V. L. & Clark, H. H. (1993). "On the course of answering questions." Journal of
   Memory and Language, 32, 25–38. Latência de resposta e entonação como marcadores do
   "feeling of knowing".
4. Krahmer, E. & Swerts, M. "How children and adults produce and perceive uncertainty in
   audiovisual speech" e "Audiovisual prosody of uncertainty: an overview." Incerteza
   marcada por pausas, entonação ascendente e fillers.
5. "Recognizing Uncertainty in Speech" (arXiv:1103.1898). Features prosódicas de incerteza
   em linguagem falada.
6. Eyben, F. et al. (2016). "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)."
   IEEE Transactions on Affective Computing, 7(2). Base de jitter/shimmer/F0.
7. Jadoul, Y., Thompson, B. & de Boer, B. (2018). "Introducing Parselmouth: A Python
   interface to Praat." Journal of Phonetics, 71. Interface do bloco Praat.
8. McFee, B. et al. (2015). "librosa: Audio and Music Signal Analysis in Python." SciPy.
   `effects.split` (VAD), `feature.rms`.
9. de Jong, N. H. & Wempe, T. (2009). "Praat script to detect syllable nuclei and measure
   speech rate automatically." Behavior Research Methods, 41(2), 385–390. Método dos
   núcleos silábicos (pico de intensidade + vale + gate de vozeamento).
10. Cucchiarini, C., Strik, H. & Boves, L. (2000). "Quantitative assessment of second
    language learners' fluency by means of automatic speech recognition technology."
    JASA, 107(2). Taxa de articulação como medida de fluência (r=0.81–0.93).
