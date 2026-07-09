"""HesitationExtractor — marcadores vocais de INCERTEZA/HESITÂNCIA por janela.

Motor reutilizável (numpy/librosa + Praat via parselmouth) que produz um bloco de
features hand-crafted alinhadas ao construto do BAH: **hesitância/incerteza** — NÃO
disfluência clínica. É um sinal de SUPORTE ao embedding, para ajudar a distinguir
fala hesitante/insegura de fala confiante.

Base na literatura (prosódia de (in)certeza, não detecção de filled pause)
-------------------------------------------------------------------------
Fala DUVIDOSA/insegura vs CONFIANTE difere de forma sistemática (Jiang & Pell,
*The sound of confidence and doubt*, Speech Communication 2017; Brennan & Williams,
*The feeling of another's knowing*, JML 1995; Krahmer & Swerts, prosódia audiovisual
da incerteza):

- **Latência de resposta** maior antes de falar (Brennan & Williams).
- **Pausas** mais frequentes/longas (Krahmer & Swerts).
- **Entonação terminal ascendente** (soa como pergunta) sinaliza incerteza; contorno
  descendente sinaliza confiança (Jiang & Pell; Brennan & Williams).
- **Volume mais baixo e instável** na dúvida; confiança é mais alta e estável (Jiang & Pell).
- **F0 mais variável** e faixa de pitch maior na dúvida (Jiang & Pell) — o oposto do
  "pitch plano" de um filled pause; por isso este bloco NÃO usa estabilidade de
  formantes/platô de pitch/energia de alta frequência (aquilo detecta o EVENTO
  disfluência, não o ESTADO de incerteza).
- Voz com maior micro-perturbação (jitter/shimmer) como pista afetiva de tensão/nervosismo.

Contrato (igual aos embedders — README §6.3)
--------------------------------------------
    ext = HesitationExtractor(sample_rate=16000)          # ou .from_config(cfg)
    names = ext.feature_names()                            # dim determinística
    X = ext.extract([wav0, wav1, ...])                     # (n, dim) float32

A ordem das colunas é ditada por :meth:`feature_names` (lista canônica derivada só da
config); :meth:`_compute` devolve um dict {nome: valor} e o vetor é montado por lookup
— computar em qualquer ordem NUNCA desalinha nome/valor. Janelas degeneradas
(vazias/curtas) e falhas de sub-bloco caem para 0.0 sem quebrar o pipeline.
"""

from __future__ import annotations

import numpy as np

from src.logger import get_logger

log = get_logger("features.hesitation")

_SR_DEFAULT = 16000


class HesitationExtractor:
    """Extrai o vetor de incerteza/hesitância (timing + entonação + volume + voz) por janela.

    Todos os limiares são configuráveis (bloco ``hesitation`` do YAML). A dimensão é
    fixa e determinística (18 features).

    | Bloco                              | features | motor          |
    |------------------------------------|---------:|----------------|
    | Timing / pausas (latência, pausas) | 6        | librosa/numpy  |
    | Taxa de fala / articulação         | 3        | librosa/scipy  |
    | Entonação (F0, subida terminal)    | 4        | parselmouth    |
    | Volume / energia (loudness)        | 2        | librosa/numpy  |
    | Qualidade de voz (jitter/shimmer)  | 2        | parselmouth    |
    | Score de incerteza (heurístico)    | 1        | derivado       |

    Attributes:
        sample_rate: SR dos waveforms (deve casar com ``data.audio.sample_rate``).
        frame_length / hop_length: janelamento das features de quadro (RMS/F0).
        silence_rms_threshold: fração do RMS máx. p/ marcar frame como silêncio.
        top_db: limiar (dB abaixo do pico) do VAD de pausas (``librosa.effects.split``).
        min_pause_s / long_pause_s: durações mínimas p/ contar pausa / pausa "longa".
        f0_min / f0_max: faixa de F0 (Hz) do Praat.
        final_frac: fração final dos frames vozeados p/ a inclinação terminal de F0.
        use_praat: usa parselmouth (Praat) para F0/jitter/shimmer.
    """

    name: str = "hesitation"

    # Constantes do Praat p/ jitter/shimmer (defaults canônicos do Praat).
    _PP_PERIOD_FLOOR = 0.0001
    _PP_PERIOD_CEIL = 0.02
    _PP_MAX_FACTOR = 1.3
    _SHIMMER_MAX_AMP = 1.6

    def __init__(
        self,
        sample_rate: int = _SR_DEFAULT,
        *,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_rms_threshold: float = 0.1,
        top_db: float = 30.0,
        min_pause_s: float = 0.15,
        long_pause_s: float = 0.5,
        f0_min: float = 65.0,
        f0_max: float = 500.0,
        final_frac: float = 0.30,
        nuclei_silence_db: float = 25.0,
        nuclei_min_dip_db: float = 2.0,
        use_praat: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.top_db = float(top_db)
        self.min_pause_s = float(min_pause_s)
        self.long_pause_s = float(long_pause_s)
        self.f0_min = float(f0_min)
        self.f0_max = float(f0_max)
        self.final_frac = float(final_frac)
        self.nuclei_silence_db = float(nuclei_silence_db)
        self.nuclei_min_dip_db = float(nuclei_min_dip_db)
        self.use_praat = bool(use_praat)

        self._feature_names = self._build_feature_names()
        self._dim = len(self._feature_names)
        self._praat_warned = False

    # =========================================================================
    # Construção a partir da Config Hydra
    # =========================================================================

    @classmethod
    def from_config(cls, cfg=None, *, sample_rate: int = _SR_DEFAULT) -> HesitationExtractor:
        """Instancia a partir de um nó ``hesitation`` (DictConfig ou dict); tolera ``None``."""

        def _get(key, default):
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        return cls(
            sample_rate=int(_get("sample_rate", sample_rate)),
            frame_length=int(_get("frame_length", 2048)),
            hop_length=int(_get("hop_length", 512)),
            silence_rms_threshold=float(_get("silence_rms_threshold", 0.1)),
            top_db=float(_get("top_db", 30.0)),
            min_pause_s=float(_get("min_pause_s", 0.15)),
            long_pause_s=float(_get("long_pause_s", 0.5)),
            f0_min=float(_get("f0_min", 65.0)),
            f0_max=float(_get("f0_max", 500.0)),
            final_frac=float(_get("final_frac", 0.30)),
            nuclei_silence_db=float(_get("nuclei_silence_db", 25.0)),
            nuclei_min_dip_db=float(_get("nuclei_min_dip_db", 2.0)),
            use_praat=bool(_get("use_praat", True)),
        )

    # =========================================================================
    # Contrato
    # =========================================================================

    @property
    def dim(self) -> int:
        """Dimensão fixa do vetor de incerteza/hesitância (18)."""
        return self._dim

    def feature_names(self) -> list[str]:
        """Lista canônica (ordem das colunas de :meth:`extract`)."""
        return list(self._feature_names)

    def extract(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Vetor de incerteza por janela → matriz ``(n, dim)`` float32."""
        if not inputs:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = [self.extract_one(wav) for wav in inputs]
        return np.vstack(rows).astype(np.float32)

    def extract_one(self, wav: np.ndarray) -> np.ndarray:
        """Vetor de uma janela (monta por lookup na lista canônica)."""
        values = self._compute(wav)
        return np.asarray(
            [float(values.get(name, 0.0)) for name in self._feature_names],
            dtype=np.float32,
        )

    # =========================================================================
    # Cálculo (dict nome→valor; blocos isolados por try/except)
    # =========================================================================

    def _compute(self, wav: np.ndarray) -> dict[str, float]:
        """Computa todas as features de uma janela; falhas de bloco caem para 0.0."""
        wav = np.asarray(wav, dtype=np.float32).ravel()
        out: dict[str, float] = {}
        if wav.size < self.frame_length:
            return out  # janela degenerada → tudo 0.0 (dim mantida)

        try:
            out.update(self._timing_features(wav))
        except Exception as exc:  # noqa: BLE001 — robustez do pipeline
            log.debug(f"hesitation: bloco de timing/pausa falhou ({exc})")
        try:
            out.update(self._rate_features(wav))
        except Exception as exc:  # noqa: BLE001
            log.debug(f"hesitation: bloco de taxa de fala falhou ({exc})")
        try:
            out.update(self._loudness_features(wav))
        except Exception as exc:  # noqa: BLE001
            log.debug(f"hesitation: bloco de loudness falhou ({exc})")
        if self.use_praat:
            try:
                out.update(self._praat_features(wav))
            except Exception as exc:  # noqa: BLE001
                if not self._praat_warned:
                    log.warning(
                        f"hesitation: Praat/parselmouth indisponível ({exc}); "
                        f"F0/jitter = 0"
                    )
                    self._praat_warned = True

        out["hes_uncertainty_score"] = self._composite_score(out)
        return out

    # --- Bloco: timing / pausas (latência de resposta + pausas) --------------

    def _timing_features(self, wav: np.ndarray) -> dict[str, float]:
        """Latência de fala, razão de silêncio, estatísticas de pausas e vozeamento.

        Latência de resposta e pausas são marcadores centrais de incerteza
        (Brennan & Williams 1995; Krahmer & Swerts).
        """
        import librosa

        sr = self.sample_rate
        total_dur = max(wav.size / sr, 1e-6)

        rms = librosa.feature.rms(
            y=wav, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        pause_ratio = 0.0
        if rms.size:
            thr = self.silence_rms_threshold * float(rms.max())
            pause_ratio = float(np.mean(rms < thr))

        # Trechos NÃO-silenciosos (amostras) via VAD por energia (top_db abaixo do pico).
        intervals = librosa.effects.split(
            wav, top_db=self.top_db, frame_length=self.frame_length, hop_length=self.hop_length
        )

        # Latência: silêncio inicial antes do 1º trecho de fala (proxy de atraso de resposta).
        onset_latency = total_dur if len(intervals) == 0 else float(intervals[0, 0] / sr)

        pauses: list[float] = []  # gaps INTERNOS (entre trechos de fala), em s
        for k in range(len(intervals) - 1):
            gap = (intervals[k + 1, 0] - intervals[k, 1]) / sr
            if gap > 0:
                pauses.append(float(gap))

        counted = [p for p in pauses if p >= self.min_pause_s]
        long_time = sum(p for p in pauses if p > self.long_pause_s)
        voiced_frac = 1.0 - pause_ratio

        return {
            "hes_onset_latency": onset_latency,
            "hes_pause_ratio": pause_ratio,
            "hes_pause_rate": len(counted) / total_dur,
            "hes_mean_pause_dur": float(np.mean(counted)) if counted else 0.0,
            "hes_long_pause_ratio": float(long_time / total_dur),
            "hes_voiced_fraction": voiced_frac,
        }

    # --- Bloco: taxa de fala / articulação (núcleos silábicos) ---------------

    def _rate_features(self, wav: np.ndarray) -> dict[str, float]:
        """Taxa de fala e de articulação via detecção de núcleos silábicos.

        Método de Jong & Wempe (2009): núcleos = picos de intensidade (dB) acima de um
        piso de silêncio E separados por um vale (``dip``), com gate de fala (dentro de
        trecho não-silencioso do VAD). Eixo de fluência ORTOGONAL às pausas — mede a
        densidade da fala, não o silêncio. Fala mais lenta ⇒ menos confiança (Jiang &
        Pell 2017). Substitui o ``tempo`` musical do librosa (que é BPM, não isto).

        - ``hes_speech_rate`` = núcleos / duração total (síl/s, inclui pausas).
        - ``hes_articulation_rate`` = núcleos / tempo de fonação (síl/s, exclui pausas).
        - ``hes_phonation_ratio`` = tempo de fonação / duração total.
        """
        import librosa
        from scipy.signal import find_peaks

        sr = self.sample_rate
        total_dur = max(wav.size / sr, 1e-6)
        rms = librosa.feature.rms(
            y=wav, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        if rms.size < 3:
            return dict.fromkeys(
                ("hes_speech_rate", "hes_articulation_rate", "hes_phonation_ratio"), 0.0
            )

        db = 20.0 * np.log10(rms + 1e-8)
        floor = float(db.max()) - self.nuclei_silence_db  # piso de silêncio (dB rel. ao pico)
        # Picos = máximos locais acima do piso, separados por um vale >= min_dip (prominência).
        peaks, _ = find_peaks(db, height=floor, prominence=self.nuclei_min_dip_db)

        # Gate de fala: o pico deve cair dentro de um trecho não-silencioso (VAD).
        intervals = librosa.effects.split(
            wav, top_db=self.top_db, frame_length=self.frame_length, hop_length=self.hop_length
        )
        phon_time = float(sum(b - a for a, b in intervals) / sr) if len(intervals) else 0.0
        if len(intervals):
            ps = (peaks * self.hop_length)[:, None]
            starts, ends = intervals[:, 0], intervals[:, 1]
            in_speech = np.any((ps >= starts) & (ps < ends), axis=1)
            n_nuclei = int(in_speech.sum())
        else:
            n_nuclei = 0

        return {
            "hes_speech_rate": n_nuclei / total_dur,
            "hes_articulation_rate": (n_nuclei / phon_time) if phon_time > 1e-3 else 0.0,
            "hes_phonation_ratio": phon_time / total_dur,
        }

    # --- Bloco: volume / energia (loudness) ----------------------------------

    def _loudness_features(self, wav: np.ndarray) -> dict[str, float]:
        """Nível e instabilidade de volume.

        Fala confiante é mais alta e estável; dúvida tem volume menor e mais instável
        (Jiang & Pell 2017). ``loudness_cv`` (coef. de variação) é invariante a ganho de
        gravação; ``loudness_mean`` é o nível de energia (relativo entre janelas).
        """
        import librosa

        rms = librosa.feature.rms(
            y=wav, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        loud_mean = float(np.mean(rms)) if rms.size else 0.0
        loud_cv = float(np.std(rms) / (loud_mean + 1e-8)) if rms.size else 0.0
        return {"hes_loudness_mean": loud_mean, "hes_loudness_cv": loud_cv}

    # --- Bloco: entonação / voz (Praat via parselmouth) ----------------------

    def _praat_features(self, wav: np.ndarray) -> dict[str, float]:
        """F0 (variabilidade/faixa/contorno/subida terminal) e jitter/shimmer.

        Incerteza: F0 MAIS variável e faixa maior; contorno geral e terminal
        ascendentes (soa como pergunta) — Jiang & Pell 2017; Brennan & Williams 1995.
        """
        import parselmouth
        from parselmouth.praat import call

        sr = self.sample_rate
        snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
        time_step = self.hop_length / sr
        out: dict[str, float] = {}

        # --- F0 / entonação ---------------------------------------------------
        pitch = snd.to_pitch(
            time_step=time_step, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max
        )
        f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
        voiced = f0 > 0
        f0v = f0[voiced]
        if f0v.size:
            f0_mean = float(np.mean(f0v))
            median = float(np.median(f0v))
            out["hes_f0_cv"] = float(np.std(f0v) / (f0_mean + 1e-8))
            out["hes_f0_range_st"] = float(12.0 * np.log2((f0v.max() + 1e-8) / (f0v.min() + 1e-8)))
            # Contorno em semitons (rel. mediana) vs tempo → slope global e terminal.
            st = 12.0 * np.log2((f0v + 1e-8) / (median + 1e-8))
            t = np.nonzero(voiced)[0] * time_step
            if f0v.size >= 2:
                out["hes_f0_slope"] = float(np.polyfit(t, st, 1)[0])
                # Subida terminal: slope só na fração final (soa como pergunta = incerteza).
                n_tail = max(2, int(round(self.final_frac * f0v.size)))
                out["hes_f0_final_slope"] = float(np.polyfit(t[-n_tail:], st[-n_tail:], 1)[0])

        # --- Jitter / shimmer (tensão vocal; pista afetiva) -------------------
        pp = call(snd, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)
        jitter = call(
            pp, "Get jitter (local)", 0, 0,
            self._PP_PERIOD_FLOOR, self._PP_PERIOD_CEIL, self._PP_MAX_FACTOR,
        )
        shimmer = call(
            [snd, pp], "Get shimmer (local)", 0, 0,
            self._PP_PERIOD_FLOOR, self._PP_PERIOD_CEIL, self._PP_MAX_FACTOR, self._SHIMMER_MAX_AMP,
        )
        out["hes_jitter_local"] = 0.0 if np.isnan(jitter) else float(jitter)
        out["hes_shimmer_local"] = 0.0 if np.isnan(shimmer) else float(shimmer)
        return out

    # --- Score composto (heurístico, NÃO supervisionado) ---------------------

    def _composite_score(self, feats: dict[str, float]) -> float:
        """Score interpretável [0,1] combinando os marcadores de incerteza mais fortes.

        Heurística de INSPEÇÃO/suporte (o modelo aprende os pesos reais a partir das
        features cruas). Cada termo ∈ [0,1], "maior ⇒ mais incerteza"; combinados por
        média ponderada. Termos de F0 são "gated" por vozeamento.
        """
        voiced = feats.get("hes_voiced_fraction", 0.0)
        gate = 1.0 if voiced > 0.1 else 0.0

        s_latency = float(np.clip(feats.get("hes_onset_latency", 0.0), 0.0, 1.0))
        s_pause = float(np.clip(feats.get("hes_long_pause_ratio", 0.0), 0.0, 1.0))
        s_rise = gate * float(np.clip(feats.get("hes_f0_final_slope", 0.0) / 4.0, 0.0, 1.0))
        s_var = gate * float(np.clip(feats.get("hes_f0_cv", 0.0) / 0.3, 0.0, 1.0))
        s_vol = float(np.clip(feats.get("hes_loudness_cv", 0.0), 0.0, 1.0))
        # Lentidão: articulação lenta ⇒ menos confiança (0 síl/s → 1; ~6 síl/s → 0).
        art = feats.get("hes_articulation_rate", 0.0)
        s_slow = float(np.clip(1.0 - art / 6.0, 0.0, 1.0)) if art > 0 else 0.0

        weights = (0.20, 0.20, 0.15, 0.15, 0.15, 0.15)
        terms = (s_latency, s_pause, s_rise, s_var, s_vol, s_slow)
        return float(np.clip(sum(w * s for w, s in zip(weights, terms, strict=True)), 0.0, 1.0))

    # =========================================================================
    # Nomes canônicos (dim determinística)
    # =========================================================================

    def _build_feature_names(self) -> list[str]:
        """Lista canônica das colunas (ordem estável, derivada só da config)."""
        return [
            # Timing / pausas (latência de resposta + pausas)
            "hes_onset_latency",
            "hes_pause_ratio",
            "hes_pause_rate",
            "hes_mean_pause_dur",
            "hes_long_pause_ratio",
            "hes_voiced_fraction",
            # Taxa de fala / articulação (núcleos silábicos — de Jong & Wempe)
            "hes_speech_rate",
            "hes_articulation_rate",
            "hes_phonation_ratio",
            # Entonação (F0 variável + subida terminal)
            "hes_f0_cv",
            "hes_f0_range_st",
            "hes_f0_slope",
            "hes_f0_final_slope",
            # Volume / energia
            "hes_loudness_mean",
            "hes_loudness_cv",
            # Qualidade de voz (tensão; pista afetiva)
            "hes_jitter_local",
            "hes_shimmer_local",
            # Composto
            "hes_uncertainty_score",
        ]
