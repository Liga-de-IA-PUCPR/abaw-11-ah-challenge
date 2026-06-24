"""Features tabulares: metadados do participante + question_type + prosódia da janela.

⚠️ Regra de inferência: usar SOMENTE features disponíveis no TEST set.
- ``certainty_ah`` e ``all_cues`` são ANOTAÇÕES (não estão no test) → NUNCA viram
  feature; ficam reservadas apenas para análise/diagnóstico.
- Encoders (one-hot) são ajustados (``fit``) APENAS no split de treino e aplicados
  (``transform``) no resto, evitando vazamento.

Features produzidas por janela:
- ``question_type`` one-hot (7 tipos do BAH).
- Metadados do participante: ``age`` (numérico), ``age_range``, ``gender``,
  ``ethnicity_simplified``, ``is_student``, ``province``, ``country``.
- Prosódia engenheirada da janela: ``duration``, ``speech_rate`` (n_palavras/dur),
  ``silence_ratio`` (via limiar de RMS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.logger import get_logger

log = get_logger("features.tabular")

# 7 tipos de pergunta do BAH (README §2). Ordem fixa → one-hot determinístico.
QUESTION_TYPES: list[str] = [
    "neutral", "positive", "negative", "ambivalent", "willing", "resistant", "hesitant",
]

# Campos demográficos categóricos (one-hot via vocabulário aprendido no train).
_CATEGORICAL_META: list[str] = [
    "age_range", "gender", "ethnicity_simplified", "province", "country",
]
# Campos demográficos numéricos / binários (passam direto).
_NUMERIC_META: list[str] = ["age", "is_student"]


@dataclass
class TabularFeaturizer:
    """Constrói features tabulares por janela com encoders fit-no-train.

    Uso:
        feat = TabularFeaturizer(sample_rate=16000)
        feat.fit(train_windows)          # aprende vocabulários do train
        X_tab = feat.transform(windows, waveforms)   # (n, d_tab) p/ qualquer split

    Entradas: ``WindowSample`` (README §6.1) para metadados/texto/tempos, e os
    ``waveforms`` da janela (cortados por ``audio_io.load_segment``, FASE 2) em
    paralelo — pois ``WindowSample`` não carrega waveform. Os acessos toleram tanto
    ``WindowSample`` quanto dict.

    Attributes:
        sample_rate: SR usado nos cálculos de prosódia (silence_ratio).
        silence_rms_threshold: Limiar relativo de RMS p/ marcar frame como silêncio.
        vocab_: Vocabulários aprendidos por campo categórico (preenchido no fit).
        feature_names_: Nomes das colunas (preenchido no fit).
    """

    sample_rate: int = 16000
    silence_rms_threshold: float = 0.1  # fração do RMS máximo da janela
    vocab_: dict[str, list[str]] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    _fitted: bool = False

    # =========================================================================
    # Construção a partir da Config Hydra
    # =========================================================================

    @classmethod
    def from_config(cls, cfg=None) -> "TabularFeaturizer":
        """Instancia o featurizer a partir de um nó de config (``cfg.data.tabular``).

        Tolera ``cfg=None`` (usa os defaults) e leitura via ``.get``/atributo, então
        funciona tanto com ``DictConfig`` quanto com dict. A FASE 6
        (``mode=featurize``) usa ``TabularFeaturizer.from_config(cfg.data.tabular)``
        seguido de ``.fit(train_windows)`` (vocabulários sem vazamento).

        Args:
            cfg: nó de config opcional com ``sample_rate`` / ``silence_rms_threshold``.

        Returns:
            ``TabularFeaturizer`` (ainda **não** fitted).
        """
        def _get(key, default):
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        return cls(
            sample_rate=int(_get("sample_rate", 16000)),
            silence_rms_threshold=float(_get("silence_rms_threshold", 0.1)),
        )

    # =========================================================================
    # Dimensão (contrato comum com os embedders — README §6.3)
    # =========================================================================

    @property
    def dim(self) -> int:
        """Número de colunas tabulares (= ``len(feature_names())``); requer ``fit``."""
        if not self._fitted:
            raise RuntimeError("TabularFeaturizer.dim acessado antes de fit().")
        return len(self.feature_names_)

    # =========================================================================
    # Fit / Transform
    # =========================================================================

    def fit(self, windows: list) -> "TabularFeaturizer":
        """Aprende vocabulários categóricos a partir das janelas de TREINO.

        Args:
            windows: Lista de ``WindowSample`` (ou dicts) do split de treino.

        Returns:
            self (fitted).
        """
        for col in _CATEGORICAL_META:
            values = sorted({str(self._meta_get(w, col)) for w in windows})
            self.vocab_[col] = values

        self.feature_names_ = self._build_feature_names()
        self._fitted = True
        log.info(
            f"TabularFeaturizer fit: d_tab={len(self.feature_names_)} "
            f"(qtype=7, demográficos="
            f"{sum(len(v) for v in self.vocab_.values()) + len(_NUMERIC_META)}, prosódia=3)"
        )
        return self

    def transform(self, windows: list, waveforms: list | None = None) -> np.ndarray:
        """Converte janelas em matriz tabular ``(n, d_tab)``.

        Args:
            windows: Lista de ``WindowSample`` (ou dicts) de qualquer split.
            waveforms: Waveforms alinhados a ``windows`` (mesma ordem/comprimento),
                cortados em ``[t0, t1]`` na FASE 2. Se ``None``, a prosódia de
                silêncio cai para 0.

        Returns:
            Array ``float32`` de shape ``(n, d_tab)``.
        """
        if not self._fitted:
            raise RuntimeError("TabularFeaturizer.transform chamado antes de fit().")

        wavs = waveforms if waveforms is not None else [None] * len(windows)
        rows = [self._transform_one(w, wav) for w, wav in zip(windows, wavs)]
        result = (
            np.vstack(rows).astype(np.float32)
            if rows else np.zeros((0, len(self.feature_names_)), dtype=np.float32)
        )
        log.debug(f"TabularFeaturizer.transform: {result.shape}")
        return result

    def feature_names(self) -> list[str]:
        """Nomes das colunas tabulares (após fit)."""
        return list(self.feature_names_)

    # =========================================================================
    # Internals
    # =========================================================================

    def _transform_one(self, window, wav) -> np.ndarray:
        """Vetor tabular de uma janela."""
        feats: list[float] = []

        # --- question_type one-hot ------------------------------------------
        qt = self._field(window, "question_type", "") or ""
        feats += [1.0 if qt == t else 0.0 for t in QUESTION_TYPES]

        # --- demográficos numéricos -----------------------------------------
        age = self._meta_get(window, "age")
        feats.append(float(age) if self._is_number(age) else 0.0)
        is_student = self._meta_get(window, "is_student")
        feats.append(1.0 if str(is_student).lower() in ("true", "1", "yes") else 0.0)

        # --- demográficos categóricos (one-hot via vocab do train) ----------
        for col in _CATEGORICAL_META:
            value = str(self._meta_get(window, col))
            for known in self.vocab_[col]:
                feats.append(1.0 if value == known else 0.0)

        # --- prosódia engenheirada da janela --------------------------------
        duration, speech_rate, silence_ratio = self._window_prosody(window, wav)
        feats += [duration, speech_rate, silence_ratio]

        return np.asarray(feats, dtype=np.float32)

    def _window_prosody(self, window, wav) -> tuple[float, float, float]:
        """Calcula (duração, taxa de fala, razão de silêncio).

        - ``duration`` = t1 - t0 (s).
        - ``speech_rate`` = n_palavras / duração (palavras/s).
        - ``silence_ratio`` = fração de frames com RMS < limiar * RMS_max.

        O waveform vem do argumento ``wav`` (cortado por ``audio_io.load_segment``,
        FASE 2), NÃO de ``WindowSample.meta``. Na ausência de waveform, silêncio → 0.
        """
        t0 = float(self._field(window, "t0", 0.0))
        t1 = float(self._field(window, "t1", 0.0))
        duration = max(t1 - t0, 1e-6)

        text = self._field(window, "text", "") or ""
        speech_rate = len(text.split()) / duration

        silence_ratio = 0.0
        if wav is not None and len(wav):
            import librosa

            rms = librosa.feature.rms(y=np.asarray(wav, dtype=np.float32).ravel())[0]
            if rms.size:
                thr = self.silence_rms_threshold * float(rms.max())
                silence_ratio = float(np.mean(rms < thr))

        return duration, speech_rate, silence_ratio

    def _build_feature_names(self) -> list[str]:
        """Nomes determinísticos (alinhados a :meth:`_transform_one`)."""
        names: list[str] = [f"qtype_{t}" for t in QUESTION_TYPES]
        names += ["meta_age", "meta_is_student"]
        for col in _CATEGORICAL_META:
            names += [f"meta_{col}={v}" for v in self.vocab_[col]]
        names += ["prosody_duration", "prosody_speech_rate", "prosody_silence_ratio"]
        return names

    @staticmethod
    def _field(window, key: str, default=None):
        """Lê um campo da janela, aceitando ``WindowSample`` OU dict."""
        if isinstance(window, dict):
            return window.get(key, default)
        return getattr(window, key, default)

    @classmethod
    def _meta_get(cls, window, key: str):
        """Lê uma chave do dict de metadados da janela (tolerante a ausência)."""
        meta = cls._field(window, "meta", {}) or {}
        return meta.get(key, "")

    @staticmethod
    def _is_number(value) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
