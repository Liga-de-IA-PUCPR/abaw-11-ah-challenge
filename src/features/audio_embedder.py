"""AudioEmbedder — FACTORY por ``audio_embedder.backend`` (README §6.3 / §8).

Três backends, mesma interface :class:`BaseEmbedder`
(``extract(waveforms: list[np.ndarray]) -> np.ndarray (n, d_audio)``):

- ``"librosa"``  → ``LibrosaAudioEmbedder``: vetor PROSÓDICO/ESPECTRAL fixo, **CPU/numpy**
  (baseline interpretável para o RandomForest). MFCC+Δ+Δ²+espectrais+chroma+zcr+rms
  agregados (mean/std/min/max) + f0(pyin) + tempo. ``d_audio`` determinístico (~328).
- ``"wav2vec2"`` / ``"hubert"`` → ``DeepAudioEmbedder``: ``AutoModel`` do transformers,
  ``last_hidden_state.mean(1)`` (reusa ``AudioVectorizer`` do matheus), device-aware.
  ``d_audio = model.config.hidden_size`` (768 base) lido automaticamente.

Por que prosódia é o sinal de hesitação (backend librosa):
- Pausas/hesitação → ``voiced_fraction`` (fração de frames com f0 detectado).
- Variação de pitch → desvio-padrão de ``f0`` (entonação instável = incerteza).
- Energia irregular → estatísticas de ``rms``; ritmo → ``tempo`` / ``zcr``.
O ``std`` de cada série captura justamente essa instabilidade.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.base.embedder import BaseEmbedder
from src.conf import resolve_device
from src.logger import get_logger

log = get_logger("features.audio_embedder")

# Sample rate canônico do pipeline (README: data.audio.sample_rate = 16000).
_SR_DEFAULT = 16000

_AGG_FUNCS = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}


# =============================================================================
# Factory (entrada da FASE 6 / build_feature_components)
# =============================================================================

def create_audio_embedder(
    backend: Literal["librosa", "wav2vec2", "hubert"] = "librosa",
    *,
    model_name: str | None = None,
    feature_set: list[str] | None = None,
    n_mfcc: int = 20,
    agg_stats: list[str] | None = None,
    sample_rate: int = _SR_DEFAULT,
    batch_size: int = 8,
    device: str = "auto",
) -> "AudioEmbedder":
    """Instancia o embedder de áudio conforme ``audio_embedder.backend``.

    Args:
        backend: "librosa" (prosódia, CPU) | "wav2vec2" | "hubert" (deep, device-aware).
        model_name: Checkpoint HF p/ backends deep (default por backend).
        feature_set / n_mfcc / agg_stats: parâmetros do backend librosa.
        sample_rate: SR dos waveforms (deve casar com data.audio.sample_rate).
        batch_size: batch dos backends deep.
        device: "auto" | "cpu" | "mps" | "cuda" (só relevante p/ backends deep).

    Returns:
        Uma instância de :class:`AudioEmbedder` (subclasse concreta).
    """
    if backend == "librosa":
        return LibrosaAudioEmbedder(
            feature_set=feature_set, n_mfcc=n_mfcc, agg_stats=agg_stats, sample_rate=sample_rate
        )
    defaults = {"wav2vec2": "facebook/wav2vec2-base", "hubert": "facebook/hubert-base-ls960"}
    if backend not in defaults:
        raise ValueError(f"backend de áudio desconhecido: {backend!r}")
    return DeepAudioEmbedder(
        model_name=model_name or defaults[backend],
        sample_rate=sample_rate,
        batch_size=batch_size,
        device=device,
    )


# =============================================================================
# Base comum (alias do tipo de retorno da factory)
# =============================================================================

class AudioEmbedder(BaseEmbedder):
    """Tipo base dos embedders de áudio (factory: :func:`create_audio_embedder`)."""

    name: str = "audio"


# =============================================================================
# Backend (a): librosa — prosódia/espectral, CPU
# =============================================================================

class LibrosaAudioEmbedder(AudioEmbedder):
    """Vetor acústico/prosódico de tamanho FIXO por janela via LIBROSA (CPU/numpy).

    Composição (default ``n_mfcc=20``, ``agg_stats`` = mean/std/min/max):

    | Grupo                  | nº séries | agregação | dim |
    |------------------------|----------:|-----------|----:|
    | MFCC + Δ + Δ²          | 60        | 4 stats   | 240 |
    | Espectrais             | 5         | 4 stats   | 20  |
    | Chroma                 | 12        | 4 stats   | 48  |
    | zcr + rms              | 2         | 4 stats   | 8   |
    | f0 (pyin)              | —         | escalares | 3   |
    | tempo                  | —         | escalar   | 1   |
    | **Total (default)**    |           |           | **~328** |
    """

    name: str = "audio"

    def __init__(
        self,
        feature_set: list[str] | None = None,
        n_mfcc: int = 20,
        agg_stats: list[str] | None = None,
        sample_rate: int = _SR_DEFAULT,
    ) -> None:
        self.backend = "librosa"
        self.feature_set = feature_set or [
            "mfcc", "delta", "spectral", "chroma", "zcr", "rms", "f0", "tempo",
        ]
        self.n_mfcc = n_mfcc
        self.agg_stats = agg_stats or ["mean", "std", "min", "max"]
        self.sample_rate = sample_rate

        self._feature_names = self._build_feature_names()
        self._dim = len(self._feature_names)
        log.info(
            f"LibrosaAudioEmbedder pronto: dim={self._dim}, n_mfcc={n_mfcc}, "
            f"agg={self.agg_stats}, feature_set={self.feature_set}"
        )

    @property
    def dim(self) -> int:
        """Dimensão fixa do vetor de áudio."""
        return self._dim

    def extract(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Extrai o vetor acústico para cada waveform de janela (cortado em [t0,t1])."""
        if len(inputs) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = [self._extract_one(wav) for wav in inputs]
        result = np.vstack(rows).astype(np.float32)
        log.debug(f"LibrosaAudioEmbedder.extract: {result.shape[0]} janelas -> dim {result.shape[1]}")
        return result

    def feature_names(self) -> list[str]:
        """Nomes determinísticos, alinhados às colunas de :meth:`extract`."""
        return list(self._feature_names)

    # --- internals -----------------------------------------------------------

    def _extract_one(self, wav: np.ndarray) -> np.ndarray:
        """Extrai o vetor de uma única janela (waveform mono)."""
        import librosa  # import tardio (dependência pesada)

        sr = self.sample_rate
        wav = np.asarray(wav, dtype=np.float32).ravel()
        if wav.size == 0:
            return np.zeros(self._dim, dtype=np.float32)

        feats: list[float] = []

        mfcc = None
        if "mfcc" in self.feature_set or "delta" in self.feature_set:
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=self.n_mfcc)
        if "mfcc" in self.feature_set:
            feats += self._agg_series(mfcc)
        if "delta" in self.feature_set:
            feats += self._agg_series(librosa.feature.delta(mfcc))
            feats += self._agg_series(librosa.feature.delta(mfcc, order=2))

        if "spectral" in self.feature_set:
            feats += self._agg_series(librosa.feature.spectral_centroid(y=wav, sr=sr))
            feats += self._agg_series(librosa.feature.spectral_bandwidth(y=wav, sr=sr))
            feats += self._agg_series(librosa.feature.spectral_rolloff(y=wav, sr=sr))
            feats += self._agg_series(librosa.feature.spectral_flatness(y=wav))
            contrast = librosa.feature.spectral_contrast(y=wav, sr=sr)  # (bands, T)
            feats += self._agg_series(contrast.mean(axis=0, keepdims=True))

        if "chroma" in self.feature_set:
            feats += self._agg_series(librosa.feature.chroma_stft(y=wav, sr=sr))

        if "zcr" in self.feature_set:
            feats += self._agg_series(librosa.feature.zero_crossing_rate(y=wav))
        if "rms" in self.feature_set:
            feats += self._agg_series(librosa.feature.rms(y=wav))

        if "f0" in self.feature_set:
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    wav,
                    fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                    fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
                    sr=sr,
                )
            except Exception as exc:  # pyin pode falhar em janelas degeneradas
                log.debug(f"pyin falhou ({exc}); f0 = NaN")
                f0, voiced_flag = np.array([np.nan]), np.array([False])
            f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            f0_mean = float(np.mean(f0_voiced)) if f0_voiced.size else 0.0
            f0_std = float(np.std(f0_voiced)) if f0_voiced.size else 0.0
            voiced_fraction = (
                float(np.mean(voiced_flag.astype(np.float32)))
                if voiced_flag is not None and len(voiced_flag) else 0.0
            )
            feats += [f0_mean, f0_std, voiced_fraction]

        if "tempo" in self.feature_set:
            try:
                tempo = librosa.beat.tempo(y=wav, sr=sr)
                feats.append(float(tempo[0]) if len(tempo) else 0.0)
            except Exception:
                feats.append(0.0)

        vec = np.asarray(feats, dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def _agg_series(self, series: np.ndarray) -> list[float]:
        """Agrega (n_series, T) por linha com cada estatística de ``agg_stats``."""
        series = np.atleast_2d(series)
        out: list[float] = []
        for row in series:
            if row.size == 0:
                out += [0.0] * len(self.agg_stats)
                continue
            for stat in self.agg_stats:
                out.append(float(_AGG_FUNCS[stat](row)))
        return out

    def _build_feature_names(self) -> list[str]:
        """Nomes determinísticos coerentes com :meth:`_extract_one`."""
        names: list[str] = []

        def add_series(prefix: str, k: int) -> None:
            for i in range(k):
                for stat in self.agg_stats:
                    names.append(f"audio_{prefix}{i}_{stat}")

        if "mfcc" in self.feature_set:
            add_series("mfcc_", self.n_mfcc)
        if "delta" in self.feature_set:
            add_series("mfcc_delta_", self.n_mfcc)
            add_series("mfcc_delta2_", self.n_mfcc)
        if "spectral" in self.feature_set:
            for name in [
                "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                "spectral_flatness", "spectral_contrast",
            ]:
                for stat in self.agg_stats:
                    names.append(f"audio_{name}_{stat}")
        if "chroma" in self.feature_set:
            add_series("chroma_", 12)
        if "zcr" in self.feature_set:
            for stat in self.agg_stats:
                names.append(f"audio_zcr_{stat}")
        if "rms" in self.feature_set:
            for stat in self.agg_stats:
                names.append(f"audio_rms_{stat}")
        if "f0" in self.feature_set:
            names += ["audio_f0_mean", "audio_f0_std", "audio_voiced_fraction"]
        if "tempo" in self.feature_set:
            names.append("audio_tempo")
        return names


# =============================================================================
# Backend (b): wav2vec2 / hubert — deep, device-aware (reusa AudioVectorizer)
# =============================================================================

class DeepAudioEmbedder(AudioEmbedder):
    """Embedder de áudio deep via transformers (``last_hidden_state.mean(1)``).

    Reusa a ideia do ``AudioVectorizer`` da branch ``matheus`` (wav2vec2-base),
    generalizado para HuBERT e tornado device-aware (``resolve_device``). A dimensão
    é ``model.config.hidden_size`` (768 no base) lida automaticamente.

    Attributes:
        name: "audio".
        backend: "wav2vec2" | "hubert".
        model_name: checkpoint HF.
        sample_rate: SR dos waveforms (16 kHz).
        batch_size: batch de inferência.
        device: ``torch.device`` resolvido.
    """

    name: str = "audio"

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        sample_rate: int = _SR_DEFAULT,
        batch_size: int = 8,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModel

        self.backend = "hubert" if "hubert" in model_name.lower() else "wav2vec2"
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._torch = torch

        log.info(f"Carregando modelo de áudio deep: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._dim = int(self.model.config.hidden_size)
        log.info(f"DeepAudioEmbedder pronto: dim={self._dim}, backend={self.backend}, device={self.device}")

    @property
    def dim(self) -> int:
        """Dimensão = ``model.config.hidden_size`` (768 base)."""
        return self._dim

    def extract(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Extrai embedding deep por janela (mean-pool temporal do último hidden state)."""
        torch = self._torch
        if len(inputs) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size]
                # Pad à direita até o maior waveform do batch → (b, L).
                max_len = max(int(np.asarray(w).size) for w in batch) or 1
                arr = np.zeros((len(batch), max_len), dtype=np.float32)
                for i, w in enumerate(batch):
                    w = np.asarray(w, dtype=np.float32).ravel()
                    arr[i, : w.size] = w
                input_values = torch.from_numpy(arr).to(self.device)
                hidden = self.model(input_values=input_values).last_hidden_state  # (b, T, H)
                pooled = hidden.mean(dim=1)  # (b, H)
                out.append(pooled.detach().cpu().numpy().astype(np.float32))

        result = np.concatenate(out, axis=0)
        log.debug(f"DeepAudioEmbedder.extract: {result.shape[0]} janelas -> dim {result.shape[1]}")
        return result

    def feature_names(self) -> list[str]:
        """Nomes determinísticos: ['audio_emb_0', ..., 'audio_emb_{dim-1}']."""
        return [f"audio_emb_{i}" for i in range(self._dim)]
