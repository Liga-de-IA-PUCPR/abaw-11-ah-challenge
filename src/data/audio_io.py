"""I/O de áudio do pipeline BAH.

- extract_audio(mp4) -> .flac 16 kHz mono   (ffmpeg via subprocess; fallback torchaudio)
  → reusa a lógica de src/scripts/extract_audio.py (branch matheus).
- load_segment(path, t0, t1, sr) -> np.ndarray  (corte por tempo via librosa, offset/duration)

Nunca tocamos frames de vídeo — apenas a faixa de áudio do mp4 (flag ffmpeg ``-vn``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from src.logger import get_logger

log = get_logger("data.audio_io")

# Default da competição (igual ao matheus): 16 kHz mono.
SAMPLE_RATE = 16000


# ==============================================================================
# Extração mp4 → flac  (reusa scripts/extract_audio.py do matheus)
# ==============================================================================


def extract_audio(
    mp4_path: str | Path,
    out_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    backend: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    """Extrai a faixa de áudio de um ``.mp4`` para ``.flac`` 16 kHz mono.

    Espelha ``src/scripts/extract_audio.py`` (matheus): ``ffmpeg -vn -ar 16000 -ac 1``.
    Adiciona cache (``overwrite=False``) e fallback ``torchaudio`` se o binário ffmpeg
    falhar/estiver ausente.

    Args:
        mp4_path: vídeo de entrada.
        out_path: caminho do ``.flac`` de saída (ex.: ``data/interim/Audio/...``).
        sample_rate: taxa-alvo (Hz). Default 16000.
        mono: se ``True``, mixa para 1 canal.
        backend: ``"ffmpeg"`` | ``"torchaudio"``.
        overwrite: se ``False`` e o destino existir, não re-extrai.

    Returns:
        ``Path`` do ``.flac`` gerado.
    """
    mp4_path = Path(mp4_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        log.debug(f"Áudio já extraído (cache): {out_path.name}")
        return out_path

    if backend == "ffmpeg":
        try:
            _extract_ffmpeg(mp4_path, out_path, sample_rate, mono)
            return out_path
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            log.warning(f"ffmpeg falhou ({exc}); tentando fallback torchaudio.")

    _extract_torchaudio(mp4_path, out_path, sample_rate, mono)
    return out_path


def _extract_ffmpeg(
    mp4_path: Path, out_path: Path, sample_rate: int, mono: bool
) -> None:
    """Extrai áudio via binário ffmpeg (subprocess) — idêntico ao matheus."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp4_path),
        "-vn",                       # descarta vídeo (nunca usamos frames)
        "-ar", str(sample_rate),     # resample
        "-ac", "1" if mono else "2",  # mono
        str(out_path),               # extensão .flac define o codec
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log.debug(f"ffmpeg → {out_path.name}")


def _extract_torchaudio(
    mp4_path: Path, out_path: Path, sample_rate: int, mono: bool
) -> None:
    """Fallback de extração via torchaudio + resample."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(mp4_path))  # (canais, amostras)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    torchaudio.save(str(out_path), waveform.to(torch.float32), sample_rate)
    log.debug(f"torchaudio → {out_path.name}")


# ==============================================================================
# Corte de segmento por tempo
# ==============================================================================


def load_segment(
    path: str | Path,
    t0: float,
    t1: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Carrega o trecho ``[t0, t1)`` (s) de um ``.flac`` como waveform mono float32.

    Usa ``librosa.load`` com ``offset``/``duration`` (lê só o necessário). Janelas
    além do fim do áudio retornam o que houver (possivelmente vazio); o *padding* da
    última janela é resolvido por :func:`pad_or_trim` na FASE 3.

    Args:
        path: caminho do ``.flac`` (16 kHz mono).
        t0: início (s).
        t1: fim (s).
        sr: taxa-alvo (Hz).

    Returns:
        ``np.ndarray`` 1-D float32 com as amostras do segmento.
    """
    import librosa

    t0 = max(0.0, float(t0))
    duration = max(0.0, float(t1) - t0)
    if duration <= 0.0:
        return np.zeros(0, dtype=np.float32)

    waveform, _ = librosa.load(
        str(path), sr=sr, mono=True, offset=t0, duration=duration
    )
    return waveform.astype(np.float32, copy=False)


def pad_or_trim(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """Ajusta ``waveform`` para ``target_len`` amostras (zero-pad à direita ou corte).

    Garante tamanho fixo por janela (``size_s * sr``) antes da extração de features
    acústicas na FASE 3 (ex.: wav2vec2/HuBERT esperam comprimento estável por lote).
    """
    n = waveform.shape[0]
    if n == target_len:
        return waveform
    if n > target_len:
        return waveform[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = waveform
    return out
