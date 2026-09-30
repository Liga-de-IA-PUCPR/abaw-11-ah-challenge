"""ASR-erased time — 16 features determinísticas dos timestamps do Whisper.

Motivação (README/`references/improvement_plan.md` Fase 2a; `references/
visual_signal_lessons_from_top_teams.md`): o achado mais forte do 1º lugar do BAH
(IISERB) não veio do rosto, veio do **tempo apagado pelo ASR** — os *gaps* entre
chunks da transcrição Whisper (silêncio/pausa que o ASR não transcreve). Eles
reportam 16 features determinísticas com AP 0.718, acima de qualquer feature
visual, e quase zero custo computacional. Este módulo reproduz essa receita.

Entrada: ``transcript_chunks`` como parseado em ``src/data/schema.py``
(``VideoRecord.transcript_chunks: list[dict]``), cada item
``{"start": float, "end": float, "text": str, "language": str}`` (chaves extras
são ignoradas). O idioma das transcrições BAH é **inglês** (ver alerta em
``src/features/text_embedder.py``), por isso a lista de palavras contrastivas é EN.

## Reset de 30s do Whisper

O Whisper processa áudio em janelas internas de ~30s e, ao cruzar uma janela,
reinicia sua linha do tempo interna — o próximo chunk pode ter ``start`` menor
ou igual ao ``end`` do chunk anterior (não é um gap real, é uma descontinuidade
de relógio interno do decoder). Um gap só é contado entre chunks consecutivos
``(i, i+1)`` quando a linha do tempo é **monotonicamente crescente**, ou seja
``chunks[i+1]["start"] > chunks[i]["end"]``. Quando isso falha (reset ou
overlap), o par é **pulado** (não conta como gap, não conta como gap negativo,
não contamina média/desvio) — mas ambos os chunks continuam contribuindo para
as features *within-chunk* (duração, taxa de fala).

## Definições operacionais (não fixadas pelo paper original)

- **Palavras contrastivas**: lista fixa EN — "but", "however", "though",
  "although", "yet", "nevertheless", "except", "still" — checadas como a
  primeira palavra (case-insensitive, ignorando pontuação) do texto do chunk
  que *seguiu* o gap. Reaproveitável de forma independente do bloco de
  hesitação (`hesitation.py` não define uma lista textual — é puramente
  acústico), então a lista é definida aqui.
- **Restart entre chunks** (feature 15): um chunk é um "restart" quando (a) ele
  é imediatamente precedido por um gap monotônico (não um reset) e (b) sua
  duração é menor que ``restart_max_dur_s`` (default 0.3s — limiar curto o
  bastante para capturar um "uh"/palavra truncada que o Whisper emitiu como
  chunk próprio após uma pausa, mas documentado como hiperparâmetro ajustável,
  não uma verdade extraída do paper). É uma proxy heurística, não uma detecção
  real de disfluência.

Todas as 16 features são finitas em qualquer entrada, incluindo 0 ou 1 chunks
(vetor de zeros) e ausência total de gaps/palavras contrastivas (as features
dependentes ficam 0.0 em vez de NaN).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# =============================================================================
# Vocabulário e hiperparâmetros
# =============================================================================

# Palavras contrastivas/qualificadoras em inglês (transcrições BAH são EN).
CONTRASTIVE_WORDS: frozenset[str] = frozenset(
    {
        "but",
        "however",
        "though",
        "although",
        "yet",
        "nevertheless",
        "except",
        "still",
    }
)

# Duração máxima (s) de um chunk pós-gap para contar como "restart" (feature 15).
_RESTART_MAX_DUR_S = 0.3

_WORD_RE = re.compile(r"[a-zA-Z']+")

ASR_TIMING_FEATURE_NAMES: list[str] = [
    "asr_gap_count",
    "asr_gap_total_time",
    "asr_gap_time_ratio",
    "asr_gap_mean",
    "asr_gap_std",
    "asr_gap_max",
    "asr_gap_count_per_chunk",
    "asr_speaking_rate_mean",
    "asr_speaking_rate_std",
    "asr_chunk_dur_mean",
    "asr_chunk_dur_std",
    "asr_chunk_fragmentation",
    "asr_gap_before_contrastive_ratio",
    "asr_gap_before_contrastive_mean",
    "asr_restart_count",
    "asr_restart_count_per_chunk",
]

N_ASR_TIMING_FEATURES = len(ASR_TIMING_FEATURE_NAMES)  # 16


def _first_word(text: str) -> str:
    match = _WORD_RE.search(text or "")
    return match.group(0).lower() if match else ""


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def extract_asr_timing_features(
    chunks: list[dict[str, Any]],
    *,
    restart_max_dur_s: float = _RESTART_MAX_DUR_S,
) -> dict[str, float]:
    """Extrai as 16 features de "tempo apagado pelo ASR" de uma transcrição.

    Args:
        chunks: ``VideoRecord.transcript_chunks`` (ou lista equivalente) —
            ``[{"start": float, "end": float, "text": str, ...}]``, em ordem
            de emissão do Whisper (não necessariamente monotônica, ver reset
            de 30s no docstring do módulo).
        restart_max_dur_s: limiar (s) de duração para a feature 15/16
            ("restart" pós-gap). Default 0.3s.

    Returns:
        Dict ordenado (``ASR_TIMING_FEATURE_NAMES``) de 16 floats finitos.
        Vetor de zeros se ``len(chunks) < 2``.
    """
    n_chunks = len(chunks)
    if n_chunks == 0:
        return dict.fromkeys(ASR_TIMING_FEATURE_NAMES, 0.0)

    starts = [float(c["start"]) for c in chunks]
    ends = [float(c["end"]) for c in chunks]
    texts = [str(c.get("text", "")) for c in chunks]

    durations = [max(0.0, e - s) for s, e in zip(starts, ends, strict=True)]
    total_speech_time = float(sum(durations))

    speaking_rates: list[float] = []
    for text, dur in zip(texts, durations, strict=True):
        n_chars = len(text.strip())
        if dur > 0.0:
            speaking_rates.append(n_chars / dur)

    if n_chunks < 2:
        feats = dict.fromkeys(ASR_TIMING_FEATURE_NAMES, 0.0)
        feats["asr_chunk_dur_mean"] = _safe_mean(durations)
        feats["asr_chunk_dur_std"] = _safe_std(durations)
        feats["asr_speaking_rate_mean"] = _safe_mean(speaking_rates)
        feats["asr_speaking_rate_std"] = _safe_std(speaking_rates)
        return feats

    # -------------------------------------------------------------------
    # Gaps monotônicos entre chunks consecutivos (pula resets de 30s).
    # -------------------------------------------------------------------
    gaps: list[float] = []
    gap_before_contrastive: list[float] = []
    restart_count = 0

    for i in range(n_chunks - 1):
        prev_end = ends[i]
        next_start = starts[i + 1]
        if next_start <= prev_end:
            # Reset da linha do tempo interna do Whisper (ou overlap) — pula.
            continue
        gap = next_start - prev_end
        gaps.append(gap)

        next_word = _first_word(texts[i + 1])
        if next_word in CONTRASTIVE_WORDS:
            gap_before_contrastive.append(gap)

        next_dur = durations[i + 1]
        if next_dur < restart_max_dur_s:
            restart_count += 1

    gap_count = len(gaps)
    gap_total = float(sum(gaps))
    gap_time_ratio = gap_total / total_speech_time if total_speech_time > 0 else 0.0

    feats = {
        "asr_gap_count": float(gap_count),
        "asr_gap_total_time": gap_total,
        "asr_gap_time_ratio": gap_time_ratio,
        "asr_gap_mean": _safe_mean(gaps),
        "asr_gap_std": _safe_std(gaps),
        "asr_gap_max": float(max(gaps)) if gaps else 0.0,
        "asr_gap_count_per_chunk": gap_count / n_chunks,
        "asr_speaking_rate_mean": _safe_mean(speaking_rates),
        "asr_speaking_rate_std": _safe_std(speaking_rates),
        "asr_chunk_dur_mean": _safe_mean(durations),
        "asr_chunk_dur_std": _safe_std(durations),
        "asr_chunk_fragmentation": (n_chunks / total_speech_time if total_speech_time > 0 else 0.0),
        "asr_gap_before_contrastive_ratio": (
            len(gap_before_contrastive) / gap_count if gap_count > 0 else 0.0
        ),
        "asr_gap_before_contrastive_mean": _safe_mean(gap_before_contrastive),
        "asr_restart_count": float(restart_count),
        "asr_restart_count_per_chunk": restart_count / n_chunks,
    }
    return {name: float(feats[name]) for name in ASR_TIMING_FEATURE_NAMES}


def extract_asr_timing_vector(
    chunks: list[dict[str, Any]],
    *,
    restart_max_dur_s: float = _RESTART_MAX_DUR_S,
) -> np.ndarray:
    """Como :func:`extract_asr_timing_features`, mas devolve ``np.ndarray (16,)``.

    Conveniência para concatenar no vetor tabular (74-dim de hesitação + estes
    16) sem depender da ordem de iteração de um dict.
    """
    feats = extract_asr_timing_features(chunks, restart_max_dur_s=restart_max_dur_s)
    return np.array([feats[name] for name in ASR_TIMING_FEATURE_NAMES], dtype=np.float32)
