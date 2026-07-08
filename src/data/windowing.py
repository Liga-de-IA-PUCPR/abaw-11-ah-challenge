"""Janela deslizante + alinhamento áudio⟷texto + rótulo de janela.

Aqui acontece o ALINHAMENTO por timestamp (consumido na FASE 3):
- texto da janela = concatenação dos chunks de transcrição sobrepostos;
- rótulo da janela = função da sobreposição com os intervalos de time_detailed_ah.

Matemática de sobreposição de dois intervalos [a0,a1] e [b0,b1]:

    overlap = max(0, min(a1, b1) - max(a0, b0))

A janela é positiva se a soma das sobreposições com time_detailed_ah, dividida pelo
tamanho da janela, atinge ``min_overlap_for_positive``:

    frac = (Σ overlap(janela, intervalo_i)) / size_s
    label = 1 se frac >= min_overlap_for_positive senão 0
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from src.data.schema import VideoRecord, WindowSample
from src.logger import get_logger

log = get_logger("data.windowing")


# ==============================================================================
# Matemática de sobreposição
# ==============================================================================


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    """Sobreposição (s) entre ``[a0, a1]`` e ``[b0, b1]``.

    ``overlap = max(0, min(a1, b1) - max(a0, b0))``; ``0.0`` se disjuntos.
    """
    return max(0.0, min(a1, b1) - max(a0, b0))


def total_overlap_with_intervals(
    t0: float, t1: float, intervals: list[tuple[float, float]]
) -> float:
    """Soma das sobreposições da janela ``[t0, t1]`` com cada intervalo de A/H."""
    return sum(overlap_seconds(t0, t1, s, e) for s, e in intervals)


# ==============================================================================
# Gerador de janelas
# ==============================================================================


class WindowGenerator:
    """Gera ``list[WindowSample]`` por vídeo (janela deslizante + alinhamento + rótulo).

    Parâmetros (de ``cfg.data.window``; README §7):
    - ``size_s``: tamanho da janela (s), default 5,0 (≈ duração média de A/H = 4,3 s).
    - ``hop_s``: passo entre janelas (s), default 2,5 (50% de sobreposição).
    - ``min_overlap_for_positive``: fração mínima coberta por A/H p/ rótulo 1.
    - ``pad_last``: se ``True``, mantém a última janela mesmo que ultrapasse a duração.
    - ``label_source``: atributo-fonte do rótulo de janela (default ``"time_detailed_ah"``).

    Uso:
        gen = WindowGenerator.from_config(cfg)
        windows = gen.generate(record)
    """

    def __init__(
        self,
        size_s: float = 5.0,
        hop_s: float = 2.5,
        min_overlap_for_positive: float = 0.5,
        pad_last: bool = True,
        label_source: str = "time_detailed_ah",
    ):
        self.size_s = float(size_s)
        self.hop_s = float(hop_s)
        self.min_overlap_for_positive = float(min_overlap_for_positive)
        self.pad_last = bool(pad_last)
        self.label_source = label_source

    @classmethod
    def from_config(cls, cfg: DictConfig) -> WindowGenerator:
        """Cria o gerador a partir de ``cfg.data.window`` (chaves do README §7)."""
        w = cfg.data.window
        return cls(
            size_s=w.size_s,
            hop_s=w.hop_s,
            min_overlap_for_positive=w.min_overlap_for_positive,
            pad_last=w.pad_last,
            label_source=w.label_source,
        )

    # =========================================================================
    # API pública
    # =========================================================================

    def generate(self, record: VideoRecord) -> list[WindowSample]:
        """Gera as janelas de um ``VideoRecord`` (ordenadas por tempo).

        Vazia se a duração for 0. ``window_id`` é ``f"{video_id}#w{idx}"``; o ``idx``
        sequencial vira ``window_idx`` no Parquet (FASE 3) e preserva a ordem temporal
        usada pelo ``VideoSequenceDataset``.
        """
        duration = record.duration_s
        if duration <= 0.0:
            log.debug(f"Duração nula; sem janelas para {record.video_id}")
            return []

        windows: list[WindowSample] = []
        idx = 0
        t0 = 0.0
        while t0 < duration:
            t1 = t0 + self.size_s

            # Última janela ultrapassa a duração:
            if t1 > duration and not self.pad_last:
                break
            # pad_last=True: mantém a janela; padding do áudio = audio_io.pad_or_trim
            # (FASE 3). Texto/rótulo usam [t0, t1] cheios.

            text = self._align_text(record, t0, t1)
            label = self._window_label(record, t0, t1)

            windows.append(
                WindowSample(
                    window_id=f"{record.video_id}#w{idx}",
                    video_id=record.video_id,
                    participant_id=record.participant_id,
                    t0=t0,
                    t1=t1,
                    text=text,
                    label=label,
                    question_type=record.question_type,
                    split=record.split,
                    video_label=record.global_ah,  # alvo a nível de vídeo (None no test)
                    # meta = só demográficos do participante (sem waveform/global_ah).
                    meta=record.meta,
                )
            )
            idx += 1
            t0 += self.hop_s

        log.debug(f"{record.video_id}: {len(windows)} janelas")
        return windows

    # =========================================================================
    # Alinhamento texto ⟷ janela
    # =========================================================================

    def _align_text(self, record: VideoRecord, t0: float, t1: float) -> str:
        """Concatena os chunks de transcrição sobrepostos a ``[t0, t1]``.

        Principal: inclui chunk com ``overlap_seconds > 0``. Fallback (nenhum chunk
        se sobrepõe): inclui o chunk cujo *midpoint* ``(start + end) / 2`` cai na janela.
        """
        chunks = record.transcript_chunks
        if not chunks:
            return ""

        selected = [
            ch["text"] for ch in chunks if overlap_seconds(t0, t1, ch["start"], ch["end"]) > 0.0
        ]
        if not selected:
            selected = [ch["text"] for ch in chunks if t0 <= (ch["start"] + ch["end"]) / 2.0 < t1]
        return " ".join(s for s in selected if s).strip()

    # =========================================================================
    # Rótulo da janela
    # =========================================================================

    def _window_label(self, record: VideoRecord, t0: float, t1: float) -> int | None:
        """Rótulo 0/1 da janela via sobreposição com ``time_detailed_ah`` (``None`` no test).

        Regra:
            frac = total_overlap_with_intervals([t0, t1], intervals) / size_s
            label = 1 se frac >= min_overlap_for_positive senão 0
        """
        if record.split == "test":
            return None

        intervals = getattr(record, self.label_source, record.time_detailed_ah)
        if not intervals:
            return 0

        overlap = total_overlap_with_intervals(t0, t1, intervals)
        frac = overlap / self.size_s if self.size_s > 0 else 0.0
        return 1 if frac >= self.min_overlap_for_positive else 0


# ==============================================================================
# Persistência do índice de janelas (FASE 6 / preprocess ⟷ featurize)
# ==============================================================================


def save_window_index(windows: list[WindowSample], path: str | Path) -> Path:
    """Persiste ``list[WindowSample]`` num Parquet de índice (1 linha/janela).

    Grava os campos canônicos do :class:`WindowSample` (README §6.1) — **sem**
    embeddings nem waveform; estes vêm na FASE 3 (``run_featurize``). É o contrato
    entre ``mode=preprocess`` (escreve) e ``mode=featurize`` (lê via
    :func:`load_window_index`). O ``meta`` (dict de demográficos) é serializado como
    JSON p/ caber numa célula Parquet; ``load_window_index`` o desserializa de volta.

    Args:
        windows: janelas geradas pelo :class:`WindowGenerator`.
        path: destino do Parquet (ex.: ``data/interim/windows_index.parquet``).

    Returns:
        ``Path`` do arquivo escrito.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "window_id": [w.window_id for w in windows],
            "video_id": [w.video_id for w in windows],
            "participant_id": [w.participant_id for w in windows],
            "t0": [float(w.t0) for w in windows],
            "t1": [float(w.t1) for w in windows],
            "text": [w.text for w in windows],
            # -1 quando label é None (test) → mapeado de volta para None na leitura.
            "label": [int(w.label) if w.label is not None else -1 for w in windows],
            "video_label": [
                int(w.video_label) if w.video_label is not None else -1 for w in windows
            ],
            "question_type": [w.question_type for w in windows],
            "split": [w.split for w in windows],
            "meta": [json.dumps(w.meta or {}, ensure_ascii=False) for w in windows],
        },
        schema_overrides={
            "t0": pl.Float32,
            "t1": pl.Float32,
            "label": pl.Int8,
            "video_label": pl.Int8,
        },
    )
    df.write_parquet(path)
    log.info(f"Índice de janelas salvo: {path} ({df.height} janelas)")
    return path


def load_window_index(path: str | Path) -> list[WindowSample]:
    """Lê o Parquet de índice e reconstrói ``list[WindowSample]``.

    Inversa de :func:`save_window_index`: desserializa ``meta`` (JSON) e mapeia
    ``label == -1`` de volta para ``None`` (janelas de test sem rótulo).

    Args:
        path: caminho do Parquet de índice (escrito por :func:`save_window_index`).

    Returns:
        ``list[WindowSample]`` na ordem do arquivo.
    """
    df = pl.read_parquet(path)
    windows: list[WindowSample] = []
    for row in df.iter_rows(named=True):
        label = row["label"]
        vlabel = row["video_label"]
        windows.append(
            WindowSample(
                window_id=row["window_id"],
                video_id=row["video_id"],
                participant_id=row["participant_id"],
                t0=float(row["t0"]),
                t1=float(row["t1"]),
                text=row["text"] or "",
                label=None if label is None or int(label) < 0 else int(label),
                question_type=row["question_type"],
                split=row["split"],
                video_label=None if vlabel is None or int(vlabel) < 0 else int(vlabel),
                meta=json.loads(row["meta"]) if row["meta"] else {},
            )
        )
    log.info(f"Índice de janelas carregado: {path} ({len(windows)} janelas)")
    return windows
