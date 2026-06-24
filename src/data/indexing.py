"""Indexação do dataset BAH → list[VideoRecord].

Fontes (sob cfg.data.paths.data_root, default 'data/raw/data'; ver README §2):
- split/{train,val,test}.txt        : 'video_id, classe_do_video, transcrição_completa'
                                       (o split vem do NOME do arquivo .txt)
- transcription/<video_id>.json     : JSON Whisper com chunks {language, text, timestamp:(s,e)}
- video_annotation_transcript.yaml  : dict por video_id com global_ah, time_detailed_ah,
                                       certainty_ah, all_cues, frame_annotation, ...
- meta_data.yml                     : dict por participante (idade, país, gênero, ...)

O nome do arquivo de vídeo carrega participante e pergunta:
    <pid>_Question_<q>_<ts>_Video.mp4
de onde derivamos ``participant_id``, ``question_id`` (1..7) e ``question_type``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import DictConfig

from src.data.schema import VideoRecord
from src.logger import get_logger

log = get_logger("data.indexing")


# ==============================================================================
# Constantes
# ==============================================================================

# As 7 perguntas do BAH (o tipo importa como feature tabular — ver README §2).
QUESTION_TYPE_MAP: dict[int, str] = {
    1: "neutral",
    2: "positive",
    3: "negative",
    4: "ambivalent",
    5: "willing",
    6: "resistant",
    7: "hesitant",
}

# Mapa arquivo de split → rótulo de split (o split vem do NOME do .txt).
_SPLIT_FILES: dict[str, Literal["train", "val", "test"]] = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}

# <pid>_Question_<q>_..._Video.mp4
_VIDEO_NAME_RE = re.compile(r"^(?P<pid>[^_]+)_Question_(?P<q>\d+)_.*?Video", re.IGNORECASE)


# ==============================================================================
# Helpers de nome de arquivo
# ==============================================================================


def parse_video_filename(video_id: str) -> tuple[str, int, str]:
    """Deriva ``(participant_id, question_id, question_type)`` do nome do vídeo.

    Espera nomes ``<pid>_Question_<q>_<ts>_Video.mp4`` (case-insensitive).
    ``video_id`` pode ser caminho relativo; usamos apenas o nome do arquivo.

    Returns:
        ``(participant_id, question_id, question_type)``; se não casar, emite warning
        e retorna ``("unknown", 0, "unknown")``.
    """
    name = Path(video_id).name
    match = _VIDEO_NAME_RE.match(name)
    if match is None:
        log.warning(f"Nome de vídeo fora do padrão esperado: {name!r}")
        return "unknown", 0, "unknown"

    pid = match.group("pid")
    qid = int(match.group("q"))
    qtype = QUESTION_TYPE_MAP.get(qid, "unknown")
    return pid, qid, qtype


# ==============================================================================
# Parsers de cada fonte
# ==============================================================================


def parse_split_file(
    path: Path,
    split: Literal["train", "val", "test"],
) -> list[dict[str, Any]]:
    """Lê um arquivo ``split/*.txt``.

    Cada linha: ``video_id, classe_do_video, transcrição_completa``. A classe pode
    estar ausente no test (sem rótulos) → ``global_ah = None``. A transcrição pode
    conter vírgulas, então fazemos *split* só nos dois primeiros separadores.

    Returns:
        Lista de dicts ``{"video_id", "global_ah", "full_transcript", "split"}``.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        log.warning(f"Arquivo de split não encontrado: {path}")
        return rows

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", maxsplit=2)]
            video_id = parts[0]
            global_ah: int | None = None
            full_transcript = ""
            if len(parts) >= 2 and parts[1] != "":
                try:
                    global_ah = int(parts[1])
                except ValueError:
                    log.debug(f"Classe não-inteira em {path.name}: {parts[1]!r}")
            if len(parts) >= 3:
                full_transcript = parts[2]
            rows.append(
                {
                    "video_id": video_id,
                    "global_ah": None if split == "test" else global_ah,
                    "full_transcript": full_transcript,
                    "split": split,
                }
            )

    log.info(f"{path.name}: {len(rows)} vídeos (split={split})")
    return rows


def load_annotation_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Carrega ``video_annotation_transcript.yaml`` (dict por video_id).

    Extrai por vídeo: ``global_ah``, ``time_detailed_ah``, ``certainty_ah``,
    ``all_cues`` e ``frame_annotation``. Usa ``yaml.safe_load``.
    """
    if not path.exists():
        log.warning(f"Anotações não encontradas: {path}")
        return {}

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    out: dict[str, dict[str, Any]] = {}
    for video_id, ann in raw.items():
        ann = ann or {}
        out[video_id] = {
            "global_ah": _as_opt_int(ann.get("global_ah")),
            "time_detailed_ah": _normalize_intervals(ann.get("time_detailed_ah")),
            "certainty_ah": _as_int_list(ann.get("certainty_ah")),
            "all_cues": _as_dict_list(ann.get("all_cues")),
            "frame_annotation": ann.get("frame_annotation"),
        }

    log.info(f"{path.name}: anotações de {len(out)} vídeos")
    return out


def load_meta_data(path: Path) -> dict[str, dict[str, Any]]:
    """Carrega ``meta_data.yml`` (dict por participante)."""
    if not path.exists():
        log.warning(f"meta_data.yml não encontrado: {path}")
        return {}

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    meta = {str(pid): (info or {}) for pid, info in raw.items()}
    log.info(f"{path.name}: metadados de {len(meta)} participantes")
    return meta


def load_transcript_chunks(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Lê o JSON de transcrição Whisper de UM vídeo.

    Estrutura esperada: ``{"text": str, "chunks": [{"language", "text",
    "timestamp": [start, end]}, ...]}``. Normaliza cada chunk para
    ``{"start": float, "end": float, "text": str, "language": str}``.

    Returns:
        ``(chunks, full_text)``; ``([], "")`` se o arquivo não existir.
    """
    if not path.exists():
        log.debug(f"Transcrição ausente: {path}")
        return [], ""

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    full_text = str(data.get("text", "") or "")
    chunks: list[dict[str, Any]] = []
    for ch in data.get("chunks", []) or []:
        ts = ch.get("timestamp") or [None, None]
        start = ts[0]
        end = ts[1] if len(ts) > 1 else None
        if start is None:
            continue
        chunks.append(
            {
                "start": float(start),
                # Último chunk às vezes vem com end=None no Whisper → usa o start.
                "end": float(end) if end is not None else float(start),
                "text": str(ch.get("text", "") or "").strip(),
                "language": str(ch.get("language", "") or ""),
            }
        )
    return chunks, full_text


# ==============================================================================
# Construtor do índice
# ==============================================================================


def build_video_index(cfg: DictConfig) -> list[VideoRecord]:
    """Constrói a lista canônica de ``VideoRecord`` a partir de ``cfg.data.paths.data_root``.

    Ordem de montagem (por vídeo):

    1. Para cada split em {train, val, test}, lê ``split/<split>.txt`` → linhas com
       ``video_id``, ``global_ah`` (None no test) e ``full_transcript``.
    2. Deriva ``participant_id`` / ``question_id`` / ``question_type`` do nome.
    3. Anexa anotações de ``video_annotation_transcript.yaml``; o ``global_ah`` do YAML
       serve de fallback ao do split (exceto no test).
    4. Anexa chunks de transcrição (``transcription/<video_id>.json``).
    5. Anexa metadados do participante (``meta_data.yml``).
    6. Resolve caminhos de mídia (``video_path``; ``audio_path`` fica ``None`` até a
       extração — FASE 6/preprocess).

    Args:
        cfg: config Hydra composto (usa ``cfg.data.paths.data_root``; ver README §7).

    Returns:
        ``list[VideoRecord]`` (todos os splits concatenados).
    """
    raw_root = Path(cfg.data.paths.data_root)  # default: data/raw/data
    split_dir = raw_root / "split"
    transcript_dir = raw_root / "transcription"

    annotations = load_annotation_yaml(raw_root / "video_annotation_transcript.yaml")
    meta_by_participant = load_meta_data(raw_root / "meta_data.yml")

    records: list[VideoRecord] = []
    for fname, split in _SPLIT_FILES.items():
        rows = parse_split_file(split_dir / f"{fname}.txt", split)
        for row in rows:
            video_id = row["video_id"]
            pid, qid, qtype = parse_video_filename(video_id)

            ann = annotations.get(video_id, {})
            # global_ah: split manda; YAML é fallback. No test, sempre None.
            global_ah = row["global_ah"]
            if global_ah is None and split != "test":
                global_ah = ann.get("global_ah")

            chunks, full_from_json = load_transcript_chunks(transcript_dir / f"{video_id}.json")
            full_transcript = row["full_transcript"] or full_from_json

            record = VideoRecord(
                video_id=video_id,
                participant_id=pid,
                question_id=qid,
                question_type=qtype,
                split=split,
                video_path=(raw_root / "Videos" / video_id),
                audio_path=None,  # preenchido após extract_audio (FASE 6/preprocess)
                duration_s=_infer_duration(chunks, ann),
                transcript_chunks=chunks,
                full_transcript=full_transcript,
                global_ah=global_ah,
                time_detailed_ah=ann.get("time_detailed_ah", []),
                certainty_ah=ann.get("certainty_ah", []),
                all_cues=ann.get("all_cues", []),
                meta=meta_by_participant.get(pid, {}),
            )
            records.append(record)

    log.info(
        f"Índice construído: {len(records)} vídeos "
        f"(train={sum(r.split == 'train' for r in records)}, "
        f"val={sum(r.split == 'val' for r in records)}, "
        f"test={sum(r.split == 'test' for r in records)})"
    )
    return records


# ==============================================================================
# Utilitários internos
# ==============================================================================


def _infer_duration(chunks: list[dict[str, Any]], ann: dict[str, Any]) -> float:
    """Estima a duração do vídeo (s).

    Usa o maior ``end`` dos chunks; cai para o maior fim de ``time_detailed_ah`` se
    não houver chunks. ``0.0`` se nada disponível. A duração real (ffprobe) é
    preenchida em ``audio_io.extract_audio``.
    """
    ends = [c["end"] for c in chunks if c.get("end") is not None]
    if ends:
        return float(max(ends))
    intervals = ann.get("time_detailed_ah", [])
    if intervals:
        return float(max(e for _, e in intervals))
    return 0.0


def _normalize_intervals(value: Any) -> list[tuple[float, float]]:
    """Normaliza ``time_detailed_ah`` para ``list[tuple[float, float]]``."""
    if not value:
        return []
    out: list[tuple[float, float]] = []
    for item in value:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end >= start:
            out.append((start, end))
    return out


def _as_opt_int(value: Any) -> int | None:
    """Converte para int ou None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_int_list(value: Any) -> list[int]:
    """Converte uma lista heterogênea para list[int] (ignora inválidos)."""
    if not value:
        return []
    out: list[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _as_dict_list(value: Any) -> list[dict]:
    """Garante list[dict] (envolve dict solto; descarta não-dicts)."""
    if not value:
        return []
    if isinstance(value, dict):
        return [value]
    return [v for v in value if isinstance(v, dict)]
