"""Carregamento dos splits participant-wise do BAH (sem vazamento de participante).

Lê ``<data_root>/split/{train,val,test}.txt`` — cada linha:
``video_id, classe_do_video, transcrição_completa``. O ``participant_id`` é o
primeiro componente do caminho do ``video_id``. Aborta se um participante aparecer
em mais de um split. Oferece ``GroupKFold`` por participante para CV interna.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from src.logger import get_logger

log = get_logger("training.splits")


@dataclass
class SplitTable:
    """Tabela de um split: vídeos, rótulos a nível de vídeo e participantes."""

    split: str
    video_ids: list[str] = field(default_factory=list)
    video_labels: dict[str, int | None] = field(default_factory=dict)
    participant_ids: dict[str, str] = field(default_factory=dict)

    @property
    def participants(self) -> set[str]:
        """Conjunto de participantes presentes neste split."""
        return set(self.participant_ids.values())


def _parse_participant_id(video_id: str) -> str:
    """Deriva o ``participant_id`` (primeiro componente do caminho do ``video_id``)."""
    norm = video_id.replace("\\", "/").strip().lstrip("/")
    head = norm.split("/", 1)[0]
    return head if head else norm


def _parse_split_line(line: str) -> tuple[str, int | None] | None:
    """Parseia uma linha de split/*.txt -> ``(video_id, label)``.

    Só faz split nos **dois primeiros** campos (a transcrição contém vírgulas).
    No test a classe pode estar vazia -> ``label=None``.
    """
    line = line.rstrip("\n")
    if not line.strip():
        return None
    parts = line.split(",", 2)
    video_id = parts[0].strip()
    if not video_id:
        return None
    label: int | None = None
    if len(parts) >= 2 and parts[1].strip() != "":
        try:
            label = int(float(parts[1].strip()))
        except ValueError:
            label = None
    return video_id, label


def load_splits(
    split_dir: str | Path,
    files: dict[str, str] | None = None,
) -> dict[str, SplitTable]:
    """Carrega os splits participant-wise e valida ausência de vazamento.

    Args:
        split_dir: diretório com os .txt (tipicamente ``<data_root>/split``).
        files: mapeamento split -> arquivo. Default: train/val/test.txt.

    Raises:
        FileNotFoundError: arquivo de split ausente.
        ValueError: participante em mais de um split (vazamento).
    """
    split_dir = Path(split_dir)
    files = files or {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

    tables: dict[str, SplitTable] = {}
    for split, fname in files.items():
        path = split_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de split ausente: {path}")

        table = SplitTable(split=split)
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_split_line(raw)
            if parsed is None:
                continue
            video_id, label = parsed
            table.video_ids.append(video_id)
            table.video_labels[video_id] = label
            table.participant_ids[video_id] = _parse_participant_id(video_id)

        tables[split] = table
        n_pos = sum(1 for v in table.video_labels.values() if v == 1)
        log.info(
            f"Split '{split}': {len(table.video_ids)} vídeos, "
            f"{len(table.participants)} participantes, positivos={n_pos}"
        )

    _assert_no_participant_leakage(tables)
    return tables


def _assert_no_participant_leakage(tables: dict[str, SplitTable]) -> None:
    """Garante que nenhum participante apareça em mais de um split."""
    seen: dict[str, str] = {}
    for split, table in tables.items():
        for pid in table.participants:
            if pid in seen and seen[pid] != split:
                raise ValueError(
                    f"Vazamento de participante: '{pid}' em '{seen[pid]}' e '{split}'"
                )
            seen[pid] = split
    log.info("Verificação OK: nenhum vazamento de participante entre splits.")


def group_kfold_indices(
    video_ids: np.ndarray,
    participant_ids: np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds de ``GroupKFold`` agrupados por participante (CV interna sem vazamento)."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(len(video_ids))
    folds = [
        (tr, va) for tr, va in gkf.split(video_ids, dummy_y, groups=participant_ids)
    ]
    log.info(f"GroupKFold gerado: {len(folds)} folds, agrupado por participante.")
    return folds
