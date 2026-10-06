"""``mode=featurize_columns`` — grava colunas extras no Parquet de janelas existente.

Fluxo (ver :mod:`src.features.columns`):

    1. Lê o Parquet (``data.paths.parquet_path``) e o índice de janelas do ``preprocess``
       (``<interim_dir>/windows_index.parquet``).
    2. Instancia os featurizers de ``columns=[...]``; pula as colunas que já existem
       (``data.force_columns=true`` recomputa).
    3. Monta o índice de vídeos só se algum featurizer precisar (transcrição/chunks/.mp4).
    4. Junta as colunas novas por janela (``id, t0, t1``) e reescreve o Parquet de forma
       ATÔMICA (:func:`merge_window_columns`, também usada por ``featurize_face``), sob uma
       trava de arquivo — dá p/ rodar colunas diferentes em paralelo (ex.: uma por GPU).

Ex.: ``python main.py mode=featurize_columns "columns=[transcript,asr_timing,hesitation_markers]"``
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import DictConfig

from src.data.schema import WindowSample
from src.data.windowing import load_window_index
from src.logger import get_logger

log = get_logger("pipeline.featurize_columns")

_KEYS = ["id", "t0", "t1"]


def run_featurize_columns(cfg: DictConfig) -> dict[str, Any]:
    """Calcula as colunas pedidas em ``cfg.columns`` e as grava no Parquet de janelas."""
    from src.features.columns import build_column_featurizer

    parquet_path = Path(cfg.data.paths.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet ausente: {parquet_path}. Rode 'mode=featurize' antes.")
    names = list(cfg.get("columns") or [])
    if not names:
        raise ValueError("Nenhuma coluna pedida: use columns=[transcript,asr_timing,...].")

    existing = set(pl.read_parquet_schema(parquet_path))
    force = bool(cfg.data.get("force_columns", False))
    featurizers = [build_column_featurizer(n, cfg) for n in names]
    todo = [f for f in featurizers if force or f.column not in existing]
    for f in featurizers:
        if f not in todo:
            log.info(f"Coluna '{f.column}' já existe (data.force_columns=true recomputa).")
    if not todo:
        return {"parquet_path": str(parquet_path), "columns": [], "cached": True}

    windows = load_window_index(Path(cfg.data.paths.interim_dir) / "windows_index.parquet")
    records = None
    if any(f.needs_records for f in todo):
        from src.data.indexing import build_video_index

        records = {r.video_id: r for r in build_video_index(cfg)}

    values: dict[str, list[Any]] = {}
    names_by_col: dict[str, list[str]] = {}
    for f in todo:
        log.info(f"Calculando a coluna '{f.column}' ({len(windows)} janelas)...")
        values[f.column] = f.compute(windows, records)
        names_by_col[f.column] = f.feature_names()
    merge_window_columns(parquet_path, windows, values)
    _update_sidecar(parquet_path, names_by_col)
    return {"parquet_path": str(parquet_path), "columns": list(values), "n_windows": len(windows)}


def merge_window_columns(
    parquet_path: str | Path, windows: list[WindowSample], values: dict[str, list[Any]]
) -> pl.DataFrame:
    """Junta ``values`` (1 valor por janela de ``windows``) ao Parquet e o reescreve atômico.

    A junção é por ``(id, t0, t1)`` — independente da ordem das linhas. Janelas do Parquet
    sem valor (fora do índice) recebem zeros (vetores) ou ``""`` (texto), com aviso. Colunas
    homônimas já presentes são substituídas. Leitura → junção → escrita acontecem sob
    :func:`parquet_lock`, então processos que gravam colunas diferentes não se sobrescrevem.
    """
    parquet_path = Path(parquet_path)
    with parquet_lock(parquet_path):
        return _merge_locked(parquet_path, windows, values)


def _merge_locked(
    parquet_path: Path, windows: list[WindowSample], values: dict[str, list[Any]]
) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    keys = pl.DataFrame(
        {
            "id": [w.video_id for w in windows],
            "t0": [float(w.t0) for w in windows],
            "t1": [float(w.t1) for w in windows],
            **values,
        }
    ).with_columns(pl.col("t0").cast(df.schema["t0"]), pl.col("t1").cast(df.schema["t1"]))
    merged = df.drop([c for c in values if c in df.columns]).join(keys, on=_KEYS, how="left")
    for col, col_values in values.items():
        n_miss = merged[col].null_count()
        if not n_miss:
            continue
        log.warning(f"{n_miss} janelas sem '{col}' após o join — preenchendo com vazio/zeros.")
        sample = next(v for v in col_values if v is not None)
        fill = "" if isinstance(sample, str) else [0.0] * len(sample)
        merged = merged.with_columns(pl.col(col).fill_null(pl.lit(fill)))

    # Escrita atômica: o Parquet base (caro de recomputar) só é trocado com o novo completo.
    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    merged.write_parquet(tmp_path)
    tmp_path.replace(parquet_path)
    log.info(f"Colunas {list(values)} gravadas em {parquet_path}.")
    return merged


def _update_sidecar(parquet_path: Path, names_by_col: dict[str, list[str]]) -> None:
    """Registra nomes/dims das colunas novas no sidecar JSON do Parquet (se existir)."""
    sidecar = parquet_path.with_suffix(".json")
    if not sidecar.exists():
        return
    with parquet_lock(parquet_path):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        for col, names in names_by_col.items():
            if names:
                meta.setdefault("feature_names", {})[col] = names
                meta.setdefault("dims", {})[f"d_{col}"] = len(names)
        sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def parquet_lock(parquet_path: Path) -> Iterator[None]:
    """Trava exclusiva entre PROCESSOS sobre o Parquet (arquivo ``<parquet>.lock``).

    Vários ``featurize_columns`` em paralelo (ex.: áudio numa GPU, rosto na outra) gravando
    colunas diferentes no mesmo Parquet: cada um relê o arquivo DENTRO da trava antes de
    juntar a sua coluna. Sem ``fcntl`` (Windows) a trava não existe — rode em sequência.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — POSIX only
        yield
        return
    with open(Path(parquet_path).with_suffix(".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
