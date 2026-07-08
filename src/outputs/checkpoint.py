"""Resolução de diretórios de checkpoint do pipeline BAH — DUAS famílias.

Os artefatos em si são salvos pelos trainers (FASE 4):

- ``sklearn``   → ``SklearnTrainer.save``: ``model.joblib`` + ``trainer_state.json``.
- ``lightning`` → ``ModelCheckpoint`` do Lightning: ``[checkpoints/]*.ckpt``.

Layout em disco:
    outputs/{model_name}/{timestamp}/
    ├── model.joblib + trainer_state.json   (sklearn)   ── OU ──   checkpoints/best-*.ckpt
    ├── train_result.json                   (métricas de val do treino, FASE 6)
    └── eval_<split>/                       (Reporter: metrics.json, results.txt, plots)

``resolve_output_dir`` cria o caminho do run novo; ``resolve_latest_checkpoint``
localiza o artefato mais recente de qualquer família — usados pela CLI (FASE 6)
em train/evaluate/submit.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from src.logger import get_logger

log = get_logger("outputs.checkpoint")


def resolve_output_dir(
    output_root: str | Path,
    model_name: str,
    timestamp: str | None = None,
) -> Path:
    """Resolve ``outputs/{model_name}/{timestamp}/`` (não cria o diretório).

    Args:
        output_root: Raiz dos outputs (ex.: "outputs").
        model_name: Nome do modelo (``cfg.model.name``).
        timestamp: Timestamp fixo; se None, gera ``YYYYMMDD_HHMMSS_ffffff_<pid>`` (UTC).

    ⚠️ Granularidade de segundo (``%Y%m%d_%H%M%S``) colide em multirun paralelo
    (``hydra.launcher.n_jobs>1``): modelos rápidos (ex.: CatBoost com poucos
    estimators) podem terminar dentro do MESMO segundo, e dois jobs escrevem no
    MESMO diretório — um `train_result.json`/`model.joblib` sobrescreve o outro
    SILENCIOSAMENTE (sem erro, sem warning). Isso já aconteceu em
    `+sweep=catboost_stage3` (2 de 16 combinações perdidas). Por isso o timestamp
    agora inclui microssegundos + PID, o que torna a colisão praticamente
    impossível mesmo com dezenas de workers paralelos.
    """
    ts = timestamp or f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    return Path(output_root) / model_name / ts


def resolve_latest_checkpoint(
    output_root: str | Path,
    family: str | None = None,
    model_name: str | None = None,
) -> Path:
    """Localiza o artefato treinado mais recente sob ``output_root``.

    Reconhece as duas famílias: ``{model_name}/*/model.joblib`` (sklearn) e
    ``{model_name}/*/[checkpoints/]*.ckpt`` (Lightning). Quando ``family`` é dado,
    considera SÓ os artefatos daquela família. Quando ``model_name`` é dado, restringe
    ao subdiretório do modelo — assim ``evaluate model=xgboost`` nunca cai num run
    de ``random_forest`` mais recente (e vice-versa).

    Args:
        output_root: Raiz dos outputs (ex.: "outputs").
        family: ``"sklearn"`` | ``"lightning"`` p/ filtrar; ``None`` = qualquer uma.
        model_name: Nome do modelo (``cfg.model.name``) p/ filtrar; ``None`` = qualquer um.

    Returns:
        ``Path`` do **diretório do run** mais recente (o que ``load_trainer`` espera como
        ``out_dir``): contém ``model.joblib`` (sklearn) ou ``*.ckpt`` (Lightning).

    Raises:
        FileNotFoundError: Se nenhum artefato (da família/modelo pedido) existir sob
            ``output_root``.
    """
    root = Path(output_root)
    want_sklearn = family in (None, "sklearn")
    want_neural = family in (None, "lightning")
    # Prefixo do glob: filtra pelo subdiretório do modelo quando model_name é fornecido.
    # Sem isso, evaluate/submit com model=xgboost poderia resolver o checkpoint mais
    # recente de random_forest (outro modelo sklearn mais novo).
    prefix = model_name if model_name else "*"
    # (mtime, run_dir) por artefato reconhecido — run_dir é o passado a load_trainer.
    found: list[tuple[float, Path]] = []
    if want_sklearn:
        for f in root.glob(f"{prefix}/*/model.joblib"):  # sklearn (SklearnTrainer.save)
            found.append((f.stat().st_mtime, f.parent))
    if want_neural:
        for f in root.glob(f"{prefix}/*/*.ckpt"):  # Lightning (ckpt na raiz do run)
            found.append((f.stat().st_mtime, f.parent))
        for f in root.glob(f"{prefix}/*/checkpoints/*.ckpt"):  # Lightning (subpasta checkpoints/)
            found.append((f.stat().st_mtime, f.parent.parent))
    if not found:
        tag = f"model={model_name or 'qualquer'}, family={family or 'qualquer'}"
        raise FileNotFoundError(
            f"Nenhum checkpoint ({tag}: model.joblib | *.ckpt) "
            f"encontrado sob {root}. Rode 'python main.py model={model_name or 'random_forest'}' "
            "antes, ou passe checkpoint=<path>."
        )
    run_dir = max(found, key=lambda x: x[0])[1]
    log.info(
        f"Checkpoint mais recente (model={model_name or 'qualquer'}, "
        f"family={family or 'qualquer'}): {run_dir}"
    )
    return run_dir
