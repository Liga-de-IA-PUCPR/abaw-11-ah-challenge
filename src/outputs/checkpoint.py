"""Persistência do pipeline BAH treinado — DUAS famílias.

(a) ``random_forest`` (family="sklearn", CPU) → **bundle joblib** auto-contido:
    {model, scaler, feature_names, threshold, aggregation_method, embedder_cfg, config}.

(b) ``cross_attention`` (family="lightning") → **.ckpt do Lightning** (salvo pelo
    ``ModelCheckpoint`` no ``LightningTrainer``, FASE 4) **+ sidecar JSON** com o que o
    ``.ckpt`` não carrega: o limiar calibrado do sigmoid e a config dos embedders
    (p/ recomputar/casar as features na inferência).

Layout em disco (espelha o padrão da branch ``matheus`` e do visionT):
    outputs/{model_name}/{timestamp}/
    ├── bundle.joblib            (RF)            ── OU ──   checkpoints/best-*.ckpt  (neural)
    ├── sidecar.json             (neural: threshold + embedder_cfg + config)
    ├── metrics.json             (Reporter)
    ├── results.txt              (Reporter)
    ├── plots/                   (Reporter)
    └── submission.txt           (opcional)

``resolve_latest_checkpoint`` localiza o artefato mais recente de QUALQUER família
(``bundle.joblib`` ou ``*.ckpt``) — usado pela CLI (FASE 6) em evaluate/submit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger

log = get_logger("outputs.checkpoint")


# ==============================================================================
# (a) RandomForest — bundle joblib
# ==============================================================================


@dataclass
class CheckpointBundle:
    """Pacote auto-contido do pipeline ``random_forest`` (janela→agrega→vídeo).

    Serializado inteiro via joblib. Os objetos sklearn (``model``, ``scaler``) são
    picklados junto; ``config`` e ``embedder_cfg`` são dicts puros (legíveis e
    independentes das dataclasses).

    Attributes:
        model: Classificador de janela treinado (implementa ``BaseModel``).
        scaler: ``StandardScaler`` do treino, ou ``None``.
        feature_names: Nomes das colunas de ``X`` (ordem importa; len == d).
        threshold: Limiar calibrado na validação (``aggregation.threshold``).
        aggregation_method: "mean_proba" | "max_proba" | "frac_positive" | "any".
        embedder_cfg: Config dos embedders (text/audio/tabular) — garante que a
            inferência recompute/carregue as features com a MESMA configuração.
        config: Snapshot resolvido da config Hydra (dict).
        metadata: Metadados livres (timestamp, val_metrics, versões).
    """

    model: BaseModel
    scaler: Any | None
    feature_names: list[str]
    threshold: float
    aggregation_method: str
    embedder_cfg: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    family: str = "sklearn"

    # --- inferência centralizada (reaplica o scaler do treino) ---------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica o ``scaler`` do treino (ou retorna X se ``scaler is None``)."""
        if self.scaler is None:
            return np.asarray(X, dtype=np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidade da classe 1 (A/H) por janela; aplica o scaler internamente."""
        return self.model.predict_proba(self.transform(X))

    @classmethod
    def from_training(
        cls,
        model: BaseModel,
        scaler: Any | None,
        feature_names: list[str],
        threshold: float,
        aggregation_method: str,
        embedder_cfg: dict[str, Any],
        config: dict[str, Any],
        val_metrics: dict[str, float] | None = None,
    ) -> CheckpointBundle:
        """Monta o bundle a partir dos artefatos do ``SklearnTrainer`` (FASE 4)."""
        meta: dict[str, Any] = {
            "created_at": datetime.now(tz=UTC).isoformat(),
            "model_name": config.get("model", {}).get("name", "random_forest"),
            "n_features": len(feature_names),
        }
        if val_metrics:
            meta["val_metrics"] = dict(val_metrics)
        return cls(
            model=model,
            scaler=scaler,
            feature_names=list(feature_names),
            threshold=float(threshold),
            aggregation_method=str(aggregation_method),
            embedder_cfg=dict(embedder_cfg),
            config=dict(config),
            metadata=meta,
        )

    def __repr__(self) -> str:
        return (
            f"CheckpointBundle(model={self.metadata.get('model_name', '?')}, "
            f"d={len(self.feature_names)}, agg='{self.aggregation_method}', "
            f"threshold={self.threshold:.3f})"
        )


def save_rf_bundle(bundle: CheckpointBundle, output_dir: str | Path) -> Path:
    """Serializa o bundle RF em ``{output_dir}/bundle.joblib`` (compress=3)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    log.info(f"Bundle RF salvo: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_rf_bundle(path: str | Path) -> CheckpointBundle:
    """Carrega o bundle RF de ``bundle.joblib`` (aceita diretório do run)."""
    p = Path(path)
    if p.is_dir():
        p = p / "bundle.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Bundle não encontrado: {p}")
    bundle: CheckpointBundle = joblib.load(p)
    log.info(f"Bundle RF carregado: {p} -> {bundle!r}")
    return bundle


# ==============================================================================
# (b) cross_attention — sidecar do checkpoint Lightning
# ==============================================================================


@dataclass
class NeuralSidecar:
    """Metadados que o ``.ckpt`` do Lightning NÃO carrega, mas a inferência exige.

    O ``.ckpt`` (salvo pelo ``ModelCheckpoint`` no ``LightningTrainer``, FASE 4) traz
    pesos + ``hyper_parameters`` (lr, dims, num_heads). Este sidecar acrescenta:

    Attributes:
        ckpt_path: Caminho do ``best-*.ckpt`` (relativo ao run dir).
        threshold: Limiar calibrado do sigmoid p/ max Macro-F1 (calibrado na val).
        embedder_cfg: Config dos embedders (text/audio) — casa as dims das features.
        model_cfg: Hiperparâmetros do ``LitCrossAttention`` (dim_a, dim_b, common_dim,
            num_heads) — redundante c/ o ckpt, mas útil p/ reconstruir sem carregá-lo.
        config: Snapshot resolvido da config Hydra.
        metadata: Métricas de validação, timestamp, etc.
        family: "lightning".
    """

    ckpt_path: str
    threshold: float
    embedder_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    family: str = "lightning"


def save_neural_sidecar(
    sidecar: NeuralSidecar,
    output_dir: str | Path,
) -> Path:
    """Grava o sidecar do checkpoint neural em ``{output_dir}/sidecar.json``.

    Args:
        sidecar: Metadados do checkpoint Lightning.
        output_dir: Diretório do run (criado se não existir).

    Returns:
        Caminho do ``sidecar.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sidecar.json"
    payload = {
        "family": sidecar.family,
        "ckpt_path": sidecar.ckpt_path,
        "threshold": float(sidecar.threshold),
        "embedder_cfg": sidecar.embedder_cfg,
        "model_cfg": sidecar.model_cfg,
        "config": sidecar.config,
        "metadata": sidecar.metadata,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    log.info(f"Sidecar neural salvo: {path} (threshold={sidecar.threshold:.3f})")
    return path


def load_neural_sidecar(path: str | Path) -> NeuralSidecar:
    """Carrega o sidecar de ``sidecar.json`` (aceita diretório do run)."""
    p = Path(path)
    if p.is_dir():
        p = p / "sidecar.json"
    if not p.exists():
        raise FileNotFoundError(f"Sidecar não encontrado: {p}")
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    sidecar = NeuralSidecar(
        ckpt_path=d["ckpt_path"],
        threshold=float(d["threshold"]),
        embedder_cfg=d.get("embedder_cfg", {}),
        model_cfg=d.get("model_cfg", {}),
        config=d.get("config", {}),
        metadata=d.get("metadata", {}),
        family=d.get("family", "lightning"),
    )
    log.info(f"Sidecar neural carregado: {p}")
    return sidecar


# ==============================================================================
# I/O comum
# ==============================================================================


def resolve_output_dir(
    output_root: str | Path,
    model_name: str,
    timestamp: str | None = None,
) -> Path:
    """Resolve ``outputs/{model_name}/{timestamp}/`` (não cria o diretório).

    Args:
        output_root: Raiz dos outputs (ex.: "outputs").
        model_name: Nome do modelo (``cfg.model.name``).
        timestamp: Timestamp fixo; se None, gera ``YYYYMMDD_HHMMSS`` (UTC).
    """
    ts = timestamp or datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / model_name / ts


def resolve_latest_checkpoint(output_root: str | Path) -> Path:
    """Localiza o artefato treinado mais recente sob ``output_root``.

    Considera AS DUAS famílias: ``*/*/bundle.joblib`` (RF) e ``*/*/**/*.ckpt``
    (Lightning). Devolve o de modificação mais recente. Usado pela CLI (FASE 6)
    quando o checkpoint é omitido em ``evaluate`` / ``submit``.

    Args:
        output_root: Raiz dos outputs (ex.: "outputs").

    Returns:
        ``Path`` do artefato mais recente (``bundle.joblib`` OU ``.ckpt``).

    Raises:
        FileNotFoundError: Se nenhum artefato existir sob ``output_root``.
    """
    root = Path(output_root)
    candidates = [
        *root.glob("*/*/bundle.joblib"),
        *root.glob("*/*/checkpoints/*.ckpt"),
        *root.glob("*/*/*.ckpt"),
    ]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum checkpoint (bundle.joblib | *.ckpt) encontrado sob {root}. "
            "Rode 'python main.py' (RF) ou '+experiment=cross_attention' antes, "
            "ou passe checkpoint=<path>."
        )
    log.info(f"Checkpoint mais recente: {candidates[0]}")
    return candidates[0]


def _json_default(obj: Any) -> Any:
    """Serializador de fallback p/ numpy no ``json.dump``."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
