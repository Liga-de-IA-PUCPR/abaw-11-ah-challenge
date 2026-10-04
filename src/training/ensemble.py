"""EnsembleTrainer — média de probas por vídeo de N checkpoints (janela→vídeo).

Motivação (ver diagnóstico da cross-attention): o modelo satura em ~2 épocas numa val
pequena (124 vídeos) e a variância ENTRE seeds é grande (AP 0.857→0.874). Mediar as
probas de vários seeds dissolve essa variância e cruza o teto do single-model (F1 test
0.71 → 0.725+).

Reusa o ``LightningTrainer`` inteiro: herda evaluate/predict/video_outputs/calibração e
só sobrescreve ``_infer`` para MEDIAR as probas por vídeo dos membros. O limiar do
ensemble é calibrado nas probas MÉDIAS da val (os limiares individuais não valem para a
média) — o ``main`` chama ``recalibrate_on_val`` antes de avaliar/submeter.

Membros HETEROGÊNEOS (frente "áudio + texto + vídeo"): cada membro é um trainer já
carregado do seu próprio run dir — pode ser cross-attention, GNN heterogêneo, GNN facial
ou um modelo sklearn (RF). O ``loader`` recebido define QUAIS vídeos entram; um
membro com ``parquet_path`` próprio (ex.: GNN treinado no cache wav2vec2) lê as SUAS
features para esses mesmos vídeos. Sem pesos, a combinação é a média simples de sempre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger
from src.training.lightning_trainer import LightningTrainer

log = get_logger("training.ensemble")


@dataclass
class EnsembleMember:
    """Um membro do ensemble: trainer carregado + de onde ele lê as features.

    Attributes:
        trainer: ``LightningTrainer`` ou ``SklearnTrainer`` já carregado (``load_trainer``).
        name: rótulo p/ log (run dir).
        family: ``"lightning"`` | ``"sklearn"``.
        parquet_path: Parquet das features deste membro nos splits de PREDIÇÃO
            (``None`` = o mesmo do ``loader`` recebido).
        calib_parquet_path: Parquet deste membro no split de CALIBRAÇÃO
            (``None`` = ``parquet_path``).
        weight: peso na combinação (só usado se algum membro tiver peso != 1).
    """

    trainer: Any
    name: str
    family: str = "lightning"
    parquet_path: str | None = None
    calib_parquet_path: str | None = None
    weight: float = 1.0


class EnsembleTrainer(LightningTrainer):
    """Ensemble por média (ou média ponderada) de probas sobre ``members``.

    Contrato idêntico ao ``LightningTrainer`` (evaluate/predict/video_outputs), então o
    ``main`` o trata de forma uniforme. Só ``_infer`` muda: roda cada membro (no Parquet
    dele, se diferente) e devolve a combinação das probas alinhadas por ``video_id``.
    """

    def __init__(self, members: list[Any], config: Any) -> None:
        if not members:
            raise ValueError("Ensemble vazio: forneça >= 1 checkpoint.")
        # Aceita trainers "crus" (compat.: ensemble homogêneo antigo) ou EnsembleMember.
        self.members: list[EnsembleMember] = [
            m if isinstance(m, EnsembleMember) else EnsembleMember(trainer=m, name=f"m{i}")
            for i, m in enumerate(members)
        ]
        self.config = config
        self.results: dict[str, Any] = {}
        self._role = "predict"  # "calib" durante recalibrate_on_val (escolhe o Parquet)
        # Fallback antes de recalibrar: média dos limiares individuais (o main recalibra
        # nas probas médias da val, que é o correto para a média).
        thrs = [m.trainer.threshold_ for m in self.members if m.trainer.threshold_ is not None]
        self.threshold_: float | None = float(np.mean(thrs)) if thrs else 0.5
        self._weighted = any(float(m.weight) != 1.0 for m in self.members)

    def recalibrate_on_val(self, val_loader) -> float:
        """Recalibra o limiar nas probas combinadas (membros leem o Parquet de calibração)."""
        self._role = "calib"
        try:
            return super().recalibrate_on_val(val_loader)
        finally:
            self._role = "predict"

    def predict_outputs(self, loader) -> dict[str, np.ndarray]:
        """Só ``video_ids``/``proba`` combinados (membros heterogêneos não têm um embedding
        comum — o MoERouter combina embeddings a partir dos dumps de cada membro)."""
        ids, proba = self._infer(loader)
        return {"video_ids": ids, "proba": proba}

    def _infer(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Combinação das probas por vídeo entre os membros (alinhadas por ``video_id``)."""
        ref_ids: np.ndarray | None = None
        acc: np.ndarray | None = None
        w_sum = 0.0
        for i, m in enumerate(self.members):
            ids, proba = self._member_infer(m, loader)
            order = np.argsort(ids.astype(str))
            ids_sorted, proba_sorted = ids[order], proba[order].astype(np.float64)
            w = float(m.weight) if self._weighted else 1.0
            if ref_ids is None:
                ref_ids, acc = (
                    ids_sorted,
                    w * proba_sorted if self._weighted else proba_sorted.copy(),
                )
            else:
                if not np.array_equal(ref_ids, ids_sorted):
                    raise ValueError(
                        f"Ensemble: membro {i} ({m.name}) tem conjunto de video_id diferente do "
                        "1º membro — os checkpoints devem cobrir o MESMO split (confira o "
                        "parquet_path do membro)."
                    )
                acc += w * proba_sorted if self._weighted else proba_sorted
            w_sum += w
        assert ref_ids is not None and acc is not None
        denom = w_sum if self._weighted else len(self.members)
        return ref_ids, (acc / denom).astype(np.float32)

    # ------------------------------------------------------------------------------
    # Inferência de UM membro no split do loader (no Parquet dele)
    # ------------------------------------------------------------------------------

    def _member_parquet(self, m: EnsembleMember) -> str | None:
        if self._role == "calib" and m.calib_parquet_path:
            return m.calib_parquet_path
        return m.parquet_path

    def _member_infer(self, m: EnsembleMember, loader) -> tuple[np.ndarray, np.ndarray]:
        ds = loader.dataset
        video_ids = {str(v) for v in ds.video_ids}
        own_pq = self._member_parquet(m)
        src_pq = Path(str(getattr(ds, "parquet_path", "")))
        use_pq = Path(own_pq) if own_pq else src_pq
        spec = _member_spec(m)

        if m.family == "sklearn":
            from src.data.datasets import WindowMatrixView

            scores = m.trainer.predict_scores(WindowMatrixView(use_pq, video_ids))
            ids = np.asarray(list(scores.keys()))
            return ids, np.asarray([scores[v] for v in ids], dtype=np.float32)

        other_pq = own_pq and use_pq.resolve() != src_pq.resolve()
        if other_pq or not _dataset_has(ds, spec):
            return m.trainer._infer(self._loader_for(use_pq, video_ids, loader, spec))
        return m.trainer._infer(loader)

    @staticmethod
    def _loader_for(parquet_path: Path, video_ids: set[str], like_loader, spec: dict | None = None):
        """DataLoader (sem shuffle) sobre ``parquet_path`` restrito a ``video_ids``, com as
        entradas extras que o membro declara (``spec``: colunas/transcrição)."""
        from torch.utils.data import DataLoader

        from src.data.datasets import VideoSequenceDataset, collate_sequences

        return DataLoader(
            VideoSequenceDataset(parquet_path, video_ids, **(spec or {})),
            batch_size=like_loader.batch_size or 32,
            shuffle=False,
            num_workers=like_loader.num_workers,
            collate_fn=collate_sequences,
        )


def _member_spec(m: EnsembleMember) -> dict:
    """Entradas extras do modelo do membro (``data_spec``), lidas do cfg do seu trainer."""
    cfg = getattr(m.trainer, "config", None)
    model_cfg = getattr(cfg, "model", None) if cfg is not None else None
    if model_cfg is None:
        return {}
    from src.models.registry import data_spec

    return data_spec(model_cfg)


def _dataset_has(ds, spec: dict) -> bool:
    """O dataset do loader já traz o que o membro pede? (colunas extras + transcrição)."""
    have = set(getattr(ds, "columns", []) or [])
    has_text = bool(getattr(ds, "_tokens", None)) or not spec.get("transcript")
    return set(spec.get("columns") or []) <= have and has_text
