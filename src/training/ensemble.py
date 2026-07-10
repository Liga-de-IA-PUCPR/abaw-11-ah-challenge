"""EnsembleTrainer — média de probas de N checkpoints neurais (janela→vídeo).

Motivação (ver diagnóstico da cross-attention): o modelo satura em ~2 épocas numa val
pequena (124 vídeos) e a variância ENTRE seeds é grande (AP 0.857→0.874). Mediar as
probas de vários seeds dissolve essa variância e cruza o teto do single-model (F1 test
0.71 → 0.725+).

Reusa o ``LightningTrainer`` inteiro: herda evaluate/predict/video_outputs/calibração e
só sobrescreve ``_infer`` para MEDIAR as probas por vídeo dos membros. O limiar do
ensemble é calibrado nas probas MÉDIAS da val (os limiares individuais não valem para a
média) — o ``main`` chama ``recalibrate_on_val`` antes de avaliar/submeter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.logger import get_logger
from src.training.lightning_trainer import LightningTrainer

log = get_logger("training.ensemble")


class EnsembleTrainer(LightningTrainer):
    """Ensemble por média de probas sobre ``members`` (``LightningTrainer`` já carregados).

    Contrato idêntico ao ``LightningTrainer`` (evaluate/predict/video_outputs), então o
    ``main`` o trata de forma uniforme. Só ``_infer`` muda: roda cada membro e devolve a
    média das probas alinhadas por ``video_id``.
    """

    def __init__(self, members: list[LightningTrainer], config: Any) -> None:
        if not members:
            raise ValueError("Ensemble vazio: forneça >= 1 checkpoint.")
        self.members = members
        self.config = config
        self.results: dict[str, Any] = {}
        # Fallback antes de recalibrar: média dos limiares individuais (o main recalibra
        # nas probas médias da val, que é o correto para a média).
        thrs = [m.threshold_ for m in members if m.threshold_ is not None]
        self.threshold_: float | None = float(np.mean(thrs)) if thrs else 0.5

    def _infer(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Média das probas por vídeo entre os membros (alinhadas por ``video_id``)."""
        ref_ids: np.ndarray | None = None
        acc: np.ndarray | None = None
        for i, m in enumerate(self.members):
            ids, proba = m._infer(loader)
            order = np.argsort(ids.astype(str))
            ids_sorted, proba_sorted = ids[order], proba[order].astype(np.float64)
            if ref_ids is None:
                ref_ids, acc = ids_sorted, proba_sorted.copy()
            else:
                if not np.array_equal(ref_ids, ids_sorted):
                    raise ValueError(
                        f"Ensemble: membro {i} tem conjunto de video_id diferente do 1º "
                        "membro — os checkpoints devem cobrir o MESMO split."
                    )
                acc += proba_sorted
        assert ref_ids is not None and acc is not None
        return ref_ids, (acc / len(self.members)).astype(np.float32)
