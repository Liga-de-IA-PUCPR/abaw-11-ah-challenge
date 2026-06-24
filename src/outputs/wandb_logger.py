"""Tracking Weights & Biases para o caminho sklearn (RandomForest).

Por que existe (e o Lightning NÃO usa isto):
- O caminho ``cross_attention`` (family="lightning") já tem tracking nativo: o
  ``LightningTrainer`` (FASE 4) passa um ``lightning.pytorch.loggers.WandbLogger``
  ao ``L.Trainer`` (como na branch ``matheus``), que loga *loss*/*acc*/*F1*/*AP*
  por época automaticamente. Esta classe é o equivalente para o caminho
  ``random_forest``, que roda **100% em CPU, sem Lightning e sem torchmetrics** —
  logo precisa logar manualmente via ``wandb.init`` / ``wandb.log``.

Contrato de tolerância (regra da síntese):
- W&B é o tracker **primário**, mas **opcional**. Se o pacote ``wandb`` não estiver
  instalado, se não houver API key, ou se ``wandb.mode == "disabled"``, o ``WandbRun``
  vira um *no-op* silencioso. O **Reporter local (sempre roda)** garante os artefatos
  do paper independentemente do W&B.
- ``wandb.mode``: ``online`` (sobe p/ SaaS) · ``offline`` (grava em ``wandb/`` p/ sync
  posterior) · ``disabled`` (no-op). Sobe **apenas métricas/curvas/embeddings** — nunca
  áudio/vídeo bruto (OK com a EULA do desafio).

Uso (SklearnTrainer / pipeline, FASE 4/6):
    with WandbRun.from_config(cfg, run_name="rf-librosa") as run:
        run.log_metrics({"val/macro_f1": 0.61, "val/average_precision": 0.58})
        run.log_pr_curve(y_true, y_proba)
        run.log_confusion_matrix(y_true, y_pred, class_names=["sem_AH", "com_AH"])
        run.log_artifact("outputs/random_forest/.../bundle.joblib", kind="model")
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.logger import get_logger

log = get_logger("outputs.wandb")


class WandbRun(AbstractContextManager):
    """Wrapper tolerante a ausência sobre uma run do W&B (caminho sklearn).

    Implementa o protocolo de *context manager*: o ``__exit__`` sempre fecha a run
    (``wandb.finish``), mesmo em erro. Qualquer falha de import/init degrada para
    *no-op* — o pipeline nunca quebra por causa do tracker.

    Attributes:
        project: Nome do projeto W&B (``cfg.wandb.project``).
        mode: ``online`` | ``offline`` | ``disabled``.
        run_name: Nome legível da run (ex.: derivado dos hiperparâmetros).
        active: True somente se o ``wandb.init`` teve sucesso e ``mode != disabled``.
    """

    def __init__(
        self,
        project: str,
        mode: str = "online",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ):
        """Inicializa (mas ainda não abre) a run.

        Args:
            project: Projeto W&B (``cfg.wandb.project``, ex.: "abaw-ah").
            mode: "online" | "offline" | "disabled".
            run_name: Nome da run; se None, o W&B gera um aleatório.
            config: Snapshot da config (logado como hiperparâmetros).
            group: Agrupa runs (ex.: "multirun" nos sweeps Hydra+joblib).
            tags: Tags livres (ex.: ["random_forest", "librosa"]).
        """
        self.project = project
        self.mode = (mode or "online").lower()
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.tags = tags or []
        self.active = False
        self._wandb = None
        self._run = None

    # --------------------------------------------------------------------------
    # Construção a partir da config Hydra
    # --------------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> WandbRun:
        """Constrói o ``WandbRun`` a partir do nó ``cfg.wandb`` (README §7).

        Lê ``cfg.wandb.project`` e ``cfg.wandb.mode``. O snapshot completo de ``cfg``
        é serializado (resolvido) e logado como config da run.

        Args:
            cfg: Config Hydra raiz (contém ``cfg.wandb`` e ``cfg.experiment_name``).
            run_name: Nome da run; se None, deriva do modelo + embedders.
            group: Grupo de runs (use "multirun" nos sweeps ``-m``).
            tags: Tags adicionais.

        Returns:
            ``WandbRun`` pronto para ``with``.
        """
        wb = cfg.get("wandb", {})
        project = wb.get("project", cfg.get("experiment_name", "abaw-ah"))
        mode = wb.get("mode", "online")

        if run_name is None:
            model_name = cfg.get("model", {}).get("name", "model")
            audio = cfg.get("audio_embedder", {}).get("backend", "audio")
            run_name = f"{model_name}-{audio}"

        config = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        return cls(
            project=project,
            mode=mode,
            run_name=run_name,
            config=config,  # type: ignore[arg-type]
            group=group,
            tags=tags,
        )

    # --------------------------------------------------------------------------
    # Ciclo de vida (context manager)
    # --------------------------------------------------------------------------

    def __enter__(self) -> WandbRun:
        """Abre a run. Falhas (sem wandb, sem key, disabled) caem em no-op."""
        if self.mode == "disabled":
            log.info("W&B desabilitado (wandb.mode=disabled); usando Reporter local.")
            return self
        try:
            import wandb  # import lazy — wandb é dependência core, mas tolerável se faltar
        except ImportError:
            log.warning("Pacote 'wandb' ausente; tracking remoto desligado (Reporter local roda).")
            return self
        try:
            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                group=self.group,
                tags=self.tags,
                mode=self.mode,            # online | offline
                config=self.config,
                reinit=True,
            )
            self.active = True
            log.info(f"W&B run iniciada: project='{self.project}' name='{self.run_name}' mode='{self.mode}'")
        except Exception as exc:  # noqa: BLE001 — nunca derrubar o pipeline por causa do tracker
            log.warning(f"Falha ao iniciar W&B ({exc}); seguindo só com Reporter local.")
            self._run = None
            self.active = False
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Fecha a run (sempre). Retorna False p/ não suprimir exceções do bloco."""
        if self.active and self._wandb is not None:
            try:
                self._wandb.finish()
                log.info("W&B run finalizada.")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Falha ao finalizar W&B: {exc}")
        return False

    # --------------------------------------------------------------------------
    # Logging (no-op se a run não está ativa)
    # --------------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Loga um dict de métricas escalares (no-op se inativo).

        Convenção de nomes: ``split/metric`` (ex.: ``val/macro_f1``,
        ``test/average_precision``) p/ casar com o ``WandbLogger`` do Lightning.

        Args:
            metrics: Mapa nome→valor (floats).
            step: Passo opcional (ex.: índice de fold/threshold).
        """
        if not self.active or self._run is None:
            return
        try:
            self._run.log({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_metrics falhou: {exc}")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Grava métricas finais no ``run.summary`` (no-op se inativo)."""
        if not self.active or self._run is None:
            return
        try:
            for k, v in summary.items():
                self._run.summary[k] = v
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_summary falhou: {exc}")

    def log_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """Loga a curva Precision–Recall (classe positiva A/H) como objeto W&B."""
        if not self.active or self._wandb is None:
            return
        try:
            y_true = np.asarray(y_true).astype(int)
            y_proba = np.asarray(y_proba, dtype=float)
            # probs (n, 2) p/ a API de curvas do W&B
            probs = np.stack([1.0 - y_proba, y_proba], axis=1)
            self._run.log(
                {
                    "pr_curve": self._wandb.plot.pr_curve(
                        y_true, probs, labels=["sem_AH", "com_AH"]
                    )
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_pr_curve falhou: {exc}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        """Loga a matriz de confusão a nível de vídeo como objeto W&B."""
        if not self.active or self._wandb is None:
            return
        try:
            self._run.log(
                {
                    "confusion_matrix_video": self._wandb.plot.confusion_matrix(
                        y_true=np.asarray(y_true).astype(int).tolist(),
                        preds=np.asarray(y_pred).astype(int).tolist(),
                        class_names=class_names or ["sem_AH", "com_AH"],
                    )
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_confusion_matrix falhou: {exc}")

    def log_image(self, key: str, path: str | Path) -> None:
        """Sobe um PNG (ex.: plot do Reporter) como imagem da run."""
        if not self.active or self._wandb is None:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            self._run.log({key: self._wandb.Image(str(p))})
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_image falhou: {exc}")

    def log_artifact(self, path: str | Path, kind: str = "model", name: str | None = None) -> None:
        """Sobe um artefato (bundle joblib, .ckpt, submissão) como W&B Artifact.

        Args:
            path: Caminho do arquivo (ou diretório do run).
            kind: Tipo do artefato ("model" | "submission" | "report").
            name: Nome do artefato; default = nome do arquivo.
        """
        if not self.active or self._wandb is None:
            return
        p = Path(path)
        if not p.exists():
            log.warning(f"Artefato inexistente, ignorado: {p}")
            return
        try:
            art = self._wandb.Artifact(name or p.stem, type=kind)
            if p.is_dir():
                art.add_dir(str(p))
            else:
                art.add_file(str(p))
            self._run.log_artifact(art)
            log.info(f"Artefato W&B logado: {p} (type={kind})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"W&B log_artifact falhou: {exc}")
