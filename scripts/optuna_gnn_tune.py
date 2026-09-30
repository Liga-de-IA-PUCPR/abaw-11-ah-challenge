"""Optuna HPO para GNN heterogênea (HeteroGAT / multimodal_hetero_face).

Objetivo do estudo: Macro-F1 na validação (sem vazamento de teste).
Após o estudo, avalia o melhor trial no split test uma vez.

Exemplos::

    # HeteroGAT contrastive (parquet atual, sem face)
    uv run python scripts/optuna_gnn_tune.py --family hetero_gnn --n-trials 25 device=cuda

    # Face + landmarks (requer mode=featurize_face antes)
    uv run python scripts/optuna_gnn_tune.py --family hetero_face --n-trials 20 device=cuda

    # Retomar estudo
    uv run python scripts/optuna_gnn_tune.py --family hetero_gnn --storage sqlite:///outputs/optuna/hetero_gnn.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

# Garante imports de ``main`` / ``src`` quando rodado como script.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _compose_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides)


def _has_face_landmarks(parquet_path: str) -> bool:
    import polars as pl

    schema = pl.scan_parquet(parquet_path).collect_schema()
    return "face_landmarks" in schema.names()


def _ensure_face_features(parquet_path: str, device: str, force: bool) -> None:
    if _has_face_landmarks(parquet_path) and not force:
        return
    import subprocess

    cmd = [
        "uv",
        "run",
        "python",
        str(ROOT / "main.py"),
        "mode=featurize_face",
        f"device={device}",
        "wandb.mode=disabled",
    ]
    if force:
        cmd.append("+data.force=true")
    print(f"[optuna] Rodando featurize_face: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_train_trial(cfg: DictConfig, device) -> tuple[float, Path]:
    """Treina um trial e devolve (val_macro_f1, out_dir)."""
    import src.models  # noqa: F401 — registra modelos

    from main import _as_loader, _build_trainer
    from src.conf import resolve_device, seed_everything
    from src.data.datasets import load_train_val
    from src.outputs.checkpoint import resolve_output_dir

    seed_everything(cfg.seed)
    device = resolve_device(str(device))

    try:
        trainer, family = _build_trainer(cfg, device)
        train_data, val_data = load_train_val(cfg, family=family)
        train_data = _as_loader(cfg, train_data, family, "train")
        val_data = _as_loader(cfg, val_data, family, "val")

        out_dir = resolve_output_dir(cfg.data.paths.output_root, cfg.model.name)
        trainer.output_dir = str(out_dir)
        result = trainer.fit(train_data, val_data)
        val_f1 = float(result["val"]["macro_f1"])
        trainer.save(out_dir)
        return val_f1, out_dir
    finally:
        # Evita vazamento de run W&B entre trials (warning no log + possível lentidão).
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:  # noqa: BLE001
            pass


def _evaluate_test(cfg: DictConfig, device, ckpt_dir: Path) -> dict:
    from main import _as_loader, _build_trainer, _write_eval_report
    from src.conf import resolve_device, seed_everything
    from src.data.datasets import load_split
    from src.training.factory import load_trainer

    seed_everything(cfg.seed)
    device = resolve_device(str(device))
    _, family = _build_trainer(cfg, device)
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)
    data = _as_loader(cfg, load_split(cfg, "test", family=family), family, "test")
    report = trainer.evaluate(data)
    _write_eval_report(cfg, trainer, data, report, "test", ckpt_dir)
    return report


def _base_overrides(family: str, device: str) -> list[str]:
    if family == "hetero_gnn":
        experiment = "+experiment=hetero_gnn_v2_tune"
        model_name = "hetero_gnn_contrastive"
    elif family == "hetero_face":
        experiment = "+experiment=multimodal_hetero_face_v2"
        model_name = "multimodal_hetero_face"
    else:
        raise ValueError(f"family desconhecida: {family}")

    return [
        experiment,
        "mode=train",
        f"device={device}",
        "wandb.mode=disabled",
        "trainer.max_epochs=200",
        "trainer.patience=25",
        "trainer.monitor=val_f1_macro",
        "trainer.mode=max",
        f"model.name={model_name}",
    ]


def _suggest_hetero_gnn(trial: optuna.Trial) -> list[str]:
    gat_layers = 2  # 3 overfitou neste dataset
    hidden = trial.suggest_categorical("hidden_channels", [128, 160])
    heads = trial.suggest_categorical("heads", [4, 8])
    out_ch = trial.suggest_categorical("out_channels", [48, 64, 96])
    dropout = trial.suggest_float("dropout", 0.12, 0.28)
    lr = trial.suggest_float("lr", 2e-4, 8e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.02, 0.05, log=True)
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.05)
    lambda_supcon = trial.suggest_float("lambda_supcon", 0.05, 0.18)
    batch_size = trial.suggest_categorical("batch_size", [24, 32])
    use_tab_enhanced = trial.suggest_categorical("use_tab_enhanced", [False, True])
    tab_pool = trial.suggest_categorical("tab_pool", ["attention", "max", "mean"])

    overrides = [
        f"model.gat_num_layers={gat_layers}",
        f"model.hidden_channels={hidden}",
        f"model.heads={heads}",
        f"model.out_channels={out_ch}",
        f"model.dropout={dropout:.4f}",
        f"model.lr={lr:.6g}",
        f"model.weight_decay={wd:.6g}",
        f"model.loss.type={loss_type}",
        f"model.loss.label_smoothing={label_smooth:.4f}",
        f"model.contrastive.lambda_supcon={lambda_supcon:.4f}",
        f"model.use_tab_enhanced={str(use_tab_enhanced).lower()}",
        f"model.tab_pool={tab_pool}",
        f"data.batch_size={batch_size}",
        "aggregation.calibration=smooth",
    ]
    if loss_type == "focal":
        overrides.append(f"model.loss.gamma={trial.suggest_float('focal_gamma', 1.5, 3.0):.2f}")
    return overrides


def _suggest_hetero_face(trial: optuna.Trial) -> list[str]:
    overrides = _suggest_hetero_gnn(trial)
    temporal_mode = trial.suggest_categorical("face_temporal_mode", ["chain", "gnn4ts"])
    use_velocity = trial.suggest_categorical("face_use_velocity", [False, True])
    top_k = trial.suggest_categorical("face_top_k", [6, 8, 12])
    spatial_out = trial.suggest_categorical("face_spatial_out", [64, 128])
    temporal_out = trial.suggest_categorical("face_temporal_out", [64, 128])
    landmark_stride = trial.suggest_categorical("landmark_stride", [1, 2])
    # batch menor p/ face (mais RAM)
    batch_size = trial.suggest_categorical("face_batch_size", [4, 8, 12])

    overrides.extend(
        [
            f"model.face.temporal_mode={temporal_mode}",
            f"model.face.use_velocity={str(use_velocity).lower()}",
            f"model.face.top_k={top_k}",
            f"model.face.spatial_out={spatial_out}",
            f"model.face.temporal_out={temporal_out}",
            f"model.face.landmark_stride={landmark_stride}",
            f"data.batch_size={batch_size}",
        ]
    )
    return overrides


def _build_objective(family: str, device: str):
    base = _base_overrides(family, device)
    suggest = _suggest_hetero_face if family == "hetero_face" else _suggest_hetero_gnn

    def objective(trial: optuna.Trial) -> float:
        overrides = base + suggest(trial)
        cfg = _compose_cfg(overrides)
        val_f1, out_dir = _run_train_trial(cfg, device)
        trial.set_user_attr("out_dir", str(out_dir))
        trial.set_user_attr("overrides", overrides)
        return val_f1

    return objective


def main() -> int:
    import os

    os.environ.setdefault("WANDB_MODE", "disabled")
    parser = argparse.ArgumentParser(description="Optuna HPO para GNN heterogênea")
    parser.add_argument(
        "--family",
        choices=["hetero_gnn", "hetero_face"],
        default="hetero_gnn",
        help="hetero_gnn = HeteroGAT contrastive; hetero_face = landmarks + GCN temporal",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="ex.: sqlite:///outputs/optuna/hetero_gnn.db")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-featurize-face", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    args, hydra_rest = parser.parse_known_args()

    device = args.device
    if hydra_rest:
        for item in hydra_rest:
            if item.startswith("device="):
                device = item.split("=", 1)[1]

    cfg_probe = _compose_cfg(_base_overrides(args.family, device))
    parquet_path = str(cfg_probe.data.paths.parquet_path)

    if args.family == "hetero_face":
        _ensure_face_features(parquet_path, device, args.force_featurize_face)

    study_name = args.study_name or f"gnn_{args.family}"
    storage = args.storage
    if storage and storage.startswith("sqlite:///"):
        db_path = Path(storage.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(f"[optuna] family={args.family} trials={args.n_trials} storage={storage or 'in-memory'}")
    study.optimize(_build_objective(args.family, device), n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n[optuna] Melhor trial #{best.number}: val F1={best.value:.4f}")
    print(json.dumps(best.params, indent=2))

    out_root = ROOT / "outputs" / "optuna"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"{study_name}_best.json"
    payload = {
        "family": args.family,
        "best_trial": best.number,
        "val_macro_f1": best.value,
        "params": best.params,
        "out_dir": best.user_attrs.get("out_dir"),
        "overrides": best.user_attrs.get("overrides"),
    }

    if not args.skip_test_eval and best.user_attrs.get("out_dir"):
        ckpt_dir = Path(best.user_attrs["out_dir"])
        eval_overrides = _base_overrides(args.family, device) + [
            "mode=evaluate",
            f"checkpoint={ckpt_dir}",
            "split=test",
        ]
        for ov in best.user_attrs.get("overrides", []):
            if ov.startswith("model.") or ov.startswith("data.batch_size"):
                eval_overrides.append(ov)
        eval_cfg = _compose_cfg(eval_overrides)
        test_report = _evaluate_test(eval_cfg, device, ckpt_dir)
        payload["test_macro_f1"] = float(test_report["macro_f1"])
        payload["test_average_precision"] = float(test_report.get("average_precision", 0.0))
        print(f"[optuna] Test F1={payload['test_macro_f1']:.4f}")

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[optuna] Resumo salvo em {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
