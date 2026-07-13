#!/usr/bin/env python3
"""Ensemble de seeds do HeteroGAT (hetero_gnn_v2_tune) + calibração smooth.

Uso::

    uv run python scripts/gnn_seed_ensemble.py all --device cuda
    uv run python scripts/gnn_seed_ensemble.py all --experiment hetero_gnn_v2_luiz_mil --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 1, 2, 3, 4, 5, 6]
OUT_FAMILY = "hetero_gnn_contrastive"


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _latest_run_dir() -> Path:
    runs = sorted((ROOT / "outputs" / OUT_FAMILY).glob("2*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"Nenhum run em outputs/{OUT_FAMILY}/")
    return runs[-1]


def _paths(experiment: str) -> tuple[Path, str]:
    tag = experiment.replace("/", "_")
    manifest = ROOT / "outputs" / OUT_FAMILY / f"{tag}_manifest.txt"
    ensemble_name = f"{tag}_seeds"
    return manifest, ensemble_name


def _write_ensemble_yaml(ensemble_name: str, checkpoints: list[str]) -> None:
    out = ROOT / "configs" / "ensemble" / f"{ensemble_name}.yaml"
    lines = [
        f"# Gerado por gnn_seed_ensemble.py — {ensemble_name}",
        "combine: mean",
        "members:",
    ]
    for ckpt in checkpoints:
        lines.append(f"  - model: {OUT_FAMILY}")
        lines.append(f"    checkpoint: {ckpt}")
        lines.append("    weight: 1.0")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_train(experiment: str, seeds: list[int], device: str) -> None:
    manifest, ensemble_name = _paths(experiment)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("", encoding="utf-8")
    for seed in seeds:
        _run(
            [
                "uv",
                "run",
                "python",
                "main.py",
                f"+experiment={experiment}",
                "mode=train",
                f"seed={seed}",
                f"device={device}",
                "wandb.mode=disabled",
            ]
        )
        run_dir = _latest_run_dir()
        with manifest.open("a", encoding="utf-8") as f:
            f.write(str(run_dir.relative_to(ROOT)) + "\n")
        print(f"seed={seed} -> {run_dir}")
    print(f"Manifest: {manifest} ({len(seeds)} runs) → ensemble={ensemble_name}")


def cmd_evaluate(experiment: str, split: str, device: str, calibration: str) -> None:
    manifest, ensemble_name = _paths(experiment)
    if not manifest.exists() or not manifest.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Manifest vazio: rode 'train' antes ({manifest})")
    checkpoints = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    _write_ensemble_yaml(ensemble_name, checkpoints)
    _run(
        [
            "uv",
            "run",
            "python",
            "main.py",
            f"+experiment={experiment}",
            "mode=ensemble_evaluate",
            f"split={split}",
            f"device={device}",
            f"aggregation.calibration={calibration}",
            f"+ensemble={ensemble_name}",
            "wandb.mode=disabled",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble de seeds HeteroGAT")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--experiment", default="hetero_gnn_v2_tune")
        p.add_argument("--device", default="cuda")

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)

    p_eval = sub.add_parser("evaluate")
    add_common(p_eval)
    p_eval.add_argument("--split", default="test")
    p_eval.add_argument("--calibration", default="smooth")

    p_all = sub.add_parser("all")
    add_common(p_all)
    p_all.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p_all.add_argument("--split", default="test")
    p_all.add_argument("--calibration", default="smooth")

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args.experiment, args.seeds, args.device)
    elif args.cmd == "evaluate":
        cmd_evaluate(args.experiment, args.split, args.device, args.calibration)
    else:
        cmd_train(args.experiment, args.seeds, args.device)
        cmd_evaluate(args.experiment, args.split, args.device, args.calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
