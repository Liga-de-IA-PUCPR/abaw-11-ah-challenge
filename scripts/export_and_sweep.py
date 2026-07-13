#!/usr/bin/env python3
"""Exporta predictions.csv (val+test) de N checkpoints e roda weight sweep.

Serve para misturar modelos com parquets diferentes (ex.: CA librosa + GNN wav2vec2)
sem forçar um único Hydra ensemble.

Uso::

    uv run python scripts/export_and_sweep.py \\
      --member ca:+experiment=cross_attention_luiz_librosa:outputs/cross_attention/A \\
      --member ca:+experiment=cross_attention_luiz_librosa:outputs/cross_attention/B \\
      --member gnn:+experiment=hetero_gnn_v2_tune:outputs/hetero_gnn_contrastive/C \\
      --device cuda --group-mean ca --out outputs/ensemble_eval/ca_gnn_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _parse_member(raw: str) -> tuple[str, str, Path]:
    # name:hydra_experiment:checkpoint_dir
    name, experiment, ckpt = raw.split(":", 2)
    if not experiment.startswith("+experiment="):
        experiment = f"+experiment={experiment}"
    return name, experiment, Path(ckpt)


def _export_split(experiment: str, ckpt: Path, split: str, device: str) -> Path:
    out_dir = ckpt / f"eval_{split}"
    pred = out_dir / "predictions.csv"
    if pred.exists():
        print(f"reuse {pred}")
        return pred
    _run(
        [
            "uv",
            "run",
            "python",
            "main.py",
            experiment,
            "mode=evaluate",
            f"split={split}",
            f"checkpoint={ckpt}",
            f"device={device}",
            "wandb.mode=disabled",
        ]
    )
    if not pred.exists():
        raise FileNotFoundError(f"predictions não gerados: {pred}")
    return pred


def _mean_group_preds(csvs: list[Path], out: Path) -> Path:
    """Média das y_proba por video_id (mesmo y_true)."""
    scores: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for path in csvs:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row["video_id"]
                labels[vid] = int(row["y_true"])
                scores[vid].append(float(row["y_proba"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "y_true", "y_proba"])
        w.writeheader()
        for vid in sorted(scores):
            w.writerow(
                {
                    "video_id": vid,
                    "y_true": labels[vid],
                    "y_proba": sum(scores[vid]) / len(scores[vid]),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="name:+experiment=...:checkpoint_dir",
    )
    parser.add_argument(
        "--group-mean",
        action="append",
        default=[],
        help="Agrupa membros com mesmo nome via média (ex.: --group-mean ca)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("outputs/ensemble_eval/ca_gnn_sweep.json"))
    args = parser.parse_args()

    members = [_parse_member(m) for m in args.member]
    exported: list[tuple[str, Path, Path]] = []
    for name, experiment, ckpt in members:
        ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
        val_csv = _export_split(experiment, ckpt, "val", args.device)
        test_csv = _export_split(experiment, ckpt, "test", args.device)
        exported.append((name, val_csv, test_csv))

    # Agrupa por nome se --group-mean
    by_name: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for name, val_csv, test_csv in exported:
        by_name[name].append((val_csv, test_csv))

    sweep_members: list[str] = []
    work = ROOT / "outputs" / "ensemble_eval" / "_export"
    work.mkdir(parents=True, exist_ok=True)
    for name, pairs in by_name.items():
        if name in args.group_mean and len(pairs) > 1:
            val_m = _mean_group_preds([p[0] for p in pairs], work / f"{name}_val_mean.csv")
            test_m = _mean_group_preds([p[1] for p in pairs], work / f"{name}_test_mean.csv")
            sweep_members.append(f"{name}:{val_m}:{test_m}")
        else:
            for i, (val_csv, test_csv) in enumerate(pairs):
                tag = name if len(pairs) == 1 else f"{name}{i}"
                sweep_members.append(f"{tag}:{val_csv}:{test_csv}")

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/ensemble_sweep.py",
        "--step",
        str(args.step),
        "--out",
        str(args.out),
    ]
    for m in sweep_members:
        cmd.extend(["--member", m])
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
