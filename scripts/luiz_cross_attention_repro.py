#!/usr/bin/env python3
"""Reprodução do pipeline Luiz: cross-attention + features de suporte + ensemble de seeds.

O F1 test ~0.7245 documentado na branch luiz veio do **ensemble de 7 seeds**
(cross_attention regularizado) com calibração ``smooth`` na avaliação do ensemble.
Treino single-seed usa ``base_rate``; ensemble-eval usa ``smooth`` (como ENS_CALIB do Makefile luiz).

Uso::

    uv run python scripts/luiz_cross_attention_repro.py all --device cuda
    uv run python scripts/luiz_cross_attention_repro.py all --audio librosa --force-featurize --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 1, 2, 3, 4, 5, 6]


@dataclass(frozen=True)
class Variant:
    name: str
    experiment: str
    featurize: str
    manifest: Path
    ensemble_yaml: str


VARIANTS: dict[str, Variant] = {
    "wav2vec2": Variant(
        name="wav2vec2",
        experiment="cross_attention_luiz_repro",
        featurize="featurize_luiz",
        manifest=ROOT / "outputs" / "cross_attention" / "luiz_ensemble_manifest.txt",
        ensemble_yaml="cross_attention_luiz_seeds",
    ),
    "librosa": Variant(
        name="librosa",
        experiment="cross_attention_luiz_librosa",
        featurize="featurize_luiz_librosa",
        manifest=ROOT / "outputs" / "cross_attention" / "luiz_ensemble_manifest_librosa.txt",
        ensemble_yaml="cross_attention_luiz_seeds_librosa",
    ),
}


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _latest_run_dir() -> Path:
    runs = sorted((ROOT / "outputs" / "cross_attention").glob("2*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("Nenhum run em outputs/cross_attention/")
    return runs[-1]


def _write_ensemble_yaml(variant: Variant, checkpoints: list[str]) -> None:
    out = ROOT / "configs" / "ensemble" / f"{variant.ensemble_yaml}.yaml"
    lines = [
        f"# Gerado por luiz_cross_attention_repro.py ({variant.name}) — não editar à mão.",
        "combine: mean",
        "members:",
    ]
    for ckpt in checkpoints:
        lines.append("  - model: cross_attention")
        lines.append(f"    checkpoint: {ckpt}")
        lines.append("    weight: 1.0")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_train(
    variant: Variant,
    seeds: list[int],
    device: str,
    force_featurize: bool,
) -> None:
    if force_featurize:
        _run(
            [
                "uv",
                "run",
                "python",
                "main.py",
                f"+experiment={variant.featurize}",
                "mode=featurize",
                f"device={device}",
                "+data.force=true",
                "wandb.mode=disabled",
            ]
        )
    variant.manifest.parent.mkdir(parents=True, exist_ok=True)
    variant.manifest.write_text("", encoding="utf-8")
    for seed in seeds:
        _run(
            [
                "uv",
                "run",
                "python",
                "main.py",
                f"+experiment={variant.experiment}",
                "mode=train",
                f"seed={seed}",
                f"device={device}",
                "wandb.mode=disabled",
            ]
        )
        run_dir = _latest_run_dir()
        with variant.manifest.open("a", encoding="utf-8") as f:
            f.write(str(run_dir.relative_to(ROOT)) + "\n")
        print(f"seed={seed} -> {run_dir}")
    print(f"Manifest: {variant.manifest} ({len(seeds)} runs)")


def cmd_evaluate(variant: Variant, split: str, device: str, calibration: str) -> None:
    if not variant.manifest.exists() or not variant.manifest.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Manifest vazio: rode 'train' antes ({variant.manifest})")
    checkpoints = [
        ln.strip() for ln in variant.manifest.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    _write_ensemble_yaml(variant, checkpoints)
    _run(
        [
            "uv",
            "run",
            "python",
            "main.py",
            f"+experiment={variant.experiment}",
            "mode=ensemble_evaluate",
            f"split={split}",
            f"device={device}",
            f"aggregation.calibration={calibration}",
            f"+ensemble={variant.ensemble_yaml}",
            "wandb.mode=disabled",
        ]
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--audio", choices=list(VARIANTS), default="wav2vec2")
    p.add_argument("--device", default="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reprodução cross-attention Luiz")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Treina N seeds e grava manifest")
    _add_common_args(p_train)
    p_train.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p_train.add_argument("--force-featurize", action="store_true")

    p_eval = sub.add_parser("evaluate", help="Ensemble-evaluate (smooth por default)")
    _add_common_args(p_eval)
    p_eval.add_argument("--split", default="test")
    p_eval.add_argument(
        "--calibration",
        default="smooth",
        help="smooth = Luiz ENS_CALIB para ensemble; base_rate p/ single-seed",
    )

    p_all = sub.add_parser("all", help="train + evaluate")
    _add_common_args(p_all)
    p_all.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p_all.add_argument("--split", default="test")
    p_all.add_argument("--force-featurize", action="store_true")

    args = parser.parse_args()
    variant = VARIANTS[args.audio]
    if args.cmd == "train":
        cmd_train(variant, args.seeds, args.device, args.force_featurize)
    elif args.cmd == "evaluate":
        cmd_evaluate(variant, args.split, args.device, args.calibration)
    elif args.cmd == "all":
        cmd_train(variant, args.seeds, args.device, args.force_featurize)
        cmd_evaluate(variant, args.split, args.device, "smooth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
