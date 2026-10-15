"""EXP-001 — reanálise dos runs OOF e da medição única no test (só leitura).

Recalcula o que não sai pronto dos artefatos do lote de 2026-10-06: o AP médio por dobra
(contra o AP das 5 dobras juntas do ``oof_metrics.json``), o gate R1 × Exp 0 e as
comparações pareadas no test (``pred_test.csv`` de cada run; rótulos do Parquet de janelas).

Uso — num checkout com o PR #7 (``feat/moe-fusion``: ``compare_runs``, ``paired_bootstrap_ap``),
com os outputs baixados do bucket numa pasta qualquer (sem os pesos):

    uv run python experiments_notebooks/analises/001_oof_reanalise.py <pasta dos outputs> \
        [--parquet data/processed/text_audio_windows.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, f1_score

from src.eval.protocol import paired_bootstrap_ap, paired_bootstrap_macro_f1

OOF = {
    "exp0 artigo": "oof/exp0-cross-attention/20261006_201715",
    "r1 texto": "oof/moe-r1-text/20261006_201635",
    "exp1 MoE": "oof/moe-r5-face/20261006_201918",
    "exp2 artigo+vídeo": "oof/cross-attention-video/20261006_201827",
    "exp3 MoE+feat.artigo": "oof/moe-luiz-features/20261006_202016",
}
ROUTES = {
    "roteador exp0+exp1": "route/exp4-baselines/20261006_202255",
    "roteador 4 membros": "route/final-todos/20261006_202305",
}


def _ci(r: dict) -> str:
    """``Δ [IC 95%]`` de um resultado de bootstrap pareado."""
    return f"{r['observed_diff']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"


def main() -> None:
    from src.pipeline.oof import compare_runs  # PR #7

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="pasta com oof/ e route/ baixados do bucket")
    parser.add_argument("--parquet", default="data/processed/text_audio_windows.parquet")
    args = parser.parse_args()

    print("AP médio por dobra × AP das 5 dobras juntas (oof_predictions.csv):")
    for name, rel in OOF.items():
        df = pd.read_csv(args.root / rel / "oof_predictions.csv")
        per_fold = [average_precision_score(g.y_true, g.y_proba) for _, g in df.groupby("fold")]
        pooled = average_precision_score(df.y_true, df.y_proba)
        print(f"  {name:<22} por dobra={np.mean(per_fold):.4f}  juntas={pooled:.4f}")

    r = compare_runs(args.root / OOF["r1 texto"], args.root / OOF["exp0 artigo"], n_boot=1000)
    f, a = r["macro_f1"], r["ap"]
    print(f"\nOOF r1 × exp0: ΔF1={_ci(f)} p={f['p_value']:.3f} | ΔAP={_ci(a)}")

    labels = (
        pl.read_parquet(args.parquet, columns=["id", "split", "video_label"])
        .filter(pl.col("split") == "test")
        .unique("id")
        .to_pandas()
        .set_index("id")["video_label"]
    )
    proba = pd.DataFrame(
        {k: pd.read_csv(args.root / rel / "pred_test.csv").set_index("video_id")["y_proba"]
         for k, rel in {**OOF, **ROUTES}.items()}
    )  # fmt: skip
    proba["média exp0+exp1"] = proba[["exp0 artigo", "exp1 MoE"]].mean(axis=1)
    y = labels.loc[proba.index].to_numpy().astype(int)
    print(f"\nTest: {len(y)} vídeos, {y.mean():.4f} positivos")
    print("Macro-F1@0.5 | AP | positivos previstos")
    for k in proba:
        p = proba[k].to_numpy()
        f1 = f1_score(y, (p >= 0.5).astype(int), average="macro")
        print(f"  {k:<22} {f1:.4f} | {average_precision_score(y, p):.4f} | {np.mean(p >= 0.5):.4f}")

    ref = proba["exp0 artigo"].to_numpy()
    print("\nTest pareado vs exp0 (bootstrap 2000, τ=0.5):")
    for k in proba.columns.drop("exp0 artigo"):
        new = proba[k].to_numpy()
        f = paired_bootstrap_macro_f1(y, new, ref, threshold=0.5, n_boot=2000)
        a = paired_bootstrap_ap(y, new, ref, n_boot=2000)
        print(f"  {k:<22} ΔF1={_ci(f)} p={f['p_value']:.3f} | ΔAP={_ci(a)}")


if __name__ == "__main__":
    main()
