"""EXP-002 — reanálise do protocolo oficial com as estratégias de limiar (só leitura).

Recalcula o que não sai pronto dos artefatos do lote de 2026-10-14: as métricas de val de
cada seed (do ``main.log``), a média de cada estratégia de limiar nos 5 ensembles, as
comparações pareadas no test (entre ensembles e entre estratégias) e, só como diagnóstico,
o limiar ótimo do próprio test.

Uso — num checkout com o PR #7 (``feat/moe-fusion``: ``paired_bootstrap_ap``), com os
outputs baixados do bucket numa pasta qualquer (sem os pesos):

    uv run python experiments_notebooks/analises/002_oficial_reanalise.py <pasta dos outputs>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.protocol import paired_bootstrap_ap, paired_bootstrap_macro_f1
from src.training.aggregation import threshold_curve
from src.training.metrics import video_macro_f1

ENSEMBLES = {
    "exp0 artigo": "oficial-exp0",
    "exp1 MoE": "oficial-exp1",
    "exp2 artigo+vídeo": "oficial-exp2",
    "exp3 MoE+feat.artigo": "oficial-exp3",
    "exp4 exp0+exp1": "oficial-exp4",
}
STRATEGIES = ("fixo_0.5", "argmax", "smooth", "base_rate", "bootstrap")
VAL = re.compile(r"Macro-F1\(val\)=([0-9.]+) \| AP\(val\)=([0-9.]+) \| limiar=([0-9]+\.[0-9]+)")


def _ci(r: dict) -> str:
    """``Δ [IC 95%]`` de um resultado de bootstrap pareado."""
    return f"{r['observed_diff']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="pasta com oficial/ e ensemble/ baixados do bucket")
    root = parser.parse_args().root

    print("Seeds (val, fim do treino; limiar salvo = base_rate):")
    for exp in ("r1", "exp0", "exp1", "exp2", "exp3"):
        f1s, aps, epochs, thrs = [], [], [], []
        for seed_dir in sorted((root / "oficial" / exp).glob("s*")):
            run = sorted(seed_dir.glob("*/"))[-1]
            f1, ap, thr = VAL.search((run / "main.log").read_text()).groups()
            ckpt = json.loads((run / "trainer_state.json").read_text()).get("ckpt_path") or ""
            epochs.append(int(re.search(r"epoch=(\d+)", Path(ckpt).name).group(1)))
            f1s.append(float(f1))
            aps.append(float(ap))
            thrs.append(float(thr))
        print(
            f"  {exp:<5} F1 val {np.mean(f1s):.4f}±{np.std(f1s):.4f}  AP val {np.mean(aps):.4f}"
            f"±{np.std(aps):.4f}  melhor época {epochs}  limiar {min(thrs):.4f}–{max(thrs):.4f}"
        )

    preds, thr = {}, {}
    for name, ens in ENSEMBLES.items():
        d = root / "ensemble" / ens / "eval_test"
        preds[name] = pd.read_csv(d / "predictions.csv").set_index("video_id").sort_index()
        strategies = json.loads((d / "thresholds.json").read_text())["strategies"]
        thr[name] = {r["strategy"]: r["threshold"] for r in strategies}
    y = preds["exp0 artigo"]["y_true"].to_numpy()

    print("\nLimiar ótimo do PRÓPRIO test (diagnóstico — não implantável):")
    for name, p in preds.items():
        grid, f1s = threshold_curve(y, p["y_proba"].to_numpy())
        i = int(np.argmax(f1s))
        print(f"  {name:<22} τ={grid[i]:.2f} Macro-F1={f1s[i]:.4f}")

    def binar(name: str, strategy: str) -> np.ndarray:
        return (preds[name]["y_proba"].to_numpy() >= thr[name][strategy]).astype(float)

    print("\nMédia de cada estratégia nos 5 ensembles (Macro-F1 no test):")
    for s in STRATEGIES:
        vals = [video_macro_f1(y, binar(n, s).astype(int)) for n in ENSEMBLES]
        print(f"  {s:<10} {np.mean(vals):.4f}")

    def pair(a: str, sa: str, b: str, sb: str) -> str:
        r = paired_bootstrap_macro_f1(y, binar(a, sa), binar(b, sb), threshold=0.5, n_boot=2000)
        return f"ΔF1={_ci(r)} p={r['p_value']:.3f}"

    print("\nEnsembles vs exp0 (bootstrap pareado 2000):")
    ref = preds["exp0 artigo"]["y_proba"].to_numpy()
    for name in list(ENSEMBLES)[1:]:
        a = paired_bootstrap_ap(y, preds[name]["y_proba"].to_numpy(), ref, n_boot=2000)
        print(
            f"  {name:<22} smooth {pair(name, 'smooth', 'exp0 artigo', 'smooth')} | fixo "
            f"{pair(name, 'fixo_0.5', 'exp0 artigo', 'fixo_0.5')} | ΔAP={_ci(a)}"
        )

    print("\nEstratégias vs fixo_0.5 no mesmo ensemble (bootstrap pareado 2000):")
    for name in ENSEMBLES:
        cells = [
            f"{s} {pair(name, s, name, 'fixo_0.5')}" for s in ("smooth", "base_rate", "bootstrap")
        ]
        print(f"  {name:<22} " + " | ".join(cells))


if __name__ == "__main__":
    main()
