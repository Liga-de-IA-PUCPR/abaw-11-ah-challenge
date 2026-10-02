#!/usr/bin/env python3
"""Tabela das ablações (modelos do Rodrigo × pré-processamento do Luiz) + ensembles offline.

Lê ``<run>/eval_{val,test}/predictions.csv`` (gerados por ``make ablate``) de cada célula em
``outputs/ablations/<célula>/manifest.txt`` e das referências fixas (CA × 5 do artigo e o
GNN wav2vec2 do meta-router). Cada sistema = média das probas dos seus seeds.

Por que estas colunas (``references/ablation_plan.md``): o +1 pt do ensemble CA+GNN e o
0.7454 do meta-router vieram de prever MAIS positivos no test-525 (60.6% positivos), o que
não transfere para o teste externo (~47%). Então, além do F1 no limiar calibrado na val,
todo sistema é comparado no MESMO número de positivos da referência (``F1@k``, k = quantos
positivos a CA × 5 prevê) e no AP (sem limiar), com IC por bootstrap pareado vs a referência.

Uso::

    uv run python scripts/ablation_report.py                      # células isoladas
    uv run python scripts/ablation_report.py --combos             # + ensembles CA5 (+GNN) + célula
    uv run python scripts/ablation_report.py --combo ca5+A3+B3 --combo ca5:2+B3 --emit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.protocol import average_precision_score_, paired_bootstrap_macro_f1  # noqa: E402
from src.training.aggregation import calibrate_threshold_from_video_scores  # noqa: E402

W2V_PARQUET = "data/processed/text_audio_windows_w2v.parquet"
CA_MANIFEST = ROOT / "outputs/cross_attention/ensemble_manifest.txt"
GNN_RUN = ROOT / "outputs/hetero_gnn_contrastive/20260713_162753"


@dataclass
class System:
    name: str
    runs: list[Path]
    note: str = ""
    parquet: str | None = None  # Parquet dos membros sem parquet_path no trainer_state
    weight: float = 1.0
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # split → {vid: p}
    seed_test: list[dict[str, float]] = field(default_factory=list)  # probas de cada seed


# ==============================================================================
# Carregamento
# ==============================================================================


def _read_preds(run: Path, split: str) -> tuple[dict[str, float], dict[str, int]]:
    path = run / f"eval_{split}" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} (rode `make eval-run RUN_DIR={run} ...`)")
    proba, label = {}, {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            proba[row["video_id"]] = float(row["y_proba"])
            label[row["video_id"]] = int(row["y_true"])
    return proba, label


def _load(system: System, labels: dict[str, dict[str, int]], splits=("val", "test")) -> System:
    """Média das probas dos seeds por vídeo (só os vídeos presentes em todos os seeds)."""
    for split in splits:
        per_run = []
        for run in system.runs:
            proba, lab = _read_preds(run, split)
            labels.setdefault(split, {}).update(lab)
            per_run.append(proba)
        ids = sorted(set.intersection(*(set(p) for p in per_run)))
        system.scores[split] = {v: float(np.mean([p[v] for p in per_run])) for v in ids}
        if split == "test":
            system.seed_test = per_run
    return system


def _restrict_to_common_videos(systems: list[System]) -> None:
    """Todos os sistemas no MESMO conjunto de vídeos por split (comparação justa, k comum)."""
    for split in ("val", "test"):
        common = set.intersection(*(set(s.scores[split]) for s in systems))
        if any(len(s.scores[split]) != len(common) for s in systems):
            print(
                f"[aviso] {split}: sistemas cobrem vídeos diferentes; usando {len(common)} comuns",
                file=sys.stderr,
            )
        for s in systems:
            s.scores[split] = {v: p for v, p in s.scores[split].items() if v in common}
        if split == "test":
            for s in systems:
                s.seed_test = [{v: p for v, p in d.items() if v in common} for d in s.seed_test]


def _default_refs() -> list[System]:
    refs = []
    if CA_MANIFEST.exists():
        runs = [ROOT / ln.strip() for ln in CA_MANIFEST.read_text().splitlines() if ln.strip()]
        refs.append(System("ca5", runs, "CA × 5 seeds do artigo (librosa + 74 sup. token+MIL)"))
    if GNN_RUN.exists():
        refs.append(System("gnn_w2v", [GNN_RUN], "HeteroGAT wav2vec2 do meta-router", W2V_PARQUET))
    return refs


def _cells(root: Path) -> list[System]:
    cells = []
    for manifest in sorted(root.glob("*/manifest.txt")):
        cell_dir = manifest.parent
        runs = [ROOT / ln.split()[1] for ln in manifest.read_text().splitlines() if ln.strip()]
        note_file = cell_dir / "overrides.txt"
        note = note_file.read_text().strip() if note_file.exists() else ""
        if runs:
            cells.append(System(cell_dir.name, runs, note))
    return cells


# ==============================================================================
# Métricas
# ==============================================================================


def _aligned(scores: dict[str, float], labels: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    ids = _ids(scores, labels)
    return np.array([labels[v] for v in ids]), np.array([scores[v] for v in ids])


def _ids(scores: dict[str, float], labels: dict[str, int]) -> list[str]:
    return sorted(v for v in scores if v in labels)


def _f1_top_k_curve(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Macro-F1 prevendo positivos os top-k scores, para k = 0..n (vetorizado)."""
    y_sorted = y[np.argsort(-s, kind="stable")]
    pos, n = int(y.sum()), len(y)
    k = np.arange(n + 1)
    tp = np.concatenate([[0], np.cumsum(y_sorted)])
    fp, fn = k - tp, pos - tp
    tn = (n - pos) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_pos + f1_neg) / 2


def _top_k(s: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(s))
    pred[np.argsort(-s, kind="stable")[:k]] = 1.0
    return pred


def _evaluate(sys_val: dict[str, float], sys_test: dict[str, float], labels) -> dict:
    y_v, s_v = _aligned(sys_val, labels["val"])
    y_t, s_t = _aligned(sys_test, labels["test"])
    thr, _ = calibrate_threshold_from_video_scores(s_v, y_v, selection="smooth")
    k_thr = int((s_t >= thr).sum())
    curve = _f1_top_k_curve(y_t, s_t)  # curve[k] = F1 com os top-k scores positivos
    return {
        "ids": _ids(sys_test, labels["test"]),
        "y": y_t,
        "s": s_t,
        "ap": average_precision_score_(y_t, s_t),
        "auc": float(roc_auc_score(y_t, s_t)),
        "thr": float(thr),
        "f1_thr": float(curve[k_thr]),
        "k_thr": k_thr,
        "pos_rate": k_thr / len(s_t),
        "curve": curve,
        "val_prev": float(y_v.mean()),
    }


def _ranks(scores: dict[str, float]) -> dict[str, float]:
    ids = list(scores)
    order = np.argsort(np.argsort([scores[v] for v in ids]))
    return {v: float(r) / max(len(ids) - 1, 1) for v, r in zip(ids, order, strict=True)}


def _combine(systems: list[System], split: str, rank: bool) -> dict[str, float]:
    parts = [(_ranks(s.scores[split]) if rank else s.scores[split], s.weight) for s in systems]
    ids = sorted(set.intersection(*(set(p) for p, _ in parts)))
    w_sum = sum(w for _, w in parts)
    return {v: sum(w * p[v] for p, w in parts) / w_sum for v in ids}


# ==============================================================================
# Relatório
# ==============================================================================


def _row(name: str, m: dict, *, n_seeds: int, k_ref: int, ref: dict | None, seed_aps) -> dict:
    f1_k = float(m["curve"][k_ref])
    row = {
        "sistema": name,
        "seeds": n_seeds,
        "AP": m["ap"],
        "AP/seed": f"{np.mean(seed_aps):.4f}±{np.std(seed_aps):.4f}" if len(seed_aps) > 1 else "",
        "AUC": m["auc"],
        "F1@thr_val": m["f1_thr"],
        "%pos": f"{100 * m['pos_rate']:.1f}",
        f"F1@k={k_ref}": f1_k,
        "teto F1": float(m["curve"].max()),
        "Δ vs ref [IC95]": "",
    }
    if ref is not None and ref is not m and ref["ids"] == m["ids"]:
        boot = paired_bootstrap_macro_f1(
            m["y"], _top_k(m["s"], k_ref), _top_k(ref["s"], k_ref), threshold=0.5
        )
        row["Δ vs ref [IC95]"] = (
            f"{boot['observed_diff']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"
        )
    return row


def _markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _member_parquet(run: Path, fallback: str | None) -> str | None:
    state = run / "trainer_state.json"
    pq = json.loads(state.read_text()).get("parquet_path") if state.exists() else None
    return pq or fallback


def _ensemble_command(name: str, systems: list[System]) -> str:
    """Comando ``main.py`` que reproduz a média (pesos por sistema, cada membro no seu cache)."""
    members, weights = [], []
    for s in systems:
        for run in s.runs:
            rel = run.relative_to(ROOT) if run.is_relative_to(ROOT) else run
            pq = _member_parquet(run, s.parquet)
            members.append(f"{{checkpoint:{rel},parquet_path:{pq}}}" if pq else str(rel))
            weights.append(f"{s.weight / len(s.runs):.6g}")
    return (
        "uv run python main.py mode=evaluate split=test +experiment=cross_attention "
        "aggregation.calibration=smooth "
        f'"ensemble=[{",".join(members)}]" "ensemble_weights=[{",".join(weights)}]" '
        f"+ensemble_name={name.replace('+', '_').replace(':', 'w')}"
    )


def _parse_combo(spec: str, by_name: dict[str, System]) -> list[System]:
    out = []
    for tok in spec.split("+"):
        name, _, w = tok.partition(":")
        if name not in by_name:
            raise SystemExit(f"--combo {spec!r}: sistema {name!r} inexistente ({sorted(by_name)})")
        base = by_name[name]
        out.append(
            System(base.name, base.runs, base.note, base.parquet, float(w or 1.0), base.scores)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "outputs/ablations")
    ap.add_argument("--ref", default="ca5", help="sistema de referência p/ k e Δ (default ca5)")
    ap.add_argument("--combos", action="store_true", help="ensembles padrão: CA5(+GNN) + célula")
    ap.add_argument("--combo", action="append", default=[], help="ex.: ca5+A3+B3 ou ca5:2+B3")
    ap.add_argument("--rank", action="store_true", help="média de RANKS em vez de probas")
    ap.add_argument("--emit", action="store_true", help="imprime o comando main.py de cada combo")
    args = ap.parse_args()

    labels: dict[str, dict[str, int]] = {}
    refs, cells = _default_refs(), _cells(args.root)
    systems = [_load(s, labels) for s in refs + cells]
    if not systems:
        raise SystemExit(f"Nada para reportar: sem referências nem células em {args.root}.")
    _restrict_to_common_videos(systems)
    by_name = {s.name: s for s in systems}

    metrics = {s.name: _evaluate(s.scores["val"], s.scores["test"], labels) for s in systems}
    ref = metrics.get(args.ref)
    # Sem a referência: k = prevalência da val aplicada ao test.
    val_prev = metrics[systems[0].name]["val_prev"]
    k_ref = ref["k_thr"] if ref else round(val_prev * len(labels["test"]))

    def seed_aps(s: System) -> list[float]:
        return [average_precision_score_(*_aligned(p, labels["test"])) for p in s.seed_test]

    rows = [
        _row(
            s.name, metrics[s.name], n_seeds=len(s.runs), k_ref=k_ref, ref=ref, seed_aps=seed_aps(s)
        )
        for s in systems
    ]
    out = [
        f"## Sistemas isolados (test: {len(metrics[systems[0].name]['y'])} vídeos; limiar smooth"
        f" na val; k = positivos da ref '{args.ref}')",
        "",
        _markdown(rows),
    ]

    combos = list(args.combo)
    if args.combos:
        cell_names = [c.name for c in cells]
        combos = (
            (["ca5+gnn_w2v"] if {"ca5", "gnn_w2v"} <= set(by_name) else [])
            + [f"ca5+{c}" for c in cell_names]
            + [f"ca5+gnn_w2v+{c}" for c in cell_names if "gnn_w2v" in by_name]
            + combos
        )
    if combos:
        crow, cmds = [], []
        for spec in combos:
            members = _parse_combo(spec, by_name)
            val = _combine(members, "val", args.rank)
            test = _combine(members, "test", args.rank)
            m = _evaluate(val, test, labels)
            n = sum(len(s.runs) for s in members)
            crow.append(_row(spec, m, n_seeds=n, k_ref=k_ref, ref=ref, seed_aps=[]))
            cmds.append(f"# {spec}\n{_ensemble_command(spec, members)}")
        kind = "ranks" if args.rank else "probas"
        out += ["", f"## Ensembles (média de {kind}, peso igual por sistema)", "", _markdown(crow)]
        if args.emit:
            out += ["", "## Comandos main.py (reproduz/submete pelo caminho canônico)", "", *cmds]

    out += ["", "## Legenda", ""] + [
        f"- **{s.name}** ({len(s.runs)} run): {s.note}" for s in systems
    ]
    text = "\n".join(out)
    print(text)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "report.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n→ {args.root / 'report.md'}")


if __name__ == "__main__":
    main()
