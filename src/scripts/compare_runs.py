"""Tabela comparativa de todos os runs treinados/avaliados.

Lê ``train_result.json`` (gerado automaticamente no treino) e
``eval_val/metrics.json`` (gerado pelo ``mode=evaluate``). Para cada run exibe
Macro-F1, AP, limiar, janela, embedders (texto/áudio), modelo e timestamp,
ordenados pelo Macro-F1.

Uso:
    python -m src.scripts.compare_runs              # raiz do projeto
    python -m src.scripts.compare_runs --root outputs
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _context(d: dict) -> tuple[str, str, str]:
    """Extrai (janela, text_embedder, audio_embedder) de um registro de run.

    Dois formatos possíveis:
    - ``eval_val/metrics.json``: ``config`` é o cfg Hydra COMPLETO (chaves
      aninhadas ``data.window.*``, ``text_embedder.name``, ``audio_embedder.name``).
    - ``train_result.json``: ``config`` é só ``cfg.model``; janela/embedders vêm
      de chaves de topo ``window``/``text_embedder``/``audio_embedder`` (ver
      ``main.py::_run_train``). Runs treinados ANTES dessa mudança não têm essas
      chaves — aparecem como "?" (não quebram o script).
    """
    cfg = d.get("config") or {}
    if isinstance(cfg, dict) and "data" in cfg:
        window = cfg.get("data", {}).get("window", {}) or {}
        text_name = (cfg.get("text_embedder") or {}).get("name", "?")
        audio_name = (cfg.get("audio_embedder") or {}).get("name", "?")
    else:
        window = d.get("window") or {}
        text_name = d.get("text_embedder", "?")
        audio_name = d.get("audio_embedder", "?")

    win_label = f"{window['size_s']:g}s" if isinstance(window, dict) and "size_s" in window else "?"
    return win_label, text_name or "?", audio_name or "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Diretório raiz dos runs")
    args = parser.parse_args()
    root = args.root

    # eval_val/metrics.json tem precedência (métricas mais completas + plots).
    eval_files = glob.glob(f"{root}/*/*/eval_val/metrics.json")
    train_files = glob.glob(f"{root}/*/*/train_result.json")

    eval_keys = {(Path(f).parts[-4], Path(f).parts[-3]) for f in eval_files}

    rows = []
    for f in eval_files:
        p = Path(f)
        d = _load(f)
        m = d.get("metrics", d)
        win, txt, aud = _context(d)
        rows.append(
            (
                m.get("macro_f1", 0),
                p.parts[-4],
                p.parts[-3],
                m.get("average_precision", 0),
                m.get("threshold", "?"),
                "eval",
                win,
                txt,
                aud,
            )
        )

    for f in train_files:
        p = Path(f)
        key = (p.parts[-3], p.parts[-2])
        if key in eval_keys:
            continue
        d = _load(f)
        m = d.get("metrics", d)
        win, txt, aud = _context(d)
        rows.append(
            (
                m.get("macro_f1", 0),
                p.parts[-3],
                p.parts[-2],
                m.get("average_precision", 0),
                m.get("threshold", "?"),
                "train",
                win,
                txt,
                aud,
            )
        )

    if not rows:
        print("Nenhum resultado encontrado — rode 'make train' ou 'make evaluate' primeiro.")
        return

    rows.sort(reverse=True)
    print(
        f"{'Macro-F1':>10}  {'AP':>8}  {'thr':>6}  {'src':>5}  "
        f"{'janela':>7}  {'texto':<15}  {'áudio':<10}  {'modelo':<22}  run"
    )
    print("-" * 130)
    for f1, model, ts, ap, thr, src, win, txt, aud in rows:
        print(
            f"{f1:>10.4f}  {ap:>8.4f}  {str(thr):>6}  {src:>5}  "
            f"{win:>7}  {txt:<15}  {aud:<10}  {model:<22}  {ts}"
        )


if __name__ == "__main__":
    main()
