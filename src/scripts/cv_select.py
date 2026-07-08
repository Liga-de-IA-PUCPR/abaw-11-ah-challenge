"""Seleção de família de modelo via GroupKFold no conjunto de treino (Fase 2).

Avalia cada modelo com GroupKFold agrupado por participante no split de TREINO —
sem tocar o val. Para cada família reporta Macro-F1 médio ± desvio entre os folds,
além do limiar médio calibrado. Use o resultado para escolher 1–2 famílias antes
de qualquer sweep de hiperparâmetros.

Uso:
    python -m src.scripts.cv_select
    python -m src.scripts.cv_select --models xgboost lightgbm random_forest extra_trees
    python -m src.scripts.cv_select --n-splits 5 --agg mean_proba
    python -m src.scripts.cv_select --out cv_results.json
    python -m src.scripts.cv_select --parquet data/processed/text_audio_windows.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.logger import get_logger
from src.training.aggregation import calibrate_threshold
from src.training.splits import group_kfold_indices

log = get_logger("scripts.cv_select")

# Modelos avaliados por default — cross_attention é omitido (família lightning, não sklearn).
DEFAULT_MODELS: list[str] = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "extra_trees",
    "logistic_regression",
    "mlp",
]

DEFAULT_PARQUET = Path("data/processed/text_audio_windows.parquet")
DEFAULT_N_SPLITS = 5
DEFAULT_AGG = "mean_proba"


# ==============================================================================
# Carregamento de dados
# ==============================================================================


def _load_train_features(parquet_path: Path) -> dict[str, Any]:
    """Carrega a matriz de features do Parquet, filtrada para split='train'.

    Returns:
        dict com chaves:
            X              (n_windows, d) float32 — audio ‖ text ‖ tabular
            y_window       (n_windows,) int32  — rótulo por janela
            video_ids      (n_windows,) str    — video_id por janela
            participant_ids (n_windows,) str   — participant_id por janela
            video_labels   dict[str, int]      — rótulo a nível de vídeo
    """
    log.info(f"Carregando features de {parquet_path} (split='train')…")
    df = pl.read_parquet(parquet_path).filter(pl.col("split") == "train")

    if len(df) == 0:
        raise ValueError(
            f"Nenhuma janela com split='train' em {parquet_path}. Rode 'make data' antes."
        )

    audio = np.vstack(df["audio_emb"].to_list()).astype(np.float32)
    text = np.vstack(df["text_emb"].to_list()).astype(np.float32)
    tabular = np.vstack(df["tabular"].to_list()).astype(np.float32)
    X = np.concatenate([audio, text, tabular], axis=1)

    y_window = np.asarray(df["label"].to_list(), dtype=np.int32)
    video_ids = np.asarray(df["id"].to_list())
    participant_ids = np.asarray(df["participant_id"].to_list())

    # Rótulo a nível de vídeo: todas as janelas do mesmo vídeo têm o mesmo video_label.
    vid_lbl = df.select(["id", "video_label"]).unique(subset=["id"])
    video_labels: dict[str, int] = {
        vid: int(lbl)
        for vid, lbl in zip(vid_lbl["id"].to_list(), vid_lbl["video_label"].to_list(), strict=True)
    }

    log.info(
        f"{len(df)} janelas · {len(video_labels)} vídeos · "
        f"d={X.shape[1]} (audio={audio.shape[1]}, text={text.shape[1]}, tab={tabular.shape[1]})"
    )
    return {
        "X": X,
        "y_window": y_window,
        "video_ids": video_ids,
        "participant_ids": participant_ids,
        "video_labels": video_labels,
    }


# ==============================================================================
# Avaliação de um fold
# ==============================================================================


def _eval_fold(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    va_video_ids: np.ndarray,
    va_video_labels: dict[str, int],
    agg_method: str,
) -> tuple[float, float]:
    """Treina o modelo num fold e retorna (macro_f1, limiar_calibrado)."""
    from src.models.registry import create_model

    model, _ = create_model(model_name, {})
    model.fit(X_tr, y_tr)

    proba_va = model.predict_proba(X_va)

    # Calibra limiar no fold val (sem tocar o val externo do dataset).
    threshold, f1 = calibrate_threshold(
        val_proba=proba_va,
        val_video_ids=va_video_ids,
        val_video_labels=va_video_labels,
        method=agg_method,
        selection="smooth",
    )
    return f1, threshold


# ==============================================================================
# Avaliação completa de um modelo (todos os folds)
# ==============================================================================


def cv_evaluate(
    model_name: str,
    data: dict[str, Any],
    n_splits: int = DEFAULT_N_SPLITS,
    agg_method: str = DEFAULT_AGG,
) -> dict[str, Any]:
    """GroupKFold completo para um modelo. Retorna estatísticas por fold.

    O GroupKFold é aplicado a nível de VÍDEO (agrupado por participante), e os
    índices são mapeados para janelas — garantindo que nenhuma janela de um
    participante do fold val apareça no treino.

    Returns:
        dict com mean_f1, std_f1, fold_f1s, mean_threshold, n_splits, model.
    """
    X = data["X"]
    y_window = data["y_window"]
    video_ids = data["video_ids"]
    participant_ids = data["participant_ids"]
    video_labels = data["video_labels"]

    # GroupKFold a nível de vídeo (sem janelas duplicadas entre folds).
    unique_vids = np.unique(video_ids)
    unique_pids = np.array([participant_ids[video_ids == v][0] for v in unique_vids])
    folds = group_kfold_indices(unique_vids, unique_pids, n_splits=n_splits)

    fold_f1s: list[float] = []
    fold_thrs: list[float] = []

    for fold_idx, (tr_vid_idx, va_vid_idx) in enumerate(folds):
        tr_vids = set(unique_vids[tr_vid_idx].tolist())
        va_vids = set(unique_vids[va_vid_idx].tolist())

        # Máscaras de janela por split do fold.
        tr_mask = np.array([v in tr_vids for v in video_ids])
        va_mask = np.array([v in va_vids for v in video_ids])

        X_tr, y_tr = X[tr_mask], y_window[tr_mask]
        X_va = X[va_mask]
        va_video_ids_fold = video_ids[va_mask]

        # Rótulos de vídeo apenas para os vídeos do fold val.
        va_video_labels = {v: video_labels[v] for v in va_vids if v in video_labels}

        f1, thr = _eval_fold(
            model_name, X_tr, y_tr, X_va, va_video_ids_fold, va_video_labels, agg_method
        )
        fold_f1s.append(f1)
        fold_thrs.append(thr)
        log.info(
            f"  [{model_name}] fold {fold_idx + 1}/{n_splits}: "
            f"F1={f1:.4f}  thr={thr:.3f}  "
            f"(treino={len(tr_vids)} vídeos, val={len(va_vids)} vídeos)"
        )

    return {
        "model": model_name,
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "fold_f1s": [float(f) for f in fold_f1s],
        "mean_threshold": float(np.mean(fold_thrs)),
        "n_splits": n_splits,
        "agg_method": agg_method,
    }


# ==============================================================================
# Relatório
# ==============================================================================


def _print_ranking(results: list[dict[str, Any]]) -> None:
    """Imprime a tabela comparativa ordenada por Macro-F1 médio."""
    ranked = sorted(results, key=lambda r: r["mean_f1"], reverse=True)
    print()
    print(f"{'Rank':>4}  {'Modelo':<24}  {'F1 médio':>9}  {'± desvio':>9}  {'thr médio':>9}")
    print("-" * 68)
    for i, r in enumerate(ranked, 1):
        marker = "  ◀ recomendado" if i == 1 else ""
        print(
            f"{i:>4}  {r['model']:<24}  {r['mean_f1']:>9.4f}  "
            f"{r['std_f1']:>9.4f}  {r['mean_threshold']:>9.3f}{marker}"
        )
    print()
    # Detalhe dos folds do melhor modelo.
    best = ranked[0]
    fold_str = "  ".join(f"{f:.4f}" for f in best["fold_f1s"])
    print(f"Folds do melhor ({best['model']}): {fold_str}")
    print()


# ==============================================================================
# CLI
# ==============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fase 2 — seleção de família de modelo via GroupKFold no treino."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=f"Famílias a avaliar (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help=f"Cache de features (default: {DEFAULT_PARQUET}).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Número de folds GroupKFold (default: {DEFAULT_N_SPLITS}).",
    )
    parser.add_argument(
        "--agg",
        default=DEFAULT_AGG,
        choices=["mean_proba", "max_proba", "frac_positive"],
        help=f"Método de agregação janela→vídeo (default: {DEFAULT_AGG}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Salva resultados em JSON (opcional).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(
            f"Cache de features não encontrado: {args.parquet}\n"
            "Rode 'make data' antes de cv_select."
        )

    data = _load_train_features(args.parquet)
    results: list[dict[str, Any]] = []

    for model_name in args.models:
        log.info(f"Avaliando '{model_name}' com {args.n_splits}-fold GroupKFold…")
        try:
            result = cv_evaluate(
                model_name=model_name,
                data=data,
                n_splits=args.n_splits,
                agg_method=args.agg,
            )
        except ImportError as exc:
            log.warning(f"'{model_name}' pulado — dependência ausente: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            log.error(f"'{model_name}' falhou: {exc}")
            continue
        results.append(result)
        log.info(f"  → {model_name}: F1={result['mean_f1']:.4f} ± {result['std_f1']:.4f}")

    if not results:
        log.error("Nenhum modelo avaliado com sucesso.")
        return

    _print_ranking(results)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Resultados salvos em: {args.out}")


if __name__ == "__main__":
    main()
