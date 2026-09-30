"""Protocolo de avaliação honesto — CV agrupada por participante (FASE 0).

Objetivo: parar de sobreajustar ao val de 124 vídeos. Este módulo fornece:

- ``make_group_folds``: 5 folds ``StratifiedGroupKFold`` sobre train+val
  (902 vídeos), agrupados por ``participant_id`` e estratificados pelo
  rótulo binário A/H (``video_label``/``label``). Nenhum participante
  aparece em dois folds.
- ``run_oof``: gera predições **out-of-fold** (OOF) para cada vídeo de
  train+val, a partir de uma *factory* de modelo (treina no fold de
  treino, devolve uma função ``predict_proba``-like para o fold de teste).
- ``macro_f1_at_threshold`` / ``average_precision_score_``: métricas com
  limiar **fixo** (default 0.5) — nada de tuning de limiar no val/test.
- ``paired_bootstrap_macro_f1``: IC por bootstrap pareado (1000
  reamostragens) da diferença de Macro-F1 entre dois vetores de score.
- ``bootstrap_ci_macro_f1``: IC por bootstrap (não pareado) do Macro-F1
  de um único vetor de score.

Regra do protocolo: nenhum hiperparâmetro é escolhido olhando o public
test. O test é medido uma única vez por fase.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.logger import get_logger

log = get_logger("eval.protocol")

# Assinatura de uma factory de modelo: recebe o DataFrame de treino do
# fold e devolve uma função ``predict(df_test) -> np.ndarray`` de scores
# (probabilidade da classe 1) para as linhas do fold de teste.
ModelFactory = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


# ==============================================================================
# Folds agrupados por participante
# ==============================================================================


def make_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    participant_col: str = "participant_id",
    label_col: str = "label",
    seed: int = 42,
) -> np.ndarray:
    """Atribui cada linha de ``df`` a um fold (0..n_splits-1).

    Usa ``StratifiedGroupKFold`` (sklearn): agrupa por
    ``df[participant_col]`` (nenhum participante aparece em dois folds) e
    estratifica por ``df[label_col]`` (proporção A/H preservada por fold).

    Args:
        df: uma linha por vídeo, com colunas ``participant_col`` e
            ``label_col``.
        n_splits: número de folds (default 5).
        participant_col: coluna de agrupamento (id do participante).
        label_col: coluna do rótulo binário A/H (0/1).
        seed: semente do embaralhamento (``shuffle=True``).

    Returns:
        Array ``int`` de tamanho ``len(df)`` com o fold de cada linha.
    """
    if participant_col not in df.columns:
        raise KeyError(f"coluna de participante ausente: {participant_col!r}")
    if label_col not in df.columns:
        raise KeyError(f"coluna de rótulo ausente: {label_col!r}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=np.int64)
    groups = df[participant_col].to_numpy()
    labels = df[label_col].to_numpy()
    for fold_id, (_, test_idx) in enumerate(sgkf.split(df, labels, groups=groups)):
        folds[test_idx] = fold_id

    if (folds < 0).any():
        raise RuntimeError("StratifiedGroupKFold deixou linhas sem fold atribuído")

    # Sanidade: nenhum participante em dois folds.
    per_participant = pd.Series(folds, index=df.index).groupby(df[participant_col]).nunique()
    bad = per_participant[per_participant > 1]
    if len(bad) > 0:
        raise RuntimeError(f"participantes em mais de um fold: {list(bad.index)[:5]}")

    log.info(f"{n_splits} folds agrupados por participante sobre {len(df)} vídeos")
    return folds


# ==============================================================================
# Predições OOF
# ==============================================================================


def run_oof(
    df: pd.DataFrame,
    model_factory: ModelFactory,
    n_splits: int = 5,
    participant_col: str = "participant_id",
    label_col: str = "label",
    seed: int = 42,
    folds: np.ndarray | None = None,
) -> np.ndarray:
    """Gera predições **out-of-fold** para cada vídeo de ``df``.

    Para cada fold: treina ``model_factory`` nas linhas dos outros folds
    (train) e prediz nas linhas do fold atual (held-out). O resultado
    cobre cada vídeo exatamente uma vez, sem vazamento de participante
    entre treino e predição.

    Args:
        df: uma linha por vídeo (índice 0..len(df)-1 ou qualquer índice
            estável; ``iloc`` é usado internamente).
        model_factory: ``df_train -> predict_fn``; ``predict_fn(df_test)``
            devolve um array de scores (proba da classe 1) alinhado com
            ``df_test``.
        n_splits: número de folds (ignorado se ``folds`` for passado).
        participant_col: coluna de agrupamento.
        label_col: coluna do rótulo binário.
        seed: semente (ignorada se ``folds`` for passado).
        folds: folds pré-computados (``make_group_folds``); se ``None``,
            são computados aqui.

    Returns:
        Array ``float`` de tamanho ``len(df)`` com o score OOF de cada
        vídeo (mesma ordem de ``df``).
    """
    if folds is None:
        folds = make_group_folds(
            df,
            n_splits=n_splits,
            participant_col=participant_col,
            label_col=label_col,
            seed=seed,
        )

    oof = np.full(len(df), np.nan, dtype=np.float64)
    unique_folds = sorted(set(int(f) for f in folds))
    for fold_id in unique_folds:
        test_mask = folds == fold_id
        train_mask = ~test_mask
        df_train = df.iloc[train_mask].reset_index(drop=True)
        df_test = df.iloc[test_mask].reset_index(drop=True)

        predict_fn = model_factory(df_train)
        scores = np.asarray(predict_fn(df_test), dtype=np.float64)
        if scores.shape[0] != df_test.shape[0]:
            raise ValueError(
                f"fold {fold_id}: predict_fn devolveu {scores.shape[0]} scores "
                f"para {df_test.shape[0]} linhas de teste"
            )
        oof[test_mask] = scores

    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(f"{missing} vídeos sem predição OOF")

    log.info(f"OOF gerado para {len(df)} vídeos em {len(unique_folds)} folds")
    return oof


# ==============================================================================
# Métricas com limiar fixo
# ==============================================================================


def macro_f1_at_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> float:
    """Macro-F1 (classes 0/1) com limiar **fixo** (default 0.5).

    Nada de busca de limiar no val/test — o limiar é um argumento fixo,
    não um hiperparâmetro otimizado nos dados de avaliação.
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score = np.asarray(list(y_score), dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))


def average_precision_score_(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Average Precision (AP) da classe positiva. 0.0 se só houver uma classe."""
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score = np.asarray(list(y_score), dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        log.warning("AP indefinida (uma única classe verdadeira); retornando 0.0")
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ==============================================================================
# Bootstrap
# ==============================================================================


def bootstrap_ci_macro_f1(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """IC por bootstrap (não pareado) do Macro-F1 de um único vetor de score.

    Reamostra ``(y_true, y_score)`` com reposição ``n_boot`` vezes,
    calcula o Macro-F1 (limiar fixo) em cada reamostra e devolve
    média/percentis.

    Returns:
        Dict com ``mean``, ``std``, ``ci_low``, ``ci_high`` (percentis
        ``(1-ci)/2`` e ``1-(1-ci)/2``).
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score = np.asarray(list(y_score), dtype=np.float64)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = macro_f1_at_threshold(y_true[idx], y_score[idx], threshold=threshold)

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if n_boot > 1 else 0.0,
        "ci_low": float(np.quantile(scores, alpha)),
        "ci_high": float(np.quantile(scores, 1.0 - alpha)),
    }


def paired_bootstrap_macro_f1(
    y_true: Iterable[int],
    y_score_a: Iterable[float],
    y_score_b: Iterable[float],
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """IC por bootstrap **pareado** da diferença de Macro-F1 (A - B).

    A mesma reamostra de índices é usada para ``y_score_a`` e
    ``y_score_b`` (bootstrap pareado), o que controla a variância comum
    entre os dois modelos e é o teste correto para comparar dois membros
    do mesmo conjunto de vídeos.

    Returns:
        Dict com ``mean_diff``, ``std_diff``, ``ci_low``, ``ci_high``
        (da diferença ``f1_a - f1_b``) e ``p_value`` (fração de
        reamostras em que a diferença troca de sinal em relação à
        observada — teste bicaudal aproximado).
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_score_a = np.asarray(list(y_score_a), dtype=np.float64)
    y_score_b = np.asarray(list(y_score_b), dtype=np.float64)
    n = len(y_true)
    if len(y_score_a) != n or len(y_score_b) != n:
        raise ValueError("y_true, y_score_a e y_score_b devem ter o mesmo tamanho")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_a = macro_f1_at_threshold(y_true[idx], y_score_a[idx], threshold=threshold)
        f1_b = macro_f1_at_threshold(y_true[idx], y_score_b[idx], threshold=threshold)
        diffs[b] = f1_a - f1_b

    observed = macro_f1_at_threshold(
        y_true, y_score_a, threshold=threshold
    ) - macro_f1_at_threshold(y_true, y_score_b, threshold=threshold)

    alpha = (1.0 - ci) / 2.0
    # p-value bicaudal: fração de reamostras que cruzam 0 relativo ao sinal observado.
    if observed >= 0:
        p_value = float(2.0 * min((diffs <= 0).mean(), 0.5))
    else:
        p_value = float(2.0 * min((diffs >= 0).mean(), 0.5))

    return {
        "observed_diff": float(observed),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)) if n_boot > 1 else 0.0,
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha)),
        "p_value": p_value,
    }
