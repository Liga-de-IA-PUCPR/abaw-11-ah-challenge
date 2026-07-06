"""Geração do arquivo de submissão do BAH A/H Challenge (áudio + texto).

Protocolo do desafio (README §9):
- **Métrica oficial:** Macro-F1 a nível de VÍDEO (média das classes presença/ausência
  de A/H). Também é reportado o AP da classe positiva.
- **Public test:** avaliação local (sklearn / torchmetrics).
- **Private test:** predições enviadas **por e-mail** aos organizadores; **≤ 5 trials
  por semana**, **melhor submissão conta**. Logo: gere, valide o formato localmente
  (``validate_submission``) e só então envie — não gaste trials com arquivos malformados.

Formato:
    Uma linha por vídeo: ``video_id, pred``  (pred ∈ {0, 1}).
    Cabeçalho ``video_id, pred`` por padrão (``with_header=True``).

``predict_from_checkpoint`` **reaproveita o artefato salvo na FASE 5** (bundle joblib
do RF OU ``.ckpt`` + sidecar do cross_attention) p/ gerar a submissão **sem retreinar**,
consumindo o cache de features (``data/processed/text_audio_windows.parquet``, FASE 3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger
from src.outputs.checkpoint import (
    CheckpointBundle,
    load_neural_sidecar,
    load_rf_bundle,
)
from src.training.aggregation import aggregate_to_video

log = get_logger("outputs.submission")


# ==============================================================================
# Escrita / validação do arquivo de submissão
# ==============================================================================


def write_submission(
    video_preds: dict[str, int],
    path: str | Path,
    with_header: bool = True,
) -> Path:
    """Escreve o arquivo de submissão (``video_id, pred``), ordenado por video_id.

    Args:
        video_preds: Mapa ``{video_id: pred}`` (saída de ``aggregate_to_video`` ou
            da predição direta a nível de vídeo do ``cross_attention``).
        path: Caminho de saída (ex.: ``outputs/submission.txt``).
        with_header: Se True, escreve o cabeçalho ``video_id, pred``.

    Returns:
        Caminho do arquivo escrito.

    Raises:
        ValueError: Se alguma predição não estiver em {0, 1}.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if with_header:
        lines.append("video_id, pred")
    for vid in sorted(video_preds):
        pred = int(video_preds[vid])
        if pred not in (0, 1):
            raise ValueError(f"Predição inválida para {vid}: {pred} (esperado 0/1)")
        lines.append(f"{vid}, {pred}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_pos = sum(1 for v in video_preds.values() if int(v) == 1)
    log.info(
        f"Submissão escrita: {out} | {len(video_preds)} vídeos "
        f"({n_pos} positivos, {len(video_preds) - n_pos} negativos)"
    )
    return out


def validate_submission(
    path: str | Path,
    expected_video_ids: set[str] | None = None,
) -> bool:
    """Valida o formato do arquivo de submissão ANTES de enviar por e-mail.

    Checa: cabeçalho, ``video_id, pred`` por linha, pred ∈ {0,1}, sem duplicatas e
    (opcional) cobertura exata do test. ≤ 5 trials/semana → não desperdice trials.

    Args:
        path: Caminho do arquivo de submissão.
        expected_video_ids: Conjunto esperado de ``video_id``; se dado, exige cobertura.

    Returns:
        True se válido; caso contrário registra os erros e retorna False.
    """
    p = Path(path)
    if not p.exists():
        log.error(f"Arquivo não encontrado: {p}")
        return False

    seen: set[str] = set()
    ok = True
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        if i == 0 and line.lower().replace(" ", "") == "video_id,pred":
            continue  # cabeçalho
        parts = [tok.strip() for tok in line.split(",")]
        if len(parts) != 2:
            log.error(f"Linha {i + 1} malformada: {line!r}")
            ok = False
            continue
        vid, pred = parts
        if pred not in ("0", "1"):
            log.error(f"Linha {i + 1}: pred inválido {pred!r} (esperado 0/1)")
            ok = False
        if vid in seen:
            log.error(f"Linha {i + 1}: video_id duplicado {vid!r}")
            ok = False
        seen.add(vid)

    if expected_video_ids is not None:
        missing = expected_video_ids - seen
        extra = seen - expected_video_ids
        if missing:
            log.error(f"Faltam {len(missing)} vídeos (ex.: {sorted(missing)[:3]})")
            ok = False
        if extra:
            log.error(f"{len(extra)} vídeos inesperados (ex.: {sorted(extra)[:3]})")
            ok = False

    log.info(f"Validação da submissão: {'OK' if ok else 'FALHOU'} ({len(seen)} vídeos)")
    return ok


# ==============================================================================
# Reaproveitamento do checkpoint: gerar submissão SEM retreinar (2 famílias)
# ==============================================================================


def predict_from_checkpoint(
    checkpoint: str | Path,
    out_path: str | Path,
    *,
    # caminho RF (janela→agrega):
    X: np.ndarray | None = None,
    window_video_ids: np.ndarray | None = None,
    # caminho neural (sequência por vídeo):
    seq_dataset: Any | None = None,
    device: str = "auto",
    # overrides:
    threshold: float | None = None,
    aggregation_method: str | None = None,
    with_header: bool = True,
) -> dict[str, int]:
    """Gera a submissão a partir de um checkpoint salvo, **sem retreinar**.

    Detecta a família pelo arquivo: ``bundle.joblib`` → RF; ``.ckpt``/``sidecar.json``
    → cross_attention.

    - **RF:** requer ``X`` (n_windows, d) + ``window_video_ids`` (do cache, FASE 3) →
      ``predict_proba`` por janela (scaler do bundle) → ``aggregate_to_video`` (método +
      limiar do bundle) → ``write_submission``.
    - **cross_attention:** requer ``seq_dataset`` (``VideoSequenceDataset`` do split de
      teste, FASE 2/4) → reconstrói a fusão via ``CrossAttentionFusion`` (registry lazy,
      ``model_cfg`` do sidecar) e carrega os pesos do ``.ckpt`` com
      ``CrossAttentionFusion.load_from_checkpoint`` (import LAZY de torch) → 1 logit/vídeo
      → sigmoid > limiar (do sidecar) → ``write_submission``.

    Args:
        checkpoint: Caminho do ``bundle.joblib`` (RF) ou do ``.ckpt`` / diretório do run.
        out_path: Caminho do arquivo de submissão a escrever.
        X: (RF) matriz de features de janela do teste (n_windows, d).
        window_video_ids: (RF) ``video_id`` por janela (n_windows,).
        seq_dataset: (neural) dataset de sequências por vídeo (split test).
        device: (neural) "auto"|"cpu"|"mps"|"cuda" (resolvido via ``resolve_device``).
        threshold: Sobrescreve o limiar do checkpoint.
        aggregation_method: (RF) sobrescreve o método.
        with_header: Repassado a ``write_submission``.

    Returns:
        Mapa ``{video_id: pred}`` (também persistido em ``out_path``).
    """
    ckpt_path = Path(checkpoint)
    is_rf = ckpt_path.name == "bundle.joblib" or (
        ckpt_path.is_dir() and (ckpt_path / "bundle.joblib").exists()
    )

    if is_rf:
        video_preds = _predict_rf(
            load_rf_bundle(ckpt_path), X, window_video_ids, threshold, aggregation_method
        )
    else:
        video_preds = _predict_neural(ckpt_path, seq_dataset, device, threshold)

    write_submission(video_preds, out_path, with_header=with_header)
    log.info(f"Submissão gerada do checkpoint ({'RF' if is_rf else 'neural'}) → {out_path}")
    return video_preds


def _predict_rf(
    bundle: CheckpointBundle,
    X: np.ndarray | None,
    window_video_ids: np.ndarray | None,
    threshold: float | None,
    aggregation_method: str | None,
) -> dict[str, int]:
    """Inferência do caminho RandomForest (janela→agrega→vídeo)."""
    if X is None or window_video_ids is None:
        raise ValueError("Caminho RF exige X e window_video_ids (cache da FASE 3).")
    if X.shape[1] != len(bundle.feature_names):
        raise ValueError(
            f"X tem {X.shape[1]} colunas, mas o bundle espera {len(bundle.feature_names)}. "
            "Recompute o cache (FASE 3) com a mesma config de features (embedder_cfg)."
        )
    thr = bundle.threshold if threshold is None else float(threshold)
    method = bundle.aggregation_method if aggregation_method is None else aggregation_method
    window_proba = bundle.predict_proba(X)
    return aggregate_to_video(
        window_proba=np.asarray(window_proba, dtype=np.float32),
        window_video_ids=np.asarray(window_video_ids),
        method=method,
        threshold=thr,
    )


def _predict_neural(
    ckpt_path: Path,
    seq_dataset: Any | None,
    device: str,
    threshold: float | None,
) -> dict[str, int]:
    """Inferência do caminho cross_attention (1 logit por vídeo; sem agregação)."""
    if seq_dataset is None:
        raise ValueError("Caminho neural exige seq_dataset (VideoSequenceDataset do teste).")

    # imports LAZY — lightning/torch só entram quando o caminho neural é usado.
    import torch
    from torch.utils.data import DataLoader

    from src.conf import resolve_device  # helper modular (README §3)
    from src.data.datasets import collate_sequences  # collate module-level (FASE 2)
    from src.models import create_model  # factory lazy (FASE 4 registry)

    run_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    sidecar = load_neural_sidecar(run_dir)
    thr = sidecar.threshold if threshold is None else float(threshold)

    ckpt_file = ckpt_path
    if ckpt_path.is_dir():
        ckpts = sorted(ckpt_path.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise FileNotFoundError(f"Nenhum .ckpt sob {ckpt_path}")
        ckpt_file = ckpts[0]

    dev = resolve_device(device)
    # ``LitCrossAttention`` é uma classe closure NÃO exportada (FASE 4): não dá p/
    # importar nem chamar ``LitCrossAttention.load_from_checkpoint``. Em vez disso
    # reconstruímos o nn.Module de fusão pela fachada ``CrossAttentionFusion`` (criada
    # via registry lazy com o ``model_cfg`` do sidecar) e carregamos os pesos do .ckpt.
    fusion, _family = create_model("cross_attention", sidecar.model_cfg)
    module = fusion.load_from_checkpoint(str(ckpt_file), map_location=dev)
    module.to(dev).eval()

    # collate_sequences (FASE 2) devolve sequências padded + key_padding_mask + ids.
    loader = DataLoader(seq_dataset, batch_size=8, shuffle=False, collate_fn=collate_sequences)
    video_preds: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            feat_a = batch["audio_seq"].to(dev)  # (B, T, d_audio)
            feat_b = batch["text_seq"].to(dev)  # (B, T, d_text)
            mask = batch["key_padding_mask"].to(dev)  # (B, T) True onde é padding
            logits = module(feat_a, feat_b, key_padding_mask=mask)  # (B, 1)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            for vid, p in zip(batch["video_id"], probs, strict=True):
                video_preds[str(vid)] = int(p >= thr)
    return video_preds


def submission_summary(video_preds: dict[str, int], path: str | Path | None = None) -> dict:
    """Resumo da distribuição de classes da submissão (sanity check)."""
    n = len(video_preds)
    n_pos = sum(1 for v in video_preds.values() if int(v) == 1)
    summary = {
        "n_videos": n,
        "n_positivos": n_pos,
        "n_negativos": n - n_pos,
        "frac_positivos": (n_pos / n) if n else 0.0,
    }
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Resumo da submissão: {summary}")
    return summary
