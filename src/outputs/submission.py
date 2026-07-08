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

A geração SEM retreinar é feita pela CLI (``mode=submit``, FASE 6): ela resolve o
checkpoint mais recente via ``resolve_latest_checkpoint``, recarrega o trainer com
``load_trainer`` e chama ``trainer.predict`` → ``write_submission``.
"""

from __future__ import annotations

from pathlib import Path

from src.logger import get_logger

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
